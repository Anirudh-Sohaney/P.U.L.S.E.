"""Discover and stage new official CMS Part D geography-by-drug releases."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .config import settings
from .signal_store import connect, initialize


CATALOG_URL = "https://data.cms.gov/data.json"
DATASET_TITLE = "Medicare Part D Prescribers - by Geography and Drug"
LANDING_URL = ("https://data.cms.gov/provider-summary-by-type-of-service/"
               "medicare-part-d-prescribers/medicare-part-d-prescribers-by-geography-and-drug")
SOURCE_KIND = "cms_partd_catalog_source"
REQUIRED_COLUMNS = {"Prscrbr_Geo_Lvl", "Prscrbr_Geo_Cd", "Gnrc_Name", "Tot_Clms"}


def runtime_source_path() -> Path:
    return Path(settings.DATA_PATH) / "cms_partd_claims.csv.gz"


def runtime_manifest_path() -> Path:
    return Path(settings.DATA_PATH) / "cms_partd_source_manifest.json"


def current_source_path() -> Path:
    return source_snapshot()[0]


def source_snapshot() -> tuple[Path, dict]:
    """Resolve the source and provenance from one committed manifest read."""
    manifest_path = current_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    def verified(path: Path, expected: str | None) -> tuple[Path, dict]:
        if not path.is_file():
            raise FileNotFoundError(f"CMS committed source file is missing: {path}")
        if expected:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("CMS source checksum disagrees with its manifest")
        return path, manifest

    if manifest_path == runtime_manifest_path():
        filename = manifest.get("source_file")
        if filename:
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or not filename.startswith("cms_partd_claims.")
                    or not filename.endswith(".csv.gz")):
                raise ValueError("CMS runtime manifest has an invalid source file")
            path = manifest_path.parent / filename
            filename_hash = filename[len("cms_partd_claims."):-len(".csv.gz")]
            if len(filename_hash) != 64 or any(char not in "0123456789abcdef" for char in filename_hash):
                raise ValueError("CMS runtime source file lacks a SHA-256 name")
            if manifest.get("normalized_sha256") not in (None, filename_hash):
                raise ValueError("CMS source checksum disagrees with its manifest")
            return verified(path, filename_hash)
        legacy = runtime_source_path()
        if not legacy.is_file():
            raise FileNotFoundError("CMS runtime manifest has no matching source file")
        return verified(legacy, manifest.get("normalized_sha256"))
    return verified(Path(settings.CMS_PARTD_SOURCE_PATH), manifest.get("normalized_sha256"))


def current_manifest_path() -> Path:
    path = runtime_manifest_path()
    return path if path.is_file() else Path(settings.CMS_PARTD_SOURCE_MANIFEST_PATH)


def _official_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname == "data.cms.gov" and not parsed.username


def discover_releases() -> dict[int, str]:
    with urlopen(Request(CATALOG_URL, headers={"User-Agent": "PULSE/0.1"}),
                 timeout=30) as response:
        catalog = json.load(response)
    series = [row for row in catalog.get("datasetSeries", [])
              if row.get("title") == DATASET_TITLE]
    if len(series) != 1:
        raise ValueError("CMS catalog has no unique Part D geography-and-drug series")
    releases: dict[int, set[str]] = {}
    for row in catalog.get("dataset", []):
        landing = row.get("landingPage") or {}
        if landing.get("accessURL") != LANDING_URL:
            continue
        periods = row.get("temporal") or []
        if len(periods) != 1:
            continue
        end = periods[0].get("endDate", "")
        if not end.endswith("-12-31"):
            continue
        year = int(end[:4])
        csv_urls = [distribution.get("downloadURL") for distribution in
                    row.get("distribution", [])
                    if distribution.get("mediaType") == "text/csv"
                    and distribution.get("downloadURL")]
        if len(csv_urls) != 1 or not _official_url(csv_urls[0]):
            raise ValueError(f"CMS release {year} lacks a unique official CSV")
        releases.setdefault(year, set()).add(csv_urls[0])
    if not releases:
        raise ValueError("CMS catalog has no Part D geography-and-drug releases")
    for year, urls in releases.items():
        if len(urls) != 1:
            raise ValueError(f"CMS catalog has conflicting releases for {year}")
    return {year: next(iter(urls)) for year, urls in sorted(releases.items())}


def discover_latest_release() -> tuple[int, str]:
    releases = discover_releases()
    year = max(releases)
    return year, releases[year]


def revision_timestamp(url: str) -> str | None:
    """Read the official CSV's modification time without downloading its rows."""
    if not _official_url(url):
        raise ValueError("CMS source URL must use HTTPS on data.cms.gov")
    with urlopen(Request(url, headers={"User-Agent": "PULSE/0.1"}, method="HEAD"),
                 timeout=30) as response:
        header = response.headers.get("Last-Modified")
    if not header:
        return None
    parsed = parsedate_to_datetime(header)
    if parsed.tzinfo is None:
        raise ValueError("CMS Last-Modified timestamp has no timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def download_arkansas_year(url: str, year: int) -> pd.DataFrame:
    if not _official_url(url):
        raise ValueError("CMS source URL must use HTTPS on data.cms.gov")
    totals: defaultdict[str, int] = defaultdict(int)
    rows = 0
    with urlopen(Request(url, headers={"User-Agent": "PULSE/0.1"}), timeout=90) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig", newline=""))
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("CMS CSV schema is missing required columns")
        for item in reader:
            if item["Prscrbr_Geo_Lvl"] != "State" or item["Prscrbr_Geo_Cd"] != "05":
                continue
            drug = item["Gnrc_Name"].strip()
            claims = item["Tot_Clms"].strip()
            if not drug or not claims or not claims.isdecimal():
                raise ValueError("CMS Arkansas row has an invalid generic name or claims count")
            totals[drug] += int(claims)
            rows += 1
    if not rows:
        raise ValueError("CMS CSV has no Arkansas state rows")
    return pd.DataFrame([{"year": year, "state": "arkansas", "drug_key": drug,
                          "demand_claims": count}
                         for drug, count in sorted(totals.items())])


def _stage_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                        dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        return Path(name)
    except Exception:
        os.unlink(name)
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def refresh_cms_partd_source() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        source_path, manifest = source_snapshot()
        existing = pd.read_csv(source_path)
        required = {"year", "state", "drug_key", "demand_claims"}
        if not required.issubset(existing):
            raise ValueError("Existing CMS normalized source has an invalid schema")
        current_year = int(existing["year"].max())
        years = list(manifest.get("years") or [])
        urls = list(manifest.get("source_urls") or [])
        actual_years = sorted(int(year) for year in existing["year"].unique())
        if (len(years) != len(urls) or years != actual_years
                or current_year not in years):
            raise ValueError("CMS source and manifest years disagree")
        releases = discover_releases()
        latest_year = max(releases)
        latest_url = releases[latest_year]
        latest_revision = revision_timestamp(latest_url)
        if latest_year < current_year:
            raise ValueError(f"CMS catalog latest year {latest_year} predates local year {current_year}")
        published_new_years = sorted(year for year in releases if year > current_year)
        verify_current = (current_year in releases and
                          (releases[current_year] != urls[years.index(current_year)]
                           or manifest.get("latest_source_verified_url") != releases[current_year]
                           or (latest_year == current_year and
                               (latest_revision is None or
                                latest_revision != manifest.get(
                                    "latest_source_verified_last_modified")))))
        metadata_changed = (latest_revision is not None
                            and latest_revision != manifest.get("latest_source_last_modified"))
        changed = bool(published_new_years or verify_current or metadata_changed)
        rows_written = 0
        if changed:
            replacements = []
            if verify_current:
                downloaded = download_arkansas_year(releases[current_year], current_year)
                columns = ["year", "state", "drug_key", "demand_claims"]
                prior = existing[existing["year"] == current_year][columns].sort_values(
                    "drug_key").reset_index(drop=True)
                checked = downloaded[columns].sort_values("drug_key").reset_index(drop=True)
                if not prior.equals(checked):
                    replacements.append(downloaded)
                    existing = existing[existing["year"] != current_year]
                urls[years.index(current_year)] = releases[current_year]
            for year in published_new_years:
                replacements.append(download_arkansas_year(releases[year], year))
                years.append(year)
                urls.append(releases[year])
            combined = (pd.concat([existing, *replacements], ignore_index=True)
                        if replacements else existing)
            if combined.duplicated(["year", "drug_key"]).any():
                raise ValueError("CMS normalized source has duplicate drug-year rows")
            if replacements:
                combined = combined.sort_values(["year", "drug_key"])
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
                    combined.to_csv(compressed, index=False)
                source_bytes = buffer.getvalue()
            else:
                # Establish header provenance without changing a source's
                # content hash or creating a duplicate forecast generation.
                source_bytes = source_path.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            committed_source = Path(settings.DATA_PATH) / f"cms_partd_claims.{source_hash}.csv.gz"
            manifest.update({"years": years, "source_urls": urls, "rows": len(combined),
                             "source_file": committed_source.name,
                             "normalized_sha256": source_hash,
                             "normalized_output": str(committed_source),
                             "latest_source_verified_url": latest_url,
                             "latest_source_verified_last_modified": latest_revision})
            if latest_revision is not None:
                manifest["latest_source_last_modified"] = latest_revision
            else:
                manifest.pop("latest_source_last_modified", None)
            staged_manifest = _stage_bytes(runtime_manifest_path(),
                                           (json.dumps(manifest, indent=2) + "\n").encode())
            try:
                staged_source = _stage_bytes(committed_source, source_bytes)
            except Exception:
                staged_manifest.unlink()
                raise
            # The source generation is durable before the manifest names it.
            # If interrupted here, readers keep using the previous manifest.
            try:
                os.replace(staged_source, committed_source)
                _sync_directory(committed_source.parent)
                os.replace(staged_manifest, runtime_manifest_path())
                _sync_directory(runtime_manifest_path().parent)
            finally:
                staged_source.unlink(missing_ok=True)
                staged_manifest.unlink(missing_ok=True)
            rows_written = sum(len(frame) for frame in replacements)
        finished = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (finished, rows_written, run_id))
        return {"source": SOURCE_KIND, "source_year": latest_year,
                "source_url": latest_url, "rows_written": rows_written,
                "finished_at": finished}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
