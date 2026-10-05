"""Detect new official HHS NDC releases without treating them as model inputs."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


SOURCE_NAME = "hhs_ndc_release_catalog"
MANIFEST = Path(__file__).resolve().parents[1] / "catalog" / "hhs_ndc_source.json"
FLIGHT_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,("(?:\\.|[^"\\])*")\]\)')
SHA256 = re.compile(r"sha256:([0-9a-f]{64})\Z")


def parse_release_page(page: str, *, slug: str) -> dict:
    """Read the dataset metadata embedded in HHS's server-rendered page."""
    dataset = None
    for match in FLIGHT_CHUNK.finditer(page):
        chunk = json.loads(match.group(1))
        marker = chunk.find('"dataset":')
        if marker < 0:
            continue
        candidate, _ = json.JSONDecoder().raw_decode(chunk, marker + len('"dataset":'))
        if isinstance(candidate, dict) and candidate.get("slug") == slug:
            dataset = candidate
            break
    if dataset is None:
        raise ValueError("HHS page has no matching server-rendered dataset metadata")
    versions = dataset.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError("HHS NDC dataset has no release versions")
    dated = []
    for release in versions:
        version = release.get("version")
        try:
            dated.append((date.fromisoformat(version), release))
        except (TypeError, ValueError) as exc:
            raise ValueError("HHS NDC release has an invalid version date") from exc
    _, latest = max(dated, key=lambda item: item[0])
    artifacts = [item for item in latest.get("artifacts", [])
                 if item.get("name") == f"{slug}.csv.zip" and item.get("format") == "zip"]
    if len(artifacts) != 1:
        raise ValueError("HHS NDC release lacks a unique ZIP artifact")
    artifact = artifacts[0]
    url = artifact.get("url") or ""
    parsed = urlparse(url)
    prefix = f"/datasets/{slug}/{latest['version']}/dataset/{slug}.csv.zip"
    if (parsed.scheme != "https" or parsed.hostname != "stopendataprod.blob.core.windows.net"
            or parsed.path != prefix or parsed.username or parsed.query or parsed.fragment):
        raise ValueError("HHS NDC ZIP URL has an unexpected origin or path")
    checksum = SHA256.fullmatch(artifact.get("checksum") or "")
    if not checksum:
        raise ValueError("HHS NDC ZIP lacks a SHA-256 checksum")
    return {"version": latest["version"], "zip_url": url,
            "zip_sha256": checksum.group(1)}


def discover_release(*, timeout: int = 30) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    url = manifest["source_url"]
    if url != "https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/":
        raise ValueError("HHS NDC metadata URL changed unexpectedly")
    request = Request(url, headers={"User-Agent": "PULSE-public-source-monitor/1.0"})
    with urlopen(request, timeout=timeout) as response:
        page = response.read(2_000_001)
    if len(page) > 2_000_000:
        raise ValueError("HHS NDC metadata page exceeds 2 MB")
    return parse_release_page(page.decode("utf-8"), slug=manifest["dataset_slug"])


def refresh_hhs_ndc_release() -> dict:
    """Record a failed source check when a new release needs explicit validation."""
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_NAME, started)).lastrowid
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        release = discover_release()
        if (release["version"] != manifest["processed_version"]
                or release["zip_sha256"] != manifest["processed_zip_sha256"]):
            raise RuntimeError(
                f"HHS NDC release {release['version']} ({release['zip_sha256']}) differs "
                "from the processed source; download, validate, and evaluate it before publication")
        finished = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success'
                WHERE id=?""", (finished, run_id))
        return {"source": SOURCE_NAME, "status": "success", "version": release["version"],
                "rows_written": 0}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
