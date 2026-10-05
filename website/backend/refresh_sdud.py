"""Capture the published Arkansas State Drug Utilization Data without imputation.

This is a quarterly Medicaid prescription count source, not the monthly
pharmacy-provider claim-line target used by the historical ATC model.
Suppressed counts remain NULL. A new snapshot is stored only if source content
changes, preserving revisions and the first time each version was retrieved.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


CATALOG_URL = "https://data.medicaid.gov/api/1/metastore/schemas/dataset/items"
SOURCE_KIND = "medicaid_sdud_arkansas"
PAGE_SIZE = 5000
TITLE_PATTERN = re.compile(r"State Drug Utilization Data (20\d{2})")


@dataclass(frozen=True)
class SourceRelease:
    year: int
    dataset_id: str
    modified_at: str

    @property
    def source_url(self) -> str:
        return f"https://data.medicaid.gov/dataset/{self.dataset_id}"

    @property
    def api_url(self) -> str:
        return f"https://data.medicaid.gov/api/1/datastore/query/{self.dataset_id}/0"


def select_latest_release(catalog: list[dict], *, today_year: int) -> SourceRelease:
    releases: list[SourceRelease] = []
    for item in catalog:
        match = TITLE_PATTERN.fullmatch(str(item.get("title") or ""))
        if not match:
            continue
        year = int(match.group(1))
        if year > today_year:
            continue
        dataset_id = str(item.get("identifier") or "")
        try:
            UUID(dataset_id)
            modified = datetime.fromisoformat(str(item.get("modified") or ""))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"SDUD catalog has invalid metadata for {year}") from exc
        if modified.tzinfo is None or modified.astimezone(timezone.utc) > datetime.now(timezone.utc):
            raise ValueError(f"SDUD catalog has invalid modification time for {year}")
        releases.append(SourceRelease(year, dataset_id, modified.astimezone(timezone.utc).isoformat()))
    if not releases:
        raise ValueError("Medicaid catalog has no dated SDUD release")
    latest_year = max(row.year for row in releases)
    latest = [row for row in releases if row.year == latest_year]
    if len(latest) != 1:
        raise ValueError(f"Medicaid catalog has no unique SDUD release for {latest_year}")
    return latest[0]


def discover_latest_release() -> SourceRelease:
    request = Request(CATALOG_URL, headers={
        "User-Agent": "PULSE-public-signal-monitor/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=40) as response:
        catalog = json.load(response)
    if not isinstance(catalog, list):
        raise ValueError("Medicaid metastore did not return a dataset list")
    return select_latest_release(catalog, today_year=datetime.now(timezone.utc).year)


def _url(release: SourceRelease, offset: int) -> str:
    return release.api_url + "?" + urlencode({
        "conditions[0][property]": "state",
        "conditions[0][value]": "AR",
        "conditions[0][operator]": "=",
        "limit": PAGE_SIZE,
        "offset": offset,
    })


def fetch_rows(release: SourceRelease) -> list[dict]:
    rows: list[dict] = []
    expected: int | None = None
    while expected is None or len(rows) < expected:
        request = Request(_url(release, len(rows)), headers={
            "User-Agent": "PULSE-public-signal-monitor/1.0",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        page, count = payload.get("results"), payload.get("count")
        if not isinstance(page, list) or not isinstance(count, int) or count < 1:
            raise ValueError("SDUD returned an invalid or empty result")
        if expected is not None and expected != count:
            raise ValueError("SDUD source changed during pagination")
        if not page or len(page) > PAGE_SIZE:
            raise ValueError("SDUD pagination ended before its reported count")
        expected = count
        rows.extend(page)
        if len(rows) > expected:
            raise ValueError("SDUD pagination exceeded its reported count")
    return rows


def normalize(rows: list[dict], *, source_year: int) -> list[tuple]:
    if not rows:
        raise ValueError("SDUD has no Arkansas rows")
    output: list[tuple] = []
    keys: set[tuple] = set()
    for row in rows:
        if row.get("state") != "AR":
            raise ValueError("SDUD returned a non-Arkansas row")
        year, quarter = int(row["year"]), int(row["quarter"])
        if year != source_year or quarter not in (1, 2, 3, 4):
            raise ValueError("SDUD returned an unexpected year or quarter")
        utilization = str(row["utilization_type"]).strip()
        ndc = str(row["ndc"]).strip()
        product = str(row["product_name"]).strip()
        if utilization not in ("FFSU", "MCOU") or len(ndc) != 11 or not ndc.isdigit() or not product:
            raise ValueError("SDUD returned an invalid drug identity")
        suppressed_raw = str(row["suppression_used"]).lower()
        if suppressed_raw not in ("true", "false"):
            raise ValueError("SDUD returned an unknown suppression flag")
        suppressed = int(suppressed_raw == "true")
        raw_count = row.get("number_of_prescriptions")
        if suppressed:
            if raw_count not in (None, ""):
                raise ValueError("SDUD suppressed row unexpectedly contains a count")
            prescriptions = None
        else:
            if raw_count is None or not str(raw_count).isdigit():
                raise ValueError("SDUD returned an invalid prescription count")
            prescriptions = int(raw_count)
        key = (year, quarter, utilization, ndc)
        if key in keys:
            raise ValueError("SDUD contains a duplicate NDC/utilization/period")
        keys.add(key)
        output.append((year, quarter, utilization, ndc, product, prescriptions, suppressed))
    return sorted(output)


def refresh_sdud() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        release = discover_latest_release()
        normalized = normalize(fetch_rows(release), source_year=release.year)
        latest_quarter = max(row[1] for row in normalized)
        digest = hashlib.sha256(json.dumps(normalized, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()
        suppressed = sum(row[6] for row in normalized)
        reported = sum(row[5] or 0 for row in normalized)
        retrieved = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            exists = db.execute("SELECT 1 FROM sdud_snapshots WHERE content_hash=?",
                                (digest,)).fetchone()
            written = 0
            if not exists:
                db.execute("""INSERT INTO sdud_snapshots
                    (content_hash, source_year, latest_quarter, retrieved_at, last_seen_at,
                     source_modified_at, source_url,
                     row_count, suppressed_rows, reported_prescriptions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (digest, release.year, latest_quarter, retrieved, retrieved,
                     release.modified_at, release.source_url,
                     len(normalized), suppressed, reported))
                db.executemany("""INSERT INTO sdud_rows
                    (content_hash, year, quarter, utilization_type, ndc,
                     product_name, prescriptions, suppressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(digest, *row) for row in normalized])
                written = len(normalized)
            else:
                db.execute("""UPDATE sdud_snapshots SET last_seen_at=?,
                    source_modified_at=? WHERE content_hash=?""",
                    (retrieved, release.modified_at, digest))
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, written, run_id))
        return {"source": SOURCE_KIND, "source_year": release.year,
                "source_url": release.source_url,
                "source_modified_at": release.modified_at,
                "latest_quarter": latest_quarter,
                "rows": len(normalized), "suppressed_rows": suppressed,
                "reported_prescriptions": reported, "rows_written": written,
                "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
