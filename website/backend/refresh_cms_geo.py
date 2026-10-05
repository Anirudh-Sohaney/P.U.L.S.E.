"""Check the official annual CMS geographic-variation source for nine catalog IDs."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


DATASET_ID = "6219697b-8f6c-4164-bed4-cd9317c58ebc"
API_URL = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"
SOURCE_KIND = "cms_geographic_variation_api"
FIELDS = {
    "medicare_beneficiary_count": "BENES_TOTAL_CNT",
    "medicare_per_capita_spending": "TOT_MDCR_PYMT_PC",
    "inpatient_utilization": "IP_CVRD_STAYS_PER_1000_BENES",
    "outpatient_utilization": "OP_VISITS_PER_1000_BENES",
    "emergency_department_utilization": "ER_VISITS_PER_1000_BENES",
    "home_health_utilization": "HH_VISITS_PER_1000_BENES",
    "post_acute_utilization": "SNF_CVRD_STAYS_PER_1000_BENES",
    "dual_eligible_rate": "BENE_DUAL_PCT",
    "medicare_advantage_participation_rate": "MA_PRTCPTN_RATE",
}


def _url(offset: int, size: int = 100) -> str:
    return API_URL + "?" + urlencode({
        "size": size, "offset": offset,
        "filter[BENE_GEO_DESC]": "National", "filter[BENE_AGE_LVL]": "All",
    })


def fetch_national_rows() -> tuple[list[dict], str]:
    rows: list[dict] = []
    offset = 0
    while True:
        request = Request(_url(offset), headers={
            "User-Agent": "PULSE-public-signal-monitor/1.0", "Accept": "application/json"})
        with urlopen(request, timeout=35) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise ValueError("CMS API returned a non-list response")
        if any(row.get("BENE_GEO_DESC") != "National" or row.get("BENE_AGE_LVL") != "All"
               for row in page):
            raise ValueError("CMS API returned a row outside the national all-age filter")
        rows.extend(page)
        if len(page) < 100:
            break
        offset += len(page)
    if not rows:
        raise ValueError("CMS API returned no national all-age rows")
    return rows, _url(0)


def parse_rows(rows: list[dict]) -> list[tuple[str, str, float]]:
    seen: set[str] = set()
    observations: list[tuple[str, str, float]] = []
    for row in rows:
        if row.get("BENE_GEO_DESC") != "National" or row.get("BENE_AGE_LVL") != "All":
            raise ValueError("CMS API returned a row outside the national all-age filter")
        year = str(row.get("YEAR") or "")
        if len(year) != 4 or not year.isdigit() or int(year) > datetime.now(timezone.utc).year:
            raise ValueError(f"Invalid CMS year: {year!r}")
        if year in seen:
            raise ValueError(f"Duplicate CMS national all-age year: {year}")
        seen.add(year)
        for signal_name, field in FIELDS.items():
            raw = row.get(field)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = float(str(raw).replace(",", ""))
            except ValueError as exc:
                raise ValueError(f"Invalid CMS value for {field} in {year}") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid CMS value for {field} in {year}")
            observations.append((signal_name, f"{year}-12-31", value))
    latest_year = max(seen)
    if not all(any(name == item[0] and item[1][:4] == latest_year for item in observations)
               for name in FIELDS):
        raise ValueError(f"CMS latest year {latest_year} is missing a catalog field")
    return observations


def refresh_cms_geo() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        rows, source_url = fetch_national_rows()
        observations = parse_rows(rows)
        latest_year = max(day[:4] for _, day, _ in observations)
        retrieved = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with connect() as db:
            definitions = {}
            for signal_name in FIELDS:
                definition = db.execute("""SELECT id FROM signal_definitions
                    WHERE signal_origin='model_external_state_feature'
                    AND signal_id=? AND cadence='annual'
                    AND geography_level='national' AND geography_id='US'""",
                    (signal_name,)).fetchone()
                if definition is None:
                    raise ValueError(f"Catalog definition missing for {signal_name}")
                definitions[signal_name] = definition["id"]
            for signal_name, day, value in observations:
                uid = definitions[signal_name]
                latest = db.execute("""SELECT value, source_kind FROM signal_records
                    WHERE signal_uid=? AND observation_date=?
                    ORDER BY source_timestamp DESC, ingested_at DESC LIMIT 1""",
                    (uid, day)).fetchone()
                if (latest and latest["source_kind"] == SOURCE_KIND and
                    math.isclose(latest["value"], value, rel_tol=1e-12, abs_tol=1e-12)):
                    continue
                row_hash = hashlib.sha256(json.dumps(
                    [uid, day, value, retrieved, SOURCE_KIND], separators=(",", ":")
                ).encode()).hexdigest()
                db.execute("""INSERT INTO signal_records
                    (row_hash, signal_uid, signal_date, observation_date, value,
                     source_timestamp, ingested_at, data_quality, missingness,
                     forecast_horizon, source_kind, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'published;publication_time_unknown', '', '', ?, ?)""",
                    (row_hash, uid, day, day, value, retrieved, retrieved,
                     SOURCE_KIND, source_url))
                inserted += 1
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, inserted, run_id))
        return {"source": SOURCE_KIND, "latest_source_year": int(latest_year),
                "catalog_signals": len(FIELDS), "values_seen": len(observations),
                "rows_written": inserted, "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
