"""Refresh Arkansas monthly Medicaid/CHIP measures from the official data API.

The API publishes reporting periods and revision flags, but no row publication
timestamp. The first retrieval time is recorded as the known availability time.
"""

from __future__ import annotations

import hashlib
import json
import math
from calendar import monthrange
from datetime import date, datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


DATASET_ID = "6165f45b-ca93-5bb5-9d06-db29c692a360"
API_URL = f"https://data.medicaid.gov/api/1/datastore/query/{DATASET_ID}/0"
SOURCE_KIND = "medicaid_state_performance_api"
FIELDS = {
    "adult_medicaid_enrollment": "total_adult_medicaid_enrollment",
    "chip_enrollment": "total_chip_enrollment",
    "medicaid_call_center_abandonment_rate": "average_call_center_abandonment_rate",
    "medicaid_call_center_wait_time": "average_call_center_wait_time_minutes",
    "medicaid_chip_child_enrollment": "medicaid_and_chip_child_enrollment",
    "medicaid_chip_new_applications": "new_applications_submitted_to_medicaid_and_chip_agencies",
    "medicaid_chip_total_enrollment": "total_medicaid_and_chip_enrollment",
    "medicaid_enrollment": "total_medicaid_enrollment",
}


def _url(offset: int, limit: int = 500) -> str:
    query = urlencode({
        "conditions[0][property]": "state_abbreviation",
        "conditions[0][value]": "AR",
        "conditions[0][operator]": "=",
        "limit": limit,
        "offset": offset,
    })
    return f"{API_URL}?{query}"


def fetch_arkansas_rows() -> tuple[list[dict], str]:
    rows: list[dict] = []
    count: int | None = None
    offset = 0
    while count is None or offset < count:
        url = _url(offset)
        request = Request(url, headers={"User-Agent": "PULSE-public-signal-monitor/1.0",
                                        "Accept": "application/json"})
        with urlopen(request, timeout=35) as response:
            payload = json.load(response)
        page = payload.get("results")
        page_count = payload.get("count")
        if not isinstance(page, list) or not isinstance(page_count, int) or page_count < 1:
            raise ValueError("Medicaid API returned an invalid or empty page")
        if count is not None and page_count != count:
            raise ValueError("Medicaid API count changed during pagination")
        count = page_count
        if not page:
            raise ValueError("Medicaid API pagination ended before the reported count")
        if any(row.get("state_abbreviation") != "AR" for row in page):
            raise ValueError("Medicaid API returned a non-Arkansas row")
        rows.extend(page)
        offset += len(page)
    if len(rows) != count:
        raise ValueError("Medicaid API result count mismatch")
    return rows, _url(0)


def _period(row: dict) -> str:
    raw = str(row.get("reporting_period") or "")
    if len(raw) != 6 or not raw.isdigit():
        raise ValueError(f"Invalid Medicaid reporting period: {raw!r}")
    year, month = int(raw[:4]), int(raw[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid Medicaid reporting period: {raw!r}")
    end = date(year, month, monthrange(year, month)[1])
    if end > datetime.now(timezone.utc).date():
        raise ValueError(f"Medicaid API returned a future period: {raw}")
    return end.isoformat()


def _rank(row: dict) -> int:
    flag = str(row.get("preliminary_or_updated") or "").upper()
    final = str(row.get("final_report") or "").upper()
    if flag == "U" and final == "Y":
        return 2
    if flag == "U":
        return 1
    if flag == "P":
        return 0
    raise ValueError(f"Unknown Medicaid revision status: {flag!r}")


def select_periods(rows: list[dict]) -> list[tuple[str, dict]]:
    selected: dict[str, tuple[int, dict]] = {}
    for row in rows:
        if row.get("state_abbreviation") != "AR":
            raise ValueError("Medicaid API returned a non-Arkansas row")
        day, rank = _period(row), _rank(row)
        previous = selected.get(day)
        if previous and rank == previous[0] and any(
            row.get(field) != previous[1].get(field) for field in FIELDS.values()
        ):
            raise ValueError(f"Conflicting Medicaid rows for {day} at the same revision level")
        if previous is None or rank > previous[0]:
            selected[day] = (rank, row)
    return [(day, selected[day][1]) for day in sorted(selected)]


def _value(raw: object) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Medicaid numeric value: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid Medicaid numeric value: {raw!r}")
    return value


def refresh_medicaid() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        rows, source_url = fetch_arkansas_rows()
        periods = select_periods(rows)
        if not periods:
            raise ValueError("Medicaid API has no valid Arkansas periods")
        # Parse and validate before any signal write, including the newest row.
        observations = []
        for day, row in periods:
            quality = ("final" if _rank(row) == 2 else
                       "updated" if _rank(row) == 1 else "preliminary")
            for signal_name, field in FIELDS.items():
                if field not in row:
                    raise ValueError(f"Medicaid API omitted {field}")
                value = _value(row[field])
                if value is not None:
                    observations.append((signal_name, day, value, quality))
        latest_day = periods[-1][0]
        if not any(item[1] == latest_day for item in observations):
            raise ValueError("Latest Medicaid reporting period has no usable values")
        retrieved = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with connect() as db:
            definitions = {}
            for signal_name in FIELDS:
                definition = db.execute("""SELECT id FROM signal_definitions
                    WHERE signal_origin='model_external_state_feature'
                      AND signal_id=? AND cadence='monthly'
                      AND geography_level='state' AND geography_id='AR'""",
                    (signal_name,)).fetchone()
                if definition is None:
                    raise ValueError(f"Catalog definition missing for {signal_name}")
                definitions[signal_name] = definition["id"]
            for signal_name, day, value, quality in observations:
                uid = definitions[signal_name]
                latest = db.execute("""SELECT value, source_kind, data_quality
                    FROM signal_records WHERE signal_uid=? AND observation_date=?
                    ORDER BY source_timestamp DESC, ingested_at DESC LIMIT 1""",
                    (uid, day)).fetchone()
                marker = quality + ";publication_time_unknown"
                if (latest and latest["source_kind"] == SOURCE_KIND and
                    math.isclose(latest["value"], value, rel_tol=1e-12, abs_tol=1e-12) and
                    latest["data_quality"] == marker):
                    continue
                row_hash = hashlib.sha256(json.dumps(
                    [uid, day, value, retrieved, quality], separators=(",", ":")
                ).encode()).hexdigest()
                db.execute("""INSERT INTO signal_records
                    (row_hash, signal_uid, signal_date, observation_date, value,
                     source_timestamp, ingested_at, data_quality, missingness,
                     forecast_horizon, source_kind, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?)""",
                    (row_hash, uid, day, day, value, retrieved, retrieved,
                     marker, SOURCE_KIND, source_url))
                inserted += 1
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, inserted, run_id))
        return {"source": SOURCE_KIND, "periods": len(periods),
                "catalog_signals": len(FIELDS), "values_seen": len(observations),
                "rows_written": inserted, "latest_period": latest_day,
                "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
