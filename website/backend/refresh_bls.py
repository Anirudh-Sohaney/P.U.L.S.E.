"""Refresh BLS observations and derived catalog signals from the public v1 API.

BLS v1 supplies published month/value pairs without a publication timestamp.
The first retrieval time is therefore the known availability time in PULSE.
Revisions are retained as new rows; unchanged daily fetches add no rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from calendar import monthrange
from datetime import date, datetime, timezone
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


BLS_SERIES = {
    "arkansas_unemployment_rate": "LAUST050000000000003",
    "consumer_price_index_all_items": "CUUR0000SA0",
    "national_unemployment_rate": "LNS14000000",
}
SOURCE_KIND = "bls_public_api_v1"
DERIVED_KIND = "bls_public_api_v1_derived_v2"
INVALID_DERIVED_KIND = "bls_public_api_v1_derived"


def fetch_series(series_id: str, *, timeout: int = 25) -> list[dict]:
    if series_id not in BLS_SERIES.values():
        raise ValueError("Unexpected BLS series ID")
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    rows: list[dict] = []
    current_year = datetime.now(timezone.utc).year
    for start in range(2000, current_year + 1, 10):
        end = min(start + 9, current_year)
        payload_body = json.dumps({"seriesid": [series_id],
                                   "startyear": str(start), "endyear": str(end)}).encode()
        request = Request(url, data=payload_body,
                          headers={"User-Agent": "PULSE-public-signal-monitor/1.0",
                                   "Content-Type": "application/json", "Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS request failed for {series_id}: {payload.get('message')}")
        series = (payload.get("Results") or {}).get("series") or []
        if len(series) != 1 or series[0].get("seriesID") != series_id:
            raise ValueError(f"BLS response omitted requested series {series_id}")
        page = series[0].get("data")
        if not isinstance(page, list) or not page:
            raise ValueError(f"BLS response contains no observations for {series_id}, {start}-{end}")
        rows.extend(page)
    return rows


def parse_months(rows: list[dict]) -> list[tuple[str, float, str]]:
    """Accept only valid monthly observations, preserving preliminary flags."""
    parsed: dict[str, tuple[float, str]] = {}
    for row in rows:
        period = str(row.get("period") or "")
        year = str(row.get("year") or "")
        if len(year) != 4 or not year.isdigit() or len(period) != 3 or not period.startswith("M"):
            continue  # BLS may also include M13 annual averages.
        try:
            month = int(period[1:])
            if not 1 <= month <= 12:
                continue
            value = float(str(row.get("value") or "").replace(",", ""))
            if not math.isfinite(value):
                continue
            end = date(int(year), month, monthrange(int(year), month)[1])
        except (ValueError, TypeError):
            continue
        if end > datetime.now(timezone.utc).date():
            continue
        footnotes = row.get("footnotes") or []
        preliminary = any(str(note.get("code") or "").upper() == "P"
                          for note in footnotes if isinstance(note, dict))
        day = end.isoformat()
        if day in parsed and parsed[day][0] != value:
            raise ValueError(f"BLS returned conflicting values for {day}")
        parsed[day] = (value, "preliminary" if preliminary else "published")
    return [(day, *parsed[day]) for day in sorted(parsed)]


def derive_months(base_name: str, months: list[tuple[str, float, str]]) -> dict[str, list[tuple[str, float, str]]]:
    """Match the catalog's preceding-observation shifts and rolling formulas."""
    months = sorted(months)
    result: dict[str, list[tuple[str, float, str]]] = {base_name: months}
    for width in (1, 4, 12):
        result[f"{base_name}_lag_{width}"] = []
        result[f"{base_name}_change_{width}"] = []
    for width in (4, 12):
        result[f"{base_name}_rolling_mean_{width}"] = []
        result[f"{base_name}_rolling_std_{width}"] = []
    result[f"{base_name}_seasonal_baseline"] = []
    result[f"{base_name}_seasonal_anomaly"] = []
    prior_by_month: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for index, (day, value, quality) in enumerate(months):
        for width in (1, 4, 12):
            if index < width:
                continue
            _, prior_value, prior_quality = months[index - width]
            combined = "preliminary" if "preliminary" in (quality, prior_quality) else "published"
            result[f"{base_name}_lag_{width}"].append((day, prior_value, combined))
            result[f"{base_name}_change_{width}"].append((day, value - prior_value, combined))
        for width in (4, 12):
            if index < width:
                continue
            window = months[index - width:index]
            values = [item[1] for item in window]
            combined = "preliminary" if any(item[2] == "preliminary" for item in window) else "published"
            result[f"{base_name}_rolling_mean_{width}"].append((day, statistics.mean(values), combined))
            result[f"{base_name}_rolling_std_{width}"].append((day, statistics.pstdev(values), combined))
        prior = prior_by_month[day[5:7]]
        if prior:
            baseline = statistics.mean(item[0] for item in prior)
            combined = "preliminary" if quality == "preliminary" or any(
                item[1] == "preliminary" for item in prior) else "published"
            result[f"{base_name}_seasonal_baseline"].append((day, baseline, combined))
            result[f"{base_name}_seasonal_anomaly"].append((day, value - baseline, combined))
        prior.append((value, quality))
    return result


def refresh_bls() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        fetched: dict[str, list[tuple[str, float, str]]] = {}
        for signal_name, series_id in BLS_SERIES.items():
            months = parse_months(fetch_series(series_id))
            if not months:
                raise ValueError(f"BLS has no valid monthly values for {series_id}")
            fetched.update(derive_months(signal_name, months))
        retrieved = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with connect() as db:
            # The earlier calendar-offset formula did not match catalog
            # semantics. These rows were generated by this adapter only.
            db.execute("DELETE FROM signal_records WHERE source_kind=?",
                       (INVALID_DERIVED_KIND,))
            for signal_name, months in fetched.items():
                definition = db.execute("""SELECT id FROM signal_definitions
                    WHERE signal_origin='model_external_state_feature'
                      AND signal_id=? AND cadence='monthly'""",
                    (signal_name,)).fetchone()
                if definition is None:
                    raise ValueError(f"Catalog definition missing for {signal_name}")
                uid = definition["id"]
                kind = SOURCE_KIND if signal_name in BLS_SERIES else DERIVED_KIND
                base = next(name for name in BLS_SERIES if signal_name == name or signal_name.startswith(name + "_"))
                source_url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
                db.execute("""UPDATE signal_records SET source_url=?
                    WHERE signal_uid=? AND source_kind LIKE 'bls_public_api_v1%'
                      AND source_url=''""", (source_url, uid))
                for day, value, quality in months:
                    latest = db.execute("""SELECT value, source_kind FROM signal_records
                        WHERE signal_uid=? AND observation_date=?
                        ORDER BY source_timestamp DESC, ingested_at DESC LIMIT 1""",
                        (uid, day)).fetchone()
                    if latest and latest["source_kind"] == kind and math.isclose(latest["value"], value, rel_tol=1e-12, abs_tol=1e-12):
                        continue
                    row_hash = hashlib.sha256(json.dumps(
                        [uid, day, value, retrieved, kind], separators=(",", ":")
                    ).encode()).hexdigest()
                    db.execute("""INSERT INTO signal_records
                        (row_hash, signal_uid, signal_date, observation_date, value,
                         source_timestamp, ingested_at, data_quality, missingness,
                         forecast_horizon, source_kind, source_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?)""",
                        (row_hash, uid, day, day, value, retrieved, retrieved,
                         quality + ";publication_time_unknown", kind, source_url))
                    inserted += 1
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, inserted, run_id))
        return {"source": SOURCE_KIND, "series": len(BLS_SERIES),
                "catalog_signals": len(fetched),
                "values_computed": sum(map(len, fetched.values())),
                "rows_written": inserted, "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
