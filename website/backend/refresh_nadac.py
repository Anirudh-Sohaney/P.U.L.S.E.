"""Capture dated NADAC rate snapshots without collapsing unlike pricing units."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


DATASET_ID = "fbb83258-11c7-47f5-8b18-5f8e79f7e704"
API_URL = f"https://data.medicaid.gov/api/1/datastore/query/{DATASET_ID}/0"
SOURCE_KIND = "medicaid_nadac_2026_api"
PAGE_SIZE = 5000
PROPERTIES = ("ndc", "nadac_per_unit", "pricing_unit",
              "classification_for_rate_setting", "as_of_date")


def _url(*, as_of_date: str | None = None, offset: int = 0,
         limit: int = PAGE_SIZE) -> str:
    query: list[tuple[str, str | int]] = [("limit", limit), ("offset", offset)]
    if as_of_date:
        query.extend((
            ("conditions[0][property]", "as_of_date"),
            ("conditions[0][operator]", "="),
            ("conditions[0][value]", as_of_date),
        ))
    else:
        query.extend((("sorts[0][property]", "as_of_date"),
                      ("sorts[0][order]", "desc")))
    query.extend((f"properties[{index}]", field) for index, field in enumerate(PROPERTIES))
    return API_URL + "?" + urlencode(query)


def _fetch(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "PULSE-public-signal-monitor/1.0",
                                    "Accept": "application/json"})
    with urlopen(request, timeout=45) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or \
            not isinstance(payload.get("count"), int):
        raise ValueError("NADAC API returned an invalid response")
    return payload


def fetch_latest_snapshot() -> tuple[str, list[dict], str]:
    latest_page = _fetch(_url(limit=1))
    if not latest_page["results"]:
        raise ValueError("NADAC API returned no rows")
    as_of = str(latest_page["results"][0].get("as_of_date") or "")
    try:
        source_day = date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(f"Invalid NADAC as_of_date: {as_of!r}") from exc
    if source_day > datetime.now(timezone.utc).date():
        raise ValueError("NADAC API returned a future snapshot date")
    rows: list[dict] = []
    count: int | None = None
    offset = 0
    while count is None or offset < count:
        payload = _fetch(_url(as_of_date=as_of, offset=offset))
        page = payload["results"]
        if count is not None and payload["count"] != count:
            raise ValueError("NADAC snapshot count changed during pagination")
        count = payload["count"]
        if not page or any(row.get("as_of_date") != as_of for row in page):
            raise ValueError("NADAC snapshot pagination is incomplete or mixed")
        rows.extend(page)
        offset += len(page)
    if len(rows) != count:
        raise ValueError("NADAC snapshot row count mismatch")
    return as_of, rows, _url(as_of_date=as_of)


def aggregate_snapshot(rows: list[dict], as_of: str) -> tuple[list[dict], str]:
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "total": Decimal(0), "minimum": None,
                 "maximum": None, "ndcs": set()})
    canonical = []
    for row in rows:
        if row.get("as_of_date") != as_of:
            raise ValueError("NADAC row has an unexpected snapshot date")
        ndc = str(row.get("ndc") or "").strip()
        unit = str(row.get("pricing_unit") or "").strip()
        classification = str(row.get("classification_for_rate_setting") or "").strip()
        try:
            price = Decimal(str(row.get("nadac_per_unit") or ""))
        except InvalidOperation as exc:
            raise ValueError("NADAC snapshot contains a nonnumeric price") from exc
        if not ndc or not unit or not classification or not price.is_finite() or price <= 0:
            raise ValueError("NADAC snapshot contains an invalid rate row")
        key = (classification, unit)
        group = grouped[key]
        group["count"] += 1
        group["total"] += price
        group["minimum"] = min(group["minimum"], price) if group["minimum"] is not None else price
        group["maximum"] = max(group["maximum"], price) if group["maximum"] is not None else price
        group["ndcs"].add(ndc)
        canonical.append((ndc, str(price), classification, unit))
    if not canonical:
        raise ValueError("NADAC snapshot contains no rates")
    groups = []
    for (classification, unit), group in sorted(grouped.items()):
        groups.append({"classification_for_rate_setting": classification,
                       "pricing_unit": unit, "row_count": group["count"],
                       "distinct_ndc_count": len(group["ndcs"]),
                       "nadac_per_unit_min": str(group["minimum"]),
                       "nadac_per_unit_max": str(group["maximum"]),
                       "nadac_per_unit_mean": str(group["total"] / group["count"])})
    digest = hashlib.sha256(json.dumps(sorted(canonical), separators=(",", ":")).encode()).hexdigest()
    return groups, digest


def refresh_nadac() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        as_of, rows, source_url = fetch_latest_snapshot()
        groups, digest = aggregate_snapshot(rows, as_of)
        retrieved = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            before = db.total_changes
            db.execute("""INSERT OR IGNORE INTO nadac_price_snapshots
                (as_of_date, content_hash, retrieved_at, total_rows, groups_json, source_url)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (as_of, digest, retrieved, len(rows), json.dumps(groups), source_url))
            inserted = db.total_changes - before
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, inserted, run_id))
        return {"source": SOURCE_KIND, "as_of_date": as_of,
                "total_rows": len(rows), "groups": len(groups),
                "snapshots_written": inserted, "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
