"""Daily immutable snapshots of the official openFDA shortage feed."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .signal_store import connect, initialize


OPENFDA_SHORTAGES_URL = "https://api.fda.gov/drug/shortages.json"


def fetch_shortages(*, timeout: int = 30) -> tuple[list[dict], str]:
    records = []
    last_updated = ""
    skip = 0
    while True:
        query = {"limit": 1000, "skip": skip}
        api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
        if api_key:
            query["api_key"] = api_key
        request = Request(f"{OPENFDA_SHORTAGES_URL}?{urlencode(query)}",
                          headers={"User-Agent": "PULSE-public-signal-monitor/1.0"})
        with urlopen(request, timeout=timeout) as response:  # fixed HTTPS URL
            payload = json.load(response)
        meta = payload.get("meta") or {}
        last_updated = str(meta.get("last_updated") or last_updated)
        page = payload.get("results")
        if not isinstance(page, list):
            raise ValueError("openFDA shortage page has no results list")
        records.extend(page)
        skip += len(page)
        total = int((meta.get("results") or {}).get("total", skip))
        if not page or skip >= total:
            break
        if skip > 25000:
            raise ValueError("openFDA shortage result exceeds supported paging limit")
    return records, last_updated


def _date(value: str) -> str | None:
    for pattern in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def refresh_shortages() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES ('openfda_shortages', ?, 'running')""", (started,)).lastrowid
    try:
        records, source_updated = fetch_shortages()
        if not records or not source_updated:
            raise ValueError("openFDA returned no dated shortage records")
        today = datetime.now(timezone.utc).date().isoformat()
        retrieved = datetime.now(timezone.utc).isoformat()
        rows = []
        for record in records:
            raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            rows.append((today, hashlib.sha256(raw.encode()).hexdigest(),
                         str(record.get("generic_name") or ""),
                         str(record.get("status") or record.get("availability") or ""),
                         _date(str(record.get("change_date") or "")),
                         source_updated, retrieved, raw))
        with connect() as db:
            before = db.total_changes
            db.executemany("""INSERT OR IGNORE INTO shortage_snapshots
                (snapshot_date, row_hash, generic_name, status, change_date,
                 source_last_updated, retrieved_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
            written = db.total_changes - before
            generation_id = db.execute("""INSERT INTO shortage_snapshot_generations
                (snapshot_date, retrieved_at, source_last_updated, row_count)
                VALUES (?, ?, ?, ?)""",
                (today, retrieved, source_updated, len({row[1] for row in rows}))).lastrowid
            db.executemany("""INSERT INTO shortage_snapshot_generation_rows
                (generation_id, row_hash) VALUES (?, ?)""",
                [(generation_id, row_hash) for row_hash in sorted({row[1] for row in rows})])
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (retrieved, written, run_id))
        return {"source": "openfda_shortages", "source_last_updated": source_updated,
                "fetched": len(records), "rows_written": written, "finished_at": retrieved}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
