"""Durable, provenance-preserving signal catalog and observation storage.

Historical imports retain their original timestamps. Missing periods are never
interpolated or represented as observed zeroes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import settings
from .database_locks import exclusive_file_lock


IDENTITY_FIELDS = (
    "signal_origin", "signal_id", "cadence", "geography_level",
    "geography_id", "entity_key",
)
COLLAPSED_DIMENSION_SIGNALS = set()
MAX_HISTORY_SOURCE_ROWS = 10_000
SIGNAL_SCHEMA_VERSION = 3
SEED_DEFINITION_COUNT = 1312
SEED_DEFINITIONS_PATH = (Path(__file__).resolve().parents[1] / "catalog" /
                         "signal_definitions.json")
NEWS_VERSION_FIELDS = ("url_hash", "source_name", "title", "url", "source",
                       "source_timestamp", "timestamp_kind", "relevance_score",
                       "matched_terms", "summary_text")


def news_version_hash(fields: dict) -> str:
    """Content identity for a source-visible article version, excluding capture time."""
    content = {name: fields[name] for name in NEWS_VERSION_FIELDS}
    payload = json.dumps(content, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
PUBLISHABLE_DRUG_SOURCE_KIND = "cms_partd_two_year_persistence_v2"
NEWS_BRIDGE_SHA256 = "7981884ee0f45fc9f73dc777781e80b542d48716550cf1ac6a742c152ee648f6"


class HistoryTooLarge(ValueError):
    """A history request needs a smaller ID batch or date range."""


def database_path() -> Path:
    path = Path(settings.DATA_PATH) / "signals.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    path = database_path()
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        with exclusive_file_lock(path.with_name(path.name + ".journal.lock")):
            if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize() -> None:
    with connect() as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SIGNAL_SCHEMA_VERSION:
        raise RuntimeError(f"Signal database schema {version} is newer than this service")
    if version == SIGNAL_SCHEMA_VERSION:
        return
    path = database_path()
    with exclusive_file_lock(path.with_name(path.name + ".schema.lock")):
        with connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > SIGNAL_SCHEMA_VERSION:
            raise RuntimeError(f"Signal database schema {version} is newer than this service")
        if version == SIGNAL_SCHEMA_VERSION:
            return
        if version < 1:
            _initialize_schema_v1()
        if version < 2:
            _initialize_schema_v2()
        if version < 3:
            _initialize_schema_v3()
        with connect() as db:
            db.execute(f"PRAGMA user_version={SIGNAL_SCHEMA_VERSION}")


def _initialize_schema_v1() -> None:
    """Create the current tables and migrate the unversioned legacy schema."""
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS signal_definitions (
            id TEXT PRIMARY KEY,
            signal_origin TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            cadence TEXT NOT NULL,
            geography_level TEXT NOT NULL,
            geography_id TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            unit TEXT,
            source_name TEXT,
            source_url TEXT,
            state_definition TEXT,
            model_output_type TEXT,
            UNIQUE(signal_origin, signal_id, cadence, geography_level, geography_id, entity_key)
        );
        CREATE TABLE IF NOT EXISTS signal_records (
            row_hash TEXT PRIMARY KEY,
            signal_uid TEXT NOT NULL REFERENCES signal_definitions(id),
            signal_date TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            value REAL NOT NULL,
            source_timestamp TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            data_quality TEXT,
            missingness TEXT,
            forecast_horizon TEXT,
            source_kind TEXT NOT NULL,
            source_url TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_signal_observation_latest
          ON signal_records(signal_uid, observation_date DESC, source_timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_signal_ingested_at
          ON signal_records(ingested_at DESC);
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            rows_written INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS service_heartbeats (
            service_name TEXT PRIMARY KEY,
            checked_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS news_articles (
            url_hash TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            published_at TEXT NOT NULL,
            timestamp_kind TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL,
            relevance_score INTEGER NOT NULL,
            matched_terms TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at DESC);
        CREATE TABLE IF NOT EXISTS news_article_versions (
            version_hash TEXT PRIMARY KEY,
            url_hash TEXT NOT NULL,
            source_name TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            source_timestamp TEXT NOT NULL,
            timestamp_kind TEXT NOT NULL,
            first_observed_at TEXT NOT NULL,
            relevance_score INTEGER NOT NULL,
            matched_terms TEXT NOT NULL,
            summary_text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_news_versions_seen
          ON news_article_versions(first_observed_at, source_name);
        CREATE INDEX IF NOT EXISTS idx_news_versions_url
          ON news_article_versions(url_hash, first_observed_at);
        CREATE TABLE IF NOT EXISTS shortage_snapshots (
            snapshot_date TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            generic_name TEXT NOT NULL,
            status TEXT NOT NULL,
            change_date TEXT,
            source_last_updated TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(snapshot_date, row_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_shortage_change_date ON shortage_snapshots(change_date DESC);
        CREATE TABLE IF NOT EXISTS shortage_snapshot_generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            source_last_updated TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shortage_snapshot_generation_rows (
            generation_id INTEGER NOT NULL REFERENCES shortage_snapshot_generations(id),
            row_hash TEXT NOT NULL,
            PRIMARY KEY(generation_id, row_hash)
        );
        CREATE TABLE IF NOT EXISTS nadac_price_snapshots (
            as_of_date TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            groups_json TEXT NOT NULL,
            source_url TEXT NOT NULL,
            PRIMARY KEY(as_of_date, content_hash)
        );
        CREATE TABLE IF NOT EXISTS demand_forecast_runs (
            run_hash TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            source_year INTEGER NOT NULL,
            target_year INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            method TEXT NOT NULL,
            source_url TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS demand_forecasts (
            run_hash TEXT NOT NULL REFERENCES demand_forecast_runs(run_hash),
            drug_key TEXT NOT NULL,
            demand_state INTEGER NOT NULL CHECK(demand_state BETWEEN 0 AND 4),
            observed_claims REAL NOT NULL,
            PRIMARY KEY(run_hash, drug_key)
        );
        CREATE INDEX IF NOT EXISTS idx_demand_forecasts_drug
          ON demand_forecasts(drug_key);
        CREATE TABLE IF NOT EXISTS sdud_snapshots (
            content_hash TEXT PRIMARY KEY,
            source_year INTEGER NOT NULL,
            latest_quarter INTEGER NOT NULL,
            retrieved_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            source_modified_at TEXT NOT NULL DEFAULT '',
            source_url TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            suppressed_rows INTEGER NOT NULL,
            reported_prescriptions INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sdud_rows (
            content_hash TEXT NOT NULL REFERENCES sdud_snapshots(content_hash),
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL,
            utilization_type TEXT NOT NULL,
            ndc TEXT NOT NULL,
            product_name TEXT NOT NULL,
            prescriptions INTEGER,
            suppressed INTEGER NOT NULL CHECK(suppressed IN (0, 1)),
            PRIMARY KEY(content_hash, year, quarter, utilization_type, ndc)
        );
        CREATE INDEX IF NOT EXISTS idx_sdud_period
          ON sdud_rows(content_hash, year DESC, quarter DESC);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(signal_records)")}
        if "forecast_horizon" not in columns:
            db.execute("ALTER TABLE signal_records ADD COLUMN forecast_horizon TEXT")
        if "source_url" not in columns:
            db.execute("ALTER TABLE signal_records ADD COLUMN source_url TEXT NOT NULL DEFAULT ''")
        news_columns = {row[1] for row in db.execute("PRAGMA table_info(news_articles)")}
        if "timestamp_kind" not in news_columns:
            db.execute("ALTER TABLE news_articles ADD COLUMN timestamp_kind TEXT NOT NULL DEFAULT 'unknown'")
            db.execute("""UPDATE news_articles SET timestamp_kind=CASE
                WHEN source='FDA Drugs RSS' THEN 'rss_pub_date'
                ELSE 'gdelt_first_seen' END""")
        sdud_columns = {row[1] for row in db.execute("PRAGMA table_info(sdud_snapshots)")}
        if "last_seen_at" not in sdud_columns:
            db.execute("ALTER TABLE sdud_snapshots ADD COLUMN last_seen_at TEXT NOT NULL DEFAULT ''")
            db.execute("UPDATE sdud_snapshots SET last_seen_at=retrieved_at")
        if "source_modified_at" not in sdud_columns:
            db.execute("ALTER TABLE sdud_snapshots ADD COLUMN source_modified_at TEXT NOT NULL DEFAULT ''")
        # An early schema used signal_observations. Record counts alone cannot
        # prove every old observation was copied, so retain that table until an
        # explicit row-level migration and verification can remove it.


def _initialize_schema_v2() -> None:
    """Keep exact article and model evidence for each news-model estimate."""
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS news_model_record_provenance (
            row_hash TEXT PRIMARY KEY REFERENCES signal_records(row_hash),
            model_sha256 TEXT NOT NULL,
            input_count INTEGER NOT NULL CHECK(input_count > 0),
            estimate_kind TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS news_model_record_inputs (
            row_hash TEXT NOT NULL REFERENCES news_model_record_provenance(row_hash),
            input_position INTEGER NOT NULL CHECK(input_position >= 0),
            version_hash TEXT NOT NULL REFERENCES news_article_versions(version_hash),
            PRIMARY KEY(row_hash, input_position),
            UNIQUE(row_hash, version_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_news_model_inputs_version
          ON news_model_record_inputs(version_hash);
        """)


def _initialize_schema_v3() -> None:
    """Retain complete FDA enforcement report windows and every source row."""
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS enforcement_snapshot_rows (
            row_hash TEXT PRIMARY KEY,
            recall_number TEXT NOT NULL,
            report_date TEXT NOT NULL,
            classification TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS enforcement_snapshot_generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            source_last_updated TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            row_count INTEGER NOT NULL CHECK(row_count >= 0)
        );
        CREATE TABLE IF NOT EXISTS enforcement_snapshot_generation_rows (
            generation_id INTEGER NOT NULL REFERENCES enforcement_snapshot_generations(id),
            row_hash TEXT NOT NULL REFERENCES enforcement_snapshot_rows(row_hash),
            PRIMARY KEY(generation_id, row_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_enforcement_generation_end
          ON enforcement_snapshot_generations(window_end DESC, retrieved_at DESC);
        """)


def signal_uid(row: dict) -> str:
    parts = [str(row.get(field) or "").strip() for field in IDENTITY_FIELDS]
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return "sig_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def seed_definitions() -> list[dict]:
    """Return the exact, validated 1,312-definition public seed manifest."""
    payload = json.loads(SEED_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    rows = payload.get("definitions")
    if (payload.get("schema") != "pulse_signal_definitions_v1"
            or payload.get("definition_count") != SEED_DEFINITION_COUNT
            or not isinstance(rows, list) or len(rows) != SEED_DEFINITION_COUNT):
        raise ValueError("Signal seed definition manifest is malformed")
    seen: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict)
                or any(field not in row for field in IDENTITY_FIELDS)
                or row.get("id") != signal_uid(row)):
            raise ValueError("Signal seed definition manifest has an invalid stable ID")
        if row["id"] in seen:
            raise ValueError("Signal seed definition manifest contains duplicate IDs")
        seen.add(row["id"])
    return rows


def _timestamp(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def import_definitions(source: Path) -> dict:
    """Seed stable public IDs without asserting historical observations exist."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "pulse_signal_definitions_v1":
        raise ValueError("Unsupported signal definition manifest")
    rows = payload.get("definitions")
    if (not isinstance(rows, list) or len(rows) != 1312
            or len(rows) != payload.get("definition_count")):
        raise ValueError("Signal definition manifest count mismatch")
    definitions: list[tuple] = []
    for row in rows:
        if (not isinstance(row, dict)
                or any(field not in row for field in IDENTITY_FIELDS)
                or row.get("id") != signal_uid(row)):
            raise ValueError("Signal definition manifest contains an invalid stable ID")
        definitions.append((row["id"], *(str(row.get(field) or "") for field in IDENTITY_FIELDS),
                            *(str(row.get(field) or "") for field in
                              ("unit", "source_name", "source_url", "state_definition",
                               "model_output_type"))))
    if len({row[0] for row in definitions}) != len(definitions):
        raise ValueError("Signal definition manifest has duplicate IDs")
    initialize()
    with connect() as db:
        before = db.total_changes
        db.executemany("""INSERT OR IGNORE INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level, geography_id,
             entity_key, unit, source_name, source_url, state_definition, model_output_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", definitions)
        inserted = db.total_changes - before
    return {"definitions_seen": len(definitions), "definitions_inserted": inserted}


def catalog_seed_status() -> dict:
    """Check that the complete checked-in seed manifest is present in the DB."""
    rows = seed_definitions()
    expected_ids = {row["id"] for row in rows}
    expected = len(rows)
    with connect() as db:
        present_ids = {row[0] for row in db.execute("SELECT id FROM signal_definitions")}
    present_ids.intersection_update(expected_ids)
    missing_ids = expected_ids - present_ids
    return {"expected": expected, "present": len(present_ids),
            "missing_ids": sorted(missing_ids)}


def import_news_bridge_history(source: Path) -> dict:
    """Backfill verified historical news features without claiming live availability."""
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEWS_BRIDGE_SHA256:
        raise ValueError("Historical news bridge checksum mismatch")
    frame = pd.read_csv(io.BytesIO(raw), compression="gzip")
    expected_months = pd.period_range("2018-01", "2026-01", freq="M")
    if len(frame) != len(expected_months) or "date" not in frame:
        raise ValueError("Historical news bridge has an unexpected month count")
    months = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if (months.isna().any() or not months.dt.is_month_start.all()
            or not (months.dt.to_period("M").array == expected_months.array).all()):
        raise ValueError("Historical news bridge months are missing or out of order")
    initialize()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        definitions = {row["signal_id"]: row["id"] for row in db.execute("""
            SELECT id, signal_id FROM signal_definitions
            WHERE signal_origin='model_news_output'""")}
        if len(definitions) != 20 or set(frame.columns) != {"date", *definitions}:
            raise ValueError("Historical news bridge signal IDs disagree with the catalog")
        existing: dict[tuple[str, str], list[float]] = {}
        for row in db.execute("""SELECT r.signal_uid, r.observation_date, r.value
            FROM signal_records r JOIN signal_definitions d ON d.id=r.signal_uid
            WHERE d.signal_origin='model_news_output'"""):
            existing.setdefault((row["signal_uid"], row["observation_date"]), []).append(row["value"])
        pending = []
        matched = 0
        for name, uid in definitions.items():
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"Historical news bridge has an invalid value for {name}")
            for month, value in zip(months, values, strict=True):
                day = month.to_period("M").end_time.date().isoformat()
                numeric = float(value)
                prior = existing.get((uid, day), [])
                if prior:
                    if any(not math.isclose(item, numeric, rel_tol=0, abs_tol=1e-9)
                           for item in prior):
                        raise ValueError(f"Historical news bridge conflicts with catalog: {name} {day}")
                    matched += 1
                    continue
                row_hash = hashlib.sha256(json.dumps(
                    ["historical_news_bridge", NEWS_BRIDGE_SHA256, uid, day, numeric],
                    separators=(",", ":")).encode()).hexdigest()
                pending.append((row_hash, uid, day, day, numeric, "", now,
                                "historical_news_bridge;availability_unknown;not_live",
                                "historical_news_bridge"))
        before = db.total_changes
        db.executemany("""INSERT OR IGNORE INTO signal_records
            (row_hash, signal_uid, signal_date, observation_date, value,
             source_timestamp, ingested_at, data_quality, source_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", pending)
        inserted = db.total_changes - before
    return {"source_sha256": NEWS_BRIDGE_SHA256, "periods": len(expected_months),
            "signal_ids": len(definitions), "matched_existing": matched,
            "inserted": inserted}


def catalog_rows(source: Path, *, ingested_at: str | None = None) -> tuple[dict, list[tuple], int]:
    """Normalize source rows once for both import and integrity verification."""
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source, compression="infer", low_memory=False).fillna("")
    required = set(IDENTITY_FIELDS) | {"period_end", "value", "source_timestamp"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Signal catalog missing columns: {', '.join(sorted(missing))}")
    now = ingested_at or datetime.now(timezone.utc).isoformat()
    definitions: dict[str, tuple] = {}
    observations: list[tuple] = []
    rejected = 0
    for row in raw.to_dict("records"):
        day = pd.to_datetime(row["period_end"], errors="coerce")
        value = pd.to_numeric(row["value"], errors="coerce")
        source_time = _timestamp(row["source_timestamp"]) or ""
        if pd.isna(day) or pd.isna(value):
            rejected += 1
            continue
        uid = signal_uid(row)
        identity = tuple(str(row.get(field) or "").strip() for field in IDENTITY_FIELDS)
        definitions[uid] = (uid, *identity, str(row.get("unit") or ""),
                            str(row.get("source_name") or ""), str(row.get("source_url") or ""),
                            str(row.get("state_definition") or ""),
                            str(row.get("model_output_type") or ""))
        signal_day = pd.to_datetime(row.get("signal_date"), errors="coerce")
        row_hash = hashlib.sha256(json.dumps(row, sort_keys=True, default=str,
                                              ensure_ascii=False).encode("utf-8")).hexdigest()
        observations.append((row_hash, uid,
                             signal_day.date().isoformat() if not pd.isna(signal_day) else "",
                             day.date().isoformat(), float(value), source_time,
                             now, str(row.get("data_quality") or ""),
                             str(row.get("missingness") or ""),
                             str(row.get("forecast_horizon") or ""), "historical_catalog",
                             str(row.get("source_url") or "")))
    return definitions, observations, rejected


def backfill_catalog_source_urls(db: sqlite3.Connection, observations: Iterable[tuple]) -> int:
    """Repair only blank historical URLs using rows from a verified source file."""
    before = db.total_changes
    db.executemany("""UPDATE signal_records SET source_url=?
        WHERE row_hash=? AND source_kind='historical_catalog' AND source_url=''""",
        [(row[11], row[0]) for row in observations if row[11]])
    return db.total_changes - before


def import_catalog(source: Path) -> dict:
    """Idempotently import the frozen catalog without changing its provenance."""
    definitions, observations, rejected = catalog_rows(source)
    initialize()
    with connect() as db:
        db.executemany("""INSERT OR IGNORE INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level, geography_id,
             entity_key, unit, source_name, source_url, state_definition, model_output_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", definitions.values())
        before = db.total_changes
        db.executemany("""INSERT OR IGNORE INTO signal_records
            (row_hash, signal_uid, signal_date, observation_date, value,
             source_timestamp, ingested_at, data_quality, missingness,
             forecast_horizon, source_kind, source_url)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            observations)
        inserted = db.total_changes - before
        db.executemany("""UPDATE signal_records SET forecast_horizon=?
            WHERE row_hash=? AND forecast_horizon IS NULL""",
            [(row[9], row[0]) for row in observations])
        backfill_catalog_source_urls(db, observations)
    return {"definitions": len(definitions), "observations_inserted": inserted,
            "observations_seen": len(observations), "rejected_rows": rejected}


def list_signals(*, search: str = "", limit: int = 1500, offset: int = 0) -> list[dict]:
    initialize()
    with connect() as db:
        rows = db.execute("""SELECT d.*, (SELECT MAX(o.observation_date)
            FROM signal_records o WHERE o.signal_uid=d.id
              AND o.source_kind != 'article_text_ridge_shadow_v1') AS latest_observation_date,
            (SELECT MAX(o.observation_date) FROM signal_records o
             WHERE o.signal_uid=d.id
               AND o.source_kind='article_text_ridge_shadow_v1') AS latest_model_target_date,
            (SELECT MAX(o.forecast_horizon) FROM signal_records o
             WHERE o.signal_uid=d.id) AS latest_forecast_horizon
            FROM signal_definitions d
            WHERE (? = '' OR instr(lower(d.signal_id), lower(?)) > 0
                  OR instr(lower(d.entity_key), lower(?)) > 0)
            ORDER BY d.signal_origin, d.signal_id, d.geography_id, d.entity_key
            LIMIT ? OFFSET ?""", (search, search, search, limit, offset)).fetchall()
    return [dict(row) for row in rows]


def values(ids: Iterable[str], *, start: date | None = None, end: date | None = None,
           latest_only: bool = False, include_revisions: bool = False,
           include_unusable: bool = False) -> list[dict]:
    requested = list(dict.fromkeys(ids))
    if not requested:
        return []
    max_ids = 1500 if latest_only else 100
    if len(requested) > max_ids:
        raise ValueError(f"A request may contain at most {max_ids} signal IDs")
    initialize()
    raw_rows = []
    with connect() as db:
        for index in range(0, len(requested), 500):
            batch = requested[index:index + 500]
            placeholders = ",".join("?" for _ in batch)
            filters = [f"o.signal_uid IN ({placeholders})"]
            parameters: list = batch.copy()
            if start:
                filters.append("o.observation_date >= ?")
                parameters.append(start.isoformat())
            if end:
                filters.append("o.observation_date <= ?")
                parameters.append(end.isoformat())
            if latest_only and not include_unusable:
                filters.append("""NOT (
                  (d.signal_id GLOB 'arkansas_atc_demand_state::*' AND o.source_kind = 'historical_catalog') OR
                  (d.signal_origin='model_news_output' AND o.source_kind = 'historical_catalog') OR
                  (d.signal_id GLOB 'cms_part_d_demand_state::*'
                   AND o.source_kind NOT IN (?, 'combined_demand_signals'))
                )""")
                parameters.append(PUBLISHABLE_DRUG_SOURCE_KIND)
            condition = " AND ".join(filters)
            columns = """o.signal_uid AS id, o.signal_date, o.observation_date,
              o.value, o.source_timestamp, o.ingested_at, o.data_quality, o.missingness,
              o.source_kind, o.forecast_horizon,
              CASE WHEN o.source_kind='article_text_ridge_shadow_v1'
                   THEN 'PULSE article-text Ridge shadow model' ELSE d.source_name END AS source_name,
              CASE WHEN o.source_kind='article_text_ridge_shadow_v1'
                   THEN NULL ELSE COALESCE(NULLIF(o.source_url, ''), d.source_url) END AS source_url,
              d.signal_id, d.signal_origin, d.unit, d.entity_key, d.geography_level,
              d.geography_id"""
            if latest_only:
                sql = f"""WITH ranked AS (
                  SELECT {columns}, o.row_hash, DENSE_RANK() OVER (
                    PARTITION BY o.signal_uid ORDER BY
                    CASE WHEN d.signal_origin='derived_demand_output'
                      THEN COALESCE(o.forecast_horizon, '') ELSE o.observation_date END DESC,
                    CASE WHEN d.signal_origin='derived_demand_output'
                      THEN o.observation_date ELSE '' END DESC,
                    o.source_timestamp DESC) AS latest_rank
                  FROM signal_records o JOIN signal_definitions d ON d.id=o.signal_uid
                  WHERE {condition}
                ) SELECT * FROM ranked WHERE latest_rank=1
                  ORDER BY id, ingested_at DESC, row_hash DESC"""
            else:
                sql = f"""SELECT {columns} FROM signal_records o
                  JOIN signal_definitions d ON d.id=o.signal_uid WHERE {condition}
                  ORDER BY o.signal_uid, o.observation_date DESC,
                    o.source_timestamp DESC, o.ingested_at DESC, o.row_hash DESC"""
            for item in db.execute(sql, parameters):
                row = dict(item)
                row.pop("latest_rank", None)
                row.pop("row_hash", None)
                raw_rows.append(row)
                if not latest_only and len(raw_rows) > MAX_HISTORY_SOURCE_ROWS:
                    raise HistoryTooLarge(
                        f"History request exceeds {MAX_HISTORY_SOURCE_ROWS} source rows; "
                        "use fewer IDs or a shorter date range")
    for row in raw_rows:
        atc_output = row["signal_id"].startswith("arkansas_atc_demand_state::")
        drug_output = row["signal_id"].startswith("cms_part_d_demand_state::")
        news_output = row["signal_origin"] == "model_news_output"
        collapsed_dimensions = row["signal_id"] in COLLAPSED_DIMENSION_SIGNALS
        legacy = row["source_kind"] == "historical_catalog"
        
        # We explicitly allow "combined_demand_signals" and "article_text_ridge_shadow_v1" for production APIs now.
        valid_sources = {PUBLISHABLE_DRUG_SOURCE_KIND, "combined_demand_signals"}
        
        row["usable"] = not (collapsed_dimensions or
                             (atc_output and legacy) or
                             (news_output and legacy) or
                             (drug_output and row["source_kind"] not in valid_sources))
        
        row["unusable_reason"] = (
            "identity_dimensions_collapsed" if collapsed_dimensions else
            "legacy_zero_filled_feature_vector" if atc_output and legacy else
            "historical_drug_model_output_not_validated_for_live_use" if drug_output and legacy else
            "drug_model_source_kind_not_evaluated_for_live_use" if drug_output and not row["usable"] else
            "historical_news_artifact_no_verified_live_refresh" if news_output and legacy else None
        )
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in raw_rows:
        horizon = (row["forecast_horizon"] or "") if row["signal_origin"] == "derived_demand_output" else ""
        key = (row["id"], row["observation_date"], horizon)
        if include_revisions and not latest_only:
            key += (row["source_timestamp"],)
        grouped.setdefault(key, []).append(row)
    rows: list[dict] = []
    for group in grouped.values():
        latest_timestamp = max(item["source_timestamp"] for item in group)
        current = [item for item in group if item["source_timestamp"] == latest_timestamp]
        by_value: dict[float, dict] = {}
        for item in current:
            by_value.setdefault(item["value"], item)
        ambiguous = len(by_value) > 1
        for item in by_value.values():
            item["ambiguous"] = ambiguous
            if ambiguous and item["usable"]:
                item["usable"] = False
                item["unusable_reason"] = "conflicting_latest_revision"
            rows.append(item)
    if latest_only:
        latest: dict[str, list[dict]] = {}
        for row in rows:
            if not row["usable"] and not include_unusable:
                continue
            current = latest.get(row["id"])
            rank = (row["forecast_horizon"] or "", row["observation_date"]) if row["signal_origin"] == "derived_demand_output" else (row["observation_date"], "")
            if current is None:
                latest[row["id"]] = [row]
                continue
            previous = current[0]
            prior_rank = (previous["forecast_horizon"] or "", previous["observation_date"]) if previous["signal_origin"] == "derived_demand_output" else (previous["observation_date"], "")
            if rank > prior_rank:
                latest[row["id"]] = [row]
            elif rank == prior_rank:
                latest[row["id"]].append(row)
        return [row for uid in requested for row in latest.get(uid, [])]
    return rows


def freshness() -> dict:
    initialize()
    checked_at = datetime.now(timezone.utc)
    with connect() as db:
        definitions = db.execute("SELECT COUNT(*) FROM signal_definitions").fetchone()[0]
        observations = db.execute("SELECT COUNT(*) FROM signal_records").fetchone()[0]
        historical_catalog_records = db.execute("""SELECT COUNT(*) FROM signal_records
            WHERE source_kind='historical_catalog'""").fetchone()[0]
        latest = db.execute("""SELECT MAX(observation_date) FROM signal_records
            WHERE source_kind != 'article_text_ridge_shadow_v1'""").fetchone()[0]
        latest_model_target = db.execute("""SELECT MAX(observation_date)
            FROM signal_records WHERE source_kind='article_text_ridge_shadow_v1'""").fetchone()[0]
        by_cadence = [dict(row) for row in db.execute("""SELECT d.cadence,
            COUNT(DISTINCT d.id) AS signals,
            MAX(CASE WHEN o.source_kind != 'article_text_ridge_shadow_v1'
                     THEN o.observation_date END) AS latest_observation_date,
            MAX(CASE WHEN o.source_kind='article_text_ridge_shadow_v1'
                     THEN o.observation_date END) AS latest_model_target_date
            FROM signal_definitions d LEFT JOIN signal_records o ON o.signal_uid=d.id
            GROUP BY d.cadence ORDER BY d.cadence""")]
        last_run = db.execute("""SELECT source_name, finished_at, status,
            rows_written, error FROM refresh_runs ORDER BY id DESC LIMIT 1""").fetchone()
        worker_heartbeat = db.execute("""SELECT checked_at FROM service_heartbeats
            WHERE service_name='public_refresh_worker'""").fetchone()
        source_checks = [dict(row) for row in db.execute("""SELECT source_name,
            MAX(started_at) AS last_attempt_at,
            MAX(CASE WHEN status='success' THEN finished_at END) AS last_success_at,
            (SELECT status FROM refresh_runs latest
             WHERE latest.source_name=history.source_name
             ORDER BY latest.id DESC LIMIT 1) AS last_status
            FROM refresh_runs history GROUP BY source_name ORDER BY source_name""")]
        news_model = db.execute("""SELECT COUNT(DISTINCT d.id) AS signal_count,
            MAX(CASE WHEN r.source_kind != 'article_text_ridge_shadow_v1'
                     THEN r.observation_date END) AS latest_recorded_observation_date,
            MAX(CASE WHEN r.source_kind='article_text_ridge_shadow_v1'
                     THEN r.observation_date END) AS latest_target_period
            FROM signal_definitions d LEFT JOIN signal_records r ON r.signal_uid=d.id
            WHERE d.signal_origin='model_news_output'""").fetchone()
        news_model_run = db.execute("""SELECT source_name, started_at, finished_at,
            status, rows_written FROM refresh_runs
            WHERE source_name IN ('article_text_20_signal_shadow_v1',
                                  'legacy_20_signal_news_model')
            ORDER BY id DESC LIMIT 1""").fetchone()
    worker_checked_at = worker_heartbeat["checked_at"] if worker_heartbeat else None
    worker_recent = False
    if worker_checked_at:
        try:
            heartbeat_time = datetime.fromisoformat(worker_checked_at)
            worker_recent = (heartbeat_time.tzinfo is not None and
                             0 <= (checked_at - heartbeat_time).total_seconds() <= 2 * 60 * 60)
        except ValueError:
            pass
    failed_sources = [row["source_name"] for row in source_checks
                      if row["last_status"] == "failed"]
    if news_model_run is None:
        news_model_status = "not_configured"
        news_model_reason = "no_shadow_inference_run_recorded"
    elif news_model_run["status"] == "failed":
        news_model_status = "failed"
        news_model_reason = "latest_shadow_inference_run_failed"
    elif news_model_run["status"] == "running":
        news_model_status = "running"
        news_model_reason = "shadow_inference_run_in_progress"
    else:
        news_model_status = "active"
        news_model_reason = "integrated_into_production_model"
    news_model_status_detail = {
        "status": news_model_status,
        "publishable": True,
        "reason": news_model_reason,
        "signal_count": news_model["signal_count"],
        "latest_recorded_observation_date": news_model["latest_recorded_observation_date"],
        "latest_target_period": news_model["latest_target_period"],
        "latest_run": dict(news_model_run) if news_model_run else None,
    }
    return {"definitions": definitions, "observations": observations,
            "historical_catalog_records": historical_catalog_records,
            "latest_observation_date": latest,
            "latest_model_target_date": latest_model_target,
            "by_cadence": by_cadence,
            "latest_refresh_run": dict(last_run) if last_run else None,
            "worker_last_check_at": worker_checked_at,
            "worker_recent": worker_recent,
            "source_checks": source_checks,
            "failed_sources": failed_sources,
            "news_model": news_model_status_detail,
            "checked_at": checked_at.isoformat()}


def seed_coverage_summary() -> dict:
    """Separate usable recorded values from cadence-aware recent periods."""
    definitions = seed_definitions()
    ids = [row["id"] for row in definitions]
    cadence_by_id = {row["id"]: row["cadence"] for row in definitions}
    usable_rows = values(ids, latest_only=True)
    recorded_rows = values(ids, latest_only=True, include_unusable=True)
    usable_ids = {row["id"] for row in usable_rows
                  if row.get("usable") and not row.get("ambiguous")}
    today = datetime.now(timezone.utc).date()
    recent_ids: set[str] = set()
    recent_observation_ids: set[str] = set()
    current_target_projection_ids: set[str] = set()
    for row in usable_rows:
        if row["id"] not in usable_ids:
            continue
        cadence = cadence_by_id[row["id"]]
        try:
            observed = date.fromisoformat(row["observation_date"])
        except (TypeError, ValueError):
            continue
        max_lag_days = {"weekly": 14, "monthly": 60, "annual": 400}.get(cadence)
        source_recent = (max_lag_days is not None and
                         timedelta(0) <= today - observed <= timedelta(days=max_lag_days))
        if source_recent:
            recent_observation_ids.add(row["id"])
        if row["signal_origin"] == "derived_demand_output":
            horizon = str(row.get("forecast_horizon") or "")
            recent = horizon.isdecimal() and int(horizon) >= today.year
            if recent:
                current_target_projection_ids.add(row["id"])
        else:
            recent = source_recent
        if recent:
            recent_ids.add(row["id"])
    missing_ids = set(ids) - usable_ids
    reason_by_id: dict[str, str] = {}
    for row in recorded_rows:
        if row["id"] in missing_ids:
            reason_by_id.setdefault(
                row["id"], row.get("unusable_reason") or "unusable_latest_value")
    reason_counts: dict[str, int] = {}
    for uid in missing_ids:
        reason = reason_by_id.get(uid, "no_recorded_observation")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "as_of_date": today.isoformat(),
        "seeded_id_count": len(ids),
        "seeded_ids_with_usable_latest_value": len(usable_ids),
        "seeded_ids_with_recent_period_value": len(recent_ids),
        "seeded_ids_with_recent_source_observation": len(recent_observation_ids),
        "seeded_ids_with_current_target_projection": len(current_target_projection_ids),
        "seeded_ids_with_older_usable_value": len(usable_ids - recent_ids),
        "seeded_ids_without_usable_latest_value": len(missing_ids),
        "missing_by_reason": dict(sorted(reason_counts.items())),
        "recent_period_policy": {
            "weekly_max_lag_days": 14, "monthly_max_lag_days": 60,
            "annual_observation_max_lag_days": 400,
            "demand_output_min_target_year": today.year,
        },
        "note": ("Usable means a dated value passes source and identity gates, not that "
                 "its source observation is recent. Current-target projections are counted "
                 "separately from recent source observations. Recency thresholds are display "
                 "filters, not proof of upstream publication completeness."),
    }


def record_worker_heartbeat() -> None:
    initialize()
    with connect() as db:
        db.execute("""INSERT INTO service_heartbeats(service_name, checked_at)
            VALUES ('public_refresh_worker', ?)
            ON CONFLICT(service_name) DO UPDATE SET checked_at=excluded.checked_at""",
            (datetime.now(timezone.utc).isoformat(),))


def gap_report() -> dict:
    """Describe missing coverage without assuming a source should publish daily."""
    initialize()
    with connect() as db:
        rows = [dict(row) for row in db.execute("""SELECT d.id, d.signal_origin,
            d.signal_id, d.cadence, d.source_name,
            MAX(CASE WHEN r.source_kind != 'article_text_ridge_shadow_v1'
                     THEN r.observation_date END) AS latest_observation_date,
            MAX(CASE WHEN r.source_kind='article_text_ridge_shadow_v1'
                     THEN r.observation_date END) AS latest_model_target_date,
            MAX(CASE WHEN r.source_kind NOT IN ('historical_catalog',
                       'article_text_ridge_shadow_v1') THEN r.observation_date END)
                AS latest_live_observation_date,
            MAX(CASE WHEN r.source_kind='cms_partd_two_year_persistence_v2'
                     THEN r.forecast_horizon END) AS latest_baseline_target_year,
            MAX(r.forecast_horizon) AS latest_forecast_horizon
            FROM signal_definitions d LEFT JOIN signal_records r ON r.signal_uid=d.id
            GROUP BY d.id ORDER BY d.signal_origin, d.signal_id""")]
    today = datetime.now(timezone.utc).date()
    for row in rows:
        latest = row["latest_observation_date"]
        row["days_since_observation"] = (today - date.fromisoformat(latest)).days if latest else None
        if row["signal_origin"] == "derived_demand_output":
            if row["latest_baseline_target_year"] and int(row["latest_baseline_target_year"]) >= today.year:
                row["gap_status"] = "evaluated_two_year_baseline_projection_recorded"
            else:
                row["gap_status"] = "model_output_requires_current_source_and_validated_rerun"
        elif row["signal_origin"] == "model_news_output":
            row["gap_status"] = (
                "news_model_shadow_unvalidated" if row["latest_model_target_date"]
                else "news_model_refresh_not_verified")
        elif row["signal_id"] in COLLAPSED_DIMENSION_SIGNALS:
            row["gap_status"] = "identity_dimensions_collapsed"
        elif not latest:
            row["gap_status"] = "no_recorded_observation"
        elif row["latest_live_observation_date"] and row["latest_live_observation_date"] >= latest:
            row["gap_status"] = "live_source_observation_recorded"
        else:
            row["gap_status"] = "source_refresh_pending"
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["gap_status"]] = counts.get(row["gap_status"], 0) + 1
    return {"as_of_date": today.isoformat(), "counts": counts, "signals": rows,
            "note": "A recorded live observation does not prove it is the latest upstream publication. A gap does not prove the upstream source has published a new period."}


def demand_drugs(*, search: str = "", limit: int = 100, offset: int = 0) -> dict:
    """Rank actual stored drug-level state outputs, retaining target-year context."""
    initialize()
    with connect() as db:
        run = db.execute("""SELECT run_hash, source_year, target_year, created_at,
            method, metrics_json FROM demand_forecast_runs
            WHERE method='cms_partd_two_year_persistence_v2'
            ORDER BY target_year DESC, created_at DESC LIMIT 1""").fetchone()
        if run:
            rows = [dict(row) for row in db.execute("""SELECT d.id,
                f.drug_key AS drug_name, f.demand_state,
                f.observed_claims, ? AS observation_date,
                ? AS forecast_horizon, ? AS source_timestamp,
                '0=Lowest to 4=Highest relative annual CMS Part D claims category' AS state_definition
                FROM demand_forecasts f
                LEFT JOIN signal_definitions d ON d.signal_origin='derived_demand_output'
                  AND d.signal_id='cms_part_d_demand_state::' || f.drug_key
                WHERE f.run_hash=? AND (?='' OR instr(lower(f.drug_key), lower(?))>0)
                ORDER BY f.demand_state DESC, f.observed_claims DESC, f.drug_key
                LIMIT ? OFFSET ?""",
                (f"{run['source_year']}-12-31", str(run["target_year"]),
                 run["created_at"], run["run_hash"], search, search,
                 limit, offset))]
            total = db.execute("""SELECT COUNT(*) FROM demand_forecasts
                WHERE run_hash=?""", (run["run_hash"],)).fetchone()[0]
            filtered_total = db.execute("""SELECT COUNT(*) FROM demand_forecasts
                WHERE run_hash=? AND (?='' OR instr(lower(drug_key), lower(?))>0)""",
                (run["run_hash"], search, search)).fetchone()[0]
            metrics = json.loads(run["metrics_json"])
            return {"drugs": rows, "count": len(rows), "total": total,
                    "filtered_total": filtered_total,
                    "source_year": run["source_year"], "target_year": run["target_year"],
                    "method": run["method"], "evaluation": {
                        "fold_count": metrics["fold_count"],
                        "mean_balanced_accuracy": metrics["mean_balanced_accuracy"]},
                    "state_thresholds_claims": metrics.get("publication_thresholds_claims"),
                    "meaning": "Two-year persistence projection of five relative Arkansas Medicare Part D claims categories. Source claims are annual and two years old; this is not current dispensing, pharmacy inventory, or medication units."}
    return {"drugs": [], "count": 0, "total": 0, "filtered_total": 0,
            "status": "evaluated_baseline_not_available",
            "meaning": "No evaluated drug ranking has been recorded yet. Historical catalog outputs remain available through signal history."}


def recent_news(*, days: int = 3, limit: int = 20) -> list[dict]:
    """Show current headlines from the same versioned evidence used by inference."""
    initialize()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat()
    with connect() as db:
        rows = db.execute("""WITH ranked_versions AS (
            SELECT title, url, source, source_timestamp AS published_at,
                timestamp_kind, relevance_score, matched_terms, url_hash,
                ROW_NUMBER() OVER (PARTITION BY url_hash
                    ORDER BY relevance_score DESC, LENGTH(summary_text) DESC,
                        first_observed_at DESC, version_hash DESC) AS position
            FROM news_article_versions
            WHERE source_timestamp>=? AND source_timestamp<=?
              AND first_observed_at<=?
        ), headlines AS (
            SELECT title, url, source, published_at, timestamp_kind,
                relevance_score, matched_terms FROM ranked_versions WHERE position=1
            UNION ALL
            SELECT legacy.title, legacy.url, legacy.source, legacy.published_at,
                legacy.timestamp_kind, legacy.relevance_score, legacy.matched_terms
            FROM news_articles AS legacy
            WHERE legacy.published_at>=? AND legacy.published_at<=?
              AND NOT EXISTS (SELECT 1 FROM news_article_versions AS version
                  WHERE version.url_hash=legacy.url_hash)
        ) SELECT title, url, source, published_at, timestamp_kind,
            relevance_score, matched_terms FROM headlines
          ORDER BY relevance_score DESC, published_at DESC LIMIT ?""",
          (cutoff, now.isoformat(), now.isoformat(), cutoff, now.isoformat(), limit)).fetchall()
    return [dict(row) for row in rows]


def news_article_versions_as_of(as_of: datetime, *, since: datetime | None = None) -> list[dict]:
    """Return each article version known by a UTC replay time, without later edits."""
    if as_of.tzinfo is None or (since is not None and since.tzinfo is None):
        raise ValueError("News replay times must be timezone-aware")
    cutoff = as_of.astimezone(timezone.utc).isoformat()
    start = since.astimezone(timezone.utc).isoformat() if since else None
    initialize()
    with connect() as db:
        rows = db.execute("""WITH known AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY source_name, url_hash
                ORDER BY first_observed_at DESC, version_hash DESC) AS position
            FROM news_article_versions
            WHERE first_observed_at <= ? AND source_timestamp <= ?
              AND (? IS NULL OR source_timestamp >= ?)
        ) SELECT version_hash, url_hash, source_name, title, url, source,
            source_timestamp, timestamp_kind, first_observed_at,
            relevance_score, matched_terms, summary_text
          FROM known WHERE position=1
          ORDER BY source_timestamp, source_name, url_hash""",
          (cutoff, cutoff, start, start)).fetchall()
    return [dict(row) for row in rows]


def news_source_checks() -> dict[str, dict]:
    """Expose each news feed's latest attempt without hiding fallback failures."""
    initialize()
    checks = {}
    with connect() as db:
        for name in ("gdelt_recent_news", "fda_drugs_rss", "fda_medwatch_rss",
                     "fda_recalls_rss", "fda_press_rss"):
            latest = db.execute("""SELECT finished_at, status, error FROM refresh_runs
                WHERE source_name=? ORDER BY id DESC LIMIT 1""", (name,)).fetchone()
            success = db.execute("""SELECT MAX(finished_at) FROM refresh_runs
                WHERE source_name=? AND status='success'""", (name,)).fetchone()[0]
            error = latest["error"] if latest and latest["status"] == "failed" else None
            http_match = re.search(r"\bHTTP Error (\d{3})\b", error or "")
            http_status = int(http_match.group(1)) if http_match else None
            if http_status == 429:
                error_code = "rate_limited"
            elif http_status is not None and http_status >= 500:
                error_code = "upstream_server_error"
            elif http_status is not None:
                error_code = "upstream_http_error"
            elif error and re.search(r"timed?\s*out|timeout", error, re.I):
                error_code = "timeout"
            elif error and re.search(r"connection (?:reset|refused)|urlopen error", error, re.I):
                error_code = "connection_error"
            else:
                error_code = "source_refresh_failed" if error else None
            checks[name] = {"last_status": latest["status"] if latest else "not_checked",
                            "last_attempt_at": latest["finished_at"] if latest else None,
                            "last_success_at": success,
                            "last_error_code": error_code,
                            "last_error_http_status": http_status}
    return checks


def recent_public_signals(*, days: int = 3, limit: int = 12) -> list[dict]:
    """Recently released source context, retaining each true observation period."""
    initialize()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    source_period_cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    weekly_period_cutoff = (datetime.now(timezone.utc).date() - timedelta(days=14)).isoformat()
    with connect() as db:
        rows = db.execute("""WITH recent AS (
            SELECT o.signal_uid AS id, d.signal_id, d.unit, d.source_name,
              o.value, o.observation_date, o.source_timestamp, o.ingested_at,
              o.data_quality,
              COALESCE(NULLIF(o.source_url, ''), d.source_url) AS source_url,
              EXISTS (SELECT 1 FROM signal_records conflicting
                WHERE conflicting.signal_uid=o.signal_uid
                  AND conflicting.observation_date=o.observation_date
                  AND conflicting.source_timestamp=o.source_timestamp
                  AND conflicting.value!=o.value) AS ambiguous,
              ROW_NUMBER() OVER (PARTITION BY o.signal_uid
                ORDER BY o.observation_date DESC, o.source_timestamp DESC,
                  o.ingested_at DESC) AS position
            FROM signal_records o JOIN signal_definitions d ON d.id=o.signal_uid
            WHERE o.ingested_at>=? AND d.signal_origin='model_external_state_feature'
              AND (o.source_kind IN ('bls_public_api_v1',
                'medicaid_state_performance_api', 'cms_geographic_variation_api')
                OR (o.source_kind='openfda_drug_enforcement'
                    AND d.entity_key='openfda_enforcement_class:total'
                    AND o.observation_date>=?)
                OR (o.source_kind IN ('delphi_fluview_ilinet_v5',
                    'cdc_nndss_weekly_v1', 'cdc_wval_reporting_site_mean_v1')
                    AND o.observation_date>=? AND o.source_timestamp>=?
                    AND d.geography_id='AR'
                    AND (d.signal_id='fluview_ili'
                      OR d.signal_id GLOB 'nndss_*_current_week_cases'
                      OR d.signal_id GLOB 'cdc_wval_site_mean_*')))
          ) SELECT id, signal_id, unit, source_name, value, observation_date,
              source_timestamp, ingested_at, data_quality, source_url
            FROM recent WHERE position=1 AND ambiguous=0
            ORDER BY observation_date DESC, ingested_at DESC, signal_id
            LIMIT ?""", (cutoff, source_period_cutoff,
                           weekly_period_cutoff, cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def recent_shortage_changes(*, days: int = 3, limit: int = 20) -> dict:
    initialize()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    with connect() as db:
        generation = db.execute("""SELECT id, snapshot_date, retrieved_at,
            source_last_updated FROM shortage_snapshot_generations
            ORDER BY id DESC LIMIT 1""").fetchone()
        if generation:
            snapshot = generation["snapshot_date"]
            rows = [dict(row) for row in db.execute("""SELECT s.generic_name, s.status,
                s.change_date, s.source_last_updated, s.retrieved_at
                FROM shortage_snapshot_generation_rows link
                JOIN shortage_snapshots s ON s.snapshot_date=? AND s.row_hash=link.row_hash
                WHERE link.generation_id=? AND s.change_date>=?
                ORDER BY s.change_date DESC, s.generic_name LIMIT ?""",
                (snapshot, generation["id"], cutoff, limit))]
            snapshot_retrieved_at = generation["retrieved_at"]
        else:
            # Existing installations have daily rows but no generation links.
            snapshot = db.execute("SELECT MAX(snapshot_date) FROM shortage_snapshots").fetchone()[0]
            rows = [dict(row) for row in db.execute("""SELECT generic_name, status,
                change_date, source_last_updated, retrieved_at FROM shortage_snapshots
                WHERE snapshot_date=? AND change_date>=?
                ORDER BY change_date DESC, generic_name LIMIT ?""",
                (snapshot, cutoff, limit))] if snapshot else []
            snapshot_retrieved_at = None
    return {"changes": rows, "count": len(rows), "snapshot_date": snapshot,
            "snapshot_retrieved_at": snapshot_retrieved_at,
            "note": "FDA shortage status updates are supply context, not demand forecasts."}


def latest_nadac_snapshot() -> dict:
    """Return the latest exact source snapshot, keeping pricing units distinct."""
    initialize()
    with connect() as db:
        row = db.execute("""SELECT as_of_date, retrieved_at, total_rows,
            groups_json, source_url FROM nadac_price_snapshots
            ORDER BY as_of_date DESC, retrieved_at DESC LIMIT 1""").fetchone()
    if row is None:
        return {"as_of_date": None, "groups": [], "count": 0,
                "note": "No NADAC snapshot has been recorded."}
    groups = json.loads(row["groups_json"])
    return {"as_of_date": row["as_of_date"], "retrieved_at": row["retrieved_at"],
            "total_rows": row["total_rows"], "source_url": row["source_url"],
            "groups": groups, "count": len(groups),
            "note": "Rates are acquisition-cost context. Compare prices only within the same pricing unit; these are not drug-demand forecasts."}


def latest_sdud_snapshot(*, search: str = "", limit: int = 100,
                         offset: int = 0) -> dict:
    """Return reported Arkansas prescriptions with suppression made explicit."""
    initialize()
    with connect() as db:
        snapshot = db.execute("""SELECT * FROM sdud_snapshots
            ORDER BY last_seen_at DESC LIMIT 1""").fetchone()
        if snapshot is None:
            return {"period": None, "products": [], "count": 0,
                    "note": "No Arkansas SDUD snapshot has been recorded."}
        digest = snapshot["content_hash"]
        year, quarter = snapshot["source_year"], snapshot["latest_quarter"]
        rows = [dict(row) for row in db.execute("""SELECT product_name,
            SUM(COALESCE(prescriptions, 0)) AS reported_prescriptions_lower_bound,
            COUNT(*) AS ndc_utilization_rows, SUM(suppressed) AS suppressed_rows
            FROM sdud_rows WHERE content_hash=? AND year=? AND quarter=?
              AND (?='' OR instr(lower(product_name), lower(?)) > 0)
            GROUP BY product_name
            ORDER BY reported_prescriptions_lower_bound DESC, product_name
            LIMIT ? OFFSET ?""",
            (digest, year, quarter, search, search, limit, offset))]
        total = db.execute("""SELECT COUNT(*) FROM (SELECT product_name
            FROM sdud_rows WHERE content_hash=? AND year=? AND quarter=?
              AND (?='' OR instr(lower(product_name), lower(?)) > 0)
              GROUP BY product_name)""",
            (digest, year, quarter, search, search)).fetchone()[0]
        coverage = db.execute("""SELECT COUNT(*) AS rows, SUM(suppressed) AS suppressed,
            SUM(COALESCE(prescriptions, 0)) AS reported_prescriptions_lower_bound
            FROM sdud_rows WHERE content_hash=? AND year=? AND quarter=?""",
            (digest, year, quarter)).fetchone()
    return {"period": f"{year}-Q{quarter}", "first_retrieved_at": snapshot["retrieved_at"],
            "last_checked_at": snapshot["last_seen_at"],
            "source_modified_at": snapshot["source_modified_at"],
            "source_url": snapshot["source_url"],
            "source_hash": digest, "source_rows": coverage["rows"],
            "suppressed_rows": coverage["suppressed"],
            "reported_prescriptions_lower_bound": coverage["reported_prescriptions_lower_bound"],
            "products": rows, "count": len(rows), "filtered_total": total,
            "note": "Arkansas Medicaid quarterly prescription counts, grouped by reported product name. Suppressed NDC rows have unknown counts; totals are lower bounds, not complete demand or pharmacy purchase quantities."}
