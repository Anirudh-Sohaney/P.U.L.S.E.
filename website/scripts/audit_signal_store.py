"""Read-only release audit of catalog identity and signal-record datestamps."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from backend.config import settings
from backend.refresh_enforcement import CLASSES as ENFORCEMENT_CLASSES
from backend.refresh_enforcement import SIGNAL_ID as ENFORCEMENT_SIGNAL_ID
from backend.refresh_enforcement import SOURCE_NAME as ENFORCEMENT_SOURCE_NAME
from backend.refresh_enforcement import _source_date as enforcement_source_date
from backend.refresh_fluview import SOURCE_KIND as FLUVIEW_SOURCE_KIND
from backend.refresh_fluview import SERIES as FLUVIEW_SERIES
from backend.refresh_fluview import _url as fluview_url
from backend.refresh_nndss import SOURCE_KIND as NNDSS_SOURCE_KIND
from backend.refresh_nndss import SOURCE_URL as NNDSS_SOURCE_URL
from backend.refresh_nndss import LABELS as NNDSS_LABELS
from backend.refresh_nndss import METRICS as NNDSS_METRICS
from backend.refresh_wastewater import SOURCE_KIND as WVAL_SOURCE_KIND
from backend.refresh_wastewater import PATHOGENS as WVAL_PATHOGENS
from backend.refresh_wastewater import _url as wval_url
from backend.signal_store import IDENTITY_FIELDS, catalog_rows, news_version_hash, signal_uid
from scripts.import_signal_catalog import DEFAULT_HISTORY, DEFAULT_HISTORY_SHA256


UNKNOWN_SOURCE_TIME_KINDS = {"historical_catalog", "historical_news_bridge"}
NEWS_MODEL_SOURCE_KIND = "article_text_ridge_shadow_v1"
DEFAULT_DEFINITIONS = (Path(__file__).resolve().parents[1] / "catalog" /
                       "signal_definitions.json")


def _date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def _aware_time(value: str) -> bool:
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except (TypeError, ValueError):
        return False


def _surveillance_problem(row: sqlite3.Row, definition: sqlite3.Row | None) -> str | None:
    """Recompute source row identities without reading a mutable live feed."""
    kind = row["source_kind"]
    if kind not in {FLUVIEW_SOURCE_KIND, NNDSS_SOURCE_KIND, WVAL_SOURCE_KIND}:
        return None
    if definition is None or not _date(row["observation_date"]):
        return "missing definition or invalid surveillance period"
    if (date.fromisoformat(row["observation_date"]).weekday() != 5
            or not _aware_time(row["source_timestamp"])
            or datetime.fromisoformat(row["source_timestamp"]).date()
                < date.fromisoformat(row["observation_date"])):
        return "surveillance period is not Saturday or release time is invalid"
    value = row["value"]
    if kind == FLUVIEW_SOURCE_KIND:
        candidates = {(f"fluview_{signal}", "state" if geo_type == "state" else "national",
                       geo_id): fluview_url(signal, geo_type)
                      for signal, geo_type, _geo_value, geo_id in FLUVIEW_SERIES}
        key = (definition["signal_id"], definition["geography_level"],
               definition["geography_id"])
        if (key not in candidates or definition["cadence"] != "weekly"
                or definition["entity_key"] or definition["unit"] != "percent_outpatient_visits"
                or row["source_url"] != candidates[key] or not 0 <= value <= 100):
            return "FluView identity, unit, source URL, or percentage changed"
        expected = hashlib.sha256(json.dumps([
            FLUVIEW_SOURCE_KIND, row["signal_uid"], row["observation_date"],
            row["source_timestamp"], value,
        ], separators=(",", ":")).encode()).hexdigest()
        if row["row_hash"] != expected:
            return "FluView source row hash changed"
    elif kind == NNDSS_SOURCE_KIND:
        quality = {part.partition("=")[0]: part.partition("=")[2]
                   for part in str(row["data_quality"] or "").split(";") if "=" in part}
        metric = quality.get("source_metric")
        flag = quality.get("source_flag")
        label = definition["entity_key"]
        geo_id = definition["geography_id"]
        if (metric not in NNDSS_METRICS or flag not in {"none", "-"}
                or label not in NNDSS_LABELS or geo_id not in {"AR", "US"}
                or definition["signal_id"] !=
                    f"nndss_{NNDSS_LABELS[label]}_{NNDSS_METRICS[metric]}"
                or definition["geography_level"] != ("state" if geo_id == "AR" else "national")
                or definition["cadence"] != "weekly" or definition["unit"] != "cases"
                or row["source_url"] != NNDSS_SOURCE_URL or value < 0):
            return "NNDSS identity, metric, source URL, or count changed"
        states = ("Arkansas",) if geo_id == "AR" else ("US RESIDENTS", "U.S. Residents")
        expected_hashes = {hashlib.sha256(json.dumps([
            NNDSS_SOURCE_KIND, row["signal_uid"], row["observation_date"],
            row["source_timestamp"], state, metric, value,
            "" if flag == "none" else flag,
        ], separators=(",", ":")).encode()).hexdigest() for state in states}
        if row["row_hash"] not in expected_hashes:
            return "NNDSS source row hash changed"
    else:
        quality = str(row["data_quality"] or "")
        count_text = next((part.split("=", 1)[1] for part in quality.split(";")
                           if part.startswith("contributing_site_rows=")), "")
        sites = int(count_text) if count_text.isdigit() else 0
        geo_id = definition["geography_id"]
        slug = next((slug for slug in WVAL_PATHOGENS.values()
                     if definition["signal_id"] == f"cdc_wval_site_mean_{slug}"), None)
        if (slug is None or geo_id not in {"AR", "US"}
                or definition["geography_level"] != ("state" if geo_id == "AR" else "national")
                or definition["entity_key"] != "reporting_site_unweighted_mean"
                or definition["cadence"] != "weekly" or definition["unit"] != "mean_site_wval"
                or row["source_url"] != wval_url(geo_id == "AR")
                or not quality.startswith("derived_unweighted_reporting_site_mean;")
                or "not_official_state_or_national_median" not in quality
                or value < 0 or sites < 1):
            return "CDC WVAL mean identity, coverage, or source URL changed"
        expected = hashlib.sha256(json.dumps([
            WVAL_SOURCE_KIND, row["signal_uid"], row["observation_date"],
            row["source_timestamp"], value, sites,
        ], separators=(",", ":")).encode()).hexdigest()
        if row["row_hash"] != expected:
            return "CDC WVAL reporting-site mean hash changed"
    if definition["signal_origin"] != "model_external_state_feature":
        return "surveillance row has an unexpected signal origin"
    return None


def audit(path: Path | None = None, *, expected_definitions: int = 1312,
          check_bundled_history: bool = True) -> dict:
    database = path or Path(settings.DATA_PATH) / "signals.sqlite3"
    if not database.is_file():
        raise ValueError(f"Signal database is missing: {database}")
    problems: list[str] = []
    source_counts: Counter[str] = Counter()
    unknown_source_times = 0
    news_versions = 0
    expected_history = {}
    if check_bundled_history:
        if hashlib.sha256(DEFAULT_HISTORY.read_bytes()).hexdigest() != DEFAULT_HISTORY_SHA256:
            raise ValueError("Bundled historical catalog checksum mismatch")
        _, source_rows, rejected = catalog_rows(DEFAULT_HISTORY, ingested_at="source-audit")
        if rejected:
            raise ValueError(f"Bundled historical catalog has {rejected} invalid rows")
        expected_history = {
            row[0]: (row[1], row[2], row[3], row[4], row[5],
                     row[7], row[8], row[9], row[10], row[11])
            for row in source_rows}
        if len(expected_history) != 4097:
            raise ValueError("Bundled historical catalog has an unexpected unique row count")
    verified_history: set[str] = set()
    truncated = False
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            problems.append("SQLite integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            problems.append("SQLite foreign key check failed")
        definition_rows = list(db.execute("SELECT * FROM signal_definitions"))
        definition_by_id = {row["id"]: row for row in definition_rows}
        definitions = len(definition_rows)
        if definitions < expected_definitions:
            problems.append(
                f"Expected at least {expected_definitions} signal definitions, found {definitions}")
        for definition in definition_rows:
            identity = {field: definition[field] for field in IDENTITY_FIELDS}
            if definition["id"] != signal_uid(identity):
                problems.append(f"{definition['id']}: signal ID does not match its identity")
                if len(problems) >= 20:
                    break
        seed_payload = json.loads(DEFAULT_DEFINITIONS.read_text(encoding="utf-8"))
        seed_rows = seed_payload.get("definitions")
        seed_ids = {row.get("id") for row in seed_rows or [] if isinstance(row, dict)}
        if (seed_payload.get("schema") != "pulse_signal_definitions_v1"
                or seed_payload.get("definition_count") != expected_definitions
                or len(seed_ids) != expected_definitions):
            raise ValueError("Signal seed definition manifest count or schema mismatch")
        present_definition_ids = {row["id"] for row in definition_rows}
        missing_seed_ids = seed_ids - present_definition_ids
        if missing_seed_ids:
            problems.append(f"{len(missing_seed_ids)} seeded signal definitions are missing")
        records = 0
        for row in db.execute("""SELECT row_hash, signal_uid, source_kind, signal_date,
                observation_date, source_timestamp, ingested_at, value,
                data_quality, missingness, forecast_horizon, source_url
                FROM signal_records"""):
            records += 1
            source_counts[row["source_kind"]] += 1
            for column in ("signal_date", "observation_date"):
                if not _date(row[column]):
                    problems.append(f"{row['row_hash']}: invalid {column}")
            if not _aware_time(row["ingested_at"]):
                problems.append(f"{row['row_hash']}: invalid ingested_at")
            if row["source_timestamp"]:
                if not _aware_time(row["source_timestamp"]):
                    problems.append(f"{row['row_hash']}: invalid source_timestamp")
            elif row["source_kind"] in UNKNOWN_SOURCE_TIME_KINDS:
                unknown_source_times += 1
            else:
                problems.append(f"{row['row_hash']}: missing live source_timestamp")
            if not isinstance(row["value"], (int, float)) or not math.isfinite(row["value"]):
                problems.append(f"{row['row_hash']}: nonfinite value")
            if row["source_kind"] in {FLUVIEW_SOURCE_KIND, NNDSS_SOURCE_KIND,
                                      WVAL_SOURCE_KIND}:
                surveillance_problem = _surveillance_problem(
                    row, definition_by_id.get(row["signal_uid"]))
                if surveillance_problem:
                    problems.append(f"{row['row_hash']}: {surveillance_problem}")
            if row["row_hash"] in expected_history:
                actual = tuple(row[column] for column in (
                    "signal_uid", "signal_date", "observation_date", "value",
                    "source_timestamp", "data_quality", "missingness",
                    "forecast_horizon", "source_kind", "source_url"))
                if actual != expected_history[row["row_hash"]]:
                    problems.append(f"{row['row_hash']}: bundled historical row changed")
                verified_history.add(row["row_hash"])
            if len(problems) >= 20:
                truncated = True
                break
        missing_history = set(expected_history) - verified_history
        if missing_history and not truncated:
            problems.append(f"{len(missing_history)} bundled historical rows are missing")
        if not truncated and db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='news_article_versions'""").fetchone():
            for row in db.execute("""SELECT version_hash, url_hash, source_name,
                    title, url, source, source_timestamp, timestamp_kind,
                    first_observed_at, relevance_score, matched_terms, summary_text
                    FROM news_article_versions"""):
                news_versions += 1
                if (not _aware_time(row["source_timestamp"])
                        or not _aware_time(row["first_observed_at"])):
                    problems.append(f"{row['version_hash']}: invalid news source or capture time")
                if row["timestamp_kind"] not in {"gdelt_first_seen", "rss_pub_date"}:
                    problems.append(f"{row['version_hash']}: invalid news timestamp kind")
                if hashlib.sha256(row["url"].encode("utf-8")).hexdigest() != row["url_hash"]:
                    problems.append(f"{row['version_hash']}: news URL hash changed")
                if news_version_hash(row) != row["version_hash"]:
                    problems.append(f"{row['version_hash']}: news evidence content changed")
                if len(problems) >= 20:
                    break
        evidence_tables = {row[0] for row in db.execute("""SELECT name FROM sqlite_master
            WHERE type='table' AND name IN
            ('news_model_record_provenance', 'news_model_record_inputs')""")}
        if len(evidence_tables) != 2:
            problems.append("News-model evidence tables are missing")
        else:
            for row in db.execute("""SELECT row_hash, signal_uid, observation_date, value,
                    source_url FROM signal_records WHERE source_kind=?""",
                    (NEWS_MODEL_SOURCE_KIND,)):
                provenance = db.execute("""SELECT model_sha256, input_count, estimate_kind
                    FROM news_model_record_provenance WHERE row_hash=?""",
                    (row["row_hash"],)).fetchone()
                inputs = list(db.execute("""SELECT input_position, version_hash
                    FROM news_model_record_inputs WHERE row_hash=?
                    ORDER BY input_position""", (row["row_hash"],)))
                if provenance is None:
                    problems.append(f"{row['row_hash']}: missing news-model provenance")
                    continue
                if (len(inputs) != provenance["input_count"]
                        or [item["input_position"] for item in inputs] != list(range(len(inputs)))):
                    problems.append(f"{row['row_hash']}: incomplete article input links")
                article_versions = sorted(item["version_hash"] for item in inputs)
                expected_hash = hashlib.sha256(json.dumps([
                    NEWS_MODEL_SOURCE_KIND, provenance["model_sha256"],
                    row["signal_uid"], row["observation_date"], article_versions,
                    row["value"],
                ], separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                if expected_hash != row["row_hash"]:
                    problems.append(f"{row['row_hash']}: model evidence hash mismatch")
                if row["source_url"]:
                    problems.append(f"{row['row_hash']}: model output misattributes a source URL")
                if len(problems) >= 20:
                    break
        for generation in db.execute("""SELECT id, window_start, window_end,
                source_last_updated, row_count FROM enforcement_snapshot_generations"""):
            valid_dates = _date(generation["window_start"]) and _date(generation["window_end"])
            if (not valid_dates or generation["window_end"] != generation["source_last_updated"]
                    or (date.fromisoformat(generation["window_end"])
                        - date.fromisoformat(generation["window_start"])).days != 29):
                problems.append(f"FDA enforcement generation {generation['id']} has invalid dates")
            captured = list(db.execute("""SELECT r.row_hash, r.recall_number,
                r.report_date, r.classification, r.payload_json
                FROM enforcement_snapshot_generation_rows link
                JOIN enforcement_snapshot_rows r ON r.row_hash=link.row_hash
                WHERE link.generation_id=?""", (generation["id"],)))
            if len(captured) != generation["row_count"]:
                problems.append(f"FDA enforcement generation {generation['id']} is incomplete")
            hashes: set[str] = set()
            by_class: dict[str, list[str]] = {name: [] for name in ENFORCEMENT_CLASSES}
            for row in captured:
                if hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() != row["row_hash"]:
                    problems.append(f"{row['row_hash']}: FDA enforcement source row changed")
                payload = json.loads(row["payload_json"])
                try:
                    payload_report_date = enforcement_source_date(
                        payload.get("report_date")).isoformat()
                except ValueError:
                    payload_report_date = None
                if (row["row_hash"] in hashes
                        or not generation["window_start"] <= row["report_date"] <= generation["window_end"]
                        or row["classification"] not in by_class
                        or row["recall_number"] != str(payload.get("recall_number") or "")
                        or row["report_date"] != payload_report_date
                        or row["classification"] != payload.get("classification")):
                    problems.append(f"{row['row_hash']}: invalid FDA enforcement window row")
                hashes.add(row["row_hash"])
                if row["classification"] in by_class:
                    by_class[row["classification"]].append(row["row_hash"])
                if len(problems) >= 20:
                    break
            for name in (*ENFORCEMENT_CLASSES, "Total"):
                key = name.casefold().replace(" ", "_")
                identity = {
                    "signal_origin": "model_external_state_feature",
                    "signal_id": ENFORCEMENT_SIGNAL_ID,
                    "cadence": "weekly", "geography_level": "national",
                    "geography_id": "US",
                    "entity_key": f"openfda_enforcement_class:{key}",
                }
                uid = signal_uid(identity)
                source_hashes = (sorted(hashes) if name == "Total"
                                 else sorted(by_class[name]))
                value = float(len(source_hashes))
                expected_hash = hashlib.sha256(json.dumps([
                    ENFORCEMENT_SOURCE_NAME, generation["window_start"],
                    generation["window_end"], generation["source_last_updated"],
                    uid, source_hashes, value,
                ], separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                signal = db.execute("""SELECT signal_uid, observation_date, value,
                    source_kind FROM signal_records WHERE row_hash=?""",
                    (expected_hash,)).fetchone()
                if (signal is None or signal["signal_uid"] != uid
                        or signal["observation_date"] != generation["window_end"]
                        or signal["value"] != value
                        or signal["source_kind"] != ENFORCEMENT_SOURCE_NAME):
                    problems.append(f"FDA enforcement generation {generation['id']} has missing or changed {name} signal")
            if len(problems) >= 20:
                break
    if problems:
        raise ValueError("Signal store audit failed: " + "; ".join(problems))
    return {"status": "passed", "definitions": definitions, "records": records,
            "verified_bundled_historical_records": len(verified_history),
            "unknown_historical_source_timestamps": unknown_source_times,
            "news_article_versions": news_versions,
            "records_by_source_kind": dict(sorted(source_counts.items()))}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
