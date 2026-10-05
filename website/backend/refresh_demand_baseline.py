"""Publish an evaluated two-year Arkansas Part D claims-state projection.

This is an annual public claims proxy. It does not estimate pharmacy orders,
inventory, or current dispensing, and it does not rewrite legacy signal rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .refresh_cms_partd_source import source_snapshot
from .signal_store import connect, initialize


SOURCE_KIND = "cms_partd_two_year_persistence_v2"
MIN_FOLDS = 4
MIN_BALANCED_ACCURACY = 0.75


def load_claims(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"CMS Part D source missing: {path}")
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("CMS Part D source is a Git LFS pointer; run git lfs pull")
    digest = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(path, compression="gzip")
    required = {"year", "state", "drug_key", "demand_claims"}
    if not required.issubset(frame):
        raise ValueError(f"CMS Part D source missing columns: {sorted(required - set(frame))}")
    frame = frame[list(required)].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["demand_claims"] = pd.to_numeric(frame["demand_claims"], errors="raise")
    if (frame["state"].astype(str).str.lower() != "arkansas").any():
        raise ValueError("CMS Part D source contains non-Arkansas rows")
    if frame["drug_key"].isna().any() or frame["drug_key"].astype(str).str.strip().eq("").any():
        raise ValueError("CMS Part D source contains an empty drug key")
    if (frame["demand_claims"] < 0).any() or not np.isfinite(frame["demand_claims"]).all():
        raise ValueError("CMS Part D source contains invalid claims")
    if frame.duplicated(["year", "drug_key"]).any():
        raise ValueError("CMS Part D source has duplicate drug/year rows")
    return frame.sort_values(["year", "drug_key"]).reset_index(drop=True), digest


def paired_years(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame[["year", "drug_key", "demand_claims"]].rename(
        columns={"demand_claims": "current_claims"})
    target = frame[["year", "drug_key", "demand_claims"]].rename(
        columns={"demand_claims": "target_claims"}).copy()
    target["year"] -= 2
    return current.merge(target, on=["year", "drug_key"], validate="one_to_one")


def evaluate(frame: pd.DataFrame) -> dict:
    pairs = paired_years(frame)
    latest_feature_year = int(pairs["year"].max())
    folds = []
    for test_year in range(latest_feature_year - 3, latest_feature_year + 1):
        # At the October forecast date in target year t+2, annual claims
        # through feature year t are available. A validation pair from t-1
        # would need target-year t+1 claims, released too late for that date.
        validation_year = test_year - 2
        training = pairs[pairs["year"] < validation_year]
        validation = pairs[pairs["year"] == validation_year]
        test = pairs[pairs["year"] == test_year]
        if training["year"].nunique() < 4 or validation.empty or test.empty:
            continue
        thresholds = np.quantile(training["target_claims"], [.2, .4, .6, .8])
        actual = np.digitize(test["target_claims"], thresholds)
        predicted = np.digitize(test["current_claims"], thresholds)
        if set(actual.tolist()) != set(range(5)):
            raise ValueError(f"Two-year test fold {test_year} lacks a demand category")
        recalls = [float(np.mean(predicted[actual == state] == state))
                   for state in range(5) if np.any(actual == state)]
        folds.append({"feature_year": test_year, "target_year": test_year + 2,
                      "train_feature_year_max": test_year - 3,
                      "validation_feature_year": validation_year,
                      "test_rows": int(len(test)),
                      "accuracy": float(np.mean(predicted == actual)),
                      "balanced_accuracy": float(np.mean(recalls)),
                      "thresholds": [float(value) for value in thresholds]})
    if len(folds) < MIN_FOLDS:
        raise ValueError(f"Two-year baseline has only {len(folds)} eligible folds")
    if any(item["balanced_accuracy"] < MIN_BALANCED_ACCURACY for item in folds):
        raise ValueError("Two-year baseline failed its chronological accuracy floor")
    return {"protocol": "rolling_origin_two_year_partd_claim_state_release_safe_v2",
            "scope": "Arkansas Medicare Part D annual generic-drug claims proxy",
            "method": "two_year_state_persistence",
            "fold_count": len(folds), "test_rows": sum(item["test_rows"] for item in folds),
            "mean_accuracy": float(np.mean([item["accuracy"] for item in folds])),
            "mean_balanced_accuracy": float(np.mean(
                [item["balanced_accuracy"] for item in folds])),
            "folds": folds}


def refresh_demand_baseline() -> dict:
    initialize()
    started = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_KIND, started)).lastrowid
    try:
        path, manifest = source_snapshot()
        frame, source_sha = load_claims(path)
        source_year = int(frame["year"].max())
        target_year = source_year + 2
        if target_year < datetime.now(timezone.utc).year:
            raise ValueError(f"CMS Part D source ends in {source_year}; two-year target {target_year} is stale")
        metrics = evaluate(frame)
        pairs = paired_years(frame)
        thresholds = np.quantile(pairs["target_claims"], [.2, .4, .6, .8])
        metrics["publication_thresholds_claims"] = [float(value) for value in thresholds]
        current = frame[frame["year"] == source_year].copy()
        if current.empty:
            raise ValueError("CMS Part D source has no latest-year drugs")
        current["demand_state"] = np.digitize(current["demand_claims"], thresholds)
        years = manifest.get("years") or []
        urls = manifest.get("source_urls") or []
        actual_years = sorted(int(year) for year in frame["year"].unique())
        if years != actual_years or len(urls) != len(years):
            raise ValueError("CMS Part D source and manifest years disagree")
        source_url = urls[years.index(source_year)]
        run_hash = hashlib.sha256(json.dumps(
            [SOURCE_KIND, source_sha, source_year, target_year],
            separators=(",", ":")).encode()).hexdigest()
        finished = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            existing = db.execute("SELECT metrics_json FROM demand_forecast_runs WHERE run_hash=?",
                                  (run_hash,)).fetchone()
            inserted = 0
            if not existing:
                db.execute("""INSERT INTO demand_forecast_runs
                    (run_hash, source_sha256, source_year, target_year, created_at,
                     method, source_url, metrics_json, row_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_hash, source_sha, source_year, target_year, finished,
                     SOURCE_KIND, source_url, json.dumps(metrics), len(current)))
                db.executemany("""INSERT INTO demand_forecasts
                    (run_hash, drug_key, demand_state, observed_claims)
                    VALUES (?, ?, ?, ?)""",
                    [(run_hash, row.drug_key, int(row.demand_state), float(row.demand_claims))
                     for row in current.itertuples()])
                inserted = len(current)
            elif (json.loads(existing["metrics_json"]).get("publication_thresholds_claims")
                  != metrics["publication_thresholds_claims"]):
                # Backfill exact category boundaries without changing predictions
                # or the original publication timestamp.
                db.execute("""UPDATE demand_forecast_runs SET metrics_json=?
                    WHERE run_hash=?""", (json.dumps(metrics), run_hash))
            definitions = {row["entity_key"]: row["id"] for row in db.execute("""
                SELECT id, entity_key FROM signal_definitions
                WHERE signal_origin='derived_demand_output'
                  AND signal_id LIKE 'cms_part_d_demand_state::%'""")}
            signal_rows = 0
            for row in current.itertuples():
                uid = definitions.get(row.drug_key)
                if uid is None:
                    continue
                row_hash = hashlib.sha256(json.dumps(
                    [uid, run_hash, target_year], separators=(",", ":")
                ).encode()).hexdigest()
                before = db.total_changes
                db.execute("""INSERT OR IGNORE INTO signal_records
                    (row_hash, signal_uid, signal_date, observation_date, value,
                     source_timestamp, ingested_at, data_quality, missingness,
                     forecast_horizon, source_kind, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)""",
                    (row_hash, uid, finished[:10], f"{source_year}-12-31",
                     int(row.demand_state), finished, finished,
                     "evaluated_two_year_claim_state_projection;not_dispensing",
                     str(target_year), SOURCE_KIND, source_url))
                signal_rows += db.total_changes - before
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (finished, inserted + signal_rows, run_id))
        return {"source": SOURCE_KIND, "source_year": source_year,
                "target_year": target_year, "drugs": len(current),
                "mean_balanced_accuracy": metrics["mean_balanced_accuracy"],
                "rows_written": inserted, "catalog_rows_written": signal_rows,
                "finished_at": finished}
    except Exception as exc:
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
