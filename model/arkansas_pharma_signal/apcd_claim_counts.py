"""Arkansas APCD monthly pharmacy-claim activity target.

The public APCD report counts pharmacy claims by prescription-fill month and
submitting entity. It is a pharmacy-activity proxy, not an NDC, inventory, or
wholesaler-allocation label. This module keeps the submitting-entity grain so
statewide totals are not silently treated as individual-pharmacy demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .event_accuracy import score_event_predictions


SOURCE_URL = (
    "https://achiapcd.atlassian.net/wiki/spaces/ADRS/pages/2778136577/"
    "Arkansas%2BAPCD%2BClaim%2BCounts%2Bby%2BMonth"
)


def load_apcd_claim_counts(source: pd.DataFrame | str) -> pd.DataFrame:
    """Normalize a public APCD claim-count CSV into monthly transitions."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = {"submitter_id", "submitter_name", "year", "month", "claim_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"APCD claim counts missing columns: {sorted(missing)}")
    frame["submitter_id"] = frame["submitter_id"].astype(str).str.strip()
    frame["submitter_name"] = frame["submitter_name"].astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame["claim_count"] = pd.to_numeric(frame["claim_count"], errors="coerce")
    frame = frame.dropna(subset=["submitter_id", "year", "month", "claim_count"])
    frame = frame[frame["submitter_id"].ne("") & frame["claim_count"].ge(0)].copy()
    frame["period"] = pd.PeriodIndex(
        frame["year"].astype(int).astype(str) + "-" +
        frame["month"].astype(int).astype(str).str.zfill(2), freq="M")
    frame = (frame.drop_duplicates(["submitter_id", "period"], keep="last")
             .sort_values(["submitter_id", "period"]))
    group = frame.groupby("submitter_id", sort=False)
    frame["next_period"] = group["period"].shift(-1)
    frame["target"] = group["claim_count"].shift(-1)
    frame["current_value"] = frame["claim_count"]
    return frame[frame["next_period"].eq(frame["period"] + 1)].reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = [
        float(np.mean(predicted[actual == state] == state))
        if np.any(actual == state) else 0.0
        for state in range(5)
    ]
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state)) for state in range(5)},
    }


def evaluate_apcd_claim_activity(
    panel: pd.DataFrame,
    *,
    train_months: int = 12,
    validation_months: int = 6,
    test_months: int = 5,
    step_months: int = 5,
) -> dict[str, Any]:
    """Evaluate next-month five-state claim activity chronologically."""
    periods = sorted(panel["period"].unique())
    folds = []
    actual_all: list[int] = []
    predicted_all: list[int] = []
    for index in range(
        train_months, len(periods) - validation_months - test_months + 1,
        max(1, step_months),
    ):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[
            index + validation_months:index + validation_months + test_months
        ]
        train = panel[panel["period"].isin(train_periods)]
        validation = panel[panel["period"].isin(validation_periods)]
        test = panel[panel["period"].isin(test_periods)]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = np.quantile(train["target"].to_numpy(float), np.arange(1, 5) / 5)
        validation_actual = np.digitize(validation["target"], thresholds)
        validation_persistence = np.digitize(validation["current_value"], thresholds)
        test_actual = np.digitize(test["target"], thresholds)
        test_prediction = np.digitize(test["current_value"], thresholds)
        actual_all.extend(test_actual.tolist())
        predicted_all.extend(test_prediction.tolist())
        folds.append({
            "train_end": str(train_periods[-1]),
            "validation_start": str(validation_periods[0]),
            "validation_end": str(validation_periods[-1]),
            "test_start": str(test_periods[0]),
            "test_end": str(test_periods[-1]),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "thresholds": [float(value) for value in thresholds],
            "validation_accuracy": float(np.mean(
                validation_actual == validation_persistence)),
            "test": _metrics(test_actual, test_prediction),
        })
    if len(folds) < 3:
        raise ValueError("APCD claim activity requires at least three chronological folds")
    actual = np.asarray(actual_all, dtype=int)
    predicted = np.asarray(predicted_all, dtype=int)
    event_metrics = score_event_predictions(
        actual, predicted, kind="state", event_states={3, 4})
    return {
        "protocol": "rolling_origin_next_month_arkansas_apcd_pharmacy_claim_activity_five_state",
        "target": "arkansas_monthly_apcd_pharmacy_claim_activity_state",
        "cadence": "monthly",
        "geography": "Arkansas APCD reporting-entity activity",
        "state_count": 5,
        "state_definition": {str(i): f"training_quantile_{i + 1}_of_5" for i in range(5)},
        "submitter_count": int(panel["submitter_id"].nunique()),
        "fold_count": len(folds),
        "test_rows": int(len(actual)),
        "mean_model_accuracy": float(np.mean([fold["test"]["accuracy"] for fold in folds])),
        "mean_model_balanced_accuracy": float(np.mean([
            fold["test"]["balanced_accuracy"] for fold in folds])),
        "event_metrics": event_metrics,
        "state_counts_in_scored_rows": {
            str(state): int(np.sum(actual == state)) for state in range(5)},
        "publishable_candidate": bool(
            len(actual) >= 25
            and all(fold["test"]["balanced_accuracy"] >= 0.65 for fold in folds)
            and (event_metrics["true_positive_precision"] >= 0.80
                 and np.mean(actual == predicted) >= 0.65)
        ),
        "scope": (
            "Arkansas APCD monthly pharmacy-claim activity proxy by reporting entity; "
            "not NDC demand, supplier allocation, or inventory"
        ),
        "source_url": SOURCE_URL,
        "folds": folds,
    }
