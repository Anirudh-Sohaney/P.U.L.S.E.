"""CDC RESP-NET weekly respiratory hospitalization proxy.

The public RESP-NET API provides a stable national weekly RSV series. This
module keeps it separate from Arkansas-local hospital data: the output is a
national disease-pressure context signal, not an Arkansas pharmacy label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .event_accuracy import score_event_predictions


RESPNET_URL = "https://data.cdc.gov/resource/kvib-3txy.json"
FEATURE_COLUMNS = [
    "current_value", "lag1_value", "lag2_value", "lag4_value",
    "rolling4_value", "week_sin", "week_cos",
]
_DIMENSION_FILTERS = {
    "surveillance_network": "RSV-NET",
    "state": "Overall",
    "age_category": "Overall",
    "race": "All",
    "sex": "All",
    "data_type": "Weekly Rate",
    "rate_type": "Observed",
}


def load_respnet_rsv_weekly(path: Path | pd.DataFrame) -> pd.DataFrame:
    """Build consecutive national RSV weekly transitions from RESP-NET."""
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_json(path)
    required = {"date", "estimate"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RESP-NET data missing columns: {sorted(missing)}")
    for column, expected in _DIMENSION_FILTERS.items():
        if column in frame and not frame[column].astype(str).str.strip().eq(expected).all():
            raise ValueError(f"RESP-NET data contains unfiltered {column} rows")
    if "estimate_type" in frame and not frame["estimate_type"].astype(str).str.strip().eq(
        "Rate per 100,000"
    ).all():
        raise ValueError("RESP-NET data contains an unsupported estimate type")
    frame["week_end"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["estimate"], errors="coerce")
    frame = frame.dropna(subset=["week_end", "value"]).sort_values("week_end")
    frame = frame.drop_duplicates("week_end", keep="last")
    frame["next_week"] = frame["week_end"].shift(-1)
    frame["target"] = frame["value"].shift(-1)
    frame = frame[frame["next_week"].sub(frame["week_end"]).eq(pd.Timedelta(days=7))].copy()
    frame["target_week_end"] = frame["next_week"]
    frame["year"] = frame["week_end"].dt.isocalendar().year.astype(int)
    frame["target_year"] = frame["target_week_end"].dt.isocalendar().year.astype(int)
    frame["week"] = frame["week_end"].dt.isocalendar().week.astype(int)
    frame["current_value"] = frame["value"]
    frame["lag1_value"] = frame["value"].shift(1).fillna(frame["value"])
    frame["lag2_value"] = frame["value"].shift(2).fillna(frame["lag1_value"])
    frame["lag4_value"] = frame["value"].shift(4).fillna(frame["lag2_value"])
    frame["rolling4_value"] = frame["value"].shift(1).rolling(4, min_periods=1).mean()
    frame["rolling4_value"] = frame["rolling4_value"].fillna(frame["value"])
    frame["week_sin"] = np.sin(2 * np.pi * frame["week"] / 52.0)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week"] / 52.0)
    return frame[["year", "target_year", "week", "week_end", "target_week_end",
                  "target", *FEATURE_COLUMNS]].reset_index(drop=True)


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [
        float(np.mean(predicted[actual == state] == state))
        for state in range(5) if np.any(actual == state)
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_respnet_rsv_state(
    view: pd.DataFrame,
    *,
    folds: tuple[tuple[int, int, int], ...] = (
        (2021, 2022, 2023), (2022, 2023, 2024),
        (2023, 2024, 2025), (2024, 2025, 2026)),
) -> dict:
    """Evaluate a five-state next-week RSV pressure signal chronologically."""
    results = []
    actual_all, predicted_all = [], []
    for train_year, validation_year, test_year in folds:
        train = view[
            view["year"].le(train_year) & view["target_year"].le(train_year)
        ]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = np.quantile(train["target"].to_numpy(float), np.arange(1, 5) / 5)
        validation_actual = np.digitize(validation["target"], thresholds)
        validation_persistence = np.digitize(validation["current_value"], thresholds)
        test_actual = np.digitize(test["target"], thresholds)
        test_prediction = np.digitize(test["current_value"], thresholds)
        validation_accuracy = float(np.mean(validation_actual == validation_persistence))
        test_accuracy = float(np.mean(test_actual == test_prediction))
        test_balanced_accuracy = _balanced_accuracy(test_actual, test_prediction)
        actual_all.extend(test_actual.tolist())
        predicted_all.extend(test_prediction.tolist())
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "selected_model": "persistence",
            "thresholds": [float(value) for value in thresholds],
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "test_balanced_accuracy": test_balanced_accuracy,
            "state_counts": {str(state): int(np.sum(test_actual == state)) for state in range(5)},
        })
    if len(results) < 3:
        raise ValueError("RESP-NET requires at least three complete chronological folds")
    actual = np.asarray(actual_all, dtype=int)
    predicted = np.asarray(predicted_all, dtype=int)
    return {
        "protocol": "rolling_origin_next_week_national_respnet_rsv_five_state_v1",
        "target": "national_weekly_respnet_rsv_hospitalization_pressure_state",
        "cadence": "weekly", "geography": "national", "pathogen": "rsv",
        "state_count": 5,
        "state_definition": {str(i): f"training_quantile_{i + 1}_of_5" for i in range(5)},
        "fold_count": len(results), "test_rows": int(len(actual)),
        "mean_model_accuracy": float(np.mean([row["test_accuracy"] for row in results])),
        "mean_model_balanced_accuracy": float(np.mean([
            row["test_balanced_accuracy"] for row in results
        ])),
        "event_metrics": score_event_predictions(actual, predicted, kind="state",
                                                   event_states={3, 4}),
        "state_counts_in_scored_rows": {
            str(state): int(np.sum(actual == state)) for state in range(5)},
        "scope": "national RSV hospitalization proxy; not Arkansas pharmacy inventory",
        "folds": results,
    }
