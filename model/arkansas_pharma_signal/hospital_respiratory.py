"""Weekly Arkansas hospital respiratory-admission pressure evaluation.

The CDC HRD series is a healthcare-utilization proxy, not pharmacy dispensing
or inventory truth. This module keeps the pathogen and Arkansas geography
explicit, forecasts the next complete week, and learns state thresholds only
from each training partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .regression import LogisticRidge, RidgeLinear
from .event_accuracy import score_event_predictions


TARGET_COLUMNS = {
    "covid": "totalconfc19newadm",
    "influenza": "totalconfflunewadm",
    "rsv": "totalconfrsvnewadm",
}
FEATURE_COLUMNS = [
    "current_value", "lag1_value", "lag2_value", "lag4_value",
    "rolling4_value", "week_sin", "week_cos",
]


def load_hospital_respiratory_weekly(path: Path | pd.DataFrame) -> pd.DataFrame:
    """Build consecutive Arkansas weekly admission rows from CDC HRD data."""
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_json(path)
    required = {"weekendingdate", "jurisdiction", *TARGET_COLUMNS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CDC HRD data missing columns: {sorted(missing)}")
    frame = frame[frame["jurisdiction"].astype(str).str.upper().eq("AR")].copy()
    frame["week_end"] = pd.to_datetime(frame["weekendingdate"], errors="coerce")
    frame = frame.dropna(subset=["week_end"]).sort_values("week_end")
    frame = frame.drop_duplicates("week_end", keep="last")
    rows = []
    for pathogen, column in TARGET_COLUMNS.items():
        series = pd.to_numeric(frame[column], errors="coerce")
        current = frame[["week_end"]].copy()
        current["value"] = series.to_numpy()
        current = current.dropna(subset=["value"]).sort_values("week_end")
        current["next_week"] = current["week_end"].shift(-1)
        current["target"] = current["value"].shift(-1)
        current = current[
            current["next_week"].sub(current["week_end"]).eq(pd.Timedelta(days=7))
        ].copy()
        group = current["value"]
        current["pathogen"] = pathogen
        current["year"] = current["week_end"].dt.isocalendar().year.astype(int)
        current["week"] = current["week_end"].dt.isocalendar().week.astype(int)
        current["current_value"] = group.to_numpy()
        current["lag1_value"] = group.shift(1).fillna(group).to_numpy()
        current["lag2_value"] = group.shift(2).fillna(group).to_numpy()
        current["lag4_value"] = group.shift(4).fillna(group).to_numpy()
        current["rolling4_value"] = group.shift(1).rolling(4, min_periods=1).mean().fillna(group).to_numpy()
        current["week_sin"] = np.sin(2 * np.pi * current["week"] / 52.0)
        current["week_cos"] = np.cos(2 * np.pi * current["week"] / 52.0)
        rows.append(current[["pathogen", "year", "week", "week_end", "target", *FEATURE_COLUMNS]])
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return result.sort_values(["pathogen", "week_end"]).reset_index(drop=True)


def _state(values: np.ndarray, *thresholds: float) -> np.ndarray:
    """Map values to fit-partition thresholds without leaking test quantiles."""
    return np.digitize(np.asarray(values, dtype=float), thresholds, right=False).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray, state_count: int = 3) -> dict:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = [
        float(np.mean(predicted[actual == state] == state))
        if np.any(actual == state) else 0.0
        for state in range(state_count)
    ]
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state)) for state in range(state_count)},
    }


def _predict_model(train: pd.DataFrame, target: pd.DataFrame,
                   alpha: float, thresholds: tuple[float, ...],
                   state_count: int = 3) -> np.ndarray:
    train_state = _state(train["target"].to_numpy(float), *thresholds)
    probabilities = []
    for state in range(state_count):
        model = LogisticRidge(alpha=alpha).fit(
            train[FEATURE_COLUMNS].to_numpy(float),
            (train_state == state).astype(float), FEATURE_COLUMNS)
        probabilities.append(model.predict_proba(target[FEATURE_COLUMNS].to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_hospital_respiratory_state(
    view: pd.DataFrame,
    *,
    pathogen: str = "influenza",
    folds: Iterable[tuple[int, int, int]] | None = None,
    state_count: int = 3,
) -> dict:
    """Evaluate next-week hospital admission pressure with fit-only quantiles."""
    if pathogen not in TARGET_COLUMNS:
        raise ValueError(f"unsupported pathogen: {pathogen}")
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    data = view[view["pathogen"].eq(pathogen)].sort_values("week_end")
    folds = tuple(folds or ((2021, 2022, 2023), (2022, 2023, 2024),
                            (2023, 2024, 2025), (2024, 2025, 2026)))
    results = []
    event_actual: list[int] = []
    event_predicted: list[int] = []
    for train_year, validation_year, test_year in folds:
        train = data[data["year"] <= train_year]
        validation = data[data["year"].eq(validation_year)]
        test = data[data["year"].eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = tuple(np.quantile(
            train["target"].to_numpy(float),
            np.arange(1, state_count) / state_count,
        ))
        validation_actual = _state(validation["target"], *thresholds)
        candidates = {}
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            candidates[alpha] = _metrics(
                validation_actual,
                _predict_model(train, validation, alpha, thresholds, state_count),
                state_count,
            )["balanced_accuracy"]
        selected_alpha = max(candidates, key=candidates.get)
        validation_persistence = _state(validation["current_value"], *thresholds)
        selected_model = "logistic_one_vs_rest"
        if _metrics(validation_actual, validation_persistence, state_count)["balanced_accuracy"] > candidates[selected_alpha]:
            selected_model = "persistence"
        actual = _state(test["target"], *thresholds)
        persistence = _state(test["current_value"], *thresholds)
        prediction = persistence if selected_model == "persistence" else _predict_model(
            train, test, selected_alpha, thresholds, state_count)
        event_actual.extend(actual.tolist())
        event_predicted.extend(prediction.tolist())
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "selected_alpha": selected_alpha, "selected_model": selected_model,
            "thresholds": {str(i): float(value) for i, value in enumerate(thresholds)},
            "model": _metrics(actual, prediction, state_count),
            "persistence": _metrics(actual, persistence, state_count),
        })
    if not results:
        raise ValueError(f"no complete hospital respiratory folds for {pathogen!r}")
    state_counts = {str(state): sum(x["model"]["state_counts"][str(state)] for x in results)
                    for state in range(state_count)}
    model_accuracy = float(np.mean([x["model"]["accuracy"] for x in results]))
    model_balanced = float(np.mean([x["model"]["balanced_accuracy"] for x in results]))
    persistence_balanced = float(np.mean([x["persistence"]["balanced_accuracy"] for x in results]))
    event_metrics = score_event_predictions(
        np.asarray(event_actual), np.asarray(event_predicted), kind="state",
        event_states=set(range(max(0, state_count - 2), state_count)))
    return {
        "protocol": f"rolling_origin_next_week_arkansas_hospital_respiratory_{state_count}_state",
        "target": f"arkansas_weekly_hospital_{pathogen}_admission_pressure_state",
        "pathogen": pathogen, "region": "AR",
        "state_count": state_count,
        "state_definition": {str(i): f"quantile_{i + 1}_of_{state_count}"
                             for i in range(state_count)},
        "fold_count": len(results), "test_rows": int(sum(x["test_rows"] for x in results)),
        "mean_model_accuracy": model_accuracy,
        "mean_model_balanced_accuracy": model_balanced,
        "mean_persistence_accuracy": float(np.mean([x["persistence"]["accuracy"] for x in results])),
        "mean_persistence_balanced_accuracy": persistence_balanced,
        "model_balanced_improvement_vs_persistence": model_balanced - persistence_balanced,
        "event_metrics": event_metrics,
        "state_counts_in_scored_rows": state_counts,
        "publishable_candidate": bool(
            len(results) >= 3 and model_accuracy >= 0.65 and model_balanced >= 0.65
            and set(state_counts) == {"0", "1", "2"}),
        "scope": "Arkansas weekly hospital respiratory-admission utilization proxy; not pharmacy dispensing truth",
        "folds": results,
    }


def _numeric_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score hospital admissions numerically with explicit zero handling."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    absolute = np.abs(actual - predicted)
    relative = absolute / np.maximum(np.abs(actual), 1.0)
    return {
        "wape": float(absolute.sum() / max(np.abs(actual).sum(), 1e-12)),
        "mae": float(absolute.mean()) if actual.size else 0.0,
        "within_5_percent_error": float(np.mean(relative < 0.05)) if actual.size else 0.0,
        "within_20_percent_error": float(np.mean(relative < 0.20)) if actual.size else 0.0,
    }


def evaluate_hospital_respiratory_numeric(
        view: pd.DataFrame, *, pathogen: str = "influenza",
        folds: Iterable[tuple[int, int, int]] | None = None) -> dict:
    """Evaluate next-week hospital admissions as a numeric proxy."""
    if pathogen not in TARGET_COLUMNS:
        raise ValueError(f"unsupported pathogen: {pathogen}")
    data = view[view["pathogen"].eq(pathogen)].sort_values("week_end")
    folds = tuple(folds or ((2021, 2022, 2023), (2022, 2023, 2024),
                            (2023, 2024, 2025), (2024, 2025, 2026)))
    results = []
    for train_year, validation_year, test_year in folds:
        train = data[data["year"] <= train_year]
        validation = data[data["year"].eq(validation_year)]
        test = data[data["year"].eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        candidates = {}
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                train[FEATURE_COLUMNS].to_numpy(float),
                np.log1p(np.maximum(train["target"].to_numpy(float), 0.0)),
                FEATURE_COLUMNS)
            pred = np.expm1(model.predict(validation[FEATURE_COLUMNS].to_numpy(float)))
            candidates[alpha] = _numeric_metrics(validation["target"], pred)["wape"]
        selected_alpha = min(candidates, key=candidates.get)
        combined = pd.concat([train, validation])
        model = RidgeLinear(alpha=selected_alpha).fit(
            combined[FEATURE_COLUMNS].to_numpy(float),
            np.log1p(np.maximum(combined["target"].to_numpy(float), 0.0)),
            FEATURE_COLUMNS)
        actual = test["target"].to_numpy(float)
        learned = np.expm1(model.predict(test[FEATURE_COLUMNS].to_numpy(float)))
        persistence = test["current_value"].to_numpy(float)
        validation_persistence = validation["current_value"].to_numpy(float)
        persistence_score = _numeric_metrics(
            validation["target"], validation_persistence)["wape"]
        selected = persistence if persistence_score <= candidates[selected_alpha] else learned
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "selected_alpha": float(selected_alpha),
            "selected_model": "persistence" if selected is persistence else "log_ridge",
            "model": _numeric_metrics(actual, selected),
            "persistence": _numeric_metrics(actual, persistence),
        })
    if not results:
        raise ValueError(f"no complete hospital numeric folds for {pathogen!r}")
    within5 = float(np.mean([x["model"]["within_5_percent_error"] for x in results]))
    return {
        "protocol": "rolling_origin_next_week_arkansas_hospital_respiratory_numeric",
        "target": f"arkansas_weekly_hospital_{pathogen}_admissions",
        "pathogen": pathogen, "region": "AR", "fold_count": len(results),
        "test_rows": int(sum(x["test_rows"] for x in results)),
        "mean_model_wape": float(np.mean([x["model"]["wape"] for x in results])),
        "mean_persistence_wape": float(np.mean([x["persistence"]["wape"] for x in results])),
        "mean_model_within_5_percent_error": within5,
        "publishable_candidate": bool(len(results) >= 3 and within5 >= 0.65),
        "scope": "Arkansas weekly hospital influenza admissions numeric proxy; not pharmacy dispensing truth",
        "folds": results,
    }
