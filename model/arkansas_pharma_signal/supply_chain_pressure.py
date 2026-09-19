"""Leak-safe monthly global supply-chain pressure state evaluation.

The New York Fed GSCPI is an upstream global context signal, not a drug or
pharmacy availability label. Thresholds are fit inside each chronological
fold, and the evaluator compares a logistic state head with persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge


FEATURES = ["current_state", "lag1_state", "lag2_state", "rolling3_state", "calendar_month"]


def load_gscpi(path: Path) -> pd.DataFrame:
    """Load and validate the public monthly GSCPI history."""
    frame = pd.read_csv(path)
    required = {"period", "gscpi"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GSCPI file missing columns: {sorted(missing)}")
    frame = frame[["period", "gscpi"]].copy()
    frame["period"] = pd.PeriodIndex(frame["period"].astype(str), freq="M")
    frame["gscpi"] = pd.to_numeric(frame["gscpi"], errors="coerce")
    frame = frame.dropna().sort_values("period").reset_index(drop=True)
    if frame["period"].duplicated().any() or not frame["period"].is_monotonic_increasing:
        raise ValueError("GSCPI periods must be unique and sorted")
    expected = pd.period_range(frame["period"].min(), frame["period"].max(), freq="M")
    if not frame["period"].array.equals(expected.array):
        raise ValueError("GSCPI history contains missing monthly periods")
    if len(frame) < 25:
        raise ValueError("GSCPI history must contain at least 25 monthly observations")
    return frame


def _states(values: pd.Series, thresholds: np.ndarray) -> pd.Series:
    return pd.cut(values, bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
                  labels=False, include_lowest=True).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(3):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state)) for state in range(3)},
    }


def _features(states: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"current_state": states.astype(float)})
    frame["lag1_state"] = frame["current_state"].shift(1)
    frame["lag2_state"] = frame["current_state"].shift(2)
    frame["rolling3_state"] = frame["current_state"].shift(1).rolling(3, min_periods=1).mean()
    frame["calendar_month"] = states.index.month if isinstance(states.index, pd.PeriodIndex) else 0
    return frame


def _predict(train: pd.DataFrame, target: pd.DataFrame, alpha: float) -> np.ndarray:
    probabilities = []
    x_train = train[FEATURES].fillna(0).to_numpy(float)
    x_target = target[FEATURES].fillna(0).to_numpy(float)
    for state in range(3):
        model = LogisticRidge(alpha=alpha).fit(
            x_train, train["target_state"].eq(state).astype(float), FEATURES)
        probabilities.append(model.predict_proba(x_target))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_gscpi_rolling(
    source: pd.DataFrame, *, min_train_months: int = 36,
    validation_months: int = 3, test_months: int = 3,
    step_months: int = 1, state_thresholds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Evaluate next-month low/mid/high GSCPI state chronologically."""
    source = source.sort_values("period").reset_index(drop=True)
    periods = list(source["period"])
    folds = []
    for index in range(min_train_months, len(periods) - validation_months - test_months + 1,
                       max(1, step_months)):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:index + validation_months + test_months]
        thresholds = np.asarray(state_thresholds, dtype=float)
        if len(set(thresholds)) < 2:
            continue
        frame = source.set_index("period", drop=False).copy()
        frame["current_state"] = _states(frame["gscpi"], thresholds)
        frame["target_state"] = frame["current_state"].shift(-1)
        feature_frame = _features(frame["current_state"])
        frame[FEATURES] = feature_frame[FEATURES]
        train = frame[frame.period.isin(train_periods)].copy()
        validation = frame[frame.period.isin(validation_periods)].copy()
        test = frame[frame.period.isin(test_periods)].copy()
        train = train.dropna(subset=["target_state"])
        validation = validation.dropna(subset=["target_state"])
        test = test.dropna(subset=["target_state"])
        if train.empty or validation.empty or test.empty:
            continue
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            scores[alpha] = _metrics(
                validation.target_state, _predict(train, validation, alpha)
            )["balanced_accuracy"]
        alpha = max(scores, key=scores.get)
        actual = test.target_state.to_numpy(int)
        learned = _predict(pd.concat([train, validation]), test, alpha)
        persistence = test.current_state.to_numpy(int)
        validation_persistence = validation.current_state.to_numpy(int)
        selected = (persistence if _metrics(validation.target_state, validation_persistence)[
            "balanced_accuracy"] >= scores[alpha] else learned)
        folds.append({
            "validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
            "test_period": f"{test_periods[0]}/{test_periods[-1]}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "state_thresholds": thresholds.tolist(),
            "selected_model": "persistence" if selected is persistence else "logistic_one_vs_rest",
            "model": _metrics(actual, selected),
            "persistence": _metrics(actual, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    accuracy = float(np.mean([fold["model"]["accuracy"] for fold in folds]))
    balanced = float(np.mean([fold["model"]["balanced_accuracy"] for fold in folds]))
    counts = {str(state): sum(fold["model"]["state_counts"][str(state)] for fold in folds)
              for state in range(3)}
    return {
        "protocol": "rolling_origin_next_month_nyfed_gscpi_state",
        "fold_count": len(folds), "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "state_counts_in_scored_rows": counts,
        "publishable_candidate": bool(len(folds) >= 3 and accuracy >= 0.65
                                       and balanced >= 0.65 and all(counts.values())),
        "scope": "global supply-chain pressure context; not drug-specific or pharmacy inventory",
        "state_definition": {"0": "low", "1": "mid", "2": "high"},
        "state_thresholds": thresholds.tolist(),
        "state_threshold_semantics": "GSCPI below 0, 0 through 1, or above 1 standardized index units",
        "folds": folds,
    }
