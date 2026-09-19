"""Weekly Arkansas wastewater viral-activity proxy evaluation.

CDC wastewater site activity is an upstream respiratory-utilization signal, not
pharmacy dispensing. Site means are aggregated by pathogen and week, then a
strict rolling-origin evaluator forecasts the next week's five-quantile state.
All thresholds are learned from the training partition for each fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge
from .event_accuracy import score_event_predictions


PATHOGENS = {
    "influenza": "Influenza A virus",
    "rsv": "RSV",
    "sars_cov_2": "SARS-CoV-2",
}
FEATURES = ("current_value", "lag1_value", "lag2_value", "lag4_value",
            "rolling4_value", "week_sin", "week_cos")


def build_wastewater_weekly_view(source: pd.DataFrame, *, pathogen: str) -> pd.DataFrame:
    """Aggregate Arkansas wastewater site values into consecutive weekly rows."""
    if pathogen not in PATHOGENS:
        raise ValueError(f"unsupported pathogen: {pathogen}")
    required = {"week_end", "pathogen_target", "site_wval"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"wastewater data missing columns: {sorted(missing)}")
    frame = source[source["pathogen_target"].eq(PATHOGENS[pathogen])].copy()
    frame["week"] = pd.to_datetime(frame["week_end"], errors="coerce").dt.to_period("W-SAT")
    frame["value"] = pd.to_numeric(frame["site_wval"], errors="coerce")
    frame = frame.dropna(subset=["week", "value"])
    frame = frame.groupby("week", as_index=False)["value"].mean().sort_values("week")
    frame["next_week"] = frame["week"].shift(-1)
    frame["target"] = frame["value"].shift(-1).where(
        frame["next_week"].eq(frame["week"] + 1))
    frame = frame.dropna(subset=["target"]).copy()
    frame["current_value"] = frame["value"]
    group = frame["value"]
    frame["lag1_value"] = group.shift(1).fillna(group)
    frame["lag2_value"] = group.shift(2).fillna(group)
    frame["lag4_value"] = group.shift(4).fillna(group)
    frame["rolling4_value"] = group.shift(1).rolling(4, min_periods=1).mean().fillna(group)
    frame["week_sin"] = np.sin(2 * np.pi * frame["week"].dt.week / 52.0)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week"].dt.week / 52.0)
    frame["year"] = frame["week"].dt.year.astype(int)
    return frame.reset_index(drop=True)


def _state(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), thresholds).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray, state_count: int = 5) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(state_count):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state))
                         for state in range(state_count)},
    }


def _predict(train: pd.DataFrame, target: pd.DataFrame,
             thresholds: tuple[float, ...], alpha: float) -> np.ndarray:
    labels = _state(train["target"].to_numpy(float), thresholds)
    probabilities = []
    for state in range(5):
        model = LogisticRidge(alpha=alpha).fit(
            train[list(FEATURES)].to_numpy(float), (labels == state).astype(float),
            list(FEATURES))
        probabilities.append(model.predict_proba(target[list(FEATURES)].to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_wastewater_state_rolling(
    view: pd.DataFrame, *, min_train_rows: int = 52,
    validation_rows: int = 13, test_rows: int = 13, step_rows: int = 13,
) -> dict[str, Any]:
    """Evaluate five-state next-week wastewater activity chronologically."""
    periods = sorted(view["week"].unique())
    folds = []
    event_actual: list[int] = []
    event_predicted: list[int] = []
    for index in range(min_train_rows, len(periods) - validation_rows - test_rows + 1,
                       max(1, step_rows)):
        train = view[view["week"].isin(periods[:index])]
        validation = view[view["week"].isin(periods[index:index + validation_rows])]
        test = view[view["week"].isin(
            periods[index + validation_rows:index + validation_rows + test_rows])]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = tuple(np.quantile(train["target"], np.arange(1, 5) / 5))
        validation_actual = _state(validation["target"], thresholds)
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            scores[alpha] = _metrics(
                validation_actual, _predict(train, validation, thresholds, alpha)
            )["balanced_accuracy"]
        alpha = max(scores, key=scores.get)
        actual = _state(test["target"], thresholds)
        learned = _predict(pd.concat([train, validation]), test, thresholds, alpha)
        persistence = _state(test["current_value"], thresholds)
        validation_persistence = _state(validation["current_value"], thresholds)
        selected = (persistence if _metrics(
            validation_actual, validation_persistence)["balanced_accuracy"] >= scores[alpha]
            else learned)
        event_actual.extend(actual.tolist())
        event_predicted.extend(selected.tolist())
        folds.append({
            "validation_start": str(validation["week"].min()),
            "validation_end": str(validation["week"].max()),
            "test_start": str(test["week"].min()),
            "test_end": str(test["week"].max()),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_alpha": float(alpha),
            "selected_model": "persistence" if selected is persistence else "logistic_one_vs_rest",
            "thresholds": [float(value) for value in thresholds],
            "model": _metrics(actual, selected),
            "persistence": _metrics(actual, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    counts = {str(state): sum(fold["model"]["state_counts"][str(state)] for fold in folds)
              for state in range(5)}
    accuracy = float(np.mean([fold["model"]["accuracy"] for fold in folds]))
    balanced = float(np.mean([fold["model"]["balanced_accuracy"] for fold in folds]))
    event_metrics = score_event_predictions(
        np.asarray(event_actual), np.asarray(event_predicted), kind="state",
        event_states={3, 4})
    return {
        "protocol": "rolling_origin_next_week_arkansas_wastewater_five_state",
        "fold_count": len(folds), "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "event_metrics": event_metrics,
        "mean_persistence_accuracy": float(np.mean([
            fold["persistence"]["accuracy"] for fold in folds])),
        "state_counts_in_scored_rows": counts, "state_count": 5,
        "publishable_candidate": bool(len(folds) >= 3 and accuracy >= 0.65
                                       and balanced >= 0.65 and all(counts.values())),
        "scope": "Arkansas statewide wastewater viral-activity proxy; not pharmacy dispensing",
        "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
        "folds": folds,
    }
