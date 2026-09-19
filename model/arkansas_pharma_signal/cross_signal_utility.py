"""Cross-domain utility test: recall severity as a shortage-pressure feature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .recall_pressure import build_recall_panel, load_recall_events
from .regression import LogisticRidge
from .shortage_pressure import (
    STATE_NAMES,
    build_next_month_pressure_view,
    build_pressure_panel,
)


BASE_FEATURES = ["active_supplier_count", "observed_supplier_count",
                 "lag1_pressure", "lag2_pressure", "rolling3_pressure",
                 "calendar_month"]
AUGMENTED_FEATURES = [*BASE_FEATURES, "recall_state"]


def build_shortage_recall_view(shortage: pd.DataFrame,
                               recall_events: pd.DataFrame) -> pd.DataFrame:
    """Join only NDC/month observations present in both public panels."""
    pressure = build_next_month_pressure_view(build_pressure_panel(shortage))
    recall = build_recall_panel(recall_events)
    recall = recall.rename(columns={"ndc": "ndc9"})[["ndc9", "month", "recall_state"]]
    pressure["ndc9"] = pressure["ndc9"].astype(str).str.zfill(9)
    recall["ndc9"] = recall["ndc9"].astype(str).str.zfill(9)
    result = pressure.merge(recall, on=["ndc9", "month"], how="inner")
    return result.sort_values(["month", "ndc9"]).reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int); predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in STATE_NAMES:
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {"accuracy": float(np.mean(actual == predicted)),
            "balanced_accuracy": float(np.mean(recalls)),
            "state_counts": {str(x): int(np.sum(actual == x)) for x in STATE_NAMES}}


def _predict(train: pd.DataFrame, target: pd.DataFrame, features: list[str],
             alpha: float) -> np.ndarray:
    probabilities = []
    for state in STATE_NAMES:
        model = LogisticRidge(alpha=alpha).fit(
            train[features].fillna(0).to_numpy(float),
            train["target_pressure_state"].eq(state).astype(float), features)
        probabilities.append(model.predict_proba(
            target[features].fillna(0).to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_shortage_recall_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Evaluate baseline versus recall-augmented shortage-state models."""
    periods = sorted(view["month"].unique())
    folds = []
    for index in range(36, len(periods) - 18 + 1, 12):
        train_periods = periods[:index]
        validation_periods = periods[index:index + 6]
        test_periods = periods[index + 6:index + 18]
        train = view[view.month.isin(train_periods)]
        validation = view[view.month.isin(validation_periods)]
        test = view[view.month.isin(test_periods)]
        if train.empty or validation.empty or test.empty:
            continue
        candidates = {}
        for features in (BASE_FEATURES, AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    validation.target_pressure_state,
                    _predict(train, validation, features, alpha)
                )["balanced_accuracy"]
            candidates[tuple(features)] = max(scores, key=scores.get)
        results = {}
        for features in (BASE_FEATURES, AUGMENTED_FEATURES):
            key = "augmented" if features == AUGMENTED_FEATURES else "base"
            results[key] = _metrics(
                test.target_pressure_state,
                _predict(pd.concat([train, validation]), test, features,
                         candidates[tuple(features)]))
        folds.append({"validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
                      "test_period": f"{test_periods[0]}/{test_periods[-1]}",
                      "train_rows": int(len(train)), "validation_rows": int(len(validation)),
                      "test_rows": int(len(test)), "base": results["base"],
                      "augmented": results["augmented"],
                      "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
                      - results["base"]["balanced_accuracy"]})
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    return {
        "protocol": "rolling_origin_shortage_pressure_recall_incremental_utility",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "ndc_count": int(view["ndc9"].nunique()),
        "mean_base_balanced_accuracy": float(np.mean([
            x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([
            x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean([
            x["balanced_accuracy_delta"] for x in folds])),
        "all_folds_improve": bool(all(x["balanced_accuracy_delta"] > 0 for x in folds)),
        "publishable_candidate": False,
        "scope": "utility experiment only; FDA evidence is not pharmacy inventory truth",
        "folds": folds,
    }
