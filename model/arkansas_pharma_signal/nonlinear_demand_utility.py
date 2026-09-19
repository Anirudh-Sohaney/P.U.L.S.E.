"""Matched nonlinear public-demand ablation.

This module tests whether a compact gradient-tree model can extract utility
from external context after controlling for the model-family improvement. It
uses the same public Arkansas NDC target and chronological folds as the Ridge
ablation; it is research evidence, not a production replacement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .public_demand_utility import BASE_FEATURES, _metrics


def _fit_predict(train: pd.DataFrame, target: pd.DataFrame,
                 features: list[str], leaves: int, learning_rate: float) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=leaves, learning_rate=learning_rate,
        l2_regularization=1.0, random_state=17,
    ).fit(train[features].to_numpy(float), np.log1p(train["target"].to_numpy(float)))
    cap = float(np.quantile(np.log1p(train["target"].to_numpy(float)), 0.995))
    return np.maximum(np.expm1(np.clip(
        model.predict(target[features].to_numpy(float)), 0.0, cap)), 0.0)


def evaluate_nonlinear_demand_context_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Compare history-only and context-augmented gradient-tree models."""
    context = [column for column in view.columns if column.startswith("prior_")]
    augmented = [*BASE_FEATURES, *context]
    folds = []
    for train_year, validation_year, test_year in (
        (2017, 2018, 2019), (2018, 2019, 2020),
        (2019, 2020, 2021), (2020, 2021, 2022)):
        train = view[view["year"] <= train_year]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        selected = {}
        for name, features in (("base", BASE_FEATURES), ("context", augmented)):
            scores = {}
            for leaves in (7, 15):
                for learning_rate in (0.03, 0.1):
                    prediction = _fit_predict(train, validation, features, leaves, learning_rate)
                    scores[(leaves, learning_rate)] = _metrics(
                        validation["target"], prediction)["wape"]
            selected[name] = min(scores, key=scores.get)
        combined = pd.concat([train, validation], ignore_index=True)
        base_prediction = _fit_predict(combined, test, BASE_FEATURES, *selected["base"])
        context_prediction = _fit_predict(combined, test, augmented, *selected["context"])
        base = _metrics(test["target"], base_prediction)
        context_metrics = _metrics(test["target"], context_prediction)
        folds.append({
            "train_last_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": base, "context": context_metrics,
            "wape_delta_context_minus_base": context_metrics["wape"] - base["wape"],
            "within_5_delta_context_minus_base": context_metrics["within_5_percent"] - base["within_5_percent"],
            "selected_base": {"max_leaf_nodes": selected["base"][0], "learning_rate": selected["base"][1]},
            "selected_context": {"max_leaf_nodes": selected["context"][0], "learning_rate": selected["context"][1]},
        })
    deltas = [fold["wape_delta_context_minus_base"] for fold in folds]
    return {
        "protocol": "rolling_origin_public_arkansas_ndc_demand_nonlinear_context_ablation",
        "model": "sklearn HistGradientBoostingRegressor",
        "context_feature_count": len(context), "fold_count": len(folds),
        "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_base_wape": float(np.mean([fold["base"]["wape"] for fold in folds])),
        "mean_context_wape": float(np.mean([fold["context"]["wape"] for fold in folds])),
        "mean_wape_delta_context_minus_base": float(np.mean(deltas)),
        "all_folds_context_wape_improves": bool(all(delta < 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "public Arkansas Medicaid NDC demand utility experiment; not private pharmacy inventory truth",
        "folds": folds,
    }
