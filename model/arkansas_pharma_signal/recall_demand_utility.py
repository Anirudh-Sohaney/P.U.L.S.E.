"""Leak-safe FDA recall context ablation for public Arkansas demand.

The public demand panel is quarterly. FDA recall pressure is observed monthly,
so this module aggregates recall states to an NDC-quarter and compares a
history-only demand model with the same model augmented by recall pressure.
The experiment tests covariate usefulness, not private pharmacy inventory
accuracy. A same-quarter feature is available at the demand forecast origin;
the lagged feature is a stricter, one-quarter-behind alternative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .public_demand_utility import BASE_FEATURES, _fit_predict, _metrics, build_public_demand_view
from .recall_pressure import build_recall_panel, load_recall_events


def build_recall_demand_view(
    demand: pd.DataFrame | Path,
    recall_events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach current and prior-quarter recall states without future joins."""
    view = build_public_demand_view(demand)
    # Public CSVs may lose leading NDC zeros during generation; restore the
    # canonical nine-digit key before joining FDA event observations.
    view["ndc9"] = view["ndc9"].astype(str).str.zfill(9)
    recall = build_recall_panel(recall_events, include_source_edge=True)
    recall = recall.rename(columns={"ndc": "ndc9"}).copy()
    recall["ndc9"] = recall["ndc9"].astype(str).str.zfill(9)
    recall["year"] = recall["month"].dt.year
    recall["quarter"] = recall["month"].dt.quarter
    quarterly = (recall.groupby(["ndc9", "year", "quarter"], as_index=False)
                 .agg(recall_state=("recall_state", "max"),
                      recall_active_months=("recall_state", lambda x: int((x > 0).sum())),
                      recall_class_i_months=("recall_state", lambda x: int((x == 2).sum()))))
    quarterly["qid"] = quarterly["year"] * 4 + quarterly["quarter"]
    quarterly = quarterly.sort_values(["ndc9", "qid"])
    prior = quarterly[["ndc9", "qid", "recall_state", "recall_active_months",
                       "recall_class_i_months"]].copy()
    prior["qid"] = prior["qid"] + 1
    prior = prior.rename(columns={
        "recall_state": "prior_recall_state",
        "recall_active_months": "prior_recall_active_months",
        "recall_class_i_months": "prior_recall_class_i_months",
    })
    view = view.merge(
        quarterly.drop(columns="qid"), on=["ndc9", "year", "quarter"], how="left")
    view["qid"] = view["year"] * 4 + view["quarter"]
    view = view.merge(prior, on=["ndc9", "qid"], how="left")
    recall_columns = ["recall_state", "recall_active_months", "recall_class_i_months",
                      "prior_recall_state", "prior_recall_active_months",
                      "prior_recall_class_i_months"]
    for column in recall_columns:
        view[column] = pd.to_numeric(view[column], errors="coerce").fillna(0.0)
    view["recall_current_observed"] = view["recall_state"]
    view["recall_lagged_observed"] = view["prior_recall_state"]
    return view


def _evaluate_variant(view: pd.DataFrame, extra_features: list[str]) -> dict[str, Any]:
    augmented = [*BASE_FEATURES, *extra_features]
    folds = []
    for train_year, validation_year, test_year in (
        (2017, 2018, 2019), (2018, 2019, 2020),
        (2019, 2020, 2021), (2020, 2021, 2022),
    ):
        train = view[view["year"] <= train_year]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        selected = {}
        for name, features in (("base", BASE_FEATURES), ("augmented", augmented)):
            scores = {
                alpha: _metrics(validation["target"], _fit_predict(
                    train, validation, features, alpha)[0])["wape"]
                for alpha in (0.1, 1.0, 10.0, 100.0)
            }
            selected[name] = min(scores, key=scores.get)
        train_validation = pd.concat([train, validation], ignore_index=True)
        base_prediction, _ = _fit_predict(
            train_validation, test, BASE_FEATURES, selected["base"])
        augmented_prediction, _ = _fit_predict(
            train_validation, test, augmented, selected["augmented"])
        base = _metrics(test["target"], base_prediction)
        augmented_metrics = _metrics(test["target"], augmented_prediction)
        folds.append({
            "train_last_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": base, "augmented": augmented_metrics,
            "wape_delta_augmented_minus_base": augmented_metrics["wape"] - base["wape"],
        })
    if not folds:
        raise ValueError("no complete recall-demand folds")
    deltas = [fold["wape_delta_augmented_minus_base"] for fold in folds]
    return {
        "feature_count": len(extra_features),
        "features": extra_features,
        "fold_count": len(folds),
        "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_base_wape": float(np.mean([fold["base"]["wape"] for fold in folds])),
        "mean_augmented_wape": float(np.mean([
            fold["augmented"]["wape"] for fold in folds])),
        "mean_wape_delta_augmented_minus_base": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta < 0 for delta in deltas)),
        "folds": folds,
    }


def evaluate_recall_demand_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Evaluate current and lagged recall context against history-only demand."""
    current = _evaluate_variant(view, [
        "recall_current_observed", "recall_active_months", "recall_class_i_months",
    ])
    lagged = _evaluate_variant(view, [
        "recall_lagged_observed", "prior_recall_active_months",
        "prior_recall_class_i_months",
    ])
    return {
        "protocol": "rolling_origin_public_arkansas_ndc_demand_recall_ablation",
        "scope": "public Arkansas Medicaid demand utility experiment; not private inventory truth",
        "current_quarter": current,
        "one_quarter_lag": lagged,
        "publishable_candidate": False,
    }


def load_recall_event_files(paths: Iterable[Path]) -> pd.DataFrame:
    """Load event files through the canonical recall parser."""
    return load_recall_events(list(paths))
