"""Public Arkansas NDC-demand ablation for external-context utility.

This is the closest openly labeled demand test for the intended pharmacy use
case. It compares a history-only Ridge baseline with the same model augmented
by prior CDC disease and news variables. It is not private pharmacy inventory
evidence and cannot establish individual-pharmacy lift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .regression import RidgeLinear


BASE_FEATURES = [
    "current_log", "lag1_log", "lag2_log", "ma2_log",
    "fda_shortage_active", "fda_shortage_supplier_count", "quarter",
]


def build_public_demand_view(source: Path | pd.DataFrame) -> pd.DataFrame:
    """Construct strict next-quarter Arkansas NDC demand rows."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, dtype={"ndc9": str})
    required = {"ndc9", "year", "quarter", "medicaid_prescriptions",
                "fda_shortage_active", "fda_shortage_supplier_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"public demand panel missing columns: {sorted(missing)}")
    frame["ndc9"] = frame["ndc9"].astype(str)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["medicaid_prescriptions"], errors="coerce")
    frame["qid"] = frame["year"] * 4 + frame["quarter"]
    frame = frame.dropna(subset=["ndc9", "year", "quarter", "value"]).sort_values(["ndc9", "qid"])
    group = frame.groupby("ndc9", sort=False)
    previous_qid = group["qid"].shift(1)
    next_qid = group["qid"].shift(-1)
    frame["target"] = group["value"].shift(-1).where(next_qid.sub(frame["qid"]).eq(1))
    frame["current_log"] = np.log1p(frame["value"].clip(lower=0))
    frame["lag1_log"] = np.log1p(group["value"].shift(1).where(
        frame["qid"].sub(previous_qid).eq(1)).fillna(frame["value"]).clip(lower=0))
    previous_two_qid = group["qid"].shift(2)
    frame["lag2_log"] = np.log1p(group["value"].shift(2).where(
        frame["qid"].sub(previous_two_qid).eq(2)).fillna(frame["value"]).clip(lower=0))
    frame["ma2_log"] = (frame["current_log"] + frame["lag1_log"]) / 2.0
    context = [column for column in frame.columns if column.startswith("prior_")]
    features = [*BASE_FEATURES, *context]
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["target"]).reset_index(drop=True)


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    relative = np.abs(actual - prediction) / np.maximum(np.abs(actual), 1e-12)
    return {
        "wape": float(np.abs(actual - prediction).sum() / max(np.abs(actual).sum(), 1e-12)),
        "mae": float(np.mean(np.abs(actual - prediction))),
        "within_5_percent": float(np.mean(relative < 0.05)),
    }


def _fit_predict(train: pd.DataFrame, target: pd.DataFrame,
                 features: list[str], alpha: float) -> tuple[np.ndarray, float]:
    model = RidgeLinear(alpha=alpha).fit(
        train[features].to_numpy(float), np.log1p(train["target"].to_numpy(float)), features)
    # Cap only from the training target distribution; this prevents a sparse
    # context coefficient from generating impossible future values.
    cap = float(np.quantile(np.log1p(train["target"].to_numpy(float)), 0.995))
    prediction = np.expm1(np.clip(model.predict(target[features].to_numpy(float)), 0.0, cap))
    return np.maximum(prediction, 0.0), cap


def evaluate_public_demand_context_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Run four chronological public NDC-demand context ablation folds."""
    context_features = [column for column in view.columns if column.startswith("prior_")]
    augmented_features = [*BASE_FEATURES, *context_features]
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
        for name, features in (("base", BASE_FEATURES), ("context", augmented_features)):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                prediction, _ = _fit_predict(train, validation, features, alpha)
                scores[alpha] = _metrics(validation["target"], prediction)["wape"]
            selected[name] = min(scores, key=scores.get)
        result = {}
        for name, features in (("base", BASE_FEATURES), ("context", augmented_features)):
            prediction, cap = _fit_predict(
                pd.concat([train, validation], ignore_index=True), test,
                features, selected[name])
            result[name] = _metrics(test["target"], prediction)
            result[name]["selected_alpha"] = float(selected[name])
            result[name]["training_prediction_cap_log"] = cap
        folds.append({
            "train_last_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "train_rows": int(len(train)),
            "validation_rows": int(len(validation)), "test_rows": int(len(test)),
            "base": result["base"], "context": result["context"],
            "wape_delta_context_minus_base": result["context"]["wape"] - result["base"]["wape"],
            "within_5_delta_context_minus_base": result["context"]["within_5_percent"] - result["base"]["within_5_percent"],
        })
    if not folds:
        raise ValueError("no complete public demand context folds")
    deltas = [row["wape_delta_context_minus_base"] for row in folds]
    return {
        "protocol": "rolling_origin_public_arkansas_ndc_demand_context_ablation",
        "fold_count": len(folds), "test_rows": int(sum(row["test_rows"] for row in folds)),
        "context_feature_count": len(context_features),
        "mean_base_wape": float(np.mean([row["base"]["wape"] for row in folds])),
        "mean_context_wape": float(np.mean([row["context"]["wape"] for row in folds])),
        "mean_wape_delta_context_minus_base": float(np.mean(deltas)),
        "mean_base_within_5_percent": float(np.mean([row["base"]["within_5_percent"] for row in folds])),
        "mean_context_within_5_percent": float(np.mean([row["context"]["within_5_percent"] for row in folds])),
        "all_folds_context_wape_improves": bool(all(delta < 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "public Arkansas Medicaid NDC demand utility experiment; not private pharmacy inventory truth",
        "folds": folds,
    }


def evaluate_public_demand_context_families(view: pd.DataFrame) -> dict[str, Any]:
    """Evaluate disease and news families separately under the same protocol."""
    groups = {
        "disease": [column for column in view.columns
                    if column.startswith("prior_ar_") or column.startswith("prior_nat_")],
        "news": [column for column in view.columns if column.startswith("prior_news_")],
    }
    output = {}
    for name, extra_features in groups.items():
        augmented = [*BASE_FEATURES, *extra_features]
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
            for label, features in (("base", BASE_FEATURES), (name, augmented)):
                scores = {
                    alpha: _metrics(
                        validation["target"],
                        _fit_predict(train, validation, features, alpha)[0],
                    )["wape"]
                    for alpha in (0.1, 1.0, 10.0, 100.0)
                }
                selected[label] = min(scores, key=scores.get)
            base_prediction, _ = _fit_predict(
                pd.concat([train, validation], ignore_index=True), test,
                BASE_FEATURES, selected["base"])
            augmented_prediction, _ = _fit_predict(
                pd.concat([train, validation], ignore_index=True), test,
                augmented, selected[name])
            base = _metrics(test["target"], base_prediction)
            augmented_metrics = _metrics(test["target"], augmented_prediction)
            folds.append({
                "train_last_year": train_year, "validation_year": validation_year,
                "test_year": test_year, "test_rows": int(len(test)),
                "base": base, "augmented": augmented_metrics,
                "wape_delta_context_minus_base": augmented_metrics["wape"] - base["wape"],
            })
        deltas = [fold["wape_delta_context_minus_base"] for fold in folds]
        output[name] = {
            "feature_count": len(extra_features), "fold_count": len(folds),
            "test_rows": int(sum(fold["test_rows"] for fold in folds)),
            "mean_base_wape": float(np.mean([fold["base"]["wape"] for fold in folds])),
            "mean_augmented_wape": float(np.mean([fold["augmented"]["wape"] for fold in folds])),
            "mean_wape_delta_context_minus_base": float(np.mean(deltas)),
            "all_folds_improve": bool(all(delta < 0 for delta in deltas)),
            "folds": folds,
        }
    return {
        "protocol": "rolling_origin_public_arkansas_ndc_demand_context_family_ablation",
        "scope": "public Arkansas Medicaid NDC demand utility experiment; not private pharmacy inventory truth",
        "families": output,
    }
