"""Leakage-safe evaluation for Arkansas Medicaid-exposed shortage states."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .evaluate import (
    _auprc, _auroc, _binary_metrics, _brier, _select_balanced_binary_threshold,
    _select_binary_threshold,
)
from .regression import LogisticRidge


FEATURE_COLUMNS = [
    "current_shortage_active",
    "lag1_shortage_active",
    "lag2_shortage_active",
    "trailing_shortage_4",
    "current_supplier_count",
    "lag1_supplier_count",
    "current_archive_row_observed",
    "lag1_archive_row_observed",
    "log_medicaid_prescriptions",
    "quarter",
]


def build_next_quarter_exposure_view(panel: pd.DataFrame,
                                     onset_only: bool = False) -> pd.DataFrame:
    """Build NDC9 feature-quarter -> next consecutive target-quarter rows.

    With ``onset_only``, rows already in an observed FDA shortage at feature
    time are excluded and the target is a new next-quarter observed event.
    This removes persistence from the acceptance diagnostic.
    """
    required = {"ndc9", "year", "quarter", "medicaid_prescriptions",
                "fda_shortage_active"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"exposure panel missing columns: {sorted(missing)}")
    frame = panel.copy().sort_values(["ndc9", "year", "quarter"])
    # These are optional in legacy fixtures but are observed at feature time
    # in the archived FDA exposure panel.
    for column in ("fda_shortage_supplier_count", "fda_archive_row_observed"):
        if column not in frame.columns:
            frame[column] = 0.0
    frame["q_id"] = (frame["year"] - 2012) * 4 + frame["quarter"] - 1
    groups = frame.groupby("ndc9", sort=False)
    previous_q = groups["q_id"].shift(1)
    next_q = groups["q_id"].shift(-1)
    frame["lag1_shortage_active"] = groups["fda_shortage_active"].shift(1)
    frame["lag2_shortage_active"] = groups["fda_shortage_active"].shift(2)
    frame["trailing_shortage_4"] = groups["fda_shortage_active"].transform(
        lambda series: series.shift(1).rolling(4, min_periods=1).sum())
    frame["current_shortage_active"] = frame["fda_shortage_active"]
    frame["current_supplier_count"] = pd.to_numeric(
        frame["fda_shortage_supplier_count"], errors="coerce").fillna(0.0)
    frame["lag1_supplier_count"] = groups["fda_shortage_supplier_count"].shift(1)
    frame["current_archive_row_observed"] = pd.to_numeric(
        frame["fda_archive_row_observed"], errors="coerce").fillna(0.0)
    frame["lag1_archive_row_observed"] = groups["fda_archive_row_observed"].shift(1)
    frame["target"] = groups["fda_shortage_active"].shift(-1)
    frame["target_year"] = groups["year"].shift(-1)
    valid = next_q.sub(frame["q_id"]).eq(1)
    frame = frame[valid].copy()
    if onset_only:
        frame = frame[frame["current_shortage_active"].eq(0)].copy()
        frame["target"] = frame["target"].eq(1).astype(int)
    for column in ("lag1_shortage_active", "lag2_shortage_active",
                   "lag1_supplier_count", "lag1_archive_row_observed",
                   "trailing_shortage_4"):
        frame[column] = frame[column].fillna(0.0)
    frame["log_medicaid_prescriptions"] = np.log1p(
        pd.to_numeric(frame["medicaid_prescriptions"], errors="coerce").clip(lower=0))
    output_columns = list(dict.fromkeys(
        ["ndc9", "year", "quarter", "target_year", *FEATURE_COLUMNS, "target"]
    ))
    return frame.dropna(subset=["target"])[output_columns].reset_index(drop=True)


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    binary = _binary_metrics(y_true, probability, threshold)
    return {
        "accuracy": binary["binary_accuracy"],
        "balanced_accuracy": binary["balanced_accuracy"],
        "precision": binary["binary_precision"],
        "recall": binary["binary_recall"],
        "f1": binary["binary_f1"],
        "auroc": _auroc(y_true, probability),
        "auprc": _auprc(y_true, probability),
        "brier": _brier(y_true, np.clip(probability, 0.0, 1.0)),
        "threshold": float(threshold),
        "positive_rows": int(y_true.sum()),
        "rows": int(len(y_true)),
    }


def evaluate_next_quarter_exposure(view: pd.DataFrame, train_end_year: int = 2019,
                                   validation_year: int = 2020,
                                   test_start_year: int = 2021,
                                   task: str = "active_state") -> dict:
    """Chronological evaluation; threshold selection uses validation only."""
    train = view[view["year"] <= train_end_year]
    validation = view[view["year"] == validation_year]
    test = view[view["year"] >= test_start_year]
    if train.empty or validation.empty or test.empty:
        raise ValueError("exposure chronological split is missing a required period")
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_validation = validation[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    y_validation = validation["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)
    sample_weight = None
    threshold_selector = _select_binary_threshold
    model_variant = "logistic_ridge"
    if task == "new_shortage_onset":
        positives = max(float(np.sum(y_train > 0)), 1.0)
        negatives = max(float(np.sum(y_train <= 0)), 1.0)
        # Inverse-frequency weighting makes rare onset events visible during
        # fitting; the operating threshold is still selected on validation.
        sample_weight = np.where(y_train > 0, negatives / positives, 1.0)
        threshold_selector = _select_balanced_binary_threshold
        model_variant = "class_weighted_logistic_ridge"
    base_model = LogisticRidge(alpha=10.0, max_iter=100).fit(
        x_train, y_train, FEATURE_COLUMNS)
    base_validation_probability = base_model.predict_proba(x_validation)
    base_selector = (_select_balanced_binary_threshold
                     if task == "new_shortage_onset"
                     else _select_binary_threshold)
    base_threshold = base_selector(y_validation, base_validation_probability)
    model = base_model
    validation_probability = base_validation_probability
    threshold = base_threshold
    if sample_weight is not None:
        weighted_model = LogisticRidge(alpha=10.0, max_iter=100).fit(
            x_train, y_train, FEATURE_COLUMNS, sample_weight=sample_weight)
        weighted_validation_probability = weighted_model.predict_proba(x_validation)
        weighted_threshold = threshold_selector(
            y_validation, weighted_validation_probability)
        base_validation_metrics = _metrics(
            y_validation, base_validation_probability, base_threshold)
        weighted_validation_metrics = _metrics(
            y_validation, weighted_validation_probability, weighted_threshold)
        base_key = (base_validation_metrics["balanced_accuracy"],
                    base_validation_metrics["auroc"])
        weighted_key = (weighted_validation_metrics["balanced_accuracy"],
                        weighted_validation_metrics["auroc"])
        if weighted_key > base_key:
            model = weighted_model
            validation_probability = weighted_validation_probability
            threshold = weighted_threshold
        else:
            model_variant = "logistic_ridge_baseline_after_weighted_selection"
    probability = model.predict_proba(x_test)
    persistence = test["current_shortage_active"].to_numpy(dtype=float)
    model_metrics = _metrics(y_test, probability, threshold)
    persistence_metrics = _metrics(y_test, persistence, 0.5)
    return {
        "split": "strict_next_consecutive_quarter_arkansas_medicaid_exposure",
        "task": task,
        "model_variant": model_variant,
        "train_years": [int(train["year"].min()), int(train["year"].max())],
        "validation_year": int(validation_year),
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "logistic_ridge": model_metrics,
        "previous_quarter_persistence": persistence_metrics,
        "balanced_accuracy_improvement": (
            model_metrics["balanced_accuracy"]
            - persistence_metrics["balanced_accuracy"]
        ),
        "auroc_improvement": model_metrics["auroc"] - persistence_metrics["auroc"],
        "skill_claim_supported": bool(
            model_metrics["balanced_accuracy"]
            >= persistence_metrics["balanced_accuracy"] + 0.01
            and model_metrics["auroc"] >= persistence_metrics["auroc"] + 0.01
        ),
        "scope": "Arkansas Medicaid-exposed NDC9; FDA shortage evidence is national",
    }


def evaluate_next_quarter_exposure_rolling(
    view: pd.DataFrame,
    min_train_years: int = 4,
    task: str = "active_state",
) -> dict:
    """Evaluate shortage state over chronological one-year test folds.

    Each fold fits through the year before validation, selects its threshold on
    the validation year, and scores only the immediately following year. This
    prevents the fixed split from hiding temporal prevalence or policy drift.
    """
    if min_train_years < 1:
        raise ValueError("min_train_years must be positive")
    years = sorted(int(x) for x in view["year"].dropna().unique())
    folds = []
    for validation_year in years[min_train_years:]:
        test_year = validation_year + 1
        if test_year not in years:
            continue
        fold_view = view[view["year"].le(test_year)].copy()
        try:
            result = evaluate_next_quarter_exposure(
                fold_view,
                train_end_year=validation_year - 1,
                validation_year=validation_year,
                test_start_year=test_year,
                task=task,
            )
        except ValueError:
            continue
        model_metrics = result["logistic_ridge"]
        persistence_metrics = result["previous_quarter_persistence"]
        folds.append({
            "validation_year": validation_year,
            "test_year": test_year,
            "train_rows": result["train_rows"],
            "validation_rows": result["validation_rows"],
            "test_rows": result["test_rows"],
            "model": model_metrics,
            "persistence": persistence_metrics,
        })
    if not folds:
        return {
            "split": "rolling_origin_next_consecutive_quarter_arkansas_exposure",
            "task": task, "fold_count": 0, "folds": [],
            "requested_accuracy_goal_met": False,
            "promotion_candidate": False,
        }
    model_accuracy = float(np.mean([f["model"]["accuracy"] for f in folds]))
    model_balanced = float(np.mean([f["model"]["balanced_accuracy"] for f in folds]))
    model_auroc = float(np.nanmean([f["model"]["auroc"] for f in folds]))
    persistence_accuracy = float(np.mean([
        f["persistence"]["accuracy"] for f in folds]))
    persistence_balanced = float(np.mean([
        f["persistence"]["balanced_accuracy"] for f in folds]))
    persistence_auroc = float(np.nanmean([
        f["persistence"]["auroc"] for f in folds]))
    requested_goal = bool(model_accuracy >= 0.75 and model_balanced >= 0.75)
    skill_supported = bool(
        model_balanced >= persistence_balanced + 0.01
        and model_auroc >= persistence_auroc + 0.01)
    return {
        "split": "rolling_origin_next_consecutive_quarter_arkansas_exposure",
        "task": task,
        "min_train_years": int(min_train_years),
        "fold_count": len(folds),
        "folds": folds,
        "mean_model_accuracy": model_accuracy,
        "mean_model_balanced_accuracy": model_balanced,
        "mean_model_auroc": model_auroc,
        "mean_persistence_accuracy": persistence_accuracy,
        "mean_persistence_balanced_accuracy": persistence_balanced,
        "mean_persistence_auroc": persistence_auroc,
        "balanced_accuracy_improvement": model_balanced - persistence_balanced,
        "auroc_improvement": model_auroc - persistence_auroc,
        "requested_accuracy_threshold": 0.75,
        "requested_accuracy_goal_met": requested_goal,
        "skill_claim_supported": skill_supported,
        "promotion_candidate": bool(requested_goal and skill_supported),
        "scope": "Arkansas Medicaid-exposed NDC9; FDA shortage evidence is national",
    }
