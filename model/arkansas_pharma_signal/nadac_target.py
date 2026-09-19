"""Leak-safe short-horizon evaluation for CMS NADAC price pressure.

This is an auxiliary numeric target. NADAC is a national acquisition-cost
series, not an observation of Arkansas demand, local inventory, or shortage.
Rows are retained only when the next observation for the same NDC is exactly
seven days later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from .regression import RidgeLinear

FEATURE_COLUMNS = ["log_price", "log_price_delta", "price_obs"]


def _ndc9(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(9)[:9] if digits else ""


def load_nadac_transitions(
    path: Path, ndc_filter: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Return NDC rows whose next observed NADAC date is exactly seven days."""
    frame = pd.read_csv(path, usecols=["ndc", "nadac_per_unit", "as_of_date"])
    frame["ndc"] = frame["ndc"].map(_ndc9)
    frame["price"] = pd.to_numeric(frame["nadac_per_unit"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame[frame["ndc"].ne("") & frame["price"].gt(0)
                  & frame["date"].notna()].copy()
    if ndc_filter is not None:
        allowed: Set[str] = {_ndc9(value) for value in ndc_filter}
        frame = frame[frame["ndc"].isin(allowed)]
    # Multiple source rows can describe one NDC/date observation.
    frame = (frame.groupby(["ndc", "date"], as_index=False)["price"].mean()
             .sort_values(["ndc", "date"]))
    group = frame.groupby("ndc", sort=False)
    frame["next_date"] = group["date"].shift(-1)
    frame["target_price"] = group["price"].shift(-1)
    frame["gap_days"] = (frame["next_date"] - frame["date"]).dt.days
    frame = frame[frame["gap_days"].eq(7)].copy()
    frame["log_price"] = np.log1p(frame["price"])
    frame["log_price_delta"] = group["price"].shift(0).div(
        group["price"].shift(1)
    ).replace([np.inf, -np.inf], np.nan).sub(1).fillna(0.0)
    frame["price_obs"] = 1.0
    return frame.reset_index(drop=True)


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1e-12))


def evaluate_nadac_transitions(
    transitions: pd.DataFrame,
    train_end: str = "2021-06-30",
    validation_end: str = "2021-12-31",
) -> dict:
    """Fit on the past, select alpha on validation, and score the later test."""
    data = transitions.sort_values("date").reset_index(drop=True)
    train = data[data["date"] <= pd.Timestamp(train_end)]
    validation = data[(data["date"] > pd.Timestamp(train_end))
                      & (data["date"] <= pd.Timestamp(validation_end))]
    test = data[data["date"] > pd.Timestamp(validation_end)]
    if train.empty or validation.empty or test.empty:
        raise ValueError("NADAC transition data must contain train, validation, and test rows")
    model = RidgeLinear(alpha=1.0).fit(
        train[FEATURE_COLUMNS].to_numpy(),
        np.log1p(train["target_price"].to_numpy()),
        feature_names=FEATURE_COLUMNS,
    )
    # Alpha is intentionally selected without looking at the held-out test.
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    best_alpha, best_error = None, float("inf")
    for alpha in candidates:
        candidate = RidgeLinear(alpha=alpha).fit(
            train[FEATURE_COLUMNS].to_numpy(),
            np.log1p(train["target_price"].to_numpy()),
            feature_names=FEATURE_COLUMNS,
        )
        prediction = np.maximum(np.expm1(candidate.predict(
            validation[FEATURE_COLUMNS].to_numpy())), 0.0)
        error = _wape(validation["target_price"].to_numpy(), prediction)
        if error < best_error:
            best_alpha, best_error = alpha, error
    model = RidgeLinear(alpha=best_alpha).fit(
        train[FEATURE_COLUMNS].to_numpy(),
        np.log1p(train["target_price"].to_numpy()),
        feature_names=FEATURE_COLUMNS,
    )
    actual = test["target_price"].to_numpy()
    model_prediction = np.maximum(np.expm1(model.predict(
        test[FEATURE_COLUMNS].to_numpy())), 0.0)
    persistence = test["price"].to_numpy()

    def metrics(prediction: np.ndarray) -> dict:
        ape = np.abs(actual - prediction) / np.maximum(actual, 1e-12)
        return {
            "wape": _wape(actual, prediction),
            "under_5_percent_error": float(np.mean(ape < 0.05)),
            "median_absolute_percentage_error": float(np.median(ape)),
        }

    model_metrics = metrics(model_prediction)
    persistence_metrics = metrics(persistence)
    return {
        "target": "next_observed_nadac_price_exactly_7_days",
        "scope": "national_acquisition_cost_proxy",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_ndcs": int(test["ndc"].nunique()),
        "selected_alpha": best_alpha,
        "model": model_metrics,
        "persistence": persistence_metrics,
        "model_beats_persistence_wape": model_metrics["wape"] < persistence_metrics["wape"],
    }


def evaluate_nadac_change_state_rolling(
    transitions: pd.DataFrame,
    folds: Optional[Iterable[tuple[str, str, str]]] = None,
) -> dict:
    """Screen a five-state weekly NADAC movement target.

    Quantile cut-points are fit-only. Duplicate cut-points are reported as a
    structural rejection rather than silently producing fewer than five states.
    """
    if folds is None:
        folds = (
            ("2021-12-31", "2022-12-31", "2023-12-31"),
            ("2022-12-31", "2023-12-31", "2024-12-31"),
            ("2023-12-31", "2024-12-31", "2025-12-31"),
        )
    data = transitions.sort_values("date").copy()
    data["target_change"] = data["target_price"] / data["price"] - 1.0
    results = []
    rejection_reasons = []
    for train_end, validation_end, test_end in folds:
        fit = data[data["date"] <= pd.Timestamp(train_end)]
        test = data[(data["date"] > pd.Timestamp(validation_end))
                    & (data["date"] <= pd.Timestamp(test_end))]
        if fit.empty or test.empty:
            continue
        thresholds = np.quantile(fit["target_change"], np.arange(1, 5) / 5)
        actual = np.digitize(test["target_change"], thresholds)
        prediction = np.digitize(np.zeros(len(test)), thresholds)
        observed_states = sorted(np.unique(actual).astype(int).tolist())
        recalls = [
            float(np.mean(prediction[actual == state] == state))
            if np.any(actual == state) else 0.0
            for state in range(5)
        ]
        results.append({
            "train_end": str(train_end), "validation_end": str(validation_end),
            "test_end": str(test_end), "test_rows": int(len(test)),
            "thresholds": [float(value) for value in thresholds],
            "observed_states": observed_states,
            "exact_accuracy": float(np.mean(actual == prediction)),
            "balanced_accuracy": float(np.mean(recalls)),
            "state_counts": {str(state): int(np.sum(actual == state))
                             for state in range(5)},
        })
        if len(np.unique(thresholds)) < 4 or len(observed_states) < 5:
            rejection_reasons.append(
                "fit quantiles collapse and fewer than five states are observed")
    return {
        "protocol": "rolling_origin_next_week_nadac_relative_change_five_state_screen",
        "target": "next_observed_nadac_relative_change_exactly_7_days",
        "scope": "national acquisition-cost movement proxy for Arkansas-exposed NDCs",
        "state_count": 5,
        "fold_count": len(results),
        "test_rows": int(sum(row["test_rows"] for row in results)),
        "folds": results,
        "mean_exact_accuracy": float(np.mean([row["exact_accuracy"] for row in results]))
        if results else None,
        "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in results]))
        if results else None,
        "rejected": bool(rejection_reasons),
        "rejection_reasons": sorted(set(rejection_reasons)),
    }


def evaluate_nadac_rolling(
    transitions: pd.DataFrame,
    folds: Optional[Iterable[tuple[str, str, str]]] = None,
) -> dict:
    """Evaluate exact-seven-day price transitions over chronological folds.

    Each tuple contains inclusive train, validation, and test end dates. Alpha
    is selected on validation only; the test period is never used for model
    selection. The returned metrics retain persistence as the required
    short-horizon baseline.
    """
    if folds is None:
        folds = (
            ("2018-12-31", "2019-12-31", "2020-12-31"),
            ("2019-12-31", "2020-12-31", "2021-12-31"),
            ("2020-12-31", "2021-12-31", "2022-12-31"),
        )
    data = transitions.sort_values("date").reset_index(drop=True)
    results = []
    for train_end, validation_end, test_end in folds:
        train = data[data["date"] <= pd.Timestamp(train_end)]
        validation = data[(data["date"] > pd.Timestamp(train_end))
                          & (data["date"] <= pd.Timestamp(validation_end))]
        test = data[(data["date"] > pd.Timestamp(validation_end))
                     & (data["date"] <= pd.Timestamp(test_end))]
        if train.empty or validation.empty or test.empty:
            continue
        candidates = {}
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                train[FEATURE_COLUMNS].to_numpy(),
                np.log1p(train["target_price"].to_numpy()),
                feature_names=FEATURE_COLUMNS)
            prediction = np.maximum(np.expm1(model.predict(
                validation[FEATURE_COLUMNS].to_numpy())), 0.0)
            candidates[alpha] = _wape(validation["target_price"].to_numpy(), prediction)
        selected_alpha = min(candidates, key=candidates.get)
        model = RidgeLinear(alpha=selected_alpha).fit(
            train[FEATURE_COLUMNS].to_numpy(),
            np.log1p(train["target_price"].to_numpy()),
            feature_names=FEATURE_COLUMNS)
        actual = test["target_price"].to_numpy()
        model_prediction = np.maximum(np.expm1(model.predict(
            test[FEATURE_COLUMNS].to_numpy())), 0.0)
        persistence = test["price"].to_numpy()

        def metrics(prediction: np.ndarray) -> dict:
            ape = np.abs(actual - prediction) / np.maximum(actual, 1e-12)
            return {
                "wape": _wape(actual, prediction),
                "under_5_percent_error": float(np.mean(ape < 0.05)),
                "median_absolute_percentage_error": float(np.median(ape)),
            }

        model_metrics, persistence_metrics = metrics(model_prediction), metrics(persistence)
        results.append({
            "train_end": str(train_end),
            "validation_end": str(validation_end),
            "test_end": str(test_end),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "test_ndcs": int(test["ndc"].nunique()),
            "selected_alpha": selected_alpha,
            "model": model_metrics,
            "persistence": persistence_metrics,
            "model_beats_persistence_wape": model_metrics["wape"] < persistence_metrics["wape"],
        })
    if not results:
        raise ValueError("no complete NADAC rolling folds")
    return {
        "target": "next_observed_nadac_price_exactly_7_days",
        "scope": "national_acquisition_cost_proxy_for_arkansas_exposed_ndcs",
        "fold_count": len(results),
        "minimum_required_folds": 3,
        "folds": results,
        "mean_model_wape": float(np.mean([r["model"]["wape"] for r in results])),
        "mean_persistence_wape": float(np.mean([r["persistence"]["wape"] for r in results])),
        "mean_model_under_5_percent_error": float(np.mean([
            r["model"]["under_5_percent_error"] for r in results])),
        "mean_persistence_under_5_percent_error": float(np.mean([
            r["persistence"]["under_5_percent_error"] for r in results])),
        "model_beats_persistence_all_folds": bool(all(
            r["model_beats_persistence_wape"] for r in results)),
        "publishable_rolling_candidate": bool(
            len(results) >= 3 and all(r["model_beats_persistence_wape"]
                                      for r in results)),
        "promotion_reason": (
            "at least three complete folds beat persistence"
            if len(results) >= 3 and all(r["model_beats_persistence_wape"]
                                         for r in results)
            else "research-only: insufficient folds or no all-fold improvement"),
    }
