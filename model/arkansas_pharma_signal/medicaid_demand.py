"""Chronological evaluation of Arkansas Medicaid quarterly prescription demand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .regression import RidgeLinear


FEATURES = ["lag1", "lag2", "lag4", "rolling4", "quarter_sin", "quarter_cos"]


def load_arkansas_medicaid_quarterly(path: Path, utilization_type: str = "FFSU") -> pd.DataFrame:
    """Load one complete Arkansas quarterly utilization series."""
    frame = pd.read_csv(path)
    required = {"state", "source_year", "quarter", "utilization_type", "number_of_prescriptions"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Medicaid SDUD file missing columns: {sorted(missing)}")
    frame = frame[(frame["state"].astype(str).str.upper() == "AR")
                  & frame["utilization_type"].astype(str).eq(utilization_type)].copy()
    frame["year"] = pd.to_numeric(frame["source_year"], errors="coerce")
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="coerce")
    frame["prescriptions"] = pd.to_numeric(frame["number_of_prescriptions"], errors="coerce")
    frame = frame.dropna(subset=["year", "quarter", "prescriptions"])
    frame["period"] = pd.PeriodIndex(
        frame["year"].astype(int).astype(str) + "Q" + frame["quarter"].astype(int).astype(str),
        freq="Q",
    )
    frame = frame[["period", "year", "quarter", "prescriptions"]].sort_values("period")
    if frame["period"].duplicated().any():
        raise ValueError("Arkansas Medicaid SDUD contains duplicate utilization periods")
    expected = pd.period_range(frame["period"].min(), frame["period"].max(), freq="Q")
    if not frame["period"].array.equals(expected.array):
        raise ValueError("Arkansas Medicaid SDUD history contains missing quarters")
    if len(frame) < 25:
        raise ValueError("Arkansas Medicaid SDUD history must contain at least 25 quarters")
    return frame.reset_index(drop=True)


def _features(values: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=values.index)
    frame["lag1"] = values.shift(1)
    frame["lag2"] = values.shift(2)
    frame["lag4"] = values.shift(4)
    frame["rolling4"] = values.shift(1).rolling(4, min_periods=1).mean()
    quarters = values.index.quarter if isinstance(values.index, pd.PeriodIndex) else np.ones(len(values))
    frame["quarter_sin"] = np.sin(2 * np.pi * quarters / 4)
    frame["quarter_cos"] = np.cos(2 * np.pi * quarters / 4)
    return frame


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    relative_error = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1.0)
    return {
        "within_5_percent": float(np.mean(relative_error <= 0.05)),
        "mean_absolute_percentage_error": float(np.mean(relative_error)),
        "mean_absolute_error": float(np.mean(np.abs(predicted - actual))),
    }


def evaluate_arkansas_medicaid_rolling(
    source: pd.DataFrame, *, min_train_quarters: int = 16,
    validation_quarters: int = 4, test_quarters: int = 4, step_quarters: int = 1,
) -> dict[str, Any]:
    """Evaluate next-quarter prescription volume with rolling-origin folds."""
    frame = source.set_index("period", drop=False).sort_index().copy()
    frame["target"] = frame["prescriptions"].shift(-1)
    feature_frame = _features(frame["prescriptions"])
    frame[FEATURES] = feature_frame[FEATURES]
    periods = list(frame["period"])
    folds = []
    for index in range(min_train_quarters,
                       len(periods) - validation_quarters - test_quarters + 1,
                       max(1, step_quarters)):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_quarters]
        test_periods = periods[index + validation_quarters:index + validation_quarters + test_quarters]
        train = frame[frame.period.isin(train_periods)].dropna(subset=["target"])
        validation = frame[frame.period.isin(validation_periods)].dropna(subset=["target"])
        test = frame[frame.period.isin(test_periods)].dropna(subset=["target"])
        if train.empty or validation.empty or test.empty:
            continue
        model_scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                train[FEATURES].fillna(0).to_numpy(float), train["target"].to_numpy(float), FEATURES)
            model_scores[alpha] = _score(
                validation["target"], model.predict(validation[FEATURES].fillna(0).to_numpy(float))
            )["within_5_percent"]
        alpha = max(model_scores, key=model_scores.get)
        model = RidgeLinear(alpha=alpha).fit(
            pd.concat([train, validation])[FEATURES].fillna(0).to_numpy(float),
            pd.concat([train, validation])["target"].to_numpy(float), FEATURES)
        learned = model.predict(test[FEATURES].fillna(0).to_numpy(float))
        persistence = test["prescriptions"].to_numpy(float)
        seasonal = frame.loc[[period - 4 for period in test["period"]], "prescriptions"].to_numpy(float)
        actual = test["target"].to_numpy(float)
        candidates = {
            "learned_ridge": _score(actual, learned),
            "persistence": _score(actual, persistence),
            "same_quarter_last_year": _score(actual, seasonal),
        }
        validation_actual = validation["target"].to_numpy(float)
        validation_learned = RidgeLinear(alpha=alpha).fit(
            train[FEATURES].fillna(0).to_numpy(float),
            train["target"].to_numpy(float), FEATURES,
        ).predict(validation[FEATURES].fillna(0).to_numpy(float))
        validation_candidates = {
            "learned_ridge": validation_learned,
            "persistence": validation["prescriptions"].to_numpy(float),
            "same_quarter_last_year": frame.loc[
                [period - 4 for period in validation["period"]], "prescriptions"
            ].to_numpy(float),
        }
        selected_name = max(
            validation_candidates,
            key=lambda name: _score(validation_actual, validation_candidates[name])[
                "within_5_percent"
            ],
        )
        folds.append({
            "validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
            "test_period": f"{test_periods[0]}/{test_periods[-1]}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_model": selected_name,
            "model": candidates[selected_name], "learned_ridge": candidates["learned_ridge"],
            "persistence": candidates["persistence"],
            "same_quarter_last_year": candidates["same_quarter_last_year"],
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    selected_scores = [fold["model"]["within_5_percent"] for fold in folds]
    return {
        "protocol": "rolling_origin_next_quarter_arkansas_medicaid_prescriptions",
        "fold_count": len(folds), "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_model_within_5_percent": float(np.mean(selected_scores)),
        "mean_learned_ridge_within_5_percent": float(np.mean(
            [fold["learned_ridge"]["within_5_percent"] for fold in folds])),
        "mean_persistence_within_5_percent": float(np.mean(
            [fold["persistence"]["within_5_percent"] for fold in folds])),
        "mean_seasonal_within_5_percent": float(np.mean(
            [fold["same_quarter_last_year"]["within_5_percent"] for fold in folds])),
        "publishable_candidate": bool(len(folds) >= 3 and np.mean(selected_scores) >= 0.65),
        "scope": "Arkansas Medicaid FFS prescription-volume demand proxy; not all-payer pharmacy demand",
        "target_semantics": "next-quarter number of prescriptions",
        "folds": folds,
    }
