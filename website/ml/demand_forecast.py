"""Chronological per-drug demand forecasting for the web prototype.

This module intentionally follows the benchmark's important boundaries: each
drug has its own XGBoost regressor, features use only prior sales, and dated
external signals are only made available after their source month closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


REQUIRED_COLUMNS = {"date", "drug_name", "units_sold"}
MIN_TRAINING_ROWS = 30


class DemandDataError(ValueError):
    """Raised when an uploaded demand-history file cannot be forecast safely."""


@dataclass
class DemandForecastResult:
    forecasts: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_demand_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalize a daily medication-sales upload."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DemandDataError(f"Missing required columns: {', '.join(missing)}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["drug_name"] = work["drug_name"].astype(str).str.strip()
    work["units_sold"] = pd.to_numeric(work["units_sold"], errors="coerce")
    if work[["date", "drug_name", "units_sold"]].isna().any().any():
        raise DemandDataError("date, drug_name, and units_sold must be populated and parseable")
    if (work["units_sold"] < 0).any():
        raise DemandDataError("units_sold cannot be negative")
    if not work["date"].dt.normalize().eq(work["date"]).all():
        raise DemandDataError("date values must be daily dates without a time component")
    if (work["drug_name"] == "").any():
        raise DemandDataError("drug_name cannot be blank")
    if work.duplicated(["date", "drug_name"]).any():
        raise DemandDataError("each drug may have at most one row per date")

    warnings: list[str] = []
    for drug, part in work.groupby("drug_name", sort=False):
        dates = part["date"].sort_values()
        if len(part) < MIN_TRAINING_ROWS + 15:
            raise DemandDataError(
                f"{drug} has {len(part)} rows; at least {MIN_TRAINING_ROWS + 15} daily rows are required"
            )
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
        if len(expected) != len(dates):
            raise DemandDataError(
                f"{drug} has missing calendar dates. Include zero-unit days explicitly."
            )
    return work.sort_values(["drug_name", "date"]).reset_index(drop=True), warnings


class ModelSignalStore:
    """Loads the dated production news-signal artifact without importing model code."""

    def __init__(self, source: Path | None = None):
        default = _project_root() / "model" / "artifacts" / "news" / "news_only_catalog_features.csv.gz"
        self.source = source or default
        self.frame = pd.DataFrame(columns=["available_on"])
        self.columns: list[str] = []

    def load(self) -> None:
        if not self.source.exists():
            return
        raw = pd.read_csv(self.source, compression="infer")
        if "date" not in raw.columns:
            return
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"]).sort_values("date")
        numeric = [c for c in raw.columns if c != "date"]
        for column in numeric:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        numeric = [c for c in numeric if raw[c].notna().any()]
        # A monthly value becomes usable after its observed month closes.
        raw["available_on"] = raw["date"] + pd.offsets.MonthBegin(1)
        self.columns = [f"signal__{c}" for c in numeric]
        self.frame = raw[["available_on", *numeric]].rename(
            columns=dict(zip(numeric, self.columns))
        ).sort_values("available_on")

    @property
    def available(self) -> bool:
        return bool(self.columns)

    def join(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.available:
            return frame.copy()
        left = frame.sort_values("date").copy()
        merged = pd.merge_asof(
            left, self.frame, left_on="date", right_on="available_on", direction="backward"
        ).drop(columns="available_on")
        merged[self.columns] = merged[self.columns].fillna(0.0)
        return merged


class DemandForecaster:
    """One-day XGBoost regressors with recursive daily forecast generation."""

    def __init__(self, *, use_model_signals: bool = True, signal_source: Path | None = None):
        self.signals = ModelSignalStore(signal_source)
        self.use_model_signals = use_model_signals
        self.models: dict[str, XGBRegressor] = {}
        self.feature_columns: list[str] = []
        self.history = pd.DataFrame()
        self.metrics: dict[str, Any] = {}

    @staticmethod
    def _base_features(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy().sort_values("date")
        sales = work["units_sold"].astype(float)
        for lag in (1, 7, 14, 28):
            work[f"lag_{lag}"] = sales.shift(lag)
        for window in (7, 14, 28):
            work[f"mean_{window}"] = sales.shift(1).rolling(window, min_periods=window).mean()
            work[f"std_{window}"] = sales.shift(1).rolling(window, min_periods=window).std()
        work["dow"] = work["date"].dt.dayofweek
        work["month"] = work["date"].dt.month
        day = work["date"].dt.dayofyear
        work["sin_year"] = np.sin(2 * np.pi * day / 365.25)
        work["cos_year"] = np.cos(2 * np.pi * day / 365.25)
        return work

    def _training_frame(self, part: pd.DataFrame) -> pd.DataFrame:
        work = self._base_features(part)
        if self.use_model_signals:
            work = self.signals.join(work)
        work["target_next_day"] = work["units_sold"].shift(-1)
        return work

    def _new_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=220,
            max_depth=2,
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=10,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=2,
            random_state=20250915,
        )

    @staticmethod
    def _scores(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        error = np.abs(y - prediction)
        return {
            "mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "wape": float(error.sum() / max(np.abs(y).sum(), 1e-9)),
        }

    def fit(self, raw_history: pd.DataFrame) -> dict[str, Any]:
        history, warnings = validate_demand_history(raw_history)
        if self.use_model_signals:
            self.signals.load()
        all_actual: list[float] = []
        all_predicted: list[float] = []
        per_drug: dict[str, dict[str, float]] = {}
        models: dict[str, XGBRegressor] = {}

        for drug, part in history.groupby("drug_name", sort=True):
            work = self._training_frame(part).reset_index(drop=True)
            feature_columns = [
                c for c in work.columns
                if c.startswith(("lag_", "mean_", "std_", "signal__"))
            ] + ["dow", "month", "sin_year", "cos_year"]
            usable = work.dropna(subset=[*feature_columns, "target_next_day"])
            if len(usable) < MIN_TRAINING_ROWS:
                raise DemandDataError(f"{drug} has insufficient usable history after lag creation")
            split = max(MIN_TRAINING_ROWS, int(len(usable) * 0.80))
            split = min(split, len(usable) - 1)
            train, holdout = usable.iloc[:split], usable.iloc[split:]
            evaluation_model = self._new_model().fit(train[feature_columns], train["target_next_day"])
            predicted = np.maximum(evaluation_model.predict(holdout[feature_columns]), 0.0)
            actual = holdout["target_next_day"].to_numpy(float)
            per_drug[str(drug)] = self._scores(actual, predicted)
            all_actual.extend(actual)
            all_predicted.extend(predicted)
            models[str(drug)] = self._new_model().fit(usable[feature_columns], usable["target_next_day"])
            self.feature_columns = feature_columns

        self.models = models
        self.history = history
        self.metrics = {
            "pooled": self._scores(np.asarray(all_actual), np.asarray(all_predicted)),
            "per_drug": per_drug,
            "drugs_trained": len(models),
            "history_start": history["date"].min().date().isoformat(),
            "history_end": history["date"].max().date().isoformat(),
            "signal_feature_count": len(self.signals.columns) if self.use_model_signals else 0,
            "signal_source": str(self.signals.source) if self.signals.available else None,
            "warnings": warnings,
        }
        return self.metrics

    def forecast(self, horizon_days: int = 14) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Demand forecaster has not been trained")
        if not 1 <= horizon_days <= 56:
            raise DemandDataError("horizon_days must be between 1 and 56")
        rows: list[dict[str, Any]] = []
        for drug, model in self.models.items():
            part = self.history[self.history["drug_name"] == drug][["date", "drug_name", "units_sold"]].copy()
            current = part.copy()
            last_date = current["date"].max()
            for step in range(1, horizon_days + 1):
                forecast_date = last_date + timedelta(days=step)
                next_row = pd.DataFrame({"date": [forecast_date], "drug_name": [drug], "units_sold": [0.0]})
                feature_frame = self._base_features(pd.concat([current, next_row], ignore_index=True))
                if self.use_model_signals:
                    feature_frame = self.signals.join(feature_frame)
                features = feature_frame.iloc[[-1]][self.feature_columns].fillna(0.0)
                predicted = float(max(0.0, model.predict(features)[0]))
                current = pd.concat([
                    current,
                    pd.DataFrame({"date": [forecast_date], "drug_name": [drug], "units_sold": [predicted]}),
                ], ignore_index=True)
                rows.append({
                    "date": forecast_date.date().isoformat(),
                    "drug_name": drug,
                    "predicted_units": round(predicted, 2),
                    "horizon_day": step,
                })
        return pd.DataFrame(rows).sort_values(["date", "drug_name"]).reset_index(drop=True)
