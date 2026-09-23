"""Per-drug direct 14-day XGBoost demand forecasting.

The feature set, signal selection, and model parameters mirror the final
benchmark in test/run_publishable_benchmark.py. Daily values exposed to the
dashboard are an allocation of each direct 14-day prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


REQUIRED_COLUMNS = {"date", "drug_name", "units_sold"}
MIN_TRAINING_ROWS = 30
MIN_HISTORY_ROWS = 130  # 56-day warmup, 14-day target, signal lags, 30+ training rows.
DEMO_LEAD_TIME_DAYS = 7
DEMO_TARGET_COVER_DAYS = 14


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
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
        if len(expected) != len(dates):
            raise DemandDataError(
                f"{drug} has missing calendar dates. Include zero-unit days explicitly."
            )
        if len(part) < MIN_HISTORY_ROWS:
            raise DemandDataError(
                f"{drug} has {len(part)} rows; at least {MIN_HISTORY_ROWS} daily rows are required"
            )
    return work.sort_values(["drug_name", "date"]).reset_index(drop=True), warnings


class ModelSignalStore:
    """Loads the frozen long-format signal catalog used by the final benchmark."""

    def __init__(self, source: Path | None = None):
        default = _project_root() / "test" / "test_Signals" / "signals_2023_2025.csv.gz"
        self.source = source or default
        self.frame = pd.DataFrame(columns=["available_on"])
        self.columns: list[str] = []
        self.candidate_count = 0
        self.metadata: dict[str, dict[str, Any]] = {}
        self.rows = pd.DataFrame()

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")[:90]

    def load(self) -> None:
        if not self.source.exists():
            return
        raw = pd.read_csv(self.source, compression="infer", low_memory=False)
        if "date" in raw.columns:
            # Backward-compatible path for a wide dated model artifact.
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"]).sort_values("date")
            numeric = [c for c in raw.columns if c != "date"]
            for column in numeric:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            numeric = [c for c in numeric if raw[c].notna().any()]
            raw["available_on"] = raw["date"] + pd.offsets.MonthBegin(1)
            self.columns = [f"signal__{c}" for c in numeric]
            self.frame = raw[["available_on", *numeric]].rename(
                columns=dict(zip(numeric, self.columns))
            ).sort_values("available_on")
            self.candidate_count = len(self.columns)
            return
        required = {"value", "period_end", "signal_origin", "signal_id", "cadence",
                    "geography_level", "geography_id", "entity_key"}
        if not required.issubset(raw.columns):
            return
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw["period_end"] = pd.to_datetime(raw["period_end"], errors="coerce")
        raw = raw.dropna(subset=["value", "period_end"])
        raw = raw[raw["period_end"].dt.year.between(2022, 2025)].copy()
        raw["feature_key"] = (
            raw["signal_origin"].fillna("") + "|" + raw["signal_id"].fillna("") + "|"
            + raw["cadence"].fillna("") + "|" + raw["geography_level"].fillna("") + "|"
            + raw["geography_id"].fillna("").astype(str) + "|" + raw["entity_key"].fillna("")
        ).map(self._safe_name)
        self.rows = raw.copy()
        raw = raw.sort_values(["feature_key", "period_end"]).drop_duplicates(
            ["feature_key", "period_end"], keep="last"
        )
        columns: dict[str, pd.Series] = {}
        for key, group in raw.groupby("feature_key", sort=True):
            series = group.set_index("period_end")["value"].sort_index()
            column = f"signal__{key}"
            columns[column] = series[~series.index.duplicated(keep="last")]
            first = group.iloc[0]
            self.metadata[column] = {
                name: str(first.get(name, "")) for name in
                ("signal_id", "signal_origin", "cadence", "geography_level", "geography_id", "entity_key", "source_name", "unit")
            }
        self.columns = list(columns)
        self.candidate_count = len(self.columns)
        self.frame = pd.DataFrame(columns).sort_index().reset_index(names="available_on").fillna(0.0)

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
    """One direct next-14-day XGBoost regressor per medication."""

    def __init__(self, *, use_model_signals: bool = True, signal_source: Path | None = None):
        self.signals = ModelSignalStore(signal_source)
        self.use_model_signals = use_model_signals
        self.models: dict[str, XGBRegressor] = {}
        self.feature_columns: list[str] = []
        self.feature_columns_by_drug: dict[str, list[str]] = {}
        self.history = pd.DataFrame()
        self.metrics: dict[str, Any] = {}
        self.selected_signals: dict[str, list[str]] = {}
        self.signal_scores: dict[str, dict[str, float]] = {}

    @staticmethod
    def _base_features(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy().sort_values("date")
        sales = work["units_sold"].astype(float)
        for lag in (1, 2, 3, 7, 14, 21, 28, 56):
            work[f"lag_{lag}"] = sales.shift(lag)
        for window in (3, 7, 14, 28, 56):
            work[f"mean_{window}"] = sales.shift(1).rolling(window, min_periods=window).mean()
            work[f"std_{window}"] = sales.shift(1).rolling(window, min_periods=window).std()
        work["dow"] = work["date"].dt.dayofweek
        work["weekofyear"] = work["date"].dt.isocalendar().week.astype(int)
        work["month"] = work["date"].dt.month
        day = work["date"].dt.dayofyear
        work["sin_year"] = np.sin(2 * np.pi * day / 365.25)
        work["cos_year"] = np.cos(2 * np.pi * day / 365.25)
        if "unit_price_usd" not in work:
            work["unit_price_usd"] = 0.0
        if "stockout_flag" not in work:
            work["stockout_flag"] = 0.0
        work["price_lag1"] = pd.to_numeric(work["unit_price_usd"], errors="coerce").fillna(0).shift(1)
        work["stockout_lag1"] = pd.to_numeric(work["stockout_flag"], errors="coerce").fillna(0).shift(1)
        return work

    def _training_frame(self, part: pd.DataFrame) -> pd.DataFrame:
        work = self._base_features(part)
        if self.use_model_signals:
            work = self.signals.join(work)
        target_14d = work["units_sold"].shift(-1).rolling(14, min_periods=14).sum().shift(-13).rename("target_14d")
        return pd.concat([work, target_14d], axis=1)

    @staticmethod
    def _signal_lags(frame: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
        lagged = {
            f"{signal}__lag{lag}": frame[signal].shift(lag)
            for signal in selected for lag in (1, 7, 14)
        }
        return pd.concat([frame, pd.DataFrame(lagged, index=frame.index)], axis=1)

    @staticmethod
    def _signal_correlations(frame: pd.DataFrame, candidates: list[str]) -> dict[str, float]:
        y = frame["target_14d"].to_numpy(float)
        scores = []
        for column in candidates:
            x = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(float)
            if np.std(x) <= 1e-12:
                continue
            correlation = np.corrcoef(x, y)[0, 1]
            if np.isfinite(correlation):
                scores.append((column, abs(float(correlation))))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return dict(scores)

    def _new_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=220,
            max_depth=2,
            learning_rate=0.04,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.5,
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
            if not self.signals.available:
                raise DemandDataError(f"Signal catalog is unavailable: {self.signals.source}")
        all_actual: list[float] = []
        all_predicted: list[float] = []
        all_baseline_predicted: list[float] = []
        per_drug: dict[str, dict[str, float]] = {}
        models: dict[str, XGBRegressor] = {}
        first, last = history["date"].min(), history["date"].max()
        validation_end = first + (last - first) * 0.80

        for drug, part in history.groupby("drug_name", sort=True):
            work = self._training_frame(part).reset_index(drop=True)
            base_columns = [
                c for c in work.columns
                if c.startswith(("lag_", "mean_", "std_")) and int(c.split("_")[1]) <= 56
            ] + ["dow", "weekofyear", "month", "sin_year", "cos_year", "price_lag1", "stockout_lag1"]
            candidate_columns = [c for c in work.columns if c.startswith("signal__")]
            usable = work.dropna(subset=[*base_columns, "target_14d"]).copy()
            if len(usable) < MIN_TRAINING_ROWS + 15:
                raise DemandDataError(f"{drug} has insufficient usable history after lag creation")
            selection_train = usable[usable["date"] < validation_end]
            if len(selection_train) < MIN_TRAINING_ROWS:
                raise DemandDataError(f"{drug} needs more history before the validation period")
            scores = self._signal_correlations(selection_train, candidate_columns)
            selected = list(scores)[:5]
            self.signal_scores[str(drug)] = scores
            work = self._signal_lags(usable, selected)
            signal_columns = [f"{column}__lag{lag}" for column in selected for lag in (1, 7, 14)]
            usable = work.dropna(subset=[*signal_columns, "target_14d"])
            train = usable[usable["date"] < validation_end]
            holdout = usable[usable["date"] >= validation_end]
            if len(train) < MIN_TRAINING_ROWS or holdout.empty:
                raise DemandDataError(f"{drug} needs more history for training and validation")
            feature_columns = base_columns + signal_columns
            evaluation_model = self._new_model().fit(train[feature_columns], train["target_14d"], verbose=False)
            baseline_model = self._new_model().fit(train[base_columns], train["target_14d"], verbose=False)
            predicted = np.maximum(evaluation_model.predict(holdout[feature_columns]), 0.0)
            baseline_predicted = np.maximum(baseline_model.predict(holdout[base_columns]), 0.0)
            actual = holdout["target_14d"].to_numpy(float)
            per_drug[str(drug)] = self._scores(actual, predicted)
            all_actual.extend(actual)
            all_predicted.extend(predicted)
            all_baseline_predicted.extend(baseline_predicted)
            models[str(drug)] = self._new_model().fit(usable[feature_columns], usable["target_14d"], verbose=False)
            self.feature_columns = feature_columns
            self.feature_columns_by_drug[str(drug)] = feature_columns
            self.selected_signals[str(drug)] = selected

        selected_score = self._scores(np.asarray(all_actual), np.asarray(all_predicted))
        baseline_score = self._scores(np.asarray(all_actual), np.asarray(all_baseline_predicted))
        self.models = models
        self.history = history
        self.metrics = {
            "pooled": selected_score,
            "sales_only_pooled": baseline_score,
            "wape_improvement_vs_sales_only": float(1 - selected_score["wape"] / baseline_score["wape"]) if baseline_score["wape"] else None,
            "per_drug": per_drug,
            "drugs_trained": len(models),
            "history_start": history["date"].min().date().isoformat(),
            "history_end": history["date"].max().date().isoformat(),
            "validation_start": validation_end.date().isoformat(),
            "forecast_target": "next_14_calendar_days_total_units",
            "daily_forecast_method": "day_of_week_allocation_of_direct_14_day_prediction",
            "signal_candidate_count": self.signals.candidate_count if self.use_model_signals else 0,
            "signal_feature_count": int(np.mean([len(x) for x in self.selected_signals.values()])) if self.use_model_signals and self.selected_signals else 0,
            "signal_selection_rule": "top_5_absolute_training_correlation",
            "signal_lags_days": [1, 7, 14],
            "signal_source": str(self.signals.source) if self.signals.available else None,
            "warnings": warnings,
        }
        return self.metrics

    def forecast(self, horizon_days: int = 14) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Demand forecaster has not been trained")
        if not 1 <= horizon_days <= 14:
            raise DemandDataError("horizon_days must be between 1 and 14")
        rows: list[dict[str, Any]] = []
        for drug, model in self.models.items():
            part = self.history[self.history["drug_name"] == drug].copy()
            last_date = part["date"].max()
            feature_frame = self._base_features(part)
            if self.use_model_signals:
                feature_frame = self.signals.join(feature_frame)
                feature_frame = self._signal_lags(feature_frame, self.selected_signals[drug])
            feature_columns = self.feature_columns_by_drug[drug]
            features = feature_frame.iloc[[-1]][feature_columns]
            predicted_14d = float(max(0.0, model.predict(features)[0]))
            recent = part.tail(56)
            weekday_means = recent.groupby(recent["date"].dt.dayofweek)["units_sold"].mean()
            dates = [last_date + timedelta(days=step) for step in range(1, 15)]
            weights = np.asarray([max(0.0, float(weekday_means.get(date.dayofweek, 0))) for date in dates])
            if weights.sum() <= 0:
                weights = np.ones(14)
            allocated = np.round(predicted_14d * weights / weights.sum(), 2)
            allocated[-1] = max(0.0, round(round(predicted_14d, 2) - allocated[:-1].sum(), 2))
            for step in range(1, horizon_days + 1):
                forecast_date = dates[step - 1]
                rows.append({
                    "date": forecast_date.date().isoformat(),
                    "drug_name": drug,
                    "predicted_units": float(allocated[step - 1]),
                    "horizon_day": step,
                })
        return pd.DataFrame(rows).sort_values(["date", "drug_name"]).reset_index(drop=True)


def build_demo_inventory_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    """Derive one auditable synthetic on-hand snapshot from the sales history.

    This is a demonstration replenishment state, not a record of any Arkansas
    pharmacy's inventory. The policy uses a 28-day trailing demand estimate, a
    seven-day lead time, and normal-demand safety stock. Product ordering fixes
    the scenario variation deterministically so the same source always yields
    the same snapshot.
    """
    normalized, _ = validate_demand_history(history)
    rows: list[dict[str, Any]] = []
    for position, (drug, part) in enumerate(normalized.groupby("drug_name", sort=True)):
        trailing = part.tail(28)["units_sold"].astype(float)
        average = float(trailing.mean())
        deviation = float(trailing.std(ddof=0))
        safety = int(np.ceil(1.65 * deviation * np.sqrt(DEMO_LEAD_TIME_DAYS)))
        reorder = int(np.ceil(average * DEMO_LEAD_TIME_DAYS + safety))
        target = int(np.ceil(average * DEMO_TARGET_COVER_DAYS + safety))
        # Fixed coverage bands create a transparent mix of healthy and
        # understocked demonstration records without random hidden state.
        coverage_multiplier = (0.45, 0.70, 0.95, 1.15, 1.40)[position % 5]
        on_hand = max(0, int(np.ceil(target * coverage_multiplier)))
        rows.append({
            "as_of_date": part["date"].max().date().isoformat(),
            "facility_id": "AR-SYNTH-CLINIC-001",
            "drug_name": drug,
            "therapeutic_class": str(part["therapeutic_class"].iloc[-1]) if "therapeutic_class" in part else "unspecified",
            "on_hand_units": on_hand,
            "on_order_units": 0,
            "lead_time_days": DEMO_LEAD_TIME_DAYS,
            "safety_stock_units": safety,
            "reorder_point_units": reorder,
            "target_stock_units": target,
            "trailing_28d_avg_units": round(average, 2),
        })
    return pd.DataFrame(rows)


def build_replenishment_plan(inventory: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Join fixed inventory to the recursive forecast and calculate coverage."""
    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict(orient="records"):
        forecast = forecasts[forecasts["drug_name"] == record["drug_name"]].sort_values("horizon_day")
        if forecast.empty:
            continue
        daily = forecast["predicted_units"].to_numpy(float)
        cumulative = np.cumsum(daily)
        available = float(record["on_hand_units"] + record["on_order_units"])
        depleted = np.flatnonzero(cumulative > available)
        stockout_day = int(depleted[0] + 1) if len(depleted) else None
        demand_1 = float(cumulative[min(0, len(cumulative) - 1)])
        demand_4 = float(cumulative[min(3, len(cumulative) - 1)])
        demand_7 = float(cumulative[min(6, len(cumulative) - 1)])
        demand_14 = float(cumulative[-1])
        rows.append({
            **record,
            "forecast_1d_units": round(demand_1, 2),
            "forecast_4d_units": round(demand_4, 2),
            "forecast_7d_units": round(demand_7, 2),
            "forecast_14d_units": round(demand_14, 2),
            "days_until_stockout": stockout_day,
            "minimum_buy_1d_units": max(0, int(np.ceil(demand_1 - available))),
            "minimum_buy_7d_units": max(0, int(np.ceil(demand_7 - available))),
            "minimum_buy_14d_units": max(0, int(np.ceil(demand_14 - available))),
            "recommended_order_units": max(
                0,
                int(np.ceil(record["target_stock_units"] - available)),
            ),
            "inventory_status": "reorder_now" if record["on_hand_units"] <= record["reorder_point_units"] else "covered",
        })
    return pd.DataFrame(rows).sort_values(["recommended_order_units", "drug_name"], ascending=[False, True]).reset_index(drop=True)
