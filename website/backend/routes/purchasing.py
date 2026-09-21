"""Inventory-aware demand forecasts for the single pharmacy dashboard.

The forecasting implementation intentionally mirrors ``test/run_xgb_signals.py``:
the signal matrix, chronological split, per-drug top-15 selection, and XGBoost
hyperparameters are kept aligned with that evaluated experiment.  The dashboard
then refits those selected per-drug models on all available history and produces
direct cumulative-demand forecasts for four purchasing horizons.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from xgboost import XGBRegressor

router = APIRouter()
ROOT = Path(__file__).resolve().parents[3]
SALES_PATH = ROOT / "data/synthetic_pharmacy_data/arkansas_clinic_daily_pharmacy_sales.csv"
SIGNALS_PATH = ROOT / "test/test_Signals/signals_2023_2025.csv.gz"
INVENTORY_PATH = ROOT / "website/data/pharmacy_risk_data.csv"
HORIZONS = (1, 3, 7, 14)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")[:90]


def _make_sales_frame() -> pd.DataFrame:
    raw = pd.read_csv(SALES_PATH, parse_dates=["date"])
    raw = raw.sort_values(["drug_name", "date"]).reset_index(drop=True)
    grouped = raw.groupby("drug_name", sort=False)["units_sold"]
    for horizon in HORIZONS:
        raw[f"target_{horizon}d"] = grouped.transform(
            lambda values: values.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-horizon + 1)
        )
    for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
        raw[f"lag_{lag}"] = grouped.shift(lag)
    for window in [3, 7, 14, 28, 56]:
        shifted = grouped.shift(1)
        raw[f"mean_{window}"] = shifted.groupby(raw["drug_name"]).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        raw[f"std_{window}"] = shifted.groupby(raw["drug_name"]).transform(
            lambda values: values.rolling(window, min_periods=window).std()
        )
    raw["dow"] = raw.date.dt.dayofweek
    raw["weekofyear"] = raw.date.dt.isocalendar().week.astype(int)
    raw["month"] = raw.date.dt.month
    raw["day_of_year"] = raw.date.dt.dayofyear
    raw["sin_year"] = np.sin(2 * np.pi * raw.day_of_year / 365.25)
    raw["cos_year"] = np.cos(2 * np.pi * raw.day_of_year / 365.25)
    raw["price_lag1"] = raw.groupby("drug_name").unit_price_usd.shift(1)
    raw["stockout_lag1"] = raw.groupby("drug_name").stockout_flag.shift(1)
    return raw


def _signal_matrix(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, int]:
    signals = pd.read_csv(SIGNALS_PATH, parse_dates=["signal_date", "period_end"], low_memory=False)
    signals["value"] = pd.to_numeric(signals["value"], errors="coerce")
    signals = signals.dropna(subset=["value", "period_end"])
    signals = signals[signals.period_end.dt.year.between(2022, 2025)].copy()
    keys = (
        signals["signal_origin"].fillna("")
        + "|" + signals["signal_id"].fillna("")
        + "|" + signals["cadence"].fillna("")
        + "|" + signals["geography_level"].fillna("")
        + "|" + signals["geography_id"].fillna("").astype(str)
        + "|" + signals["entity_key"].fillna("")
    )
    signals["feature_key"] = keys.map(_safe_name)
    signals = signals.sort_values(["feature_key", "period_end"]).drop_duplicates(
        ["feature_key", "period_end"], keep="last"
    )
    wide = pd.DataFrame(index=dates)
    for key, group in signals.groupby("feature_key", sort=True):
        series = group.set_index("period_end")["value"].sort_index()
        wide[key] = series[~series.index.duplicated(keep="last")].reindex(dates, method="ffill")
    return wide.fillna(0.0), int(wide.shape[1])


def _xgb() -> XGBRegressor:
    # Exact estimator settings from test/run_xgb_signals.py.
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


def _select_top15(train: pd.DataFrame, target: str, candidates: list[str]) -> list[str]:
    scores: list[tuple[str, float]] = []
    y = train[target].to_numpy(float)
    for column in candidates:
        x = pd.to_numeric(train[column], errors="coerce").fillna(0).to_numpy(float)
        if np.std(x) <= 1e-12:
            continue
        correlation = np.corrcoef(x, y)[0, 1]
        if np.isfinite(correlation):
            scores.append((column, abs(float(correlation))))
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return [column for column, _ in scores[:15]]


def _normalise_drug(value: str) -> str:
    value = value.lower()
    for token in ("tablet", "capsule", "inhaler", "spray", "pack", "pen"):
        value = value.replace(token, "")
    return re.sub(r"[^a-z0-9]", "", value)


def _inventory_by_drug() -> dict[str, dict[str, object]]:
    inventory = pd.read_csv(INVENTORY_PATH)
    inventory["key"] = inventory["medication_name"].map(_normalise_drug)
    result: dict[str, dict[str, object]] = {}
    for key, group in inventory.groupby("key", sort=False):
        row = group.iloc[0]
        result[key] = {
            "inventory": float(row["current_inventory"]),
            "reorder_point": float(row["reorder_point"]),
            "lead_time_days": float(row["supplier_lead_time_days"]),
            "source_name": str(row["medication_name"]),
            "source_rows": int(len(group)),
        }
    return result


@lru_cache(maxsize=1)
def build_dashboard() -> dict:
    if not SALES_PATH.exists() or not SIGNALS_PATH.exists():
        raise RuntimeError("The evaluated sales or signal artifact is unavailable.")

    sales = _make_sales_frame()
    dates = pd.DatetimeIndex(sorted(sales.date.unique()))
    signals, candidate_count = _signal_matrix(dates)
    sales = sales.set_index("date").join(signals, how="left").reset_index()
    cutoff = sales.date.min() + (sales.date.max() - sales.date.min()) * 0.70
    base = [
        column for column in sales.columns
        if (column.startswith(("lag_", "mean_", "std_")) and (
            (column.startswith("lag_") and int(column.split("_")[1]) <= 56)
            or (column.startswith(("mean_", "std_")) and int(column.split("_")[1]) <= 56)
        ))
    ] + ["dow", "weekofyear", "month", "sin_year", "cos_year", "price_lag1", "stockout_lag1"]
    signal_columns = list(signals.columns)
    inventory = _inventory_by_drug()
    rows = []
    selected_total = 0

    for drug, original in sales.groupby("drug_name", sort=True):
        work = original.copy()
        usable_14 = work.dropna(subset=base + ["target_14d"])
        train_14 = usable_14[usable_14.date <= cutoff]
        top = _select_top15(train_14, "target_14d", signal_columns)
        selected_total += len(top)
        model_columns: list[str] = []
        for signal in top:
            for lag in (1, 7, 14):
                name = f"{signal}__lag{lag}"
                work[name] = work[signal].shift(lag)
                model_columns.append(name)
        usable = work.dropna(subset=base + model_columns).copy()
        latest = usable.sort_values("date").tail(1)
        if latest.empty or not top:
            continue
        forecasts: dict[str, float] = {}
        for horizon in HORIZONS:
            target = f"target_{horizon}d"
            train = usable.dropna(subset=[target])
            if len(train) < 100:
                raise RuntimeError(f"Insufficient training history for {drug}.")
            model = _xgb()
            model.fit(train[base + model_columns], train[target], verbose=False)
            prediction = float(max(0.0, model.predict(latest[base + model_columns])[0]))
            forecasts[str(horizon)] = prediction

        stock = inventory.get(_normalise_drug(drug))
        current_inventory = stock["inventory"] if stock else None
        reorder_point = stock["reorder_point"] if stock else None
        purchases = {
            str(horizon): int(max(0, np.ceil(forecasts[str(horizon)] + reorder_point - current_inventory)))
            if current_inventory is not None else None
            for horizon in HORIZONS
        }
        rows.append({
            "drug_name": drug,
            "current_inventory": current_inventory,
            "reorder_point": reorder_point,
            "lead_time_days": stock["lead_time_days"] if stock else None,
            "inventory_source": stock["source_name"] if stock else None,
            "inventory_source_rows": stock["source_rows"] if stock else 0,
            "forecast_units": forecasts,
            "purchase_units": purchases,
            "status": "buy" if any(value and value > 0 for value in purchases.values() if value is not None) else (
                "covered" if current_inventory is not None else "inventory unavailable"
            ),
            "selected_signal_count": len(top),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "candidate_signal_features": candidate_count,
            "selected_features_per_drug": 15,
            "selected_signal_features_total": selected_total,
            "variant": "top15_lagged_1_7_14",
            "training_cutoff": str(cutoff.date()),
            "history_start": str(sales.date.min().date()),
            "history_end": str(sales.date.max().date()),
            "drugs": len(rows),
        },
        "rows": rows,
    }


@router.get("/dashboard")
def get_purchasing_dashboard() -> dict:
    try:
        payload = build_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to build purchasing dashboard: {exc}") from exc
    rows = payload["rows"]
    payload["summary"] = {
        "forecast_drugs": len(rows),
        "inventory_backed": sum(row["current_inventory"] is not None for row in rows),
        "needs_purchase": sum(row["status"] == "buy" for row in rows),
        "inventory_unavailable": sum(row["status"] == "inventory unavailable" for row in rows),
        "purchase_units": {
            str(horizon): sum((row["purchase_units"][str(horizon)] or 0) for row in rows)
            for horizon in HORIZONS
        },
    }
    return payload


@router.post("/dashboard/refresh")
def refresh_purchasing_dashboard() -> dict:
    build_dashboard.cache_clear()
    return get_purchasing_dashboard()
