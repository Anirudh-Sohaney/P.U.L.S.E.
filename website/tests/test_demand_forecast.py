from pathlib import Path

import pandas as pd
import pytest

from ml.demand_forecast import (
    DemandDataError,
    DemandForecaster,
    build_demo_inventory_snapshot,
    build_replenishment_plan,
    validate_demand_history,
)


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    for drug, offset in (("Medication A", 0), ("Medication B", 3)):
        for i, day in enumerate(dates):
            rows.append({"date": day, "drug_name": drug, "units_sold": 8 + offset + (i % 7)})
    return pd.DataFrame(rows)


def test_demand_history_requires_daily_rows():
    frame = _history().drop(index=5)
    with pytest.raises(DemandDataError, match="missing calendar dates"):
        validate_demand_history(frame)


def test_forecaster_trains_direct_14_day_model_and_allocates_daily_series():
    forecaster = DemandForecaster(use_model_signals=False)
    metrics = forecaster.fit(_history())
    forecast = forecaster.forecast(7)

    assert metrics["drugs_trained"] == 2
    assert metrics["signal_feature_count"] == 0
    assert metrics["forecast_target"] == "next_14_calendar_days_total_units"
    assert metrics["sales_only_pooled"]["wape"] >= 0
    assert len(forecast) == 14
    assert forecast["predicted_units"].ge(0).all()
    assert forecast.groupby("drug_name")["date"].nunique().eq(7).all()


def test_full_signal_catalog_is_selected_per_drug():
    root = Path(__file__).resolve().parents[2]
    sales = pd.read_csv(root / "data" / "synthetic_pharmacy_data" / "arkansas_clinic_daily_pharmacy_sales.csv")
    history = sales[sales["drug_name"] == "Azithromycin 250 mg tablet"].copy()
    forecaster = DemandForecaster(use_model_signals=True)
    metrics = forecaster.fit(history)

    assert metrics["signal_candidate_count"] == 1312
    assert metrics["signal_feature_count"] == 5
    assert metrics["signal_selection_rule"] == "top_5_absolute_training_correlation"
    assert metrics["signal_lags_days"] == [1, 7, 14]
    assert all(len(columns) == 5 for columns in forecaster.selected_signals.values())
    assert all(len(columns) == 40 for columns in forecaster.feature_columns_by_drug.values())


def test_synthetic_sales_contract_is_accepted_by_demand_forecaster():
    root = Path(__file__).resolve().parents[2]
    sales = pd.read_csv(root / "data" / "synthetic_pharmacy_data" / "arkansas_clinic_daily_pharmacy_sales.csv")
    subset = sales[sales["drug_name"].isin(["Azithromycin 250 mg tablet", "Cephalexin 500 mg capsule"])].copy()
    subset = subset.groupby("drug_name", group_keys=False).head(180)

    forecaster = DemandForecaster(use_model_signals=False)
    metrics = forecaster.fit(subset)

    assert metrics["drugs_trained"] == 2
    assert len(forecaster.forecast(14)) == 28


def test_inventory_plan_covers_required_horizons_and_order_guidance():
    history = _history()
    forecaster = DemandForecaster(use_model_signals=False)
    forecaster.fit(history)
    inventory = build_demo_inventory_snapshot(history)
    plan = build_replenishment_plan(inventory, forecaster.forecast(14))

    assert len(plan) == 2
    assert {"forecast_1d_units", "forecast_4d_units", "forecast_7d_units", "forecast_14d_units", "minimum_buy_1d_units", "minimum_buy_7d_units", "minimum_buy_14d_units"} <= set(plan)
    assert plan["recommended_order_units"].ge(0).all()
    assert plan["minimum_buy_7d_units"].ge(0).all()
