from pathlib import Path

import pandas as pd
import pytest

from ml.demand_forecast import DemandDataError, DemandForecaster, validate_demand_history


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    rows = []
    for drug, offset in (("Medication A", 0), ("Medication B", 3)):
        for i, day in enumerate(dates):
            rows.append({"date": day, "drug_name": drug, "units_sold": 8 + offset + (i % 7)})
    return pd.DataFrame(rows)


def test_demand_history_requires_daily_rows():
    frame = _history().drop(index=5)
    with pytest.raises(DemandDataError, match="missing calendar dates"):
        validate_demand_history(frame)


def test_forecaster_trains_and_generates_daily_series():
    forecaster = DemandForecaster(use_model_signals=False)
    metrics = forecaster.fit(_history())
    forecast = forecaster.forecast(7)

    assert metrics["drugs_trained"] == 2
    assert metrics["signal_feature_count"] == 0
    assert len(forecast) == 14
    assert forecast["predicted_units"].ge(0).all()
    assert forecast.groupby("drug_name")["date"].nunique().eq(7).all()


def test_production_signal_artifact_can_be_joined():
    root = Path(__file__).resolve().parents[2]
    source = root / "model" / "artifacts" / "news" / "news_only_catalog_features.csv.gz"
    forecaster = DemandForecaster(use_model_signals=True, signal_source=source)
    metrics = forecaster.fit(_history())

    assert metrics["signal_feature_count"] == 20


def test_synthetic_sales_contract_is_accepted_by_demand_forecaster():
    root = Path(__file__).resolve().parents[2]
    sales = pd.read_csv(root / "data" / "synthetic_pharmacy_data" / "arkansas_clinic_daily_pharmacy_sales.csv")
    subset = sales[sales["drug_name"].isin(["Azithromycin 250 mg tablet", "Cephalexin 500 mg capsule"])].copy()
    subset = subset.groupby("drug_name", group_keys=False).head(90)

    forecaster = DemandForecaster(use_model_signals=False)
    metrics = forecaster.fit(subset)

    assert metrics["drugs_trained"] == 2
    assert len(forecaster.forecast(14)) == 28
