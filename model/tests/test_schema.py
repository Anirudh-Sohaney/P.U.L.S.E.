"""Output-contract and text-signal tests.

Heavy integration checks are gated on prebuilt artifacts so this suite runs
fast in CI and still verifies the 100+ row output contract when artifacts
exist.
"""

import json
from pathlib import Path

import pandas as pd

from arkansas_pharma_signal import io
from arkansas_pharma_signal.config import Config
from arkansas_pharma_signal.features import (
    MEDICAID_ALL_COLUMNS,
    MEDICAID_EXACT_COLUMNS,
    MEDICAID_QUARTERLY_COLUMNS,
    build_medicaid_quarterly_annual_features,
)
from arkansas_pharma_signal.forecast import OUTPUT_COLUMNS, DRIVER_FEATURES


CONTRACT_COLUMNS = [
    "forecast_run_id", "forecast_created_at", "forecast_date",
    "horizon_days", "geography_level", "geography_id", "geography_name",
    "drug_key", "drug_name", "ingredient_key", "ingredient_name",
    "supplier_key", "supplier_name", "disease_key", "disease_name",
    "target", "prediction", "prediction_interval_low",
    "prediction_interval_high", "risk_score", "model_family",
    "driver_summary_json", "source_feature_window_start",
    "source_feature_window_end",
]


def test_output_columns_match_contract():
    assert OUTPUT_COLUMNS == CONTRACT_COLUMNS


def test_driver_features_reference_real_panel_columns():
    # Every driver feature must be in the documented external/event set, or a
    # wildcard layer ("disease_burden" -> any nndss_* feature).
    known = {
        "ar_ili_mean", "ar_wili_mean", "nat_ili_mean", "nat_wili_mean",
        "ww_flu_ar", "ww_flu_nat", "ww_covid_ar", "ww_covid_nat",
        "ww_rsv_ar", "ww_rsv_nat", "news_covid_articles", "news_influenza_articles",
        "na_shortage_active_mean", "na_recall_active_mean",
        "shortage_active_total", "shortage_current_total",
        "recall_count_x", "recall_count_y", "recall_firms", "shortage_events",
        "shortage_active", "event_count", "event_severity_mean",
        "event_confidence_mean", "ar_unemployment_mean", "na_ppi_mean",
        "us_tariff_rate_mean", "y_last_log", "demand_claims_lag1_log",
        "demand_claims_lag2_log", "demand_claims_ma2_log", "delta_log",
        "delta2_log",
        "ar_population_total", "global_gscpi_mean", "ar_disaster_active_mean",
        "ar_disaster_severity_max", "recall_class_1", "recall_class_2",
        "recall_class_3", "shortage_reason_demand", "shortage_reason_discontinuation",
        "shortage_reason_ingredient", "shortage_reason_other",
        "labeler_shortage_n", "labeler_recall_n",
    }
    referenced = {f for feats in DRIVER_FEATURES.values() for f in feats}
    assert referenced <= known
    assert "disease_burden" in DRIVER_FEATURES


def test_forecast_artifact_schema():
    forecast = Path(__file__).resolve().parents[1] / "artifacts" / "forecasts" / "forecast.csv"
    if not forecast.exists():
        return
    df = pd.read_csv(forecast)
    assert len(df) >= 100
    assert set(df.columns) == set(CONTRACT_COLUMNS)
    assert set(df["target"]) == {
        "demand_claims", "demand_cost", "demand_shock_index",
        "supply_disruption_risk", "arkansas_shortage_impact",
    }
    bad = df[df["driver_summary_json"].isna()]
    assert bad.empty


def test_quarterly_forecast_artifact_schema():
    forecast = Path(__file__).resolve().parents[1] / "artifacts" / "forecasts" / "quarterly_forecast.csv"
    if not forecast.exists():
        return
    df = pd.read_csv(forecast)
    assert len(df) >= 100
    assert set(df.columns) == set(CONTRACT_COLUMNS)
    assert set(df["target"]) == {"medicaid_prescription_count"}
    assert set(df["geography_level"]) == {"state"}
    assert set(df["geography_id"]) == {"AR"}
    assert set(df["model_family"]) == {"quarterly_medicaid_selected_transition"}


def test_medicaid_quarterly_feature_layer():
    cfg = Config(root=str(Path(__file__).resolve().parents[2]))
    features = build_medicaid_quarterly_annual_features(cfg)
    if features.empty:
        return
    assert {"year", "drug_key"} | set(MEDICAID_ALL_COLUMNS) <= set(features.columns)
    assert features["drug_key"].astype(str).str.len().gt(0).all()
    assert features["medicaid_rx_annual"].ge(0).all()
    assert features["medicaid_rx_quarters_observed"].between(1, 4).all()
    assert features["medicaid_rx_bridge_records"].ge(1).all()
    assert features["medicaid_rx_bridge_families"].ge(1).all()
    assert features["medicaid_rx_bridge_families"].max() >= 2
    exact = features.dropna(subset=["medicaid_exact_rx_annual"])
    assert not exact.empty
    assert exact["medicaid_exact_rx_bridge_families"].ge(1).all()


def test_panel_has_medicaid_quarterly_layer():
    panel = Path(__file__).resolve().parents[1] / "artifacts" / "panel" / "panel.csv"
    if not panel.exists():
        return
    cols = pd.read_csv(panel, nrows=0).columns
    assert set(MEDICAID_ALL_COLUMNS) <= set(cols)
    df = pd.read_csv(panel, usecols=MEDICAID_ALL_COLUMNS)
    assert df.notna().any(axis=1).sum() > 0
    assert df[MEDICAID_EXACT_COLUMNS].notna().any(axis=1).sum() > 0
    assert df[MEDICAID_QUARTERLY_COLUMNS].notna().any(axis=1).sum() > 0


if __name__ == "__main__":
    test_output_columns_match_contract()
    test_driver_features_reference_real_panel_columns()
    test_forecast_artifact_schema()
    test_quarterly_forecast_artifact_schema()
    test_medicaid_quarterly_feature_layer()
    test_panel_has_medicaid_quarterly_layer()
    print("ok")
