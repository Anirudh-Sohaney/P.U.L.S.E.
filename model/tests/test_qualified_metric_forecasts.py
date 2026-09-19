import numpy as np
import pandas as pd

from arkansas_pharma_signal.qualified_metric_forecasts import (
    build_numeric_metric_candidates,
    build_qualified_metric_forecasts,
    forecast_latest_county_demand_five_state,
    forecast_latest_national_fluview_five_state,
    forecast_latest_nssp_influenza_five_state,
    forecast_latest_respnet_rsv_five_state,
    forecast_latest_apcd_claim_activity,
    forecast_latest_therapeutic_class_demand,
)


def test_national_fluview_adapter_emits_five_state_filterable_row():
    dates = pd.date_range("2026-01-05", periods=6, freq="7D")
    panel = pd.DataFrame([{
        "region": "nat", "week_start": date, "current_value": float(index + 1),
        "target": float(index + 1),
    } for index, date in enumerate(dates)])
    result = forecast_latest_national_fluview_five_state(panel)
    assert len(result) == 1
    assert result.iloc[0]["target"] == "national_weekly_fluview_respiratory_pressure_state"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_county_demand_adapter_emits_five_state_filterable_rows():
    demand = pd.DataFrame([
        {"year": year, "county_fips": "05001", "drug_key": "d",
         "demand_claims": float(year)}
        for year in range(2020, 2025)
    ])
    result = forecast_latest_county_demand_five_state(demand)
    assert len(result) == 1
    assert result.iloc[0]["target"] == "arkansas_county_annual_demand_five_state"
    assert result.iloc[0]["geography_level"] == "county"
    assert result.iloc[0]["county_fips"] == "05001"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_nssp_adapter_emits_pathogen_filterable_five_state_row():
    dates = pd.date_range("2026-01-03", periods=6, freq="7D")
    panel = pd.DataFrame([{
        "week_end": date, "county": "All",
        "percent_visits_influenza": float(index + 1),
    } for index, date in enumerate(dates)])
    result = forecast_latest_nssp_influenza_five_state(panel)
    assert len(result) == 1
    assert result.iloc[0]["target"] == "arkansas_weekly_nssp_influenza_ed_pressure_state"
    assert result.iloc[0]["pathogen"] == "influenza"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_respnet_adapter_emits_national_rsv_five_state_row():
    dates = pd.date_range("2020-01-04", periods=8, freq="7D")
    panel = pd.DataFrame([{
        "date": date, "estimate": float(index + 1),
    } for index, date in enumerate(dates)])
    result = forecast_latest_respnet_rsv_five_state(panel)
    assert len(result) == 1
    assert result.iloc[0]["target"] == (
        "national_weekly_respnet_rsv_hospitalization_pressure_state")
    assert result.iloc[0]["pathogen"] == "rsv"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_apcd_adapter_emits_reporting_entity_five_state_rows():
    periods = pd.period_range("2020-01", periods=7, freq="M")
    panel = pd.DataFrame([
        {"submitter_id": "s1", "submitter_name": "Entity",
         "year": period.year, "month": period.month,
         "claim_count": float(index + 1)}
        for index, period in enumerate(periods)
    ])
    result = forecast_latest_apcd_claim_activity(panel)
    assert len(result) == 1
    assert result.iloc[0]["target"] == (
        "arkansas_monthly_apcd_pharmacy_claim_activity_state")
    assert result.iloc[0]["geography_level"] == "apcd_submitter"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_therapeutic_class_adapter_emits_filterable_five_state_rows():
    demand = pd.DataFrame([
        {"month": str(month), "drug_key": "1", "demand_claim_lines": float(index + 1),
         "demand_paid": 1.0, "pharmacy_provider_count": 1}
        for index, month in enumerate(pd.period_range("2020-01", "2020-07", freq="M"))
    ])
    mapping = pd.DataFrame([{
        "ndc11": "00000000001", "class_id": "J05AH",
        "class_name": "Neuraminidase inhibitors", "mapping_status": "mapped",
    }])
    result = forecast_latest_therapeutic_class_demand(demand, mapping)
    assert len(result) == 1
    assert result.iloc[0]["target"] == (
        "arkansas_monthly_atc_therapeutic_demand_state")
    assert result.iloc[0]["geography_id"] == "AR"
    assert result.iloc[0]["prediction"] in {0, 1, 2, 3, 4}


def test_qualified_metric_forecasts_emit_all_filterable_surfaces():
    shortage = pd.DataFrame([
        {"ndc9": "1", "supplier": "a", "month": "2026-01", "shortage_active": 0},
        {"ndc9": "1", "supplier": "a", "month": "2026-02", "shortage_active": 1},
        {"ndc9": "2", "supplier": "a", "month": "2026-01", "shortage_active": 1},
        {"ndc9": "2", "supplier": "b", "month": "2026-01", "shortage_active": 1},
    ])
    flu = pd.DataFrame([{
        "region": "ar", "year": 2026, "week": 1,
        "week_start": pd.Timestamp("2026-01-05"), "target": 1.0,
        "current_value": 1.0, "lag1_value": 1.0, "lag2_value": 1.0,
        "lag4_value": 1.0, "rolling4_value": 1.0,
        "week_sin": 0.0, "week_cos": 1.0,
    }])
    demand = pd.DataFrame([
        {"year": year, "arkansas_region": region, "drug_key": "d",
         "county_fips": f"05{index + 1:03d}",
         "demand_claims": float(year + index + 1)}
        for year in range(2020, 2024)
        for index, region in enumerate(("central", "northeast", "northwest", "southeast", "southwest"))
    ])
    recall_events = pd.DataFrame([{
        "ndc": "1", "supplier": "Acme", "start": pd.Period("2026-01", freq="M"),
        "end": pd.NaT, "severity": 2,
    }])
    hospital = pd.DataFrame([
        {"weekendingdate": "2026-01-03", "jurisdiction": "AR",
         "totalconfc19newadm": 1, "totalconfflunewadm": 1,
         "totalconfrsvnewadm": 1},
        {"weekendingdate": "2026-01-10", "jurisdiction": "AR",
         "totalconfc19newadm": 4, "totalconfflunewadm": 4,
         "totalconfrsvnewadm": 4},
        {"weekendingdate": "2026-01-17", "jurisdiction": "AR",
         "totalconfc19newadm": 8, "totalconfflunewadm": 8,
         "totalconfrsvnewadm": 8},
    ])
    arcos = pd.DataFrame([
        {"year": 2024 + quarter // 4, "quarter": quarter % 4 + 1,
         "period": f"{2024 + quarter // 4}-Q{quarter % 4 + 1}",
         "period_index": (2024 + quarter // 4) * 4 + quarter % 4,
         "drug_code": drug, "drug_name": "x", "zip3": zip3,
         "grams": float((quarter + 1) * (index + 1)),
         "log_grams": np.log1p(float((quarter + 1) * (index + 1)))}
        for index, (drug, zip3) in enumerate((("1", "722"), ("2", "727")))
        for quarter in range(8)
    ])
    result = build_qualified_metric_forecasts(
        shortage_source=shortage, flu_panel=flu, county_demand=demand,
        recall_events=recall_events,
        hospital_panel=hospital,
        arcos_panel=arcos,
        model_version="test", forecast_timestamp="2026-08-16T00:00:00Z")
    assert set(result["target"]) == {
        "ndc_monthly_shortage_supplier_count",
        "ndc_monthly_shortage_pressure_state",
        "arcos_zip3_drug_next_quarter_distribution_state",
        "arkansas_region_annual_demand_five_state",
        "arkansas_county_annual_demand_five_state",
        "ndc_monthly_recall_severity",
        "supplier_ndc_monthly_recall_severity",
    }
    assert result["forecast_period"].notna().all()
    assert result["horizon"].gt(0).all()
    assert result["driver_attribution"].str.contains("persistence").any()
    numeric_targets = result["target"].str.contains("recall_severity") | result["target"].eq(
        "ndc_monthly_shortage_supplier_count")
    assert result.loc[numeric_targets,
                      "target_promotion_status"].eq("qualified_numeric_proxy").all()
    five_state = result["target"].eq("arkansas_region_annual_demand_five_state")
    assert result.loc[five_state,
                      "target_promotion_status"].eq("qualified_five_state_proxy").all()
    county_state = result["target"].eq("arkansas_county_annual_demand_five_state")
    assert result.loc[county_state,
                      "target_promotion_status"].eq("qualified_five_state_proxy").all()
    arcos_state = result["target"].eq("arcos_zip3_drug_next_quarter_distribution_state")
    assert result.loc[arcos_state,
                      "target_promotion_status"].eq("qualified_five_state_proxy").all()


def test_numeric_metric_candidates_are_filterable_but_not_promoted():
    shortage = pd.DataFrame([
        {"ndc9": "1", "supplier": "a", "month": "2026-01", "shortage_active": 0},
        {"ndc9": "1", "supplier": "a", "month": "2026-02", "shortage_active": 1},
    ])
    flu = pd.DataFrame([{
        "region": "ar", "year": 2026, "week": 1,
        "week_start": pd.Timestamp("2026-01-05"), "target": 1.0,
        "current_value": 1.0, "lag1_value": 1.0, "lag2_value": 1.0,
        "lag4_value": 1.0, "rolling4_value": 1.0,
        "week_sin": 0.0, "week_cos": 1.0,
    }])
    demand = pd.DataFrame([{
        "year": year, "arkansas_region": "central", "drug_key": "d",
        "demand_claims": float(year + 1),
    } for year in range(2020, 2024)])
    result = build_numeric_metric_candidates(
        shortage_source=shortage, flu_panel=flu, county_demand=demand,
        model_version="test", forecast_timestamp="2026-08-16T00:00:00Z")
    assert set(result["target"]) == {
        "ndc_monthly_shortage_supplier_count",
        "arkansas_weekly_fluview_wili",
        "arkansas_region_annual_demand_claims",
    }
    assert result["prediction"].map(float).notna().all()
    assert result.loc[result["target"].eq("ndc_monthly_shortage_supplier_count"),
                      "target_promotion_status"].eq("qualified_numeric_proxy").all()
    assert result.loc[~result["target"].eq("ndc_monthly_shortage_supplier_count"),
                      "target_promotion_status"].eq("candidate_numeric_proxy").all()
