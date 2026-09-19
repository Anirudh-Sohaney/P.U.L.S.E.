"""Tests for the input-variable disposition contract.

Verifies representative names from ``data/final_data/VARIABLE_LIST.md`` and
the audit's explicit non-live markers, plus fail-closed behavior for unknown
names.
"""

from arkansas_pharma_signal.input_contract import (
    DERIVED_TRAINING_VARIABLE,
    DISPOSITIONS,
    LIVE_INPUT,
    LOCAL_PHARMACY_HISTORY_INPUT,
    NEAR_REAL_TIME_INPUT,
    PERIODIC_TRAINING_ONLY,
    STATIC_IDENTITY_CONTEXT,
    audit_feature_names,
    disposition_for_variable,
    is_operational_feature,
    is_local_history_feature,
    operational_dispositions,
    summarize_dispositions,
)
from arkansas_pharma_signal.input_sources import (
    NEAR_REAL_TIME_SOURCE_REGISTRY, RESEARCH_ONLY_SOURCE_REGISTRY,
    audit_near_real_time_sources,
    audit_operational_sources,
)
from arkansas_pharma_signal.forecast import forecast_input_contract
from arkansas_pharma_signal.cli import _require_operational_forecast
from arkansas_pharma_signal.datasets import ablation_features, feature_columns_for_mode

import pandas as pd
import pytest


def test_constants_are_distinct():
    assert len(set(DISPOSITIONS)) == 6
    assert LIVE_INPUT != NEAR_REAL_TIME_INPUT != PERIODIC_TRAINING_ONLY
    assert STATIC_IDENTITY_CONTEXT != DERIVED_TRAINING_VARIABLE


def test_static_identity_fields():
    for name in ["drug_name", "drug_key", "ndc", "variable_id", "year",
                 "city", "geography_id", "provider_id", "npi", "zip_code"]:
        assert disposition_for_variable(name) == STATIC_IDENTITY_CONTEXT, name


def test_derived_transforms():
    for name in ["demand_claims_lag1_log", "demand_claims_ma2_log",
                 "y_last_log", "temperature_mean_lag_1", "ili_rolling_mean_4",
                 "PM2_5_mean_seasonal_anomaly", "arcos_grams_lag",
                 "drug_quarter_delta"]:
        assert disposition_for_variable(name) == DERIVED_TRAINING_VARIABLE, name


def test_periodic_annual_demand_measures():
    for name in ["ge65_tot_30day_fills", "tot_prscrbrs",
                 "medicaid_amount_reimbursed", "arcos_grams", "num_providers",
                 "population_total", "arthritis_prevalence",
                 "contract_obligations", "gscpi", "medicare_beneficiary_count"]:
        assert disposition_for_variable(name) == PERIODIC_TRAINING_ONLY, name


def test_historical_only_series():
    for name in ["covid_cases_per_100k_7_day", "humidity",
                 "arkansas_employed_population",
                 "consumer_price_index_medical_care"]:
        assert disposition_for_variable(name) == PERIODIC_TRAINING_ONLY, name


def test_near_real_time_sources():
    for name in ["temperature_mean", "precipitation", "disaster_active",
                 "ili", "num_ili", "wili", "current_week_cases",
                 "news_mentions"]:
        assert disposition_for_variable(name) == NEAR_REAL_TIME_INPUT, name


def test_unverified_who_fields_are_research_only():
    for name in ["geospread", "ili_activity", "ili_case", "ili_nb_sites",
                 "ili_outpatients", "intensity", "outbreak_event"]:
        assert disposition_for_variable(name) == PERIODIC_TRAINING_ONLY, name
        assert is_operational_feature(name) is False


def test_canonical_openfda_variables_are_live():
    assert disposition_for_variable("shortage_active") == LIVE_INPUT
    assert disposition_for_variable("recall_active") == LIVE_INPUT
    audit = audit_operational_sources(["shortage_active", "recall_active"])
    assert audit["all_sources_documented"] is True
    assert audit["all_adapters_ready"] is True


def test_source_prefixed_near_real_time_precedence():
    # news_/article_/ww_ prefixes win over periodic and historical tokens.
    for name in ["ww_covid_ar", "ww_rsv_nat", "news_policy_trade_present",
                 "news_arkansas_medicaid_articles"]:
        assert disposition_for_variable(name) == NEAR_REAL_TIME_INPUT, name


def test_historical_covid_preserved():
    # Non-prefixed historical covid series stay training-only.
    assert disposition_for_variable("covid_cases_per_100k_7_day") == PERIODIC_TRAINING_ONLY


def test_derived_precedence_over_surveillance():
    assert disposition_for_variable("ili_lag_1") == DERIVED_TRAINING_VARIABLE


def test_prior_context_precedence_over_source_tokens():
    # Prior joins are derived features, even when their names contain live-feed
    # tokens such as news, influenza, or shortage.
    for name in ["prior_news_influenza", "prior_ar_ili_mean",
                 "prior_shortage_active", "prior_context_missing"]:
        assert disposition_for_variable(name) == DERIVED_TRAINING_VARIABLE, name
    assert is_operational_feature("prior_news_influenza") is True
    assert is_operational_feature("prior_ar_ili_mean") is True
    assert is_operational_feature("prior_shortage_active") is True
    assert is_operational_feature("prior_context_missing") is False
    assert is_operational_feature("prior_medicaid_amount_reimbursed") is False
    assert disposition_for_variable("prior_ndc") == DERIVED_TRAINING_VARIABLE


def test_derived_periodic_sources_are_not_operational():
    assert disposition_for_variable("medicaid_exact_rx_log") == DERIVED_TRAINING_VARIABLE
    assert is_operational_feature("medicaid_exact_rx_log") is False
    assert is_operational_feature("PM2_5_mean_lag_1") is False
    assert is_operational_feature("natural_change_rate") is False
    assert is_operational_feature("arkansas_employed_population_lag_1") is False
    assert is_operational_feature("ili_lag_1") is True
    assert is_operational_feature("temperature_mean_lag_1") is True


def test_derived_operational_sources_inherit_raw_family():
    for name in ["arkansas_unemployment_rate_lag_1",
                 "consumer_price_index_all_items_change_1",
                 "pharma_producer_price_index_rolling_mean_4"]:
        assert disposition_for_variable(name) == DERIVED_TRAINING_VARIABLE, name
        assert is_operational_feature(name) is True


def test_derived_quality_counts_are_not_static_identity():
    assert disposition_for_variable("nadac_ndc_price_row_count") == DERIVED_TRAINING_VARIABLE
    assert disposition_for_variable("fda_shortage_supplier_count") == DERIVED_TRAINING_VARIABLE
    assert is_operational_feature("fda_shortage_supplier_count") is True


def test_local_history_is_operational_but_distinguished():
    assert disposition_for_variable("value") == LOCAL_PHARMACY_HISTORY_INPUT
    assert disposition_for_variable("quarter") == LOCAL_PHARMACY_HISTORY_INPUT
    assert is_local_history_feature("value") is True
    assert is_operational_feature("value") is True
    assert is_local_history_feature("global_gscpi_mean") is False


def test_unknown_fail_closed():
    for name in ["totally_unknown_feature_xyz", "mystery_variable",
                 "unclassified_price_signal"]:
        assert disposition_for_variable(name) == PERIODIC_TRAINING_ONLY, name


def test_operational_nws_and_nadac_inputs_are_not_training_only():
    assert disposition_for_variable("nws_alert_severity") == LIVE_INPUT
    assert disposition_for_variable("nadac_price_mean") == NEAR_REAL_TIME_INPUT


def test_source_prefixed_openfda_inputs_are_live():
    for name in [
        "openfda_shortages_shortage_active",
        "openfda_enforcement_recall_active",
        "openfda_shortages_recent_shortage_active",
    ]:
        assert disposition_for_variable(name) == LIVE_INPUT
        assert is_operational_feature(name) is True


def test_empty_name_fail_closed():
    assert disposition_for_variable("") == PERIODIC_TRAINING_ONLY
    assert disposition_for_variable("   ") == PERIODIC_TRAINING_ONLY


def test_audit_feature_names_rows():
    names = ["drug_name", "ili_lag_1", "shortage_active", "mystery_variable"]
    rows = audit_feature_names(names)
    assert len(rows) == len(names)
    assert [r["variable"] for r in rows] == names
    for row in rows:
        assert set(row) == {"variable", "disposition", "rationale"}
        assert row["disposition"] in DISPOSITIONS
        assert isinstance(row["rationale"], str) and row["rationale"]
    assert rows[0]["disposition"] == STATIC_IDENTITY_CONTEXT
    assert rows[1]["disposition"] == DERIVED_TRAINING_VARIABLE
    assert rows[2]["disposition"] == LIVE_INPUT
    assert rows[3]["disposition"] == PERIODIC_TRAINING_ONLY


def test_operational_dispositions():
    assert operational_dispositions() == {
        LIVE_INPUT, NEAR_REAL_TIME_INPUT, LOCAL_PHARMACY_HISTORY_INPUT}


def test_summarize_dispositions_counts():
    names = ["drug_name", "ili_lag_1", "shortage_active", "mystery_variable"]
    summary = summarize_dispositions(names)
    assert len(summary["rows"]) == len(names)
    assert summary["counts"] == {
        LIVE_INPUT: 1,
        NEAR_REAL_TIME_INPUT: 0,
        LOCAL_PHARMACY_HISTORY_INPUT: 0,
        PERIODIC_TRAINING_ONLY: 1,
        STATIC_IDENTITY_CONTEXT: 1,
        DERIVED_TRAINING_VARIABLE: 1,
    }
    assert summary["operational_feature_count"] == 1
    assert summary["periodic_training_only_count"] == 1
    assert summary["derived_training_variable_count"] == 1
    assert summary["derived_non_operational_count"] == 0
    assert summary["static_identity_context_count"] == 1
    assert summary["operational_ready"] is False


def test_summarize_dispositions_operational_ready_all_live():
    summary = summarize_dispositions(["shortage_active", "ili", "temperature_mean"])
    assert summary["periodic_training_only_count"] == 0
    assert summary["disposition_operational_ready"] is True
    assert summary["operational_ready"] is True
    assert summary["operational_source_audit"]["all_sources_documented"] is True
    assert summary["operational_source_audit"]["all_adapters_ready"] is True
    assert summary["operational_feature_count"] == 3


def test_near_real_time_source_registry_covers_public_inputs():
    expected = {
        "current_week_cases", "disaster_active", "disaster_severity",
        "extreme_cold_day", "extreme_heat_day", "precipitation", "snow_depth",
        "snowfall", "temperature_max", "temperature_mean", "temperature_min", "wind",
        "ili", "num_ili", "wili", "nadac_per_unit_max", "nadac_per_unit_mean",
        "nadac_per_unit_min", "respnet_rsv_rate", "arcos_distribution_grams",
        "arcos_retail_units", "regional_distribution_pressure",
    }
    assert set(NEAR_REAL_TIME_SOURCE_REGISTRY) == expected
    audit = audit_near_real_time_sources(expected)
    assert audit["variable_count"] == 22
    assert audit["all_sources_documented"] is True
    assert audit["all_adapters_ready"] is True
    assert set(RESEARCH_ONLY_SOURCE_REGISTRY) == {
        "geospread", "ili_activity", "ili_case", "ili_nb_sites",
        "ili_outpatients", "intensity", "outbreak_event",
    }


def test_cms_fills_not_operational_ready():
    summary = summarize_dispositions(["ge65_tot_30day_fills", "cms_fills",
                                      "shortage_active"])
    assert summary["periodic_training_only_count"] == 2
    assert summary["operational_ready"] is False


def test_summarize_dispositions_mixed_contract():
    summary = summarize_dispositions(
        ["shortage_active", "cms_fills", "ili_lag_1", "drug_name"])
    assert summary["periodic_training_only_count"] == 1
    assert summary["derived_training_variable_count"] == 1
    assert summary["static_identity_context_count"] == 1
    assert summary["operational_feature_count"] == 1
    assert summary["operational_ready"] is False
    assert set(summary["counts"]) == set(DISPOSITIONS)


def test_encoded_identity_prefixes():
    for name in ["ingredient::ondansetron",
                 "labeler::a_s_medication_solut",
                 "dosage_form::tablet", "route::oral", "market_cat::rx",
                 "dea::c_ii", "pharm_class::opioid", "therapeutic::analgesic"]:
        assert disposition_for_variable(name) == STATIC_IDENTITY_CONTEXT, name


def test_interaction_derived():
    for name in ["interaction::ingredient::ondansetron::labeler::a_s_medication_solut",
                 "interaction::drug_a::drug_b"]:
        assert disposition_for_variable(name) == DERIVED_TRAINING_VARIABLE, name


def test_summarize_dispositions_counts_encoded():
    names = ["ingredient::ondansetron",
             "interaction::ingredient::ondansetron::labeler::a_s_medication_solut",
             "shortage_active", "mystery_variable"]
    summary = summarize_dispositions(names)
    assert summary["counts"] == {
        LIVE_INPUT: 1,
        NEAR_REAL_TIME_INPUT: 0,
        LOCAL_PHARMACY_HISTORY_INPUT: 0,
        PERIODIC_TRAINING_ONLY: 1,
        STATIC_IDENTITY_CONTEXT: 1,
        DERIVED_TRAINING_VARIABLE: 1,
    }
    assert summary["static_identity_context_count"] == 1
    assert summary["derived_training_variable_count"] == 1
    assert summary["operational_feature_count"] == 1
    assert summary["operational_ready"] is False


def test_forecast_input_contract_fallback_mixed():
    contract = forecast_input_contract(
        {"demand_claims": {"feature_cols": ["shortage_active", "cms_fills"]}})
    assert contract["operational_ready"] is False
    assert contract["periodic_training_only_count"] == 1


def test_forecast_input_contract_uses_saved():
    saved = summarize_dispositions(["shortage_active"])
    trained = {"input_contract": saved, "demand_claims": {"feature_cols": ["cms_fills"]}}
    assert forecast_input_contract(trained) is saved


def test_forecast_input_contract_fallback():
    trained = {"demand_claims": {"feature_cols": ["drug_name", "cms_fills",
                                                  "shortage_active"]}}
    contract = forecast_input_contract(trained)
    assert contract["periodic_training_only_count"] == 1
    assert contract["operational_ready"] is False
    assert len(contract["rows"]) == 3


def test_forecast_input_contract_fallback_empty():
    contract = forecast_input_contract({})
    assert contract["periodic_training_only_count"] == 0
    assert contract["operational_ready"] is True
    assert contract["rows"] == []


def test_forecast_runtime_guard_requires_explicit_research_override():
    contract = {"operational_ready": False}
    with pytest.raises(RuntimeError, match="periodic or historical-only"):
        _require_operational_forecast(contract)
    _require_operational_forecast(contract, allow_training_only_features=True)


def test_forecast_runtime_guard_accepts_operational_contract():
    _require_operational_forecast({"operational_ready": True})


def test_feature_columns_for_mode_operational_drops_periodic_only():
    view = pd.DataFrame({c: [0.0] for c in [
        "y_last_log", "ingredient::metformin", "shortage_active",
        "news_mentions", "medicaid_rx_annual", "n_provider_types",
        "demand_benes", "cost_per_fill",
        "interaction::shortage_active::ingredient_metformin",
    ]})
    op = feature_columns_for_mode(view, "operational")
    assert "medicaid_rx_annual" not in op
    assert "n_provider_types" not in op
    assert "demand_benes" not in op
    assert "cost_per_fill" not in op
    for kept in ("y_last_log", "ingredient::metformin", "shortage_active",
                 "news_mentions",
                 "interaction::shortage_active::ingredient_metformin"):
        assert kept in op, kept
    assert all(disposition_for_variable(c) != PERIODIC_TRAINING_ONLY for c in op)


def test_feature_columns_for_mode_full_equals_ablation_full():
    view = pd.DataFrame({c: [0.0] for c in [
        "y_last_log", "shortage_active", "medicaid_rx_annual",
        "ingredient::metformin",
    ]})
    assert feature_columns_for_mode(view, "full") == ablation_features(view)["full"]


def test_feature_columns_for_mode_invalid_mode_errors():
    view = pd.DataFrame({"y_last_log": [0.0], "shortage_active": [0.0]})
    try:
        feature_columns_for_mode(view, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")
