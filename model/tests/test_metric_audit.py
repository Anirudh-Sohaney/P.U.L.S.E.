from pathlib import Path

from arkansas_pharma_signal.metric_audit import audit_metric_library
from arkansas_pharma_signal.universal_forecast import TARGET_METADATA


def test_metric_audit_rejects_binary_shortage_and_reports_five_state_pressure():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    shortage = next(x for x in result["candidates"]
                    if x["metric"] == "fda_shortage_continuation")
    assert shortage["status"] == "rejected"
    assert "binary-only target is prohibited; use at least 5 states" in shortage["reasons"]
    pressure = next(x for x in result["candidates"]
                    if x["metric"] == "ndc_monthly_shortage_pressure_state")
    assert pressure["status"] == "qualified_proxy"
    assert pressure["target_type"] == "five_state"
    assert pressure["event_definition"] == {"kind": "state", "states": [3, 4]}
    assert pressure["drug_count"] >= 100
    assert pressure["drug_count_at_or_above_75_percent"] >= 100
    assert result["part_one_gates"]["per_drug_100_plus_75_percent"] is True

    regional = next(x for x in result["candidates"]
                    if x["metric"] == "arkansas_region_annual_demand_state")
    assert regional["region_count_at_or_above_75_percent"] == 5


def test_metric_audit_promotes_five_state_arcos_proxy():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    arcos = next(x for x in result["candidates"]
                 if x["metric"] == "arcos_zip3_drug_next_quarter_distribution")
    assert arcos["held_out_rows"] >= 25
    assert arcos["rolling_folds"] >= 3
    assert arcos["drug_count"] == 39
    assert "accuracy<65%" in arcos["reasons"]

    arcos_state = next(x for x in result["candidates"] if x["metric"] ==
                       "arcos_zip3_drug_next_quarter_distribution_state")
    assert arcos_state["status"] == "qualified_proxy"
    assert arcos_state["reasons"] == []
    assert arcos_state["target_type"] == "five_state"
    assert arcos_state["held_out_rows"] >= 25
    assert arcos_state["rolling_folds"] >= 3


def test_metric_audit_separates_proxy_accuracy_from_incremental_utility():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    flu = next(x for x in result["candidates"]
               if x["metric"] == "arkansas_weekly_fluview_respiratory_pressure_state")
    shortage = next(x for x in result["candidates"]
                    if x["metric"] == "ndc_monthly_shortage_pressure_state")
    assert flu["status"] == "rejected"
    assert "balanced_accuracy<65%" in flu["reasons"]
    assert flu["baseline_skill"]["status"] == "persistence_dominated"
    assert flu["baseline_skill"]["incremental_utility_status"] == "not_demonstrated"
    assert shortage["baseline_skill"]["incremental_utility_status"] == "mixed_cross_signal_test"
    assert shortage["baseline_skill"]["all_folds_beating_persistence"] is False


def test_metric_audit_rejects_imbalanced_hospital_state_targets():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    covid = next(x for x in result["candidates"]
                 if x["metric"] == "arkansas_weekly_hospital_covid_admission_pressure_state")
    influenza = next(x for x in result["candidates"]
                     if x["metric"] == "arkansas_weekly_hospital_influenza_admission_pressure_state")
    assert covid["status"] == "rejected"
    assert "balanced_accuracy<65%" in covid["reasons"]
    assert influenza["status"] == "rejected"
    assert "fewer than 5 target states are observed in scored rows" in influenza["reasons"]


def test_metric_audit_uses_reconstructed_nadac_history():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    nadac = next(x for x in result["candidates"]
                 if x["metric"] == "nadac_next_observed_price")
    assert nadac["rolling_folds"] == 3
    assert nadac["held_out_rows"] == 1113045
    assert nadac["status"] == "qualified_proxy"
    assert "rolling_folds<3" not in nadac["reasons"]
    assert nadac["accuracy_within_5pct_or_exact_state"] >= 0.75


def test_metric_audit_records_publishable_provenance_fields():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    required = {
        "source_url", "source_license_access", "target_semantics",
        "feature_timestamp_boundary", "geography_scope",
        "supplier_resolution", "missingness_censoring",
    }
    for candidate in result["candidates"]:
        assert required <= candidate.keys(), candidate["metric"]
        assert candidate["source_license_access"]
        assert candidate["target_semantics"]
        assert candidate["feature_timestamp_boundary"]


def test_metric_audit_keeps_qualified_targets_proxy_only():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    assert result["target_validity"]["passed"] is True
    assert result["target_validity"]["qualified_targets_are_proxy_only"] is True
    assert result["target_validity"]["qualified_direct_inventory_claims"] == []
    for candidate in result["candidates"]:
        assert candidate["target_is_direct_pharmacy_observation"] is False
        assert candidate["target_claim_scope"] == "proxy_only_not_pharmacy_inventory"


def test_metric_audit_records_regional_wastewater_level_and_change_rejections():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    level = next(x for x in result["candidates"] if x["metric"] ==
                 "arkansas_region_weekly_wastewater_influenza_level_pressure_state")
    change = next(x for x in result["candidates"] if x["metric"] ==
                  "arkansas_region_weekly_wastewater_influenza_change_pressure_state")
    assert level["status"] == "rejected"
    assert change["status"] == "rejected"
    assert "event true-positive route not met" in level["reasons"]
    assert level["held_out_rows"] == 195
    assert change["held_out_rows"] == 195


def test_metric_audit_promotes_integrated_numeric_recall_signals():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    ndc = next(x for x in result["candidates"]
               if x["metric"] == "ndc_monthly_recall_severity")
    supplier = next(x for x in result["candidates"]
                    if x["metric"] == "supplier_ndc_monthly_recall_severity")
    assert ndc["status"] == "qualified_proxy"
    assert supplier["status"] == "qualified_proxy"
    shortage = next(x for x in result["candidates"]
                    if x["metric"] == "ndc_monthly_shortage_supplier_count")
    assert shortage["status"] == "qualified_proxy"
    regional = next(x for x in result["candidates"]
                    if x["metric"] == "arkansas_region_annual_demand_five_state")
    assert regional["status"] == "qualified_proxy"
    national_flu = next(x for x in result["candidates"]
                        if x["metric"] == "national_weekly_fluview_respiratory_pressure_state")
    assert national_flu["status"] == "qualified_proxy"
    assert national_flu["accuracy_within_5pct_or_exact_state"] >= 0.75
    assert national_flu["balanced_accuracy"] >= 0.65
    county = next(x for x in result["candidates"]
                  if x["metric"] == "arkansas_county_annual_demand_five_state")
    assert county["status"] == "qualified_proxy"
    assert county["target_type"] == "five_state"
    assert county["drug_count"] == 1326
    assert county["arkansas_region_count"] == 73
    assert county["accuracy_within_5pct_or_exact_state"] >= 0.65
    assert county["event_definition"] == {"kind": "state", "states": [3, 4]}
    assert county["event_metrics"]["true_positive_precision"] >= 0.80
    nssp = next(x for x in result["candidates"]
                if x["metric"] == "arkansas_weekly_nssp_influenza_ed_pressure_state")
    assert nssp["status"] == "qualified_proxy"
    assert nssp["target_type"] == "five_state"
    assert nssp["event_metrics"]["true_positive_precision"] >= 0.80
    assert result["qualified_metric_count"] == 14
    assert result["part_one_gates"]["per_arkansas_region_75_percent"] is True
    assert result["coverage_gates"]["geography"]["passed"] is True
    assert result["coverage_gates"]["drug"]["passed"] is True
    assert result["coverage_gates"]["supplier"]["passed"] is True
    assert result["coverage_gates"]["disease_symptom"]["passed"] is True


def test_all_qualified_event_metrics_have_machine_readable_event_scores():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    for candidate in result["candidates"]:
        if candidate["status"] != "qualified_proxy":
            continue
        definition = candidate["event_definition"]
        if definition["kind"] == "not_applicable":
            assert candidate["event_metrics"] is None
            continue
        metrics = candidate["event_metrics"]
        assert metrics and metrics["event_applicable"] is True
        assert metrics["predicted_event_count"] > 0
        assert metrics["true_positive_precision"] >= 0.0


def test_qualified_event_definitions_match_operational_target_metadata():
    result = audit_metric_library(Path("model/artifacts/evaluation"))
    qualified = [candidate for candidate in result["candidates"]
                 if candidate["status"] == "qualified_proxy"]
    for candidate in qualified:
        definition = candidate["event_definition"]
        metadata_definition = TARGET_METADATA[candidate["metric"]]["event"]
        assert definition["kind"] == metadata_definition["kind"]
        if definition["kind"] == "state":
            assert definition["states"] == metadata_definition["event_states"]
        elif definition["kind"] == "numeric":
            assert definition["threshold"] == metadata_definition["event_threshold"]
        else:
            assert definition["kind"] == "not_applicable"
