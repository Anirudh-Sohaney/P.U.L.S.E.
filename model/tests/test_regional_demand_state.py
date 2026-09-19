import pandas as pd

from arkansas_pharma_signal.regional_demand_state import (
    build_regional_demand_view, build_state_demand_view, evaluate_regional_demand_numeric,
    evaluate_regional_demand_state,
)


def test_regional_demand_state_preserves_regions_and_three_states():
    rows = []
    regions = ["central", "northeast", "northwest", "southeast", "southwest"]
    for year in range(2010, 2020):
        for region_index, region in enumerate(regions):
            for drug in ["a", "b", "c"]:
                rows.append({"year": year, "arkansas_region": region,
                             "drug_key": drug,
                             "demand_claims": float(100 + year + region_index * 10 + len(drug))})
    view = build_regional_demand_view(pd.DataFrame(rows))
    result = evaluate_regional_demand_state(view, min_train_years=4)
    assert result["region_count"] == 5
    assert result["test_rows"] >= 25
    assert set(result["state_definition"]) == {"0", "1", "2"}


def test_regional_demand_numeric_evaluator_reports_error_metrics():
    rows = []
    regions = ["central", "northeast", "northwest", "southeast", "southwest"]
    for year in range(2010, 2020):
        for region_index, region in enumerate(regions):
            for drug in ["a", "b", "c"]:
                rows.append({"year": year, "arkansas_region": region,
                             "drug_key": drug,
                             "demand_claims": float(100 + year + region_index * 10 + len(drug))})
    result = evaluate_regional_demand_numeric(
        build_regional_demand_view(pd.DataFrame(rows)), min_train_years=4)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert 0.0 <= result["mean_model_within_5_percent_error"] <= 1.0


def test_regional_demand_five_state_contract_is_supported():
    rows = []
    regions = ["central", "northeast", "northwest", "southeast", "southwest"]
    for year in range(2010, 2020):
        for region_index, region in enumerate(regions):
            for drug_index in range(10):
                rows.append({"year": year, "arkansas_region": region,
                             "drug_key": f"drug-{drug_index}",
                             "demand_claims": float(
                                 100 + year + region_index * 10 + drug_index)})
    result = evaluate_regional_demand_state(
        build_regional_demand_view(pd.DataFrame(rows)), min_train_years=4,
        state_count=5)
    assert result["state_count"] == 5
    assert set(result["state_definition"]) == {"0", "1", "2", "3", "4"}
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}


def test_state_demand_five_state_contract_is_supported():
    rows = []
    for year in range(2010, 2020):
        for drug_index in range(10):
            rows.append({"year": year, "state": "arkansas", "drug_key": f"drug-{drug_index}",
                         "demand_claims": float(100 + year + drug_index)})
    result = evaluate_regional_demand_state(
        build_state_demand_view(pd.DataFrame(rows)), min_train_years=4, state_count=5,
        required_groups=("arkansas",), protocol_label="arkansas_state_partd")
    assert result["group_count"] == 1
    assert result["test_rows"] >= 25
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
