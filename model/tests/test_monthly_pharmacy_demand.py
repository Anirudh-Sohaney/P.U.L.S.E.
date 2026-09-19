import pandas as pd
import json
from pathlib import Path

from arkansas_pharma_signal.monthly_pharmacy_demand import (
    build_monthly_pharmacy_demand_view, evaluate_monthly_pharmacy_demand,
)
from arkansas_pharma_signal.monthly_context_utility import build_monthly_context_view
from arkansas_pharma_signal.therapeutic_class_demand import (
    build_therapeutic_class_panel, evaluate_therapeutic_class_demand,
)


def test_monthly_view_excludes_nonconsecutive_suppressed_months():
    rows = []
    for drug_index in range(8):
        for month in pd.period_range("2018-01", "2024-12", freq="M"):
            if month.month == 6 and drug_index % 2 == 0:
                continue
            rows.append({"month": str(month), "drug_key": f"ndc-{drug_index}",
                         "demand_claim_lines": 100 + drug_index + month.month,
                         "demand_paid": 1000.0, "pharmacy_provider_count": 3})
    view = build_monthly_pharmacy_demand_view(pd.DataFrame(rows))
    assert not view.empty
    assert (view["next_month"] == view["month"] + 1).all()
    assert "market_lag1_demand" in view
    assert "labeler_lag1_demand" in view


def test_monthly_evaluator_emits_five_states_and_event_definition():
    rows = []
    for drug_index in range(20):
        for month in pd.period_range("2015-01", "2024-12", freq="M"):
            rows.append({"month": str(month), "drug_key": f"ndc-{drug_index:05d}",
                         "demand_claim_lines": 100 + 5 * drug_index + month.month,
                         "demand_paid": 1000.0, "pharmacy_provider_count": 3})
    result = evaluate_monthly_pharmacy_demand(
        build_monthly_pharmacy_demand_view(pd.DataFrame(rows)),
        min_train_months=24)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}


def test_monthly_evaluator_supports_county_drug_groups():
    rows = []
    for county in ["05001", "05003", "05005"]:
        for drug_index in range(5):
            for month in pd.period_range("2015-01", "2024-12", freq="M"):
                rows.append({"month": str(month), "county_fips": county,
                             "drug_key": f"ndc-{drug_index}",
                             "demand_claim_lines": 100 + drug_index + month.month,
                             "demand_paid": 1000.0, "pharmacy_provider_count": 2})
    view = build_monthly_pharmacy_demand_view(
        pd.DataFrame(rows), group_columns=("county_fips", "drug_key"))
    result = evaluate_monthly_pharmacy_demand(
        view, group_columns=("county_fips", "drug_key"))
    assert result["county_count"] == 3
    assert result["group_columns"] == ["county_fips", "drug_key"]


def test_hhs_aggregate_candidates_do_not_bypass_event_gate():
    artifact = json.loads(Path(
        "model/artifacts/evaluation/arkansas_pharmacy_market_context_metrics.json"
    ).read_text())
    results = artifact["results"]
    assert results["monthly_claim_line_demand"]["publishable_candidate"] is False
    assert results["monthly_observed_pharmacy_provider_capacity"][
        "publishable_candidate"] is False


def test_monthly_context_is_shifted_before_demand_join():
    demand = pd.DataFrame({
        "month": ["2020-01", "2020-02", "2020-03"],
        "drug_key": ["00000000001"] * 3,
        "demand_claim_lines": [10, 12, 14],
    })
    news = pd.DataFrame({
        "year": [2020, 2020], "month": [1, 2], "article_count": [5, 9],
    })
    view = build_monthly_context_view(demand, news=news)
    february = view.loc[view["month"].astype(str).eq("2020-02")].iloc[0]
    assert february["prior_news_count"] == 5


def test_monthly_context_shifts_respnet_before_demand_join():
    demand = pd.DataFrame({
        "month": ["2020-01", "2020-02", "2020-03"],
        "drug_key": ["00000000001"] * 3,
        "demand_claim_lines": [10, 12, 14],
    })
    respnet = pd.DataFrame({
        "date": ["2020-01-04", "2020-02-01"],
        "estimate": [10.0, 20.0],
    })
    view = build_monthly_context_view(demand, respnet=respnet)
    february = view.loc[view["month"].astype(str).eq("2020-02")].iloc[0]
    assert february["prior_nat_respnet_rsv"] == 10.0


def test_monthly_context_shifts_apcd_claim_activity_before_demand_join():
    demand = pd.DataFrame({
        "month": ["2020-01", "2020-02", "2020-03"],
        "drug_key": ["00000000001"] * 3,
        "demand_claim_lines": [10, 12, 14],
    })
    apcd = pd.DataFrame({
        "year": [2020, 2020], "month": [1, 2],
        "claim_count": [100, 200],
    })
    view = build_monthly_context_view(demand, apcd_claims=apcd)
    february = view.loc[view["month"].astype(str).eq("2020-02")].iloc[0]
    assert february["prior_apcd_claim_count"] == 100.0


def test_therapeutic_class_panel_maps_ndcs_without_double_counting():
    demand = pd.DataFrame([
        {"month": "2020-01", "drug_key": "4080085",
         "demand_claim_lines": 10, "demand_paid": 20.0,
         "pharmacy_provider_count": 1},
        {"month": "2020-02", "drug_key": "4080085",
         "demand_claim_lines": 12, "demand_paid": 22.0,
         "pharmacy_provider_count": 1},
    ])
    mapping = pd.DataFrame([
        {"ndc11": "00004080085", "class_id": "J05AH",
         "class_name": "Neuraminidase inhibitors", "mapping_status": "mapped"},
        {"ndc11": "00004080085", "class_id": "A01AC",
         "class_name": "Local corticosteroids", "mapping_status": "mapped"},
    ])
    panel = build_therapeutic_class_panel(demand, mapping)
    assert panel["demand_claim_lines"].sum() == 22
    assert panel["therapeutic_class"].nunique() == 2


def test_therapeutic_class_evaluator_has_five_state_event_contract():
    rows = []
    for index, class_id in enumerate(("A01", "J01", "N02", "R05", "C01")):
        for month in pd.period_range("2015-01", "2024-12", freq="M"):
            rows.append({"month": str(month), "drug_key": class_id,
                         "demand_claim_lines": 100 + index * 10 + month.month,
                         "demand_paid": 1000.0, "pharmacy_provider_count": 3})
    mapping = pd.DataFrame([{
        "ndc11": f"0000000000{index + 1}", "class_id": class_id,
        "class_name": class_id, "mapping_status": "mapped",
    } for index, class_id in enumerate(("A01", "J01", "N02", "R05", "C01"))])
    demand = pd.DataFrame([{
        "month": str(month), "drug_key": f"{index + 1}",
        "demand_claim_lines": 100 + index * 10 + month.month,
        "demand_paid": 1000.0, "pharmacy_provider_count": 3,
    } for index in range(5) for month in pd.period_range("2015-01", "2024-12", freq="M")])
    result = evaluate_therapeutic_class_demand(demand, mapping)
    assert result["target"] == "arkansas_monthly_atc_therapeutic_demand_state"
    assert result["class_count"] == 5
    assert result["event_definition"]["event_states"] == [3, 4]
