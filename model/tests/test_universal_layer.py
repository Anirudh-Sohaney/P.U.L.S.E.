"""Focused tests for the universal graph/news/forecast contracts."""

import pandas as pd

from arkansas_pharma_signal.event_schema import EVENT_COLUMNS, empty_events, validate_events
from arkansas_pharma_signal.news_corpus import available_at
from arkansas_pharma_signal.universal_forecast import (
    UNIVERSAL_OUTPUT_COLUMNS,
    build_universal_forecast_grid,
    validate_universal_forecast,
)
from arkansas_pharma_signal.geography import REGION_COUNTIES, region_for_county


def test_arkansas_regions_are_a_disjoint_75_county_partition():
    assignments = [county for counties in REGION_COUNTIES.values()
                   for county in counties]
    assert len(assignments) == 75
    assert len(set(assignments)) == 75
    assert set(region_for_county(county) for county in assignments) == set(REGION_COUNTIES)


def test_article_time_barrier_excludes_future_articles():
    corpus = pd.DataFrame({
        "article_id": ["past", "future"],
        "published_at": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "body": ["real", "real"],
    })
    assert list(available_at(corpus, "2024-06-01")["article_id"]) == ["past"]


def test_article_time_barrier_excludes_unknown_publication_time():
    corpus = pd.DataFrame({
        "article_id": ["known", "unknown"],
        "published_at": ["2024-01-01T00:00:00Z", None],
        "body": ["real", "real"],
    })
    assert list(available_at(corpus, "2024-06-01")["article_id"]) == ["known"]


def test_event_contract_empty_frame_is_valid():
    events = empty_events()
    assert list(events.columns) == EVENT_COLUMNS
    validate_events(events)


def test_universal_forecast_keeps_unresolved_supplier_hierarchy_explicit():
    panel = pd.DataFrame([{
        "year": 2024, "drug_key": "metformin", "ingredient": "metformin",
        "labeler": "Example Labeler", "demand_claims": 100.0,
        "demand_cost": 50.0, "shortage_active": 0.0, "recall_count_x": 0.0,
    }])
    result = build_universal_forecast_grid(panel, max_rows=25)
    validate_universal_forecast(result)
    assert list(result.columns) == UNIVERSAL_OUTPUT_COLUMNS
    assert result["evidence_type"].eq("unresolved").all()
    assert result["confidence"].eq(0.35).all()
    assert result["parent_company"].eq("").all()
    assert result["factory"].eq("").all()
    assert result["api_source"].eq("").all()
    assert result["target_is_direct_pharmacy_observation"].eq(False).all()
    assert result["target_promotion_status"].notna().all()


def test_universal_forecast_uses_exact_drug_supplier_row():
    panel = pd.DataFrame([{
        "year": 2024, "drug_key": "metformin", "ingredient": "metformin",
        "labeler": "Example Labeler", "demand_claims": 100.0,
        "demand_cost": 50.0, "shortage_active": 0.0, "recall_count_x": 0.0,
    }])
    suppliers = pd.DataFrame([
        {"labeler": "Example Labeler", "drug_key": "other", "factory": "Wrong Factory"},
        {"labeler": "Example Labeler", "drug_key": "metformin", "factory": "Right Factory"},
    ])
    result = build_universal_forecast_grid(panel, max_rows=25,
                                           supplier_lookup=suppliers)
    assert result["factory"].eq("Right Factory").all()


def test_universal_forecast_supports_filterable_county_region_and_supplier():
    panel = pd.DataFrame([
        {"year": 2024, "city": "A", "county_fips": "05001", "county_name": "A",
         "arkansas_region": "central", "drug_key": "d1", "ingredient": "d1",
         "labeler": "Supplier A", "demand_claims": 100.0, "demand_cost": 50.0},
        {"year": 2024, "city": "B", "county_fips": "05003", "county_name": "B",
         "arkansas_region": "delta", "drug_key": "d2", "ingredient": "d2",
         "labeler": "Supplier B", "demand_claims": 90.0, "demand_cost": 40.0},
    ])
    result = build_universal_forecast_grid(
        panel, max_rows=100, county_fips=["05001"], regions=["central"],
        suppliers=["supplier a"])
    validate_universal_forecast(result)
    county = result[result["geography_level"].eq("county")]
    assert not county.empty
    assert county["county_fips"].eq("05001").all()
    assert county["arkansas_region"].eq("central").all()
    assert county["labeler"].eq("Supplier A").all()
    assert "shortage_state" not in set(county["target"])


def test_universal_forecast_emits_explicit_region_context_rows():
    panel = pd.DataFrame([{
        "year": 2024, "city": "A", "county_fips": "05031",
        "arkansas_region": "central", "drug_key": "drug-a",
        "labeler": "Supplier A", "demand_claims": 10, "demand_cost": 20,
    }])
    result = build_universal_forecast_grid(panel, max_rows=100)
    validate_universal_forecast(result)
    region = result[result["geography_level"].eq("region")]
    assert not region.empty
    assert region["geography_id"].eq("central").all()
    assert region["drug_key"].eq("").all()
    assert region["evidence_type"].eq("aggregated_county_context").all()
    assert "shortage_state" not in set(result["target"])


def test_universal_forecast_emits_supplier_drug_shortage_context_without_county_claim():
    panel = pd.DataFrame([{
        "year": 2024, "drug_key": "furosemide", "ingredient": "furosemide",
        "labeler": "Hospira", "demand_claims": 10, "demand_cost": 20,
    }])
    scores = pd.DataFrame([{
        "supplier": "hospira", "drug": "furosemide",
        "feature_month": "2024-12", "shortage_probability": 0.7,
    }])
    result = build_universal_forecast_grid(
        panel, max_rows=100, supplier_shortage_scores=scores)
    validate_universal_forecast(result)
    context = result[result["geography_level"].eq("supplier_drug")]
    assert len(context) == 1
    assert context.iloc[0]["target"] == "supplier_shortage_probability"
    assert context.iloc[0]["prediction"] == 0.7
    assert context.iloc[0]["county_fips"] == ""
    assert context.iloc[0]["evidence_type"] == "fda_reported_supplier_drug_event_model"
    assert context.iloc[0]["target_cadence"] == "monthly"
    assert context.iloc[0]["target_supplier_resolution"] == "FDA supplier"
    assert context.iloc[0]["target_is_direct_pharmacy_observation"] == False
    projected = result[result["geography_level"].eq("arkansas_supplier_drug")]
    assert len(projected) == 1
    assert projected.iloc[0]["geography_id"] == "AR"
    assert projected.iloc[0]["target"] == "arkansas_supplier_drug_shortage_pressure"
    assert projected.iloc[0]["confidence"] == 0.25
    assert "supplier_to_arkansas_allocation_verified" in projected.iloc[0]["driver_attribution"]


def test_supplier_filter_applies_to_both_supplier_context_surfaces():
    panel = pd.DataFrame([{
        "year": 2024, "drug_key": "furosemide", "ingredient": "furosemide",
        "labeler": "Hospira", "demand_claims": 10, "demand_cost": 20,
    }])
    scores = pd.DataFrame([
        {"supplier": "hospira", "drug": "furosemide", "feature_month": "2024-12",
         "shortage_probability": 0.7},
        {"supplier": "other supplier", "drug": "furosemide", "feature_month": "2024-12",
         "shortage_probability": 0.8},
    ])
    result = build_universal_forecast_grid(
        panel, max_rows=100, suppliers=["HOSPIRA"], supplier_shortage_scores=scores)
    supplier_levels = result[result["geography_level"].isin(
        ["supplier_drug", "arkansas_supplier_drug"])]
    assert len(supplier_levels) == 2
    assert supplier_levels["labeler"].eq("hospira").all()


def test_region_crosswalk_is_used_when_existing_region_is_empty():
    assert region_for_county("05031") == "northeast"
    panel = pd.DataFrame([{
            "year": 2024, "city": "A", "county_fips": "05119", "county_name": "Pulaski",
        "arkansas_region": "", "drug_key": "drug-a", "labeler": "Supplier A",
        "demand_claims": 10, "demand_cost": 20, "shortage_active": 0,
    }])
    result = build_universal_forecast_grid(panel, max_rows=25)
    assert result["arkansas_region"].eq("central").all()


def test_neighbor_state_context_uses_external_events_without_arkansas_demand():
    panel = pd.DataFrame([{
        "year": 2024, "city": "A", "county_fips": "05031", "drug_key": "drug-a",
        "labeler": "Supplier A", "demand_claims": 10, "demand_cost": 20,
    }])
    state_events = pd.DataFrame([{
        "event_id": "e1", "event_type": "drug_shortage", "geography": "OK",
        "start_time": "2024-01-01", "severity": 0.8, "confidence": 1.0,
    }])
    result = build_universal_forecast_grid(panel, max_rows=100, state_events=state_events)
    context = result[result["geography_level"].eq("neighbor_state")]
    assert not context.empty
    assert context["geography_id"].eq("OK").all()
    assert context["target"].eq("neighbor_state_supply_event_risk").all()
    assert context["county_fips"].eq("").all()
    assert context["drug_key"].eq("").all()


def test_untrained_risk_path_does_not_convert_events_into_forecast_probability():
    panel = pd.DataFrame([{
        "year": 2024, "city": "A", "county_fips": "05031",
        "drug_key": "drug-a", "labeler": "Supplier A",
        "demand_claims": 10, "demand_cost": 20,
    }])
    events = pd.DataFrame([{
        "event_id": "e1", "county_fips": "05031", "location": "A",
        "article_id": "a1",
    }])
    result = build_universal_forecast_grid(panel, max_rows=100, events=events)
    risk = result[(result["geography_level"] == "county")
                  & (result["target"] == "supply_disruption_risk")]
    assert not risk.empty
    assert risk["prediction"].eq(0.5).all()
    assert '"event_use": "evidence_only"' in risk.iloc[0]["driver_attribution"]


def test_universal_forecast_emits_mapped_arcos_zip3_context():
    panel = pd.DataFrame([{
        "year": 2024, "drug": "methylphenidate", "drug_key": "methylphenidate",
        "ingredient": "methylphenidate", "labeler": "Supplier A",
        "demand_claims": 10, "demand_cost": 20,
    }])
    arcos = pd.DataFrame([{
        "period": "2025-Q4", "period_index": 8103, "zip3": "722",
        "drug_code": "1724", "drug_name": "METHYLPHENIDATE", "grams": 1234.0,
    }])
    result = build_universal_forecast_grid(panel, max_rows=100, arcos_panel=arcos)
    validate_universal_forecast(result)
    context = result[result["geography_level"].eq("zip3")]
    assert len(context) == 1
    assert context.iloc[0]["geography_id"] == "722"
    assert context.iloc[0]["target"] == "regional_distribution_pressure"
    assert context.iloc[0]["prediction"] == 1234.0
    assert context.iloc[0]["evidence_type"] == "observed_distribution_proxy"
    assert "inventory_observed" in context.iloc[0]["driver_attribution"]


def test_universal_forecast_does_not_fabricate_unmapped_arcos_context():
    panel = pd.DataFrame([{
        "year": 2024, "drug": "unmapped", "drug_key": "unmapped",
        "ingredient": "unmapped", "demand_claims": 10, "demand_cost": 20,
    }])
    arcos = pd.DataFrame([{
        "period": "2025-Q4", "period_index": 8103, "zip3": "722",
        "drug_code": "1724", "drug_name": "METHYLPHENIDATE", "grams": 1234.0,
    }])
    result = build_universal_forecast_grid(panel, max_rows=100, arcos_panel=arcos)
    validate_universal_forecast(result)
    assert not result["geography_level"].eq("zip3").any()


if __name__ == "__main__":
    test_article_time_barrier_excludes_future_articles()
    test_event_contract_empty_frame_is_valid()
    test_universal_forecast_keeps_unresolved_supplier_hierarchy_explicit()
    print("ok")
