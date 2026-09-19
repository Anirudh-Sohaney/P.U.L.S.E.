import json

import pandas as pd
import pytest

from arkansas_pharma_signal.external_metric_output import (
    build_metric_rows, validate_metric_rows,
)
from arkansas_pharma_signal.universal_forecast import TARGET_METADATA


def test_legacy_three_state_rows_are_rejected_from_qualified_output():
    with pytest.raises(ValueError, match="qualified metrics"):
        build_metric_rows(pd.DataFrame([{
            "target": "arkansas_weekly_fluview_respiratory_pressure_state",
            "forecast_period": "2026-09", "prediction": 2,
            "geography_level": "state", "geography_id": "AR",
        }]), model_version="test-v1", forecast_timestamp="2026-08-16T00:00:00Z")


def test_external_metric_rows_reject_duplicate_filter_keys():
    rows = pd.DataFrame([{
        "target": "arkansas_weekly_fluview_respiratory_pressure_state",
        "forecast_period": "2026-W35", "prediction": 1,
    }] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        build_metric_rows(rows, model_version="test-v1")


def test_external_metric_rows_reject_unknown_target():
    rows = pd.DataFrame([{
        "target": "binary_shortage_flag",
        "forecast_period": "2026-09", "prediction": 1,
    }])
    with pytest.raises(ValueError, match="unknown targets"):
        build_metric_rows(rows, model_version="test-v1")


def test_external_metric_rows_reject_invalid_five_state_value():
    rows = pd.DataFrame([{
        "target": "arkansas_region_annual_demand_five_state",
        "forecast_period": "2026-09", "prediction": 5,
    }])
    with pytest.raises(ValueError, match="must be 0, 1, 2, 3, or 4"):
        build_metric_rows(rows, model_version="test-v1")


def test_external_metric_rows_accept_qualified_five_state_region_target():
    result = build_metric_rows(pd.DataFrame([{
        "target": "arkansas_region_annual_demand_five_state",
        "forecast_period": "2027", "prediction": 4,
        "geography_level": "arkansas_region", "geography_id": "central",
        "drug_key": "drug-1",
    }]), model_version="test-v1")
    assert result.iloc[0]["target_promotion_status"] == "qualified_five_state_proxy"
    assert result.iloc[0]["target_cadence"] == "annual"


def test_external_metric_rows_expose_event_definition_as_first_class_metadata():
    result = build_metric_rows(pd.DataFrame([
        {"target": "ndc_monthly_shortage_pressure_state",
         "forecast_period": "2026-09", "prediction": 4},
        {"target": "ndc_monthly_shortage_supplier_count",
         "forecast_period": "2026-09", "prediction": 2, "drug_key": "ndc-1"},
        {"target": "nadac_next_observed_price",
         "forecast_period": "2026-W35", "prediction": 4.2, "drug_key": "ndc-2"},
    ]), model_version="test-v1")
    definitions = {
        row.target: json.loads(row.target_event_definition)
        for row in result.itertuples()
    }
    assert definitions["ndc_monthly_shortage_pressure_state"] == {
        "event_states": [3, 4], "kind": "state"}
    assert definitions["ndc_monthly_shortage_supplier_count"] == {
        "event_threshold": 1.0, "kind": "numeric", "relative_tolerance": 0.05}
    assert definitions["nadac_next_observed_price"] == {"kind": "not_applicable"}


def test_external_metric_rows_reject_unvalidated_confidence():
    rows = pd.DataFrame([{
        "target": "national_weekly_fluview_respiratory_pressure_state",
        "forecast_period": "2026-09", "prediction": 1,
        "confidence": 0.5,
    }])
    with pytest.raises(ValueError, match="unvalidated confidence"):
        build_metric_rows(rows, model_version="test-v1")


def test_external_metric_rows_preserve_source_freshness():
    result = build_metric_rows(pd.DataFrame([{
        "target": "national_weekly_fluview_respiratory_pressure_state",
        "forecast_period": "2026-W35", "prediction": 1,
        "source_freshness": "2026-07",
    }]), model_version="test-v1")
    assert result.iloc[0]["source_freshness"] == "2026-07"


def test_promoted_metadata_has_explicit_state_or_numeric_contract():
    promoted = {
        name: metadata for name, metadata in TARGET_METADATA.items()
        if metadata.get("status") in {
            "qualified_five_state_proxy", "qualified_numeric_proxy"}}
    assert promoted
    for metadata in promoted.values():
        event = metadata.get("event", {})
        if metadata["status"] == "qualified_five_state_proxy":
            assert event.get("kind") == "state"
            assert event.get("event_states") == [3, 4]
        else:
            assert event.get("kind") in {"numeric", "not_applicable"}
            if event.get("kind") == "numeric":
                assert "event_threshold" in event
                assert "relative_tolerance" in event


def test_validation_uses_metadata_status_not_target_name_suffix():
    rows = build_metric_rows(pd.DataFrame([{
        "target": "ndc_monthly_shortage_supplier_count",
        "forecast_period": "2026-09", "prediction": 2,
        "drug_key": "ndc-1",
    }]), model_version="test-v1")
    rows.loc[0, "target_promotion_status"] = "qualified_five_state_proxy"
    rows.loc[0, "prediction"] = 5
    with pytest.raises(ValueError, match="promotion status"):
        validate_metric_rows(rows)
