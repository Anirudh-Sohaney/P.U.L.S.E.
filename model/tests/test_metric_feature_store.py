import pandas as pd
import pytest

from arkansas_pharma_signal.metric_feature_store import (
    attach_qualified_metric_features, build_metric_feature_store,
    join_metric_features, metric_context_vector,
)
from scripts.build_metric_feature_store import build_feature_store_file


def _rows():
    return pd.DataFrame([
        {"forecast_period": "2026-08", "geography_level": "national_ndc",
         "geography_id": "US", "county_fips": "", "drug_key": "1", "labeler": "",
         "supplier": "",
         "target": "national_weekly_fluview_respiratory_pressure_state", "prediction": 2,
         "target_promotion_status": "qualified_five_state_proxy",
         "target_semantics": "five states", "target_cadence": "weekly",
         "source_freshness": "2026-W30", "driver_attribution": "{}",
         "uncertainty_status": "not_estimated", "calibration_status": "not_calibrated"},
    ])


def test_metric_feature_store_preserves_grain_and_metadata():
    store, metadata = build_metric_feature_store(_rows())
    assert store.loc[0, "metric__national_weekly_fluview_respiratory_pressure_state"] == 2
    assert metadata["feature_count"] == 1
    base = _rows()[["forecast_period", "geography_level", "geography_id",
                    "county_fips", "drug_key", "labeler", "supplier"]].copy()
    joined = join_metric_features(base, store)
    assert joined.shape[0] == 1


def test_metric_feature_store_rejects_unqualified_rows():
    rows = _rows()
    rows.loc[0, "target_promotion_status"] = "research_only"
    with pytest.raises(ValueError, match="unqualified"):
        build_metric_feature_store(rows)


def test_metric_feature_store_rejects_invalid_state_values():
    rows = _rows()
    rows.loc[0, "prediction"] = 5
    with pytest.raises(ValueError, match="must be 0, 1, 2, 3, or 4"):
        build_metric_feature_store(rows)


def test_attach_preserves_exact_grain_and_exposes_missingness():
    base = pd.DataFrame([
        {"forecast_period": "2026-08", "geography_level": "national_ndc",
         "geography_id": "US", "county_fips": "", "drug_key": "1", "labeler": "",
         "supplier": ""},
        {"forecast_period": "2026-08", "geography_level": "county",
         "geography_id": "05001", "county_fips": "05001", "drug_key": "1", "labeler": "",
         "supplier": ""},
    ])
    joined, metadata = attach_qualified_metric_features(base, _rows())
    feature = "metric__national_weekly_fluview_respiratory_pressure_state"
    assert joined.shape[0] == 2
    assert joined.loc[0, feature] == 2
    assert pd.isna(joined.loc[1, feature])
    assert joined.loc[0, f"{feature}__missing"] == 0
    assert joined.loc[1, f"{feature}__missing"] == 1
    assert metadata["values_are_not_imputed"] is True


def test_default_feature_grain_preserves_supplier_rows():
    rows = pd.concat([_rows(), _rows().assign(supplier="supplier-b", prediction=1)],
                     ignore_index=True)
    store, metadata = build_metric_feature_store(rows)
    assert metadata["keys"][-1] == "supplier"
    assert len(store) == 2
    assert set(store["supplier"]) == {"", "supplier-b"}


def test_feature_store_script_writes_hashed_metadata(tmp_path):
    source = tmp_path / "forecast.csv.gz"
    output = tmp_path / "features.csv.gz"
    metadata = tmp_path / "features.json"
    _rows().to_csv(source, index=False, compression="gzip")
    result = build_feature_store_file(source, output, metadata)
    assert result["source_rows"] == 1
    assert result["output_rows"] == 1
    assert len(result["source_sha256"]) == 64
    assert len(result["output_sha256"]) == 64
    assert metadata.exists()


def test_context_projection_matches_statewide_context_without_drug_broadcast():
    rows = _rows()
    base = pd.DataFrame([
        {"forecast_period": "2026-08", "drug_key": "1", "county_fips": "05001"},
        {"forecast_period": "2026-08", "drug_key": "2", "county_fips": "05003"},
    ])
    joined, metadata = attach_qualified_metric_features(
        base, rows, join_mode="context")
    feature = "metric__national_weekly_fluview_respiratory_pressure_state"
    assert joined.loc[0, feature] == 2
    assert joined.loc[1, feature] == 2
    assert joined.loc[0, f"{feature}__missing"] == 0
    assert joined.loc[1, f"{feature}__missing"] == 0
    assert metadata["projected_features"][0]["destination_keys"] == [
        "forecast_period"]


def test_context_projection_supports_statewide_period_only_signal():
    rows = _rows().assign(
        target="national_weekly_fluview_respiratory_pressure_state",
        forecast_period="2026",
        drug_key="", geography_level="arkansas_state", geography_id="AR",
        target_semantics="state pressure", target_cadence="weekly",
        source_freshness="2025", prediction=1)
    base = pd.DataFrame({"forecast_period": ["2026", "2026"],
                         "county_fips": ["05001", "05003"]})
    joined, metadata = attach_qualified_metric_features(
        base, rows, join_mode="context")
    feature = "metric__national_weekly_fluview_respiratory_pressure_state"
    assert joined[feature].tolist() == [1, 1]
    assert metadata["projected_features"][0]["scope"] == "national upstream respiratory context"


def test_metric_context_vector_has_stable_value_then_missing_order():
    joined, metadata = attach_qualified_metric_features(
        _rows(), _rows()[["forecast_period", "geography_level", "geography_id",
                          "county_fips", "drug_key", "labeler", "supplier",
                          "target", "prediction", "target_promotion_status",
                          "target_semantics", "target_cadence", "source_freshness",
                          "driver_attribution", "uncertainty_status",
                          "calibration_status"]])
    row = joined.iloc[0]
    vector = metric_context_vector(row, metadata)
    assert vector.shape == (2,)
    assert vector.tolist() == [2.0, 0.0]
    assert metadata["model_context_order"] == [
        "metric__national_weekly_fluview_respiratory_pressure_state",
        "metric__national_weekly_fluview_respiratory_pressure_state__missing",
    ]


def test_metric_context_vector_marks_unavailable_feature():
    metadata = {"feature_columns": ["metric__example"]}
    vector = metric_context_vector({}, metadata)
    assert vector.tolist() == [0.0, 1.0]
