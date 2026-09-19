import json

import numpy as np
import pandas as pd
import pytest

from arkansas_pharma_signal.publishable_evaluation import (
    _binary_result, _fit_ridge, _rolling_month_splits, _rolling_year_splits,
    assess_target_adequacy,
    _within_relative_error,
    verify_suite,
)


def test_numeric_protocol_selects_alpha_on_validation_only():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "target": [2, 4, 6, 8], "baseline": [1, 3, 5, 7]})
    validation = pd.DataFrame({"x": [5, 6], "target": [10, 12], "baseline": [8, 10]})
    test = pd.DataFrame({"x": [7, 8], "target": [14, 16], "baseline": [12, 14]})
    result = _fit_ridge(train, validation, test, ["x"], "target", "baseline")
    assert result["selected_alpha"] in {0.1, 1.0, 10.0, 100.0}
    assert result["test_rows"] == 2
    assert "within_5pct" in result["ridge_log1p"]["test"]


def test_within_relative_error_is_explicit_and_strict():
    assert _within_relative_error([100.0, 100.0], [105.0, 106.0]) == 0.5


def test_binary_metrics_do_not_reduce_to_raw_accuracy():
    result = _binary_result(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert result["binary_accuracy"] == 0.75
    assert result["balanced_accuracy"] < result["binary_accuracy"]
    assert "auprc" in result and "brier" in result


def test_target_adequacy_does_not_upgrade_missing_local_inventory_labels():
    state = pd.DataFrame({"x": [1]})
    county = pd.DataFrame({"x": [1, 2]})
    supplier = pd.DataFrame({
        "target_next_right_censored": [0, 1, 0],
        "shortage_active": [0, 1, 0],
        "target_next_shortage_active": [1, 0, 0],
    })
    result = assess_target_adequacy(state, county, supplier)
    assert result["supplier_shortage_evidence"]["right_censored_rows"] == 1
    assert result["supplier_shortage_evidence"]["onset_candidate_positive_rows"] == 1
    assert result["requested_local_inventory_target"]["available"] is False
    assert result["requested_local_inventory_target"]["synthetic_labels_created"] is False


def test_rolling_year_splits_use_immediately_following_year():
    frame = pd.DataFrame({"year": [2012, 2013, 2014, 2015, 2016, 2017]})
    splits = list(_rolling_year_splits(frame, min_train_years=4))
    assert [(split[1], split[2]) for split in splits] == [
        ("2016", "2017")
    ]
    assert list(splits[0][3]["year"]) == [2012, 2013, 2014, 2015]


def test_rolling_month_splits_preserve_calendar_order():
    months = pd.date_range("2020-01-01", periods=18, freq="MS")
    frame = pd.DataFrame({"month": months.strftime("%Y-%m")})
    splits = list(_rolling_month_splits(
        frame, min_train_months=12, validation_months=3, test_months=3))
    assert len(splits) == 1
    assert splits[0][1] == "2021-01-2021-03"
    assert splits[0][2] == "2021-04-2021-06"
    assert len(splits[0][3]) == 12
    assert len(splits[0][4]) == 3
    assert len(splits[0][5]) == 3


def test_suite_verification_fails_closed_for_missing_artifact(tmp_path):
    manifest = {
        "dataset_version": "test",
        "geography_definition": {
            "name": "Arkansas DHS/TEFRA five-region county partition",
            "source_url": "https://humanservices.arkansas.gov/wp-content/uploads/TEFRAAttachI.pdf",
            "regions": ["northwest", "northeast", "central", "southwest", "southeast"],
            "county_count": 75,
            "partition_policy": "each Arkansas county FIPS is assigned exactly once",
        },
        "tables": {
            "arkansas_ndc_next_quarter_test.csv.gz": {
                "targets": ["target_next_medicaid_prescriptions",
                             "target_next_shortage_active"],
                "target_qualification": {
                    "cadence": "quarterly",
                    "geography": "Arkansas statewide",
                    "direct_observation": True,
                }
            },
            "arkansas_county_drug_next_year_test.csv.gz": {
                "targets": ["target_next_demand_claims"],
                "target_qualification": {
                    "cadence": "annual",
                    "geography": "Arkansas county",
                    "direct_observation": True,
                }
            },
            "supplier_ndc_next_month_test.csv.gz": {
                "targets": ["target_next_shortage_active"],
                "target_qualification": {
                    "cadence": "monthly",
                    "geography": "national evidence restricted to Arkansas-exposed NDCs",
                    "direct_observation": True,
                }
            },
            "arcos_zip3_drug_next_quarter_test.csv.gz": {
                "targets": ["target_next_distribution_grams"],
                "target_qualification": {
                    "cadence": "quarterly",
                    "geography": "Arkansas ZIP3",
                    "direct_observation": True,
                }
            },
        },
        "artifacts": {"missing.csv.gz": {"rows": 0, "sha256": ""}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="hash verification failed"):
        verify_suite(tmp_path)
