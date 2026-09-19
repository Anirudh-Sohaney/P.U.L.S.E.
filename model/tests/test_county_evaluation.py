"""Focused tests for the annual county x drug next-year demand evaluation."""

import numpy as np
import pandas as pd
import pytest

from arkansas_pharma_signal.county_evaluation import (
    FEATURE_MATRIX_COLS,
    _feature_matrix,
    _fit_candidates,
    build_next_year_view,
    county_rolling_cutoffs,
    evaluate_county_demand,
    evaluate_county_demand_by_region,
    evaluate_county_demand_rolling,
)


def _synthetic_outcomes(years=(2013, 2014, 2015, 2016, 2017, 2018, 2019,
                              2020, 2021, 2022, 2023, 2024),
                        n_counties=3, n_drugs=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for county in range(n_counties):
        for drug in range(n_drugs):
            base = 100.0 + 10.0 * county + 5.0 * drug
            for year in years:
                demand = max(1.0, base * (1.0 + 0.05 * (year - 2013))
                             + rng.normal(0, 8))
                rows.append({
                    "year": year,
                    "county_fips": f"05{county:03d}",
                    "county_name": f"c{county}",
                    "arkansas_region": "r",
                    "drug_key": f"drug_{drug}",
                    "ingredient": f"i{drug}",
                    "labeler": "l",
                    "demand_claims": demand,
                    "demand_fills": demand * 0.9,
                    "demand_cost": demand * 20.0,
                    "source_row_count": 5,
                    "mapped_city_count": 2,
                    "mapping_confidence": 0.95,
                    "source": "s",
                    "source_timestamp": "t",
                })
    return pd.DataFrame(rows)


def test_next_year_alignment():
    """Feature year t must predict demand_claims at t+1 for the same pair."""
    source = _synthetic_outcomes()
    view = build_next_year_view(source)
    for _, row in view.iterrows():
        t = int(row["year"])
        pair = source[(source["county_fips"] == row["county_fips"])
                      & (source["drug_key"] == row["drug_key"])
                      & (source["year"] == t + 1)]
        assert not pair.empty, "missing observed t+1 label"
        assert row["y_target"] == pytest.approx(pair.iloc[0]["demand_claims"])
        assert row["y_last"] == pytest.approx(row["demand_claims"])


def test_no_same_year_target_leakage():
    """y_target must never equal the feature-year demand; ridge features carry
    no t+1 information."""
    view = build_next_year_view(_synthetic_outcomes())
    assert (view["y_target"] != view["y_last"]).all()
    # The last observed year has no t+1 label and must be dropped.
    assert view["year"].max() < 2024


def test_time_split_and_counts():
    result = evaluate_county_demand(_synthetic_outcomes())
    assert result["split"] == "strict_next_year_county_drug_time"
    assert result["n_rows"]["validation"] > 0
    assert result["n_rows"]["test"] > 0
    assert result["n_counties"] == 3
    assert result["n_drugs"] == 4
    assert set(result["candidates"]) == {"previous_year_naive", "ridge_log1p"}
    assert result["selected_model"] in result["candidates"]
    assert "publishable_candidate" in result


def test_rolling_folds_at_least_three():
    view = build_next_year_view(_synthetic_outcomes())
    cutoffs = county_rolling_cutoffs(view, min_train_years=4)
    assert len(cutoffs) >= 3
    rolling = evaluate_county_demand_rolling(_synthetic_outcomes(),
                                             min_train_years=4)
    assert rolling["fold_count"] >= 3
    assert rolling["split"] == "rolling_origin_next_year_county_drug"


def test_rolling_folds_use_same_no_leakage_contract():
    rolling = evaluate_county_demand_rolling(_synthetic_outcomes(),
                                             min_train_years=4)
    for fold in rolling["folds"].to_dict("records"):
        assert fold["selected_model"] in {"previous_year_naive", "ridge_log1p"}
        assert fold["test_rows"] > 0
        assert fold["validation_rows"] > 0


def test_region_rolling_report_preserves_region_slices():
    source = _synthetic_outcomes()
    source.loc[source["county_fips"].eq("05000"), "arkansas_region"] = "central"
    source.loc[source["county_fips"].ne("05000"), "arkansas_region"] = "delta"
    result = evaluate_county_demand_by_region(source, min_train_years=4)
    assert result["split"] == "rolling_origin_next_year_county_drug_by_region"
    assert {row["arkansas_region"] for row in result["regions"]} == {"central", "delta"}
    assert all(row["n_counties"] > 0 for row in result["regions"])
    by_region = {row["arkansas_region"]: row for row in result["regions"]}
    assert by_region["central"]["fold_count"] == 0
    assert "no eligible" in by_region["central"]["publishability_reason"]
    assert by_region["delta"]["fold_count"] >= 3


def test_ridge_coefficient_names_match_feature_matrix_columns():
    """Model coefficient names must match the exact X column order."""
    view = build_next_year_view(_synthetic_outcomes())
    train_mask, val_mask, test_mask = (
        view["year"] <= 2020, view["year"] == 2021, view["year"] > 2021)
    train, val, test = view[train_mask], view[val_mask], view[test_mask]
    X_tr = _feature_matrix(train)
    assert X_tr.shape[1] == len(FEATURE_MATRIX_COLS)
    ridge = _fit_candidates(train, val, test)
    assert ridge["model"].feature_names == FEATURE_MATRIX_COLS


if __name__ == "__main__":
    test_next_year_alignment()
    test_no_same_year_target_leakage()
    test_time_split_and_counts()
    test_rolling_folds_at_least_three()
    test_rolling_folds_use_same_no_leakage_contract()
    test_ridge_coefficient_names_match_feature_matrix_columns()
    print("ok")
