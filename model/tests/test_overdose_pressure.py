import numpy as np
import pandas as pd
import pytest

from arkansas_pharma_signal.overdose_pressure import (
    build_overdose_observations, build_overdose_view, evaluate_overdose_pressure,
)


def _source():
    rows = []
    for county, offset in (("05001", 0), ("05003", 10), ("05005", 20)):
        for month in pd.period_range("2020-01", "2025-12", freq="M"):
            value = 30 + offset + 5 * np.sin(month.month / 12 * 2 * np.pi)
            rows.append({"st_abbrev": "AR", "fips": county,
                         "monthendingdate": month.end_time,
                         "provisional_drug_overdose": value})
    return pd.DataFrame(rows)


def test_overdose_view_does_not_bridge_suppressed_months():
    source = _source()
    source = source[~((source.fips == "05001") &
                      (pd.to_datetime(source.monthendingdate).dt.to_period("M") == "2022-03"))]
    view = build_overdose_view(source)
    assert not ((view.county_fips == "05001") &
                (view.period == pd.Period("2022-02"))).any()
    observed = build_overdose_observations(source)
    assert observed.period.max() == pd.Period("2025-12")


def test_overdose_view_rejects_missing_source_columns():
    with pytest.raises(ValueError, match="missing"):
        build_overdose_view(pd.DataFrame())


def test_overdose_evaluation_is_chronological_and_three_state():
    result = evaluate_overdose_pressure(build_overdose_view(_source()), min_train_months=12)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2"}
    assert all(fold["test_period"] > fold["validation_period"]
               for fold in result["folds"])


def test_current_snapshot_cannot_be_used_as_historical_vintage():
    source = _source()
    source["data_as_of"] = "2026-07-05"
    result = evaluate_overdose_pressure(build_overdose_view(source))
    assert result["publishable_candidate"] is False
    assert result["vintage_status"] == "historical_vintages_unavailable"
    assert result["fold_count"] == 0
