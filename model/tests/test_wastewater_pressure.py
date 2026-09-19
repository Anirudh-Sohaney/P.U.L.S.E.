import pandas as pd

from arkansas_pharma_signal.wastewater_pressure import (
    build_wastewater_weekly_view, evaluate_wastewater_state_rolling,
)


def test_wastewater_view_aggregates_sites_and_preserves_next_week_boundary():
    dates = pd.date_range("2020-01-04", periods=160, freq="7D")
    rows = []
    for index, date in enumerate(dates):
        for site, offset in (("a", 0.0), ("b", 1.0)):
            rows.append({"week_end": date, "pathogen_target": "Influenza A virus",
                         "site": site, "site_wval": float((index % 10) + offset)})
    view = build_wastewater_weekly_view(pd.DataFrame(rows), pathogen="influenza")
    assert len(view) == 159
    assert view["week"].is_monotonic_increasing
    assert view["target"].iloc[0] == view["value"].iloc[1]


def test_wastewater_evaluator_reports_five_state_rolling_metrics():
    dates = pd.date_range("2020-01-04", periods=160, freq="7D")
    rows = [{"week_end": date, "pathogen_target": "Influenza A virus",
             "site": "a", "site_wval": float((index % 10) + 1)}
            for index, date in enumerate(dates)]
    result = evaluate_wastewater_state_rolling(
        build_wastewater_weekly_view(pd.DataFrame(rows), pathogen="influenza"),
        min_train_rows=52, validation_rows=13, test_rows=13, step_rows=13)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert result["state_count"] == 5
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
