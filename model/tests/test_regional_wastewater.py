import pandas as pd

from arkansas_pharma_signal.regional_wastewater import (
    build_regional_wastewater_change_view,
    build_regional_wastewater_weekly_view,
    evaluate_regional_wastewater_change_state_rolling,
    evaluate_regional_wastewater_state_rolling,
)


def _source(rows=160):
    dates = pd.date_range("2022-01-01", periods=rows, freq="7D")
    data = []
    for index, date in enumerate(dates):
        for county, offset in (("Pulaski", 0.0), ("Benton", 0.5)):
            data.append({
                "week_end": date, "pathogen_target": "Influenza A virus",
                "site": county, "counties_served": county,
                "population_served": 1000, "site_wval": (index % 10) + offset,
            })
    return pd.DataFrame(data)


def test_regional_view_preserves_regions_and_weighted_values():
    source = _source(12)
    view = build_regional_wastewater_weekly_view(source, pathogen="Influenza A virus")
    assert set(view["region"]) == {"central", "northwest"}
    assert view["target"].notna().all()
    assert view["site_count"].eq(1).all()
    assert view["week"].is_monotonic_increasing is False


def test_regional_evaluator_reports_five_state_chronological_metrics():
    view = build_regional_wastewater_weekly_view(_source(), pathogen="Influenza A virus")
    result = evaluate_regional_wastewater_state_rolling(
        view, min_train_rows=52, validation_rows=13, test_rows=13, step_rows=13)
    assert result["fold_count"] >= 6
    assert result["test_rows"] >= 25
    assert result["state_count"] == 5
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]


def test_regional_change_evaluator_uses_zero_change_persistence():
    view = build_regional_wastewater_weekly_view(_source(), pathogen="Influenza A virus")
    result = evaluate_regional_wastewater_change_state_rolling(
        build_regional_wastewater_change_view(view),
        min_train_rows=52, validation_rows=13, test_rows=13, step_rows=13)
    assert result["fold_count"] >= 6
    assert result["state_definition"]["2"] == "stable"
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
