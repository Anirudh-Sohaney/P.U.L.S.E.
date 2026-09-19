import pandas as pd

from arkansas_pharma_signal.nssp_respiratory import (
    build_nssp_weekly_view, evaluate_nssp_state_rolling,
)


def _source():
    dates = pd.date_range("2020-01-04", periods=90, freq="7D")
    return pd.DataFrame({
        "week_end": dates,
        "county": "All",
        "percent_visits_covid": range(90),
        "percent_visits_influenza": range(90),
        "percent_visits_rsv": range(90),
    })


def test_nssp_view_is_statewide_and_pathogen_specific():
    result = build_nssp_weekly_view(_source(), pathogen="influenza")
    assert len(result) == 89
    assert result["target"].notna().all()


def test_nssp_evaluator_reports_five_state_event_metrics():
    result = evaluate_nssp_state_rolling(_source(), pathogen="influenza")
    assert result["state_count"] == 5
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
