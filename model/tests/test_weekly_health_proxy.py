import numpy as np
import pandas as pd

from arkansas_pharma_signal.weekly_health_proxy import (
    evaluate_fluview_divergence_state, evaluate_weekly_health_proxy,
    load_fluview_weekly_proxy,
)


def test_fluview_divergence_screen_is_five_state_and_filter_safe():
    dates = pd.date_range("2018-01-01", periods=15, freq="7D")
    rows = []
    for index, date in enumerate(dates):
        for region, value in (("ar", float(index)), ("nat", float(index + 1))):
            rows.append({"region": region, "week_start": date,
                         "current_value": value, "target": value})
    result = evaluate_fluview_divergence_state(
        pd.DataFrame(rows), folds=((2018, 2018, 2018),))
    assert result["state_count"] == 5
    assert result["fold_count"] == 1
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}


def test_fluview_proxy_keeps_latest_release_and_strict_next_week(tmp_path):
    dates = pd.date_range("2019-01-07", periods=20, freq="7D")
    rows = []
    for i, date in enumerate(dates):
        iso = date.isocalendar()
        rows.append({"region": "ar", "epiweek": iso.year * 100 + iso.week,
                     "issue": iso.year * 100 + iso.week + 1,
                     "wili": 1.0 + i * 0.01})
    duplicate = rows[3].copy()
    duplicate["issue"] += 10
    duplicate["wili"] = 9.0
    rows.append(duplicate)
    path = tmp_path / "flu.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    view = load_fluview_weekly_proxy(path)
    assert len(view) == 19
    assert np.isclose(view.iloc[3]["current_value"], 9.0)
    assert (view["week_start"].diff().dropna() == pd.Timedelta(days=7)).all()


def test_weekly_proxy_rolling_is_chronological():
    rows = []
    dates = pd.date_range("2018-01-01", "2021-12-27", freq="7D")
    for i, date in enumerate(dates):
        rows.append({"region": "ar", "year": date.isocalendar().year,
                     "week": date.isocalendar().week,
                     "week_start": date, "target": 1.0 + i * 0.001,
                     **{c: 1.0 + i * 0.001 for c in (
                         "current_value", "lag1_value", "lag2_value",
                         "lag4_value", "rolling4_value")},
                     "week_sin": 0.0, "week_cos": 1.0})
    result = evaluate_weekly_health_proxy(
        pd.DataFrame(rows), folds=((2018, 2019, 2020),))
    assert result["fold_count"] == 1
    assert result["folds"][0]["test_rows"] > 0
