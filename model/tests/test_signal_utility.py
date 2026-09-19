import pandas as pd

from arkansas_pharma_signal.signal_utility import (
    build_fluview_hospital_view, build_fluview_wastewater_view,
    build_fluview_weather_view, build_hospital_weather_view,
    build_hospital_disaster_view,
    evaluate_hospital_incremental_utility, evaluate_hospital_weather_incremental_utility,
    evaluate_hospital_disaster_incremental_utility,
    evaluate_wastewater_incremental_utility, evaluate_weather_incremental_utility,
)


def test_signal_utility_aligns_only_same_week_and_reports_ablation():
    dates = pd.date_range("2022-01-03", periods=160, freq="7D")
    flu = pd.DataFrame([{
        "region": "ar", "epiweek": d.isocalendar().year * 100 + d.isocalendar().week,
        "issue": d.isocalendar().year * 100 + d.isocalendar().week + 1,
        "wili": 1.0 + (i % 9) * 0.1,
    } for i, d in enumerate(dates)])
    wastewater = pd.DataFrame([{
        "week_end": d + pd.Timedelta(days=5), "pathogen_target": "Influenza A virus",
        "site": "x", "site_wval": float(i % 4),
    } for i, d in enumerate(dates)])
    view = build_fluview_wastewater_view(flu, wastewater)
    result = evaluate_wastewater_incremental_utility(view)
    assert len(view) > 25
    assert result["fold_count"] >= 1
    assert "mean_balanced_accuracy_delta" in result
    assert result["publishable_candidate"] is False


def test_hospital_utility_aligns_weekly_rows_and_reports_ablation():
    dates = pd.date_range("2020-01-06", periods=300, freq="7D")
    flu = pd.DataFrame([{
        "region": "ar", "epiweek": d.isocalendar().year * 100 + d.isocalendar().week,
        "issue": d.isocalendar().year * 100 + d.isocalendar().week + 1,
        "wili": 1.0 + (i % 9) * 0.1,
    } for i, d in enumerate(dates)])
    hospital = pd.DataFrame([{
        "weekendingdate": d + pd.Timedelta(days=5), "jurisdiction": "AR",
        "totalconfflunewadm": float(i % 12),
    } for i, d in enumerate(dates)])
    view = build_fluview_hospital_view(flu, hospital)
    result = evaluate_hospital_incremental_utility(view)
    assert len(view) > 25
    assert result["fold_count"] >= 1
    assert "mean_balanced_accuracy_delta" in result
    assert result["publishable_candidate"] is False


def test_weather_utility_aligns_exact_week_and_reports_ablation():
    dates = pd.date_range("2020-01-06", periods=300, freq="7D")
    flu = pd.DataFrame([{
        "region": "ar", "epiweek": d.isocalendar().year * 100 + d.isocalendar().week,
        "issue": d.isocalendar().year * 100 + d.isocalendar().week + 1,
        "wili": 1.0 + (i % 9) * 0.1,
    } for i, d in enumerate(dates)])
    weather = pd.DataFrame([{
        "cadence": "weekly", "period_start": d, "county_fips": "05001",
        "weather_temperature_mean": 50 + i % 10,
        "weather_temperature_min": 40 + i % 10,
        "weather_temperature_max": 60 + i % 10,
        "weather_precipitation": float(i % 3), "weather_snowfall": 0.0,
        "weather_coverage_ratio": 1.0,
    } for i, d in enumerate(dates)])
    view = build_fluview_weather_view(flu, weather)
    result = evaluate_weather_incremental_utility(view)
    assert len(view) > 25
    assert result["fold_count"] >= 1
    assert "mean_balanced_accuracy_delta" in result
    assert result["publishable_candidate"] is False


def test_hospital_weather_utility_aligns_week_end_minus_five_days():
    dates = pd.date_range("2020-01-06", periods=300, freq="7D")
    hospital = pd.DataFrame([{
        "weekendingdate": d + pd.Timedelta(days=5), "jurisdiction": "AR",
        "totalconfc19newadm": 1.0, "totalconfflunewadm": 2.0 + i % 9,
        "totalconfrsvnewadm": 1.0,
    } for i, d in enumerate(dates)])
    weather = pd.DataFrame([{
        "cadence": "weekly", "period_start": d, "county_fips": "05001",
        "weather_temperature_mean": 50 + i % 10,
        "weather_temperature_min": 40 + i % 10,
        "weather_temperature_max": 60 + i % 10,
        "weather_precipitation": float(i % 3), "weather_snowfall": 0.0,
        "weather_coverage_ratio": 1.0,
    } for i, d in enumerate(dates)])
    view = build_hospital_weather_view(hospital, weather)
    result = evaluate_hospital_weather_incremental_utility(view)
    assert len(view) > 25
    assert result["fold_count"] >= 1
    assert "mean_balanced_accuracy_delta" in result


def test_hospital_disaster_utility_aligns_declaration_week():
    dates = pd.date_range("2020-01-06", periods=300, freq="7D")
    hospital = pd.DataFrame([{
        "weekendingdate": d + pd.Timedelta(days=5), "jurisdiction": "AR",
        "totalconfc19newadm": 1.0, "totalconfflunewadm": 2.0 + i % 9,
        "totalconfrsvnewadm": 1.0,
    } for i, d in enumerate(dates)])
    disasters = pd.DataFrame([{
        "variable_id": "disaster_active", "geography_id": "05001",
        "observation_time": d, "value": 1.0,
    } for d in dates[::20]])
    disasters = pd.concat([disasters, pd.DataFrame([{
        "variable_id": "disaster_severity", "geography_id": "05001",
        "observation_time": dates[0], "value": 1.0,
    }])], ignore_index=True)
    view = build_hospital_disaster_view(hospital, disasters)
    result = evaluate_hospital_disaster_incremental_utility(view)
    assert len(view) > 25
    assert result["fold_count"] >= 1
    assert "mean_balanced_accuracy_delta" in result
