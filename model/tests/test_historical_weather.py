import json

import pandas as pd
import pytest

from arkansas_pharma_signal.historical_weather import (
    attach_weather_context, build_county_weather_panel, load_noaa_daily_summaries,
)


def _row(day, station="USC0001", tmax="80", tmin="60", prcp="0.10"):
    return {
        "DATE": day, "STATION": station, "NAME": "TEST, AR US",
        "LATITUDE": "34.7", "LONGITUDE": "-92.2", "TMAX": tmax,
        "TMIN": tmin, "PRCP": prcp, "SNOW": "0.0", "SNWD": "0.0",
    }


def test_noaa_loader_deduplicates_station_date_and_normalizes_values(tmp_path):
    path = tmp_path / "noaa_daily_summaries_test.json"
    path.write_text(json.dumps([_row("2024-01-01"), _row("2024-01-01", tmax="82"),
                                _row("2024-01-02", prcp="bad")]))
    result = load_noaa_daily_summaries(tmp_path)
    assert len(result) == 2
    assert result.iloc[0]["temperature_max"] == 82
    assert result.iloc[0]["temperature_mean"] == 71
    assert pd.isna(result.iloc[1]["precipitation"])


def test_county_weather_panel_preserves_missingness_and_grain():
    daily = pd.DataFrame([{
        "date": "2024-01-01", "station_id": "USC0001", "station_name": "TEST, AR US",
        "latitude": 34.7, "longitude": -92.2, "temperature_mean": 70,
        "temperature_min": 60, "temperature_max": 80, "precipitation": 0.1,
        "snowfall": 0, "snow_depth": 0,
    }, {
        "date": "2024-01-03", "station_id": "USC0001", "station_name": "TEST, AR US",
        "latitude": 34.7, "longitude": -92.2, "temperature_mean": 50,
        "temperature_min": 40, "temperature_max": 60, "precipitation": 0.2,
        "snowfall": 0, "snow_depth": 0,
    }])
    points = pd.DataFrame([{
        "county_fips": "05001", "county_name": "Arkansas County",
        "arkansas_region": "southeast", "latitude": 34.8, "longitude": -92.1,
    }])
    result, metadata = build_county_weather_panel(daily, points)
    assert set(result["cadence"]) == {"weekly", "monthly"}
    assert len(result) == 2
    assert result["county_fips"].eq("05001").all()
    weekly = result[result["cadence"].eq("weekly")].iloc[0]
    assert weekly["weather_observed_days"] == 2
    assert weekly["weather_expected_days"] == 7
    assert weekly["weather_coverage_ratio"] == 2 / 7
    assert weekly["precipitation"] == pytest.approx(0.3)
    assert metadata["imputation"] == "none"


def test_weather_context_requires_exact_county_period_match():
    target = pd.DataFrame([{
        "county_fips": "05001", "period_start": "2024-01-01",
        "period_end": "2024-01-07", "cadence": "weekly", "drug": "A",
    }, {
        "county_fips": "05003", "period_start": "2024-01-01",
        "period_end": "2024-01-07", "cadence": "weekly", "drug": "A",
    }])
    weather = pd.DataFrame([{
        "county_fips": "05001", "period_start": "2024-01-01",
        "period_end": "2024-01-07", "cadence": "weekly",
        "temperature_mean": 70, "station_id": "USC0001",
        "station_distance_km": 4.0,
    }])
    result = attach_weather_context(target, weather)
    assert result.loc[0, "weather_temperature_mean"] == 70
    assert result.loc[0, "weather_context_available"] == 1
    assert pd.isna(result.loc[1, "weather_temperature_mean"])
    assert result.loc[1, "weather_context_available"] == 0
