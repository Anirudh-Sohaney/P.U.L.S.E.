import pandas as pd

from arkansas_pharma_signal import cli


def test_refresh_weather_cli_writes_raw_daily_and_provenance(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    hourly = pd.DataFrame([
        {"valid_time": "2026-08-16T01:00:00Z", "latitude": 34.7,
         "longitude": -92.3, "temperature": 80,
         "precipitation_probability": 20, "relative_humidity": 60,
         "wind_speed_mph": 8, "county_fips": "05125"},
    ])
    monkeypatch.setattr(
        cli, "fetch_nws_hourly_forecast",
        lambda latitude, longitude, county_fips, timeout: (hourly, {
            "source": "test", "source_url": "https://example.test/hourly",
            "points_url": "https://example.test/points", "rows": 1,
            "cadence": "hourly", "target_semantics": "weather context only",
        }),
    )
    result = cli.main([
        "--root", str(tmp_path), "refresh-weather",
        "--latitude", "34.7", "--longitude", "-92.3",
        "--county-fips", "05125",
    ])
    assert result == 0
    raw = tmp_path / "model" / "artifacts" / "live" / "nws_weather_hourly.csv"
    daily = tmp_path / "model" / "artifacts" / "live" / "nws_weather_daily.csv"
    metadata = tmp_path / "model" / "artifacts" / "metadata" / "nws_weather_refresh.json"
    assert raw.exists() and daily.exists() and metadata.exists()
    assert pd.read_csv(daily).iloc[0]["weather_hour_count"] == 1
    assert str(pd.read_csv(daily).iloc[0]["county_fips"]).zfill(5) == "05125"
    assert '"historical_training_modified": false' in metadata.read_text()
    assert '"geography_contract": "caller_supplied_county_fips"' in metadata.read_text()


def test_build_weather_geography_cli_writes_registry(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    points = pd.DataFrame([{
        "county_fips": "05001", "county_name": "Arkansas County",
        "arkansas_region": "southeast", "latitude": 34.2,
        "longitude": -91.3, "source": "Census", "source_url": "https://example.test",
        "source_vintage": "2025", "transformation_method": "test",
    }])
    monkeypatch.setattr(
        cli, "fetch_census_county_weather_points",
        lambda year, timeout: (points, {"source": "test", "rows": 1}),
    )
    output = tmp_path / "county_points.csv"
    result = cli.main([
        "--root", str(tmp_path), "build-weather-geography",
        "--output", str(output),
    ])
    assert result == 0
    assert output.exists()
    assert pd.read_csv(output).iloc[0]["county_fips"] == 5001
