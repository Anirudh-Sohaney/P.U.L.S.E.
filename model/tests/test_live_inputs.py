import io
import json

import pandas as pd
import pytest

from arkansas_pharma_signal import live_inputs
from arkansas_pharma_signal.cli import _merge_live_sdud_target


class _Response:
    def __init__(self, payload):
        body = payload if isinstance(payload, str) else json.dumps(payload)
        self.payload = io.BytesIO(body.encode())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return self.payload.read(*args)


def test_openfda_shortage_adapter_paginates_and_preserves_freshness(monkeypatch):
    calls = []
    payloads = [
        {"meta": {"last_updated": "2026-08-14", "results": {"total": 2}},
         "results": [{"change_date": "08/07/2026", "generic_name": "Drug A",
                       "company_name": "Supplier A", "status": "Current",
                       "openfda": {}}]},
        {"meta": {"last_updated": "2026-08-14", "results": {"total": 2}},
         "results": [{"initial_posting_date": "01/02/2025", "generic_name": "Drug B",
                       "company_name": "Supplier B", "status": "Resolved",
                       "openfda": {}}]},
    ]

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response(payloads[len(calls) - 1])

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_openfda_shortages(page_size=1)
    assert len(calls) == 2
    assert list(frame["drug"]) == ["Drug A", "Drug B"]
    assert list(frame["month"]) == ["2026-08", "2025-01"]
    assert metadata["last_updated"] == "2026-08-14"


def test_openfda_enforcement_adapter_normalizes_active_status(monkeypatch):
    payload = {
        "meta": {"last_updated": "2026-08-14", "results": {"total": 2}},
        "results": [
            {"recall_number": "R1", "recalling_firm": "Supplier A",
             "product_description": "Drug A", "classification": "Class II",
             "status": "Ongoing", "recall_initiation_date": "08/07/2026"},
            {"recall_number": "R2", "recalling_firm": "Supplier B",
             "product_description": "Drug B", "classification": "Class III",
             "status": "Terminated", "recall_initiation_date": "01/02/2025",
             "termination_date": "02/01/2025"},
        ],
    }

    def fake_urlopen(request, timeout):
        assert "drug/enforcement.json" in request.full_url
        return _Response(payload)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_openfda_enforcement()
    assert list(frame["recall_active"]) == [1, 0]
    assert list(frame["supplier"]) == ["Supplier A", "Supplier B"]
    assert metadata["last_updated"] == "2026-08-14"


def test_fluview_weekly_adapter_normalizes_regions_and_preserves_issue(monkeypatch):
    payload = {
        "result": 1,
        "epidata": [{
            "release_date": "2026-08-14", "region": "AR", "issue": 202633,
            "epiweek": 202632, "lag": 1, "num_ili": 81,
            "num_patients": 1000, "num_providers": 4, "wili": 8.1, "ili": 8.1,
        }, {
            "release_date": "2026-08-14", "region": "nat", "issue": 202633,
            "epiweek": 202632, "lag": 1, "num_ili": 100,
            "num_patients": 2000, "num_providers": 8, "wili": 5.0, "ili": 5.0,
        }],
    }

    def fake_urlopen(request, timeout):
        assert "source=fluview" in request.full_url
        assert "regions=ar%2Cnat" in request.full_url
        assert "epiweeks=202601-202633" in request.full_url
        return _Response(payload)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_fluview_weekly(
        regions=("AR", "nat", "AR"), epiweeks="202601-202633", timeout=8)
    assert list(frame["region"]) == ["ar", "nat"]
    assert frame.iloc[0]["issue"] == 202633
    assert frame.iloc[0]["epiweek"] == 202632
    assert frame.iloc[0]["wili"] == 8.1
    assert metadata["cadence"] == "weekly"
    assert metadata["rows"] == 2


def test_nndss_weekly_adapter_preserves_monthly_fields_and_paginates(monkeypatch):
    payloads = [[{
        "states": "Arkansas", "year": "2026", "week": "32",
        "label": "Influenza", "m1": "12", "m1_flag": "",
        "m2": "-", "m2_flag": "N", "m3": "", "m3_flag": "-",
        "m4": "", "m4_flag": "-", "geocode": "05",
    }], [{
        "states": "US RESIDENTS", "year": "2026", "week": "32",
        "label": "Influenza", "m1": "100", "m1_flag": "",
        "m2": "", "m2_flag": "-", "m3": "", "m3_flag": "-",
        "m4": "", "m4_flag": "-", "geocode": "US",
    }], []]
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response(payloads.pop(0))

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_nndss_weekly(limit=1, timeout=11)
    assert len(calls) == 3
    assert "%24offset=1" in calls[1]
    assert list(frame["state"]) == ["Arkansas", "US RESIDENTS"]
    assert frame.iloc[0]["epiweek"] == 202632
    assert frame.iloc[0]["m1"] == 12
    assert frame.iloc[0]["m2_flag"] == "N"
    assert metadata["monthly_fields_not_weekly_total"] is True
    assert metadata["rows"] == 2


def test_noaa_daily_adapter_normalizes_observed_station_fields(monkeypatch):
    payload = [{
        "DATE": "2026-08-16", "STATION": "USW00013971",
        "NAME": "HARRISON, AR US", "LATITUDE": "36.27", "LONGITUDE": "-93.15",
        "TMAX": "100", "TMIN": "70", "PRCP": "0.25", "SNOW": "0",
        "SNWD": "0", "AWND": "8.2", "RHAV": "62",
    }]

    def fake_urlopen(request, timeout):
        assert "dataset=daily-summaries" in request.full_url
        assert "stations=USW00013971" in request.full_url
        assert "startDate=2026-08-16" in request.full_url
        assert timeout == 13
        return _Response(payload)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_noaa_daily_summaries(
        stations=("usw00013971", "USW00013971"), start_date="2026-08-16",
        end_date="2026-08-16", timeout=13)
    assert frame.iloc[0]["station_id"] == "USW00013971"
    assert frame.iloc[0]["temperature_mean"] == 85
    assert frame.iloc[0]["extreme_heat_day"] == 1
    assert frame.iloc[0]["humidity"] == 62
    assert metadata["observed_only"] is True
    assert metadata["rows"] == 1


def test_nadac_adapter_requires_dated_snapshot_and_filters_ndcs(monkeypatch):
    csv = ("NDC Description,NDC,NADAC Per Unit,Effective Date,Pricing Unit,"
           "Pharmacy Type Indicator,OTC,Explanation Code,Classification for Rate Setting,"
           "Corresponding Generic Drug NADAC Per Unit,Corresponding Generic Drug Effective Date,As of Date\n"
           "DRUG A,123-45-6789,1.25,12/18/2025,EA,C/I,N,1,G,,,01/07/2026\n"
           "DRUG B,99999999999,2.00,12/18/2025,ML,C/I,Y,2,G,,,01/07/2026\n")

    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("nadac-2026-01-07.csv")
        assert timeout == 14
        return _Response(csv)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_nadac_weekly_csv(
        as_of_date="2026-01-07", url_template="https://example.test/nadac-{date}.csv",
        ndc_filter=("123456789",), timeout=14)
    assert list(frame["ndc"]) == ["123456789"]
    assert frame.iloc[0]["nadac_per_unit"] == 1.25
    assert frame.iloc[0]["as_of_date"] == "2026-01-07"
    assert metadata["ndc_filter_applied"] is True
    assert metadata["target_semantics"].startswith("national acquisition-cost")


def test_fema_adapter_preserves_declaration_and_incident_dates(monkeypatch):
    page = {"DisasterDeclarationsSummaries": [{
        "disasterNumber": 1234, "state": "AR", "declarationType": "DR",
        "declarationDate": "2026-08-10T00:00:00.000Z",
        "incidentBeginDate": "2026-08-08T00:00:00.000Z",
        "incidentEndDate": "2026-08-20T00:00:00.000Z",
        "declarationTitle": "Example Flood", "incidentType": "Flood",
        "designatedArea": "Example County", "fipsStateCode": "05",
        "fipsCountyCode": "001",
    }]}
    payloads = [page, {"DisasterDeclarationsSummaries": []}]

    def fake_urlopen(request, timeout):
        assert "state+eq+%27AR%27" in request.full_url
        assert "%24top=1" in request.full_url
        assert timeout == 15
        return _Response(payloads.pop(0))

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_fema_arkansas_disasters(
        start_date="2026-01-01", as_of_date="2026-08-15", page_size=1, timeout=15)
    assert frame.iloc[0]["county_fips"] == "05001"
    assert frame.iloc[0]["declaration_date"] == "2026-08-10"
    assert frame.iloc[0]["incident_begin_date"] == "2026-08-08"
    assert frame.iloc[0]["disaster_active"] == 1
    assert frame.iloc[0]["disaster_severity"] == 1.0
    assert metadata["declaration_is_not_incident_onset"] is True


def test_nws_arkansas_alert_adapter_preserves_context_and_freshness(monkeypatch):
    payload = {
        "features": [{
            "id": "https://api.weather.gov/alerts/1",
            "properties": {
                "event": "Flood Warning", "headline": "Flood warning",
                "severity": "Severe", "urgency": "Immediate",
                "certainty": "Observed", "status": "Actual",
                "onset": "2026-08-15T10:00:00+00:00",
                "expires": "2026-08-15T12:00:00+00:00",
                "areaDesc": "Example County",
                "geocode": {"UGC": ["ARC001"]},
                "senderName": "NWS Little Rock",
            },
        }]
    }

    def fake_urlopen(request, timeout):
        assert "api.weather.gov" in request.full_url
        assert timeout == 7
        return _Response(payload)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_nws_arkansas_alerts(timeout=7)
    assert len(frame) == 1
    assert frame.iloc[0]["event"] == "Flood Warning"
    assert frame.iloc[0]["ugc_zones"] == "ARC001"
    assert metadata["state"] == "AR"
    assert metadata["target_semantics"].endswith("context only")


def test_nws_hourly_forecast_adapter_preserves_issue_and_validity_times(monkeypatch):
    payloads = [
        {"properties": {"forecastHourly": "https://api.weather.gov/gridpoints/LZK/1,2/forecast/hourly"}},
        {"properties": {
            "updated": "2026-08-16T00:00:00+00:00",
            "periods": [{
                "startTime": "2026-08-16T01:00:00-05:00",
                "endTime": "2026-08-16T02:00:00-05:00",
                "temperature": 82, "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"value": 40},
                "relativeHumidity": {"value": 70},
                "windSpeed": "10 mph", "windDirection": "S",
                "shortForecast": "Chance Rain Showers", "isDaytime": False,
            }],
        }},
    ]
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout, request.headers.get("User-agent")))
        return _Response(payloads[len(calls) - 1])

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_nws_hourly_forecast(34.7, -92.3, timeout=9)
    assert len(calls) == 2
    assert calls[0][0].endswith("34.7,-92.3")
    assert calls[0][1] == 9
    assert frame.iloc[0]["temperature"] == 82
    assert frame.iloc[0]["precipitation_probability"] == 40
    assert frame.iloc[0]["wind_speed_mph"] == 10
    assert frame.iloc[0]["valid_time"].tzinfo is not None
    assert metadata["issued_at"].startswith("2026-08-16")
    assert metadata["cadence"] == "hourly"
    assert metadata["target_semantics"].endswith("inventory truth")


def test_nws_hourly_forecast_can_carry_explicit_county_tag(monkeypatch):
    payloads = [
        {"properties": {"forecastHourly": "https://example.test/hourly"}},
        {"properties": {"periods": [{
            "startTime": "2026-08-16T01:00:00Z", "endTime": "2026-08-16T02:00:00Z",
            "temperature": 80, "temperatureUnit": "F",
            "probabilityOfPrecipitation": {"value": 10},
            "relativeHumidity": {"value": 60}, "windSpeed": "5 mph",
        }]}},
    ]
    monkeypatch.setattr(live_inputs, "urlopen", lambda request, timeout:
                        _Response(payloads.pop(0)))
    frame, metadata = live_inputs.fetch_nws_hourly_forecast(
        34.7, -92.3, county_fips="5125")
    assert frame.iloc[0]["county_fips"] == "05125"
    assert metadata["county_fips"] == "05125"


def test_nws_hourly_forecast_rejects_non_arkansas_county_tag():
    with pytest.raises(ValueError, match="Arkansas FIPS"):
        live_inputs.fetch_nws_hourly_forecast(34.7, -92.3, county_fips="06037")


def test_nws_hourly_features_aggregate_only_observed_periods():
    hourly = pd.DataFrame([
        {"valid_time": "2026-08-16T01:00:00Z", "latitude": 34.7,
         "longitude": -92.3, "temperature": 80,
         "precipitation_probability": 20, "relative_humidity": 60,
         "wind_speed_mph": 8},
        {"valid_time": "2026-08-16T04:00:00Z", "latitude": 34.7,
         "longitude": -92.3, "temperature": 86,
         "precipitation_probability": 70, "relative_humidity": 80,
         "wind_speed_mph": 12},
    ])
    result = live_inputs.build_nws_daily_weather_features(hourly)
    assert len(result) == 1
    assert result.iloc[0]["weather_hour_count"] == 2
    assert result.iloc[0]["weather_temperature_mean"] == 83
    assert result.iloc[0]["weather_precipitation_probability_max"] == 70
    assert result.iloc[0]["county_fips"] == ""


def test_nws_county_refresh_preserves_explicit_registry_keys(monkeypatch):
    registry = pd.DataFrame([
        {"county_fips": "05001", "latitude": 34.28, "longitude": -91.37},
        {"county_fips": "05003", "latitude": 33.19, "longitude": -91.77},
    ])

    def fake_fetch(latitude, longitude, *, county_fips, timeout):
        return pd.DataFrame([{
            "valid_time": "2026-08-16T01:00:00Z", "latitude": latitude,
            "longitude": longitude, "county_fips": county_fips,
            "temperature": 80, "precipitation_probability": 10,
            "relative_humidity": 60, "wind_speed_mph": 5,
        }]), {"county_fips": county_fips}

    monkeypatch.setattr(live_inputs, "fetch_nws_hourly_forecast", fake_fetch)
    result, metadata = live_inputs.fetch_nws_county_hourly_forecasts(registry)
    assert sorted(result["county_fips"].unique()) == ["05001", "05003"]
    assert metadata["county_count_requested"] == 2
    assert metadata["county_count_returned"] == 2


def test_medicaid_sdud_adapter_filters_state_and_suppression(monkeypatch):
    csv = ("State,NDC,Year,Quarter,Suppression Used,Product Name,"
           "Number of Prescriptions\n"
           "AR,00000000001,2025,1,false,DRUG A,12\n"
           "AR,00000000002,2025,1,true,DRUG B,99\n"
           "TX,00000000003,2025,1,false,DRUG C,40\n")

    def fake_urlopen(request, timeout):
        return _Response(csv)

    monkeypatch.setattr(live_inputs, "urlopen", fake_urlopen)
    frame, metadata = live_inputs.fetch_medicaid_sdud_csv(url="https://example.test/sdud.csv")
    assert frame[["year", "quarter", "drug", "value"]].to_dict("records") == [
        {"year": 2025, "quarter": 1, "drug": "DRUG A", "value": 12.0}
    ]
    assert metadata["suppressed_rows_excluded"] is True
    assert metadata["state"] == "AR"


def test_multi_year_sdud_refresh_aggregates_and_preserves_sources(monkeypatch):
    def fake_fetch(*, url):
        year = 2023 if "2023" in url else 2024
        frame = pd.DataFrame([{
            "year": year, "quarter": 1, "drug": "DRUG A", "value": 10.0,
            "ingredient": "DRUG A", "ndc": "001", "rxnorm_rxcui": "",
            "manufacturer": "", "mapping_confidence": "test",
        }])
        return frame, {"source_url": url, "retrieved_on": "2026-08-15", "rows": 1}

    monkeypatch.setattr("arkansas_pharma_signal.cli.fetch_medicaid_sdud_csv", fake_fetch)
    base = pd.DataFrame([{
        "year": 2023, "quarter": 1, "drug": "DRUG A", "value": 5.0,
        "ingredient": "DRUG A", "ndc": "001", "rxnorm_rxcui": "",
        "manufacturer": "", "mapping_confidence": "base",
    }])
    frame, metadata = _merge_live_sdud_target(
        base, ["https://example.test/sdud2023.csv",
               "https://example.test/sdud2024.csv"])
    assert frame.loc[(frame["year"] == 2023) & (frame["quarter"] == 1), "value"].iloc[0] == 15.0
    assert frame.loc[frame["year"] == 2024, "value"].iloc[0] == 10.0
    assert metadata["source_urls"] == [
        "https://example.test/sdud2023.csv",
        "https://example.test/sdud2024.csv",
    ]
    assert metadata["panel_period_end"] == "2024-Q1"
    assert metadata["panel_period_end_date"] == "2024-03-31"
    assert metadata["panel_age_days"] >= 0
    assert metadata["next_forecast_period"] == "2024-Q2"
