"""Small, auditable live-input adapters for operational forecasts.

The historical S_D files remain the training/evaluation source.  These
adapters are intentionally opt-in at the CLI boundary so a backtest cannot
silently change when the public endpoint changes.
"""

from __future__ import annotations

import io
import json
import re
from datetime import date, datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


OPENFDA_SHORTAGES_URL = "https://api.fda.gov/drug/shortages.json"
OPENFDA_ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
NWS_ARKANSAS_ALERTS_URL = "https://api.weather.gov/alerts/active?area=AR"
NWS_POINTS_URL = "https://api.weather.gov/points/{latitude},{longitude}"
MEDICAID_SDUD_2025_URL = (
    "https://download.medicaid.gov/data/sdud2025_updatedjuly2026.csv"
)
FLUVIEW_URL = "https://delphi.cmu.edu/epidata/api.php"
NNDSS_WEEKLY_URL = "https://data.cdc.gov/resource/x9gk-5huc.json"
NOAA_DAILY_SUMMARIES_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NADAC_DOWNLOAD_TEMPLATE = (
    "https://download.medicaid.gov/data/"
    "nadac-national-average-drug-acquisition-cost-{date}.csv"
)
FEMA_DISASTER_DECLARATIONS_URL = (
    "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
)


def _nws_json(endpoint: str, *, timeout: int) -> dict:
    """Fetch one NWS JSON document with the required identification header."""
    request = Request(
        endpoint,
        headers={
            "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
            "Accept": "application/geo+json",
        },
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310: HTTPS endpoint
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("NWS response must be a JSON object")
    return payload


def _wind_mph(value: object) -> float | None:
    """Extract the first numeric wind-speed value from NWS text."""
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value or ""))
    return float(match.group(0)) if match else None


def fetch_nws_hourly_forecast(
    latitude: float, longitude: float, *,
    county_fips: str | None = None,
    points_endpoint: str | None = None,
    forecast_endpoint: str | None = None,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch an official hourly NWS forecast for one geographic point.

    The point lookup and forecast response are retained as provenance rather
    than converted into a target. Forecast validity and issuance timestamps
    make the adapter safe to join only to observations available at the
    forecast creation time.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    if county_fips is not None:
        county_fips = str(county_fips).strip().zfill(5)
        if not re.fullmatch(r"05\d{3}", county_fips):
            raise ValueError("county_fips must be a five-digit Arkansas FIPS code")
    points_url = points_endpoint or NWS_POINTS_URL.format(
        latitude=latitude, longitude=longitude)
    point_payload = _nws_json(points_url, timeout=timeout)
    point_properties = point_payload.get("properties", {}) or {}
    endpoint = forecast_endpoint or point_properties.get("forecastHourly")
    if not endpoint:
        raise ValueError("NWS points response has no hourly forecast endpoint")
    forecast_payload = _nws_json(str(endpoint), timeout=timeout)
    properties = forecast_payload.get("properties", {}) or {}
    periods = properties.get("periods", [])
    if not isinstance(periods, list):
        raise ValueError("NWS hourly forecast periods must be a list")
    rows: list[dict] = []
    for period in periods:
        if not isinstance(period, dict) or not period.get("startTime"):
            continue
        precipitation = (period.get("probabilityOfPrecipitation") or {}).get("value")
        humidity = (period.get("relativeHumidity") or {}).get("value")
        rows.append({
            "valid_time": pd.to_datetime(period.get("startTime"), utc=True,
                                          errors="coerce"),
            "valid_time_end": pd.to_datetime(period.get("endTime"), utc=True,
                                              errors="coerce"),
            "temperature": pd.to_numeric(period.get("temperature"), errors="coerce"),
            "temperature_unit": period.get("temperatureUnit", ""),
            "precipitation_probability": pd.to_numeric(precipitation, errors="coerce"),
            "relative_humidity": pd.to_numeric(humidity, errors="coerce"),
            "wind_speed_mph": _wind_mph(period.get("windSpeed")),
            "wind_direction": period.get("windDirection", ""),
            "short_forecast": period.get("shortForecast", ""),
            "is_daytime": period.get("isDaytime"),
            "county_fips": county_fips or "",
            "latitude": latitude, "longitude": longitude,
            "source": "National Weather Service hourly forecast",
        })
    frame = pd.DataFrame(rows, columns=[
        "valid_time", "valid_time_end", "temperature", "temperature_unit",
        "precipitation_probability", "relative_humidity", "wind_speed_mph",
        "wind_direction", "short_forecast", "is_daytime", "latitude", "longitude",
        "county_fips", "source",
    ])
    retrieved = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source": "National Weather Service hourly forecast",
        "source_url": str(endpoint), "points_url": points_url,
        "retrieved_at_utc": retrieved, "latitude": latitude, "longitude": longitude,
        "county_fips": county_fips or "",
        "rows": int(len(frame)), "cadence": "hourly",
        "issued_at": properties.get("updated", ""),
        "forecast_valid_from": (str(frame["valid_time"].min()) if not frame.empty else ""),
        "forecast_valid_to": (str(frame["valid_time_end"].max()) if not frame.empty else ""),
        "target_semantics": "weather context only; not pharmacy demand or inventory truth",
    }
    return frame, metadata


def build_nws_daily_weather_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly NWS periods to point/date features for model joins.

    The function keeps the geographic point and UTC calendar date in the key.
    It does not fill missing periods or infer conditions outside the forecast
    horizon, which prevents a partial forecast from becoming a complete day.
    """
    required = {
        "valid_time", "latitude", "longitude", "temperature",
        "precipitation_probability", "relative_humidity", "wind_speed_mph",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"NWS hourly frame missing columns: {sorted(missing)}")
    frame = hourly.copy()
    if "county_fips" not in frame.columns:
        frame["county_fips"] = ""
    frame["county_fips"] = frame["county_fips"].fillna("").astype(str)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="coerce", utc=True)
    for column in ("latitude", "longitude", "temperature",
                   "precipitation_probability", "relative_humidity",
                   "wind_speed_mph"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["valid_time", "latitude", "longitude"])
    frame["valid_date"] = frame["valid_time"].dt.strftime("%Y-%m-%d")
    if frame.empty:
        return pd.DataFrame(columns=[
            "valid_date", "county_fips", "latitude", "longitude", "weather_hour_count",
            "weather_temperature_mean", "weather_temperature_min",
            "weather_temperature_max", "weather_precipitation_probability_mean",
            "weather_precipitation_probability_max", "weather_humidity_mean",
            "weather_wind_speed_mean",
        ])
    result = (frame.groupby(["valid_date", "county_fips", "latitude", "longitude"], as_index=False)
              .agg(weather_hour_count=("valid_time", "count"),
                   weather_temperature_mean=("temperature", "mean"),
                   weather_temperature_min=("temperature", "min"),
                   weather_temperature_max=("temperature", "max"),
                   weather_precipitation_probability_mean=(
                       "precipitation_probability", "mean"),
                   weather_precipitation_probability_max=(
                       "precipitation_probability", "max"),
                   weather_humidity_mean=("relative_humidity", "mean"),
                   weather_wind_speed_mean=("wind_speed_mph", "mean")))
    return result.sort_values(["valid_date", "county_fips", "latitude", "longitude"]).reset_index(drop=True)


def fetch_nws_county_hourly_forecasts(
    county_points: pd.DataFrame, *, timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one NWS hourly forecast per explicitly mapped Arkansas county.

    No spatial interpolation or county broadcasting occurs. A missing county
    request fails the refresh so a partial registry cannot be mistaken for a
    complete county surface.
    """
    required = {"county_fips", "latitude", "longitude"}
    missing = required.difference(county_points.columns)
    if missing:
        raise ValueError(f"county point registry missing columns: {sorted(missing)}")
    points = county_points.copy()
    points["county_fips"] = points["county_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if points["county_fips"].duplicated().any():
        raise ValueError("county point registry contains duplicate county_fips")
    frames = []
    metadata_rows = []
    for row in points.sort_values("county_fips").itertuples(index=False):
        frame, metadata = fetch_nws_hourly_forecast(
            row.latitude, row.longitude, county_fips=row.county_fips, timeout=timeout)
        frames.append(frame)
        metadata_rows.append(metadata)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metadata = {
        "source": "National Weather Service hourly forecasts",
        "county_count_requested": int(len(points)),
        "county_count_returned": int(result["county_fips"].nunique()) if not result.empty else 0,
        "rows": int(len(result)), "cadence": "hourly",
        "county_point_semantics": "one Census representative internal point per county",
        "county_metadata": metadata_rows,
        "target_semantics": "weather context only; not pharmacy demand or inventory truth",
    }
    if metadata["county_count_returned"] != metadata["county_count_requested"]:
        raise ValueError("NWS county refresh returned an incomplete county surface")
    return result, metadata


def fetch_nws_arkansas_alerts(*, endpoint: str = NWS_ARKANSAS_ALERTS_URL,
                              timeout: int = 30) -> tuple[pd.DataFrame, dict]:
    """Fetch active National Weather Service alerts for Arkansas.

    Alerts are operational context only.  They are not converted into demand
    labels and are never used to change a historical training artifact.
    """
    request = Request(
        endpoint,
        headers={
            "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
            "Accept": "application/geo+json",
        },
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS default
        payload = json.load(response)
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError("NWS alert response has invalid features")
    rows: list[dict] = []
    for feature in features:
        properties = feature.get("properties", {}) or {}
        alert_id = str(feature.get("id") or properties.get("id") or "").strip()
        if not alert_id:
            continue
        affected_zones = properties.get("geocode", {}).get("UGC", []) or []
        rows.append({
            "alert_id": alert_id,
            "event": properties.get("event", ""),
            "headline": properties.get("headline", ""),
            "severity": properties.get("severity", ""),
            "urgency": properties.get("urgency", ""),
            "certainty": properties.get("certainty", ""),
            "status": properties.get("status", ""),
            "onset": pd.to_datetime(properties.get("onset"), errors="coerce", utc=True),
            "expires": pd.to_datetime(properties.get("expires"), errors="coerce", utc=True),
            "area_desc": properties.get("areaDesc", ""),
            "ugc_zones": ";".join(str(zone) for zone in affected_zones),
            "sender_name": properties.get("senderName", ""),
            "source": "National Weather Service",
        })
    frame = pd.DataFrame(rows, columns=[
        "alert_id", "event", "headline", "severity", "urgency", "certainty",
        "status", "onset", "expires", "area_desc", "ugc_zones", "sender_name",
        "source",
    ])
    metadata = {
        "source": "National Weather Service active alerts",
        "endpoint": endpoint,
        "retrieved_on": date.today().isoformat(),
        "state": "AR",
        "rows": int(len(frame)),
        "cadence": "event-driven; endpoint commonly refreshes within minutes",
        "target_semantics": "live weather and disaster context only",
    }
    return frame, metadata


def fetch_medicaid_sdud_csv(*, url: str = MEDICAID_SDUD_2025_URL,
                            state: str = "AR", timeout: int = 120,
                            chunksize: int = 100_000) -> tuple[pd.DataFrame, dict]:
    """Fetch one CMS SDUD annual CSV and return a quarterly AR target panel.

    CMS publishes SDUD as an annual, state/NDC-level file.  This adapter is
    intentionally opt-in and keeps only non-suppressed prescription rows;
    it is a target refresh, not an operational feature available at forecast
    time.  Chunking avoids materializing the multi-million-row national file.
    """
    required = {
        "State Code", "NDC", "Year", "Quarter", "Suppression Used",
        "Product FDA List Name", "No. of Prescriptions",
    }
    rows: list[pd.DataFrame] = []
    with urlopen(Request(url, headers={"User-Agent": "MediTrack/1.0"}),
                timeout=timeout) as response:  # nosec B310: fixed HTTPS default
        for chunk in pd.read_csv(response, dtype={"NDC": "string"},
                                 chunksize=chunksize, low_memory=False):
            # The catalog dictionary uses descriptive labels while the
            # downloadable CSV uses shorter production headers.
            chunk = chunk.rename(columns={
                "State": "State Code",
                "Product Name": "Product FDA List Name",
                "Number of Prescriptions": "No. of Prescriptions",
            })
            missing = required.difference(chunk.columns)
            if missing:
                raise ValueError(f"CMS SDUD is missing columns: {sorted(missing)}")
            ar = chunk[chunk["State Code"].astype(str).str.upper().eq(state)].copy()
            ar = ar[~ar["Suppression Used"].astype(str).str.lower().isin(
                {"true", "1", "yes"})]
            ar["value"] = pd.to_numeric(ar["No. of Prescriptions"], errors="coerce")
            ar = ar.dropna(subset=["value", "Year", "Quarter"])
            if not ar.empty:
                rows.append(ar[["NDC", "Year", "Quarter", "Product FDA List Name",
                                "value"]])
    if not rows:
        frame = pd.DataFrame(columns=["year", "quarter", "drug", "value", "ndc"])
    else:
        raw = pd.concat(rows, ignore_index=True)
        raw["drug"] = raw["Product FDA List Name"].fillna("").astype(str).str.strip()
        raw = raw[raw["drug"].ne("")]
        frame = (raw.groupby(["Year", "Quarter", "drug"], as_index=False)
                 .agg(value=("value", "sum"), ndc=("NDC", "first"))
                 .rename(columns={"Year": "year", "Quarter": "quarter"}))
        frame["ingredient"] = frame["drug"]
        frame["rxnorm_rxcui"] = ""
        frame["manufacturer"] = ""
        frame["mapping_confidence"] = "cms_sdud_product_name"
    metadata = {
        "source": "CMS State Drug Utilization Data",
        "source_url": url,
        "retrieved_on": date.today().isoformat(),
        "state": state,
        "rows": int(len(frame)),
        "target_semantics": "quarterly Medicaid prescriptions by product name",
        "suppressed_rows_excluded": True,
    }
    return frame, metadata


def fetch_openfda_shortages(*, endpoint: str = OPENFDA_SHORTAGES_URL,
                            timeout: int = 30, page_size: int = 100) -> tuple[pd.DataFrame, dict]:
    """Fetch all openFDA shortage pages and normalize event rows.

    The endpoint permits at most 100 records per request.  Only fields needed
    by the supplier-drug context model are retained.  ``meta.last_updated``
    is returned separately and must be persisted with the artifact.
    """
    rows: list[dict] = []
    skip = 0
    metadata: dict = {}
    while True:
        query = urlencode({"limit": min(int(page_size), 100), "skip": skip})
        request = Request(f"{endpoint}?{query}", headers={"User-Agent": "MediTrack/1.0"})
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint by default
            payload = json.load(response)
        metadata = payload.get("meta", {})
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("openFDA shortage response has invalid results")
        for item in results:
            openfda = item.get("openfda", {}) or {}
            manufacturers = openfda.get("manufacturer_name") or []
            drugs = openfda.get("substance_name") or openfda.get("generic_name") or []
            supplier = (manufacturers[0] if manufacturers else item.get("company_name", ""))
            drug = drugs[0] if drugs else item.get("generic_name", "")
            event_date = item.get("change_date") or item.get("initial_posting_date")
            if not supplier or not drug or not event_date:
                continue
            rows.append({
                "supplier": supplier, "drug": drug,
                "event_date": pd.to_datetime(event_date, errors="coerce"),
                "month": pd.to_datetime(event_date, errors="coerce").strftime("%Y-%m")
                if pd.notna(pd.to_datetime(event_date, errors="coerce")) else "",
                "status": item.get("status", ""), "availability": item.get("availability", ""),
                "source": "openFDA drug shortages",
            })
        total = int((metadata.get("results") or {}).get("total", 0))
        skip += len(results)
        if not results or skip >= total:
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[frame["event_date"].notna()].reset_index(drop=True)
    metadata = {"source": "openFDA drug shortages", "endpoint": endpoint,
                "retrieved_on": date.today().isoformat(), **metadata}
    return frame, metadata


def fetch_openfda_enforcement(*, endpoint: str = OPENFDA_ENFORCEMENT_URL,
                              timeout: int = 30, page_size: int = 100) -> tuple[pd.DataFrame, dict]:
    """Fetch and normalize paginated FDA drug-enforcement recall records.

    ``recall_active`` is a source-status normalization, not a forecast label:
    completed/terminated FDA records are inactive and all other reported
    records remain active until a later source observation says otherwise.
    The raw status and termination date are retained for auditability.
    """
    rows: list[dict] = []
    skip = 0
    metadata: dict = {}
    while True:
        query = urlencode({"limit": min(int(page_size), 100), "skip": skip})
        request = Request(f"{endpoint}?{query}", headers={"User-Agent": "MediTrack/1.0"})
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.load(response)
        metadata = payload.get("meta", {})
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("openFDA enforcement response has invalid results")
        for item in results:
            status = str(item.get("status", "")).strip()
            initiation = item.get("recall_initiation_date") or item.get("report_date")
            event_date = pd.to_datetime(initiation, errors="coerce")
            if pd.isna(event_date):
                continue
            rows.append({
                "recall_number": item.get("recall_number", ""),
                "supplier": item.get("recalling_firm", ""),
                "product_description": item.get("product_description", ""),
                "product_ndc": item.get("product_ndc", ""),
                "classification": item.get("classification", ""),
                "reason_for_recall": item.get("reason_for_recall", ""),
                "status": status,
                "recall_active": int(status.lower() not in {"completed", "terminated"}),
                "event_date": event_date,
                "termination_date": pd.to_datetime(
                    item.get("termination_date"), errors="coerce"),
                "source": "openFDA drug enforcement",
            })
        total = int((metadata.get("results") or {}).get("total", 0))
        skip += len(results)
        if not results or skip >= total:
            break
    frame = pd.DataFrame(rows, columns=[
        "recall_number", "supplier", "product_description", "product_ndc",
        "classification", "reason_for_recall", "status", "recall_active",
        "event_date", "termination_date", "source",
    ])
    metadata = {"source": "openFDA drug enforcement", "endpoint": endpoint,
                "retrieved_on": date.today().isoformat(), **metadata}
    return frame, metadata


def fetch_fluview_weekly(
    *,
    endpoint: str = FLUVIEW_URL,
    regions: tuple[str, ...] = ("ar", "nat"),
    epiweeks: str | None = None,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch weekly CDC ILINet/FluView surveillance through Delphi Epidata.

    ``issue`` and ``lag`` are retained because the latest report for an
    epiweek can be revised. Callers must choose a point-in-time issue when
    constructing a historical backtest; this adapter only refreshes the
    public context feed.
    """
    normalized_regions = tuple(dict.fromkeys(
        str(region).strip().lower() for region in regions if str(region).strip()))
    if not normalized_regions:
        raise ValueError("at least one FluView region is required")
    params = {"source": "fluview", "regions": ",".join(normalized_regions)}
    if epiweeks:
        params["epiweeks"] = str(epiweeks).strip()
    separator = "&" if "?" in endpoint else "?"
    request_url = f"{endpoint}{separator}{urlencode(params)}"
    request = Request(request_url, headers={
        "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
        "Accept": "application/json",
    })
    with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
        payload = json.load(response)
    rows = payload.get("epidata", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError("FluView response has invalid epidata")
    columns = [
        "release_date", "region", "issue", "epiweek", "lag", "num_ili",
        "num_patients", "num_providers", "num_age_0", "num_age_1",
        "num_age_2", "num_age_3", "num_age_4", "num_age_5", "wili", "ili",
        "source", "source_url", "retrieved_at_utc",
    ]
    retrieved = datetime.now(timezone.utc).isoformat()
    normalized = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        row = {column: item.get(column) for column in columns}
        row["region"] = str(row["region"] or "").strip().lower()
        row["source"] = "Delphi Epidata CDC FluView"
        row["source_url"] = request_url
        row["retrieved_at_utc"] = retrieved
        normalized.append(row)
    frame = pd.DataFrame(normalized, columns=columns)
    for column in ("issue", "epiweek", "lag", "num_ili", "num_patients",
                   "num_providers", "num_age_0", "num_age_1", "num_age_2",
                   "num_age_3", "num_age_4", "num_age_5", "wili", "ili"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame.empty:
        frame["release_date"] = pd.to_datetime(
            frame["release_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame[frame["region"].isin(normalized_regions)].reset_index(drop=True)
    metadata = {
        "source": "Delphi Epidata CDC FluView",
        "source_url": request_url,
        "retrieved_at_utc": retrieved,
        "regions": list(normalized_regions),
        "epiweeks": epiweeks or "api_default",
        "rows": int(len(frame)),
        "cadence": "weekly",
        "publication_lag": "approximately_1_to_2_weeks",
        "target_semantics": "respiratory surveillance context; not pharmacy dispensing or inventory truth",
        "api_result": payload.get("result") if isinstance(payload, dict) else None,
    }
    return frame, metadata


def fetch_nndss_weekly(
    *,
    endpoint: str = NNDSS_WEEKLY_URL,
    states: tuple[str, ...] = ("Arkansas", "US RESIDENTS"),
    year_start: int = 2022,
    limit: int = 50_000,
    timeout: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Fetch provisional NNDSS rows for selected jurisdictions.

    NNDSS exposes disease rows by reporting week, but its ``m1``--``m4``
    values are monthly reporting columns. They are retained as reported and
    are never summed into a fabricated weekly case count. Suppression and
    not-reported flags remain available to downstream feature builders.
    """
    normalized_states = tuple(dict.fromkeys(
        str(state).strip() for state in states if str(state).strip()))
    if not normalized_states:
        raise ValueError("at least one NNDSS state or jurisdiction is required")
    if int(year_start) < 2022:
        raise ValueError("NNDSS current dataset begins in 2022")
    limit = max(1, min(int(limit), 50_000))
    state_clause = ", ".join(f"'{state.replace(chr(39), chr(39) * 2)}'"
                              for state in normalized_states)
    where = f"year >= {int(year_start)} AND states in ({state_clause})"
    rows: list[dict] = []
    offset = 0
    while True:
        query = urlencode({
            "$where": where,
            "$order": "year,week,states,label",
            "$limit": limit,
            "$offset": offset,
        })
        request_url = f"{endpoint}?{query}"
        request = Request(request_url, headers={
            "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.load(response)
        if not isinstance(payload, list):
            raise ValueError("NNDSS response must be a JSON array")
        rows.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < limit:
            break
        offset += len(payload)
    columns = [
        "state", "year", "week", "epiweek", "disease", "m1", "m1_flag",
        "m2", "m2_flag", "m3", "m3_flag", "m4", "m4_flag", "geocode",
        "source", "source_url", "retrieved_at_utc",
    ]
    retrieved = datetime.now(timezone.utc).isoformat()
    normalized = []
    for item in rows:
        year = pd.to_numeric(item.get("year"), errors="coerce")
        week = pd.to_numeric(item.get("week"), errors="coerce")
        normalized.append({
            "state": str(item.get("states") or "").strip(),
            "year": year,
            "week": week,
            "epiweek": (int(year) * 100 + int(week)
                        if pd.notna(year) and pd.notna(week) else pd.NA),
            "disease": str(item.get("label") or "").strip(),
            **{field: item.get(field) for field in (
                "m1", "m1_flag", "m2", "m2_flag", "m3", "m3_flag",
                "m4", "m4_flag", "geocode")},
            "source": "CDC NNDSS Weekly Data",
            "source_url": request_url,
            "retrieved_at_utc": retrieved,
        })
    frame = pd.DataFrame(normalized, columns=columns)
    for column in ("year", "week", "epiweek", "m1", "m2", "m3", "m4"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame.empty:
        frame = frame[frame["state"].isin(normalized_states)].reset_index(drop=True)
    metadata = {
        "source": "CDC NNDSS Weekly Data",
        "source_url": endpoint,
        "request_url": request_url,
        "retrieved_at_utc": retrieved,
        "states": list(normalized_states),
        "year_start": int(year_start),
        "rows": int(len(frame)),
        "cadence": "weekly reporting rows with monthly m1-m4 fields",
        "publication_lag": "weekly_reporting_lag",
        "target_semantics": "provisional notifiable-disease surveillance context; not pharmacy dispensing or inventory truth",
        "monthly_fields_not_weekly_total": True,
    }
    return frame, metadata


def fetch_noaa_daily_summaries(
    *,
    stations: tuple[str, ...],
    start_date: str,
    end_date: str,
    endpoint: str = NOAA_DAILY_SUMMARIES_URL,
    timeout: int = 120,
) -> tuple[pd.DataFrame, dict]:
    """Fetch observed NOAA/NCEI GHCN-Daily summaries for named stations.

    Station selection and date bounds are explicit inputs. The adapter does
    not interpolate stations or broadcast observations to counties; callers
    can pass the result through the existing nearest-station county mapping.
    """
    normalized_stations = tuple(dict.fromkeys(
        str(station).strip().upper() for station in stations if str(station).strip()))
    if not normalized_stations:
        raise ValueError("at least one NOAA station is required")
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("start_date and end_date must be ordered ISO dates")
    params = urlencode({
        "dataset": "daily-summaries",
        "stations": ",".join(normalized_stations),
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "format": "json",
        "units": "standard",
        "includeAttributes": "false",
        "includeStationName": "true",
        "includeStationLocation": "true",
    })
    request_url = f"{endpoint}?{params}"
    request = Request(request_url, headers={
        "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
        "Accept": "application/json",
    })
    with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("NOAA daily summaries response must be a JSON array")
    columns = [
        "date", "station_id", "station_name", "latitude", "longitude",
        "temperature_max", "temperature_min", "temperature_mean",
        "precipitation", "snowfall", "snow_depth", "wind", "humidity",
        "extreme_heat_day", "extreme_cold_day", "source", "source_url",
        "retrieved_at_utc",
    ]
    retrieved = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        tmax = pd.to_numeric(item.get("TMAX"), errors="coerce")
        tmin = pd.to_numeric(item.get("TMIN"), errors="coerce")
        station = str(item.get("STATION") or "").strip().upper()
        rows.append({
            "date": item.get("DATE"), "station_id": station,
            "station_name": str(item.get("NAME") or "").strip(),
            "latitude": item.get("LATITUDE"), "longitude": item.get("LONGITUDE"),
            "temperature_max": tmax, "temperature_min": tmin,
            "temperature_mean": ((tmax + tmin) / 2
                                  if pd.notna(tmax) and pd.notna(tmin) else pd.NA),
            "precipitation": item.get("PRCP"), "snowfall": item.get("SNOW"),
            "snow_depth": item.get("SNWD"), "wind": item.get("AWND"),
            "humidity": item.get("RHAV"),
            "extreme_heat_day": (int(tmax >= 95) if pd.notna(tmax) else pd.NA),
            "extreme_cold_day": (int(tmin <= 20) if pd.notna(tmin) else pd.NA),
            "source": "NOAA/NCEI GHCN-Daily station summaries",
            "source_url": request_url, "retrieved_at_utc": retrieved,
        })
    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("latitude", "longitude", "temperature_max", "temperature_min",
                   "temperature_mean", "precipitation", "snowfall", "snow_depth",
                   "wind", "humidity", "extreme_heat_day", "extreme_cold_day"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["station_id"].isin(normalized_stations)].reset_index(drop=True)
    metadata = {
        "source": "NOAA/NCEI GHCN-Daily station summaries",
        "source_url": endpoint, "request_url": request_url,
        "retrieved_at_utc": retrieved, "stations": list(normalized_stations),
        "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
        "rows": int(len(frame)), "cadence": "daily",
        "publication_lag": "approximately_1_to_2_days",
        "observed_only": True, "spatial_method": "station observations only",
        "target_semantics": "weather context only; not pharmacy demand or inventory truth",
    }
    return frame, metadata


def fetch_nadac_weekly_csv(
    *,
    as_of_date: str,
    url_template: str = NADAC_DOWNLOAD_TEMPLATE,
    ndc_filter: tuple[str, ...] | None = None,
    timeout: int = 180,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one explicitly dated CMS NADAC weekly reference CSV.

    NADAC is a national acquisition-cost reference, not local inventory or
    pharmacy dispensing. The dated URL is intentional: a backtest or refresh
    must identify the exact CMS snapshot rather than depending on a moving
    latest-file endpoint.
    """
    date_value = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(date_value):
        raise ValueError("as_of_date must be an ISO date")
    date_text = date_value.strftime("%Y-%m-%d")
    url = url_template.format(date=date_text)
    request = Request(url, headers={
        "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
        "Accept": "text/csv",
    })
    with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
        raw = response.read()
    frame = pd.read_csv(io.BytesIO(raw), dtype={"NDC": "string"}, low_memory=False)
    required = {"NDC", "NADAC Per Unit", "As of Date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NADAC CSV is missing columns: {sorted(missing)}")
    rename = {
        "NDC": "ndc", "NDC Description": "drug_description",
        "NADAC Per Unit": "nadac_per_unit", "Effective Date": "effective_date",
        "Pricing Unit": "pricing_unit", "Pharmacy Type Indicator": "pharmacy_type_indicator",
        "OTC": "otc", "Explanation Code": "explanation_code",
        "Classification for Rate Setting": "classification_for_rate_setting",
        "Corresponding Generic Drug NADAC Per Unit": "corresponding_generic_nadac_per_unit",
        "Corresponding Generic Drug Effective Date": "corresponding_generic_effective_date",
        "As of Date": "as_of_date",
    }
    frame = frame.rename(columns=rename)
    frame["ndc"] = frame["ndc"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    frame["nadac_per_unit"] = pd.to_numeric(frame["nadac_per_unit"], errors="coerce")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("effective_date", "corresponding_generic_effective_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
    if ndc_filter is not None:
        allowed = {str(value).replace("-", "").strip() for value in ndc_filter}
        frame = frame[frame["ndc"].isin(allowed)].copy()
    frame = frame[frame["ndc"].ne("") & frame["nadac_per_unit"].notna()].reset_index(drop=True)
    retrieved = datetime.now(timezone.utc).isoformat()
    frame["source"] = "CMS NADAC"
    frame["source_url"] = url
    frame["retrieved_at_utc"] = retrieved
    metadata = {
        "source": "CMS NADAC",
        "source_url": url,
        "retrieved_at_utc": retrieved,
        "as_of_date": date_text,
        "rows": int(len(frame)),
        "cadence": "weekly with monthly reference update",
        "publication_lag": "weekly_file_release",
        "ndc_filter_applied": ndc_filter is not None,
        "target_semantics": "national acquisition-cost reference; not local inventory, demand, shortage, or supplier allocation truth",
    }
    return frame, metadata


def fetch_fema_arkansas_disasters(
    *,
    start_date: str = "2012-01-01",
    as_of_date: str | None = None,
    endpoint: str = FEMA_DISASTER_DECLARATIONS_URL,
    page_size: int = 5_000,
    timeout: int = 120,
) -> tuple[pd.DataFrame, dict]:
    """Fetch Arkansas FEMA disaster declarations with both event timestamps.

    FEMA declarations are posted after an incident and therefore are usable
    as retrospective live context, not as an incident-onset forecast input.
    ``declaration_date`` and ``incident_begin_date`` remain separate to make
    that limitation explicit in every returned row.
    """
    start = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start):
        raise ValueError("start_date must be an ISO date")
    observed_on = (pd.to_datetime(as_of_date, errors="coerce", utc=True)
                   if as_of_date else pd.Timestamp.now(tz="UTC"))
    if pd.isna(observed_on):
        raise ValueError("as_of_date must be an ISO date when supplied")
    page_size = max(1, min(int(page_size), 5_000))
    where = ("state eq 'AR' and declarationDate ge "
             f"'{start.strftime('%Y-%m-%d')}T00:00:00.000z'")
    rows: list[dict] = []
    skip = 0
    request_urls = []
    while True:
        params = urlencode({"$filter": where, "$top": page_size, "$skip": skip})
        request_url = f"{endpoint}?{params}"
        request_urls.append(request_url)
        request = Request(request_url, headers={
            "User-Agent": "MediTrack/1.0 (research; contact unavailable)",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.load(response)
        page = payload.get("DisasterDeclarationsSummaries", []) if isinstance(payload, dict) else []
        if not isinstance(page, list):
            raise ValueError("FEMA response has invalid DisasterDeclarationsSummaries")
        rows.extend(item for item in page if isinstance(item, dict))
        if len(page) < page_size:
            break
        skip += len(page)
    columns = [
        "disaster_number", "state", "declaration_type", "declaration_date",
        "incident_begin_date", "incident_end_date", "disaster_name",
        "incident_type", "designated_area", "county_fips", "fips_state_code",
        "fips_county_code", "disaster_active", "disaster_severity", "source",
        "source_url", "retrieved_at_utc",
    ]
    retrieved = datetime.now(timezone.utc).isoformat()
    normalized = []
    for item in rows:
        declaration = pd.to_datetime(item.get("declarationDate"), errors="coerce", utc=True)
        if pd.isna(declaration):
            continue
        begin = pd.to_datetime(item.get("incidentBeginDate"), errors="coerce", utc=True)
        end = pd.to_datetime(item.get("incidentEndDate"), errors="coerce", utc=True)
        state_code = str(item.get("fipsStateCode") or "05").zfill(2)
        county_code = str(item.get("fipsCountyCode") or "").zfill(3)
        county_fips = state_code + county_code if county_code.strip("0") else ""
        active = int(pd.notna(begin) and begin <= observed_on and
                     (pd.isna(end) or end >= observed_on))
        declaration_type = str(item.get("declarationType") or "").strip()
        severity = 1.0 if declaration_type == "DR" else 0.5
        normalized.append({
            "disaster_number": item.get("disasterNumber"), "state": item.get("state", "AR"),
            "declaration_type": declaration_type,
            "declaration_date": declaration.strftime("%Y-%m-%d"),
            "incident_begin_date": begin.strftime("%Y-%m-%d") if pd.notna(begin) else "",
            "incident_end_date": end.strftime("%Y-%m-%d") if pd.notna(end) else "",
            "disaster_name": item.get("declarationTitle", ""),
            "incident_type": item.get("incidentType", ""),
            "designated_area": item.get("designatedArea", ""),
            "county_fips": county_fips, "fips_state_code": state_code,
            "fips_county_code": county_code if county_fips else "",
            "disaster_active": active, "disaster_severity": severity,
            "source": "FEMA OpenFEMA Disaster Declarations Summaries",
            "source_url": request_urls[-1], "retrieved_at_utc": retrieved,
        })
    frame = pd.DataFrame(normalized, columns=columns)
    for column in ("disaster_number", "fips_state_code", "fips_county_code",
                   "disaster_active", "disaster_severity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    metadata = {
        "source": "FEMA OpenFEMA Disaster Declarations Summaries",
        "source_url": endpoint, "request_urls": request_urls,
        "retrieved_at_utc": retrieved, "as_of_date": observed_on.strftime("%Y-%m-%d"),
        "start_date": start.strftime("%Y-%m-%d"), "rows": int(len(frame)),
        "cadence": "event-driven", "publication_lag": "posted_after_declaration",
        "state": "AR", "page_count": len(request_urls),
        "declaration_is_not_incident_onset": True,
        "target_semantics": "retrospective disaster context; not pharmacy demand, supply, or shortage truth",
    }
    return frame, metadata
