"""Leak-safe county weather observations from local NOAA daily summaries.

The repository's NOAA/NCEI daily station files are observed historical data,
not forecasts. Each Arkansas county is assigned its nearest available
Arkansas station using the Census internal point registry. The assignment is
explicit and stable; no interpolation or missing-period imputation is done.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


WEATHER_VALUE_COLUMNS = [
    "temperature_mean", "temperature_min", "temperature_max", "precipitation",
    "snowfall", "snow_depth",
]
PANEL_COLUMNS = [
    "cadence", "period_start", "period_end", "county_fips", "county_name",
    "arkansas_region", "station_id", "station_name", "station_latitude",
    "station_longitude", "station_distance_km", "weather_observed_days",
    "weather_expected_days", "weather_coverage_ratio", *WEATHER_VALUE_COLUMNS,
    "source_id", "source_name", "source_url", "transformation_method",
]
WEATHER_CONTEXT_COLUMNS = [
    "weather_temperature_mean", "weather_temperature_min", "weather_temperature_max",
    "weather_precipitation", "weather_snowfall", "weather_snow_depth",
    "weather_observed_days", "weather_expected_days", "weather_coverage_ratio",
    "weather_station_id", "weather_station_distance_km", "weather_context_available",
]


def load_noaa_daily_summaries(raw_dir: Path) -> pd.DataFrame:
    """Load local NOAA daily-summary JSON files into one deduplicated frame."""
    rows: list[dict] = []
    for path in sorted(Path(raw_dir).glob("noaa_daily_summaries_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            continue
        rows.extend(row for row in payload if isinstance(row, dict))
    if not rows:
        return pd.DataFrame(columns=["date", "station_id", "station_name", "latitude",
                                     "longitude", *WEATHER_VALUE_COLUMNS])
    raw = pd.DataFrame(rows)
    rename = {"DATE": "date", "STATION": "station_id", "NAME": "station_name",
              "LATITUDE": "latitude", "LONGITUDE": "longitude", "TMAX": "tmax",
              "TMIN": "tmin", "PRCP": "precipitation", "SNOW": "snowfall",
              "SNWD": "snow_depth"}
    raw = raw.rename(columns=rename)
    required = {"date", "station_id", "station_name", "latitude", "longitude"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"NOAA daily summaries missing columns: {sorted(missing)}")
    for column in ("latitude", "longitude", "tmax", "tmin", *WEATHER_VALUE_COLUMNS[3:]):
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["station_id"] = raw["station_id"].astype(str).str.strip()
    raw["station_name"] = raw["station_name"].fillna("").astype(str).str.strip()
    raw["temperature_mean"] = raw[["tmax", "tmin"]].mean(axis=1)
    raw = raw.rename(columns={"tmin": "temperature_min", "tmax": "temperature_max"})
    for column in WEATHER_VALUE_COLUMNS:
        if column not in raw:
            raw[column] = float("nan")
    raw = raw.dropna(subset=["date", "station_id", "latitude", "longitude"])
    raw = raw.drop_duplicates(subset=["station_id", "date"], keep="last")
    return raw[["date", "station_id", "station_name", "latitude", "longitude",
                *WEATHER_VALUE_COLUMNS]].sort_values(["station_id", "date"]).reset_index(drop=True)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance for nearest-station assignment."""
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def assign_nearest_weather_stations(
    county_points: pd.DataFrame, daily: pd.DataFrame,
) -> pd.DataFrame:
    """Return one deterministic nearest-station mapping per county."""
    required_points = {"county_fips", "county_name", "arkansas_region", "latitude", "longitude"}
    required_daily = {"station_id", "station_name", "latitude", "longitude"}
    if missing := required_points.difference(county_points.columns):
        raise ValueError(f"county registry missing columns: {sorted(missing)}")
    if missing := required_daily.difference(daily.columns):
        raise ValueError(f"weather observations missing columns: {sorted(missing)}")
    stations = daily[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates("station_id")
    rows = []
    for county in county_points.itertuples(index=False):
        candidates = [
            (station, _distance_km(float(county.latitude), float(county.longitude),
                                   float(station.latitude), float(station.longitude)))
            for station in stations.itertuples(index=False)
        ]
        if not candidates:
            continue
        station, distance = min(candidates, key=lambda item: (item[1], str(item[0].station_id)))
        rows.append({
            "county_fips": str(county.county_fips).zfill(5),
            "county_name": county.county_name,
            "arkansas_region": county.arkansas_region,
            "station_id": station.station_id,
            "station_name": station.station_name,
            "station_latitude": float(station.latitude),
            "station_longitude": float(station.longitude),
            "station_distance_km": distance,
        })
    result = pd.DataFrame(rows)
    if len(result) != len(county_points) or result["county_fips"].nunique() != len(county_points):
        raise ValueError("every county must have exactly one nearest weather station")
    return result.sort_values("county_fips").reset_index(drop=True)


def build_county_weather_panel(
    daily: pd.DataFrame, county_points: pd.DataFrame, *, cadences: tuple[str, ...] = ("weekly", "monthly"),
) -> tuple[pd.DataFrame, dict]:
    """Aggregate observed station data into county-keyed weekly/monthly rows."""
    if daily.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS), {"rows": 0, "county_count": 0}
    mapping = assign_nearest_weather_stations(county_points, daily)
    # One station may be the nearest observed source for multiple counties;
    # expand that explicit mapping, never an unkeyed geographic broadcast.
    frame = daily.merge(
        mapping.drop(columns=["station_name"]), on="station_id", how="inner",
        validate="many_to_many")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    output = []
    for cadence in cadences:
        if cadence == "weekly":
            periods = frame["date"].dt.to_period("W-SUN")
        elif cadence == "monthly":
            periods = frame["date"].dt.to_period("M")
        else:
            raise ValueError(f"unsupported weather cadence: {cadence}")
        work = frame.copy()
        work["period_start"] = periods.dt.start_time
        work["period_end"] = periods.dt.end_time.dt.normalize()
        work["cadence"] = cadence
        work["expected_days"] = (work["period_end"] - work["period_start"]).dt.days + 1
        grouped = (work.groupby([
            "cadence", "period_start", "period_end", "county_fips", "county_name",
            "arkansas_region", "station_id", "station_name", "station_latitude",
            "station_longitude", "station_distance_km", "expected_days",
        ], as_index=False).agg(
            weather_observed_days=("date", "nunique"),
            temperature_mean=("temperature_mean", "mean"),
            temperature_min=("temperature_min", "min"),
            temperature_max=("temperature_max", "max"),
            precipitation=("precipitation", "sum"), snowfall=("snowfall", "sum"),
            snow_depth=("snow_depth", "mean"),
        ).rename(columns={"expected_days": "weather_expected_days"}))
        grouped["weather_coverage_ratio"] = (
            grouped["weather_observed_days"] / grouped["weather_expected_days"])
        grouped["source_id"] = "noaa_ncei_daily_summaries"
        grouped["source_name"] = "NOAA/NCEI GHCN-Daily station summaries"
        grouped["source_url"] = "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily"
        grouped["transformation_method"] = "nearest Arkansas station; observed daily aggregation; no interpolation"
        output.append(grouped)
    result = pd.concat(output, ignore_index=True)[PANEL_COLUMNS]
    result = result.sort_values(["cadence", "period_start", "county_fips"]).reset_index(drop=True)
    metadata = {
        "source_id": "noaa_ncei_daily_summaries", "rows": int(len(result)),
        "county_count": int(result["county_fips"].nunique()),
        "station_count": int(result["station_id"].nunique()),
        "cadences": list(cadences), "observed_only": True,
        "imputation": "none", "spatial_method": "nearest station",
        "source_url": "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
    }
    return result, metadata


def attach_weather_context(
    target: pd.DataFrame, weather: pd.DataFrame, *, cadence: str | None = None,
) -> pd.DataFrame:
    """Join weather only at exact county and period grain.

    ``target`` must contain ``county_fips``, ``period_start``, and
    ``period_end``. If it has no cadence column, ``cadence`` is required. The
    join is left-sided and missing weather remains explicit.
    """
    required_target = {"county_fips", "period_start", "period_end"}
    required_weather = required_target | {"cadence"}
    if missing := required_target.difference(target.columns):
        raise ValueError(f"target missing weather join columns: {sorted(missing)}")
    if missing := required_weather.difference(weather.columns):
        raise ValueError(f"weather missing join columns: {sorted(missing)}")
    left = target.copy()
    right = weather.copy()
    left["county_fips"] = left["county_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    right["county_fips"] = right["county_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    for frame in (left, right):
        frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce").dt.normalize()
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize()
    if "cadence" not in left:
        if cadence not in {"weekly", "monthly"}:
            raise ValueError("cadence is required when target has no cadence column")
        left["cadence"] = cadence
    weather_columns = {
        "temperature_mean": "weather_temperature_mean",
        "temperature_min": "weather_temperature_min",
        "temperature_max": "weather_temperature_max",
        "precipitation": "weather_precipitation",
        "snowfall": "weather_snowfall",
        "snow_depth": "weather_snow_depth",
        "weather_observed_days": "weather_observed_days",
        "weather_expected_days": "weather_expected_days",
        "weather_coverage_ratio": "weather_coverage_ratio",
        "station_id": "weather_station_id",
        "station_distance_km": "weather_station_distance_km",
    }
    right = right[["cadence", "period_start", "period_end", "county_fips",
                   *[column for column in weather_columns if column in right]]].rename(
                       columns=weather_columns)
    keys = ["cadence", "period_start", "period_end", "county_fips"]
    if right.duplicated(keys).any():
        raise ValueError("weather panel has duplicate county-period keys")
    result = left.merge(right, on=keys, how="left", validate="many_to_one")
    result["weather_context_available"] = result["weather_station_id"].notna().astype("int8")
    return result
