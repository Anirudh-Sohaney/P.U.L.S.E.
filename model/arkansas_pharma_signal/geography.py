"""Reproducible Arkansas city-to-county resolution.

The existing CMS panel contains prescriber cities but no county.  This module
uses the official Census geocoder only to resolve a representative city address
and marks the result weakly inferred.  Ambiguous or failed results are retained
as unresolved rather than guessed.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from . import io
from .entities import normalize_name

GEOGRAPHY_COLUMNS = ["city", "city_key", "county_fips", "county_name",
                      "arkansas_region", "source", "source_timestamp",
                      "confidence", "evidence_type", "transformation_method",
                      "representative_address"]

CENSUS_COUNTY_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "{year}_Gazetteer/{year}_gaz_counties_{state_fips}.txt"
)
COUNTY_WEATHER_POINT_COLUMNS = [
    "county_fips", "county_name", "arkansas_region", "latitude", "longitude",
    "source", "source_url", "source_vintage", "transformation_method",
]

# Stable, broad reporting regions from the Arkansas DHS/TEFRA five-region
# definition. This is a transparent county-to-region crosswalk rather than a
# guessed city label. Counties not present in a bucket remain unresolved.
REGION_COUNTIES = {
    "northwest": {
        "05005", "05007", "05009", "05015", "05029", "05033", "05047",
        "05071", "05083", "05087", "05089", "05115", "05127", "05129",
        "05101", "05131", "05141", "05143", "05149",
    },
    "northeast": {
        "05021", "05023", "05031", "05035", "05037", "05049", "05055",
        "05063", "05065", "05067", "05075", "05093", "05111", "05121",
        "05135", "05137", "05145", "05147",
    },
    "central": {
        "05045", "05051", "05053", "05085", "05105", "05119", "05125",
    },
    "southwest": {
        "05013", "05019", "05027", "05039", "05057", "05059", "05061",
        "05073", "05081", "05091", "05097", "05099", "05103", "05109",
        "05113", "05133", "05139",
    },
    "southeast": {
        "05001", "05003", "05011", "05017", "05025", "05041", "05043",
        "05069", "05077", "05079", "05095", "05107", "05117", "05123",
    },
}


def region_for_county(county_fips: object) -> str:
    """Return the explicit broad Arkansas reporting region for a county."""
    value = str(county_fips or "").strip()
    if value.endswith(".0"):
        value = value[:-2]
    value = value.zfill(5) if value.isdigit() else ""
    for region, counties in REGION_COUNTIES.items():
        if value in counties:
            return region
    return ""


def _query_city(city: str, representative_address: str = "", timeout: int = 30) -> dict:
    address = representative_address or f"1 Main Street, {city}, AR"
    query = urlencode({"address": address, "benchmark": "Public_AR_Current",
                       "vintage": "Current_Current", "format": "json"})
    url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress?" + query
    try:
        with urlopen(Request(url, headers={"User-Agent": "arkansas-pharma-signal/1.0"}),
                     timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        matches = payload.get("result", {}).get("addressMatches", [])
        counties = []
        for match in matches[:1]:
            for county in match.get("geographies", {}).get("Counties", []):
                fips = str(county.get("GEOID", ""))
                if fips.startswith("05"):
                    counties.append((fips, str(county.get("NAME", ""))))
        unique = list(dict.fromkeys(counties))
        if len(unique) == 1:
            return {"city": city, "city_key": normalize_name(city),
                    "county_fips": unique[0][0], "county_name": unique[0][1],
                    "confidence": 0.95 if representative_address else 0.82,
                    "evidence_type": "mapped_address" if representative_address else "weakly_inferred",
                    "transformation_method": "Census geocoder NPPES practice address" if representative_address
                    else "Census geocoder representative city address",
                    "representative_address": address}
    except Exception as exc:  # network failures remain explicit unresolved rows
        return {"city": city, "city_key": normalize_name(city), "error": type(exc).__name__,
                "representative_address": address}
    return {"city": city, "city_key": normalize_name(city), "error": "ambiguous_or_no_match",
            "representative_address": address}


def build_city_county_crosswalk(cfg, cities: list[str], workers: int = 8) -> pd.DataFrame:
    """Resolve unique CMS cities and return a provenance-bearing crosswalk."""
    unique = sorted({str(c).strip() for c in cities if str(c).strip()})
    representatives = {}
    path = cfg.data_path(cfg.nppes_provider_locations)
    if path.exists():
        nppes = io.load_csv(path, usecols=[
            "Provider First Line Business Practice Location Address",
            "Provider Business Practice Location Address City Name",
            "Provider Business Practice Location Address State Name",
            "Provider Business Practice Location Address Postal Code",
        ])
        nppes = nppes.dropna(subset=["Provider First Line Business Practice Location Address",
                                     "Provider Business Practice Location Address City Name"])
        nppes = nppes[nppes["Provider Business Practice Location Address State Name"].astype(str).str.upper().eq("AR")]
        unique_lower = {x.lower() for x in unique}
        for _, row in nppes.iterrows():
            city = str(row["Provider Business Practice Location Address City Name"]).strip()
            if city.lower() not in unique_lower or city.lower() in {x.lower() for x in representatives}:
                continue
            zip_code = str(row.get("Provider Business Practice Location Address Postal Code", ""))[:5]
            representatives[city.lower()] = (f"{row['Provider First Line Business Practice Location Address']}, "
                                     f"{city}, {row['Provider Business Practice Location Address State Name']} {zip_code}")
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_query_city, city, representatives.get(city.lower(), "")): city for city in unique}
        for future in as_completed(futures):
            row = future.result()
            row.update({"arkansas_region": region_for_county(row.get("county_fips", "")),
                        "source": "US Census Geocoder + Arkansas county region crosswalk",
                        "source_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                        "confidence": row.get("confidence", 0.0),
                        "evidence_type": row.get("evidence_type", "unresolved"),
                        "transformation_method": row.get("transformation_method", "unresolved"),
                        "representative_address": row.get("representative_address", "")})
            rows.append({k: row.get(k, "") for k in GEOGRAPHY_COLUMNS})
    return pd.DataFrame(rows, columns=GEOGRAPHY_COLUMNS).sort_values("city_key").reset_index(drop=True)


def load_city_county_crosswalk(cfg) -> pd.DataFrame:
    path = cfg.artifact_path("geography/city_county_crosswalk.csv")
    if not path.exists():
        return pd.DataFrame(columns=GEOGRAPHY_COLUMNS)
    return io.load_csv(path)


def mapping_dict(crosswalk: pd.DataFrame) -> dict[str, dict]:
    if crosswalk.empty:
        return {}
    return {str(r.city).lower(): r.to_dict() for _, r in crosswalk.iterrows()
            if str(r.get("county_fips", "")) not in ("", "nan") and float(r.get("confidence", 0) or 0) > 0}


def fetch_census_county_weather_points(
    *, year: int = 2025, state_fips: str = "05", url: str | None = None,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch Census county internal points for an explicit weather registry.

    Census calls these representative internal points. They are a reproducible
    coordinate for querying a point weather service, not an assertion that
    weather is homogeneous across a county. The returned registry is therefore
    suitable for county-keyed context with explicit point semantics.
    """
    state_fips = str(state_fips).strip().zfill(2)
    if state_fips != "05":
        raise ValueError("the county weather registry is restricted to Arkansas (FIPS 05)")
    endpoint = url or CENSUS_COUNTY_GAZETTEER_URL.format(year=year, state_fips=state_fips)
    request = Request(endpoint, headers={"User-Agent": "arkansas-pharma-signal/1.0"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310: caller-visible HTTPS source
        payload = response.read().decode("utf-8")
    raw = pd.read_csv(StringIO(payload), sep="|", dtype=str)
    required = {"USPS", "GEOID", "NAME", "INTPTLAT", "INTPTLONG"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Census county Gazetteer missing columns: {sorted(missing)}")
    result = raw.loc[raw["USPS"].astype(str).str.upper().eq("AR"),
                     ["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]].copy()
    result = result.rename(columns={"GEOID": "county_fips", "NAME": "county_name",
                                    "INTPTLAT": "latitude", "INTPTLONG": "longitude"})
    result["county_fips"] = result["county_fips"].astype(str).str.zfill(5)
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["arkansas_region"] = result["county_fips"].map(region_for_county).fillna("")
    result["source"] = "U.S. Census Bureau county Gazetteer internal point"
    result["source_url"] = endpoint
    result["source_vintage"] = str(year)
    result["transformation_method"] = "Census county internal point; NWS point query"
    result = result[COUNTY_WEATHER_POINT_COLUMNS].sort_values("county_fips").reset_index(drop=True)
    if len(result) != 75 or result["county_fips"].nunique() != 75:
        raise ValueError(f"expected 75 unique Arkansas counties, received {len(result)}")
    if result[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Census county weather registry contains missing coordinates")
    metadata = {
        "source": "U.S. Census Bureau county Gazetteer internal points",
        "source_url": endpoint, "source_vintage": int(year), "state_fips": "05",
        "rows": int(len(result)), "county_count": int(result["county_fips"].nunique()),
        "point_semantics": "representative county internal point; not county-wide weather truth",
    }
    return result, metadata
