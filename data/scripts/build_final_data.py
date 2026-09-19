#!/usr/bin/env python3
"""Build the final external-state data package.

The output is intentionally long-format: one numeric feature per row, with
source-native categorical fields retained in dimensions JSON. This keeps the
feature store compatible with the data_infra.md contract without fabricating
unavailable county or daily data.
"""

from __future__ import annotations

import csv
import collections
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_data"
RAW = OUT / "raw_downloads"
FEATURES = OUT / "features"
EVENTS = OUT / "events"
ENTITIES = OUT / "entities"
SOURCES = OUT / "sources"
QUALITY = OUT / "quality"

RUN_AT = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
SCHEMA_VERSION = "1.0"
EXTENDED_START_YEAR = 2000
CURRENT_YEAR = 2026

PHARMA_HS_PREFIXES = ("30",)
PHARMA_HS_CODES = {
    "2936", "2937", "2939", "2941", "3001", "3002", "3003", "3004", "3005", "3006",
}

ARKANSAS_EDD_REGIONS = {
    "northwest_arkansas_edd": ["Benton", "Washington", "Madison", "Carroll", "Boone", "Newton", "Marion", "Searcy", "Baxter"],
    "north_central_arkansas_edd": ["Fulton", "Izard", "Sharp", "Stone", "Independence", "Jackson", "Van Buren", "Cleburne", "White", "Woodruff"],
    "northeast_arkansas_edd": ["Randolph", "Clay", "Lawrence", "Greene", "Craighead", "Mississippi", "Poinsett", "Cross", "Crittenden", "St. Francis", "Lee", "Phillips"],
    "southeast_arkansas_edd": ["Grant", "Jefferson", "Arkansas", "Cleveland", "Lincoln", "Desha", "Bradley", "Drew", "Chicot", "Ashley"],
    "southwest_arkansas_edd": ["Sevier", "Howard", "Little River", "Hempstead", "Nevada", "Ouachita", "Dallas", "Calhoun", "Miller", "Lafayette", "Columbia", "Union"],
    "western_arkansas_edd": ["Crawford", "Franklin", "Sebastian", "Logan", "Scott", "Polk"],
    "west_central_arkansas_edd": ["Johnson", "Pope", "Conway", "Yell", "Perry", "Montgomery", "Garland", "Pike", "Clark", "Hot Spring"],
    "central_arkansas_edd": ["Faulkner", "Saline", "Pulaski", "Lonoke", "Prairie", "Monroe"],
}

COUNTY_TO_REGION = {
    f"{county.lower()} county": region
    for region, counties in ARKANSAS_EDD_REGIONS.items()
    for county in counties
}

REGION_NAMES = {
    "northwest_arkansas_edd": "Northwest Arkansas Economic Development District",
    "north_central_arkansas_edd": "North Central Arkansas Economic Development District",
    "northeast_arkansas_edd": "Northeast Arkansas Economic Development District",
    "southeast_arkansas_edd": "Southeast Arkansas Economic Development District",
    "southwest_arkansas_edd": "Southwest Economic Development District of Arkansas",
    "western_arkansas_edd": "Western Arkansas Economic Development District",
    "west_central_arkansas_edd": "West Central Arkansas Economic Development District",
    "central_arkansas_edd": "Central Arkansas Economic Development District",
}


def ensure_dirs() -> None:
    for path in [RAW, FEATURES, EVENTS, ENTITIES, SOURCES, QUALITY]:
        path.mkdir(parents=True, exist_ok=True)


def clean_previous_outputs() -> None:
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name == "raw_downloads":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dirs()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "suppressed", "*"}:
        return None
    text = text.replace(",", "")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def period_end_from_start(period_start: str, granularity: str) -> str:
    date = dt.date.fromisoformat(period_start)
    if granularity == "year":
        return f"{date.year}-12-31"
    if granularity == "quarter":
        month = ((date.month - 1) // 3 + 1) * 3
        next_month = dt.date(date.year + (month == 12), 1 if month == 12 else month + 1, 1)
        return (next_month - dt.timedelta(days=1)).isoformat()
    if granularity == "month":
        next_month = dt.date(date.year + (date.month == 12), 1 if date.month == 12 else date.month + 1, 1)
        return (next_month - dt.timedelta(days=1)).isoformat()
    if granularity == "week":
        return (date + dt.timedelta(days=6)).isoformat()
    return period_start


def week_start(year: int, week: int) -> str:
    try:
        return dt.date.fromisocalendar(year, max(1, min(week, 53)), 1).isoformat()
    except ValueError:
        return dt.date(year, 1, 1).isoformat()


def q_start(year: int, quarter: int) -> str:
    month = max(1, min(4, quarter)) * 3 - 2
    return dt.date(year, month, 1).isoformat()


def stable_id(parts: Iterable[Any]) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def feature_record(
    *,
    domain: str,
    variable_id: str,
    value: float,
    unit: str,
    geography_level: str,
    geography_id: str,
    geography_name: str,
    observation_time: str,
    period_end: str | None = None,
    forecast_horizon: str = "current",
    source_id: str,
    source_name: str,
    source_url: str,
    source_timestamp: str | None = None,
    release_frequency: str,
    coverage_start: str,
    coverage_end: str,
    data_quality: str = "observed",
    missingness: str = "not_imputed",
    transformation: str = "source_native_numeric",
    dimensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    period_end = period_end or observation_time
    rid = stable_id([
        domain, variable_id, geography_level, geography_id, observation_time,
        period_end, forecast_horizon, source_id, json.dumps(dimensions or {}, sort_keys=True),
    ])
    return {
        "record_id": rid,
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "variable_id": variable_id,
        "value": float(value),
        "unit": unit,
        "geography_level": geography_level,
        "geography_id": geography_id,
        "geography_name": geography_name,
        "observation_time": observation_time,
        "period_end": period_end,
        "forecast_horizon": forecast_horizon,
        "source_id": source_id,
        "source_name": source_name,
        "source_url": source_url,
        "source_timestamp": source_timestamp or RUN_AT,
        "release_frequency": release_frequency,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "data_quality": data_quality,
        "missingness": missingness,
        "transformation": transformation,
        "dimensions_json": json.dumps(dimensions or {}, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
    }


FEATURE_FIELDS = [
    "record_id", "schema_version", "domain", "variable_id", "value", "unit",
    "geography_level", "geography_id", "geography_name", "observation_time", "period_end",
    "forecast_horizon", "source_id", "source_name", "source_url", "source_timestamp",
    "release_frequency", "coverage_start", "coverage_end", "data_quality", "missingness",
    "transformation", "dimensions_json",
]


class FeatureSink:
    def __init__(self, domain: str):
        self.domain = domain
        self.csv_path = FEATURES / domain / f"{domain}_features.csv.gz"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_fh = gzip.open(self.csv_path, "wt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.csv_fh, fieldnames=FEATURE_FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.count = 0
        self.seen_record_ids: set[str] = set()

    def write(self, record: dict[str, Any]) -> None:
        record_id = str(record.get("record_id") or "")
        if record_id in self.seen_record_ids:
            return
        self.seen_record_ids.add(record_id)
        self.writer.writerow(record)
        self.count += 1

    def close(self) -> None:
        self.csv_fh.close()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def fetch(url: str, label: str, suffix: str = ".json", refresh: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    path = RAW / f"{label}_{digest}{suffix}"
    if path.exists() and not refresh:
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "meditrack-final-data/1.0"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                path.write_bytes(resp.read())
            return path
        except Exception as exc:  # network and HTTP failures are recorded by caller
            last_error = exc
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def load_json_url(url: str, label: str, refresh: bool = False) -> tuple[Any, str]:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    gz_path = RAW / f"{label}_{digest}.json.gz"
    if gz_path.exists() and not refresh:
        with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
            return json.load(fh), str(gz_path.relative_to(OUT))
    path = fetch(url, label, ".json", refresh)
    return json.loads(path.read_text(encoding="utf-8")), str(path.relative_to(OUT))


def fetch_bytes_url(url: str, label: str, suffix: str, refresh: bool = False) -> tuple[bytes, str]:
    path = fetch(url, label, suffix, refresh)
    return path.read_bytes(), str(path.relative_to(OUT))


def socrata_pages(base_url: str, label: str, query: dict[str, str], refresh: bool = False) -> Iterator[tuple[list[dict[str, Any]], str]]:
    limit = 50000
    offset = 0
    while True:
        params = dict(query)
        params["$limit"] = str(limit)
        params["$offset"] = str(offset)
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        payload, raw = load_json_url(url, f"{label}_{offset}", refresh)
        if not isinstance(payload, list):
            break
        yield payload, raw
        if len(payload) < limit:
            break
        offset += limit


def normalize_identifier(text: Any) -> str:
    raw = str(text or "").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in raw]
    ident = "_".join("".join(chars).split("_"))
    return ident.strip("_") or "unknown"


def ingest_disease_us(sink: FeatureSink, stats: dict[str, Any]) -> None:
    source_dir = ROOT / "disease_us/data/by_source"
    paths = sorted(source_dir.glob("*/observations.jsonl"))
    for path in paths:
        if path.parent.name == "places_2022":
            stats["disease_us_places_2022_status"] = "skipped_in_final; replaced by current Arkansas county CDC PLACES 2024 download"
            continue
        before = sink.count
        for row in jsonl_rows(path):
            value = parse_float(row.get("value"))
            if value is None:
                continue
            period = row.get("period", {})
            year = int(period.get("year"))
            granularity = "week" if period.get("week") else "year"
            start = week_start(year, int(period.get("week") or 1)) if granularity == "week" else f"{year}-01-01"
            geo = row.get("geography", {})
            geo_id = str(geo.get("id") or "")
            geo_name = str(geo.get("name") or "")
            if geo_id.lower() not in {"ar", "nat", "us residents"} and geo_name.lower() not in {"ar", "arkansas", "nat", "us residents"}:
                continue
            prov = row.get("provenance", {})
            dims = dict(row.get("dimensions") or {})
            condition = dims.get("condition")
            if row.get("source_id") == "delphi_fluview":
                dims["disease_id"] = "influenza"
                dims["disease_name"] = "Influenza-like illness"
            elif condition:
                dims["disease_id"] = normalize_identifier(condition)
                dims["disease_name"] = condition
            sink.write(feature_record(
                domain="disease_us", variable_id=str(row.get("metric")), value=value, unit=str(row.get("unit") or "source_native"),
                geography_level=str(geo.get("level") or "unknown"), geography_id=str(geo.get("id") or ""),
                geography_name=str(geo.get("name") or ""), observation_time=start,
                period_end=period_end_from_start(start, granularity), source_id=str(row.get("source_id")),
                source_name=str(row.get("dataset_id")), source_url=str(prov.get("source_url") or ""),
                source_timestamp=str(prov.get("retrieved_at") or RUN_AT), release_frequency="weekly_or_annual",
                coverage_start="2012-01-01", coverage_end="2022-12-31",
                dimensions={"dataset_id": row.get("dataset_id"), **dims},
            ))
        stats["disease_us"] = stats.get("disease_us", 0) + (sink.count - before)


def download_cdc_covid_county_transmission(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    base = "https://data.cdc.gov/resource/nra9-vzzn.json"
    level_map = {"low": 1.0, "moderate": 2.0, "substantial": 3.0, "high": 4.0}
    try:
        query = {"state_name": "Arkansas", "$order": "date", "$limit": "50000"}
        for rows, raw_file in socrata_pages(base, "cdc_covid_county_transmission_ar", query, refresh):
            for row in rows:
                date = str(row.get("date") or "")[:10]
                fips = str(row.get("fips_code") or "")
                if not date or not fips.startswith("05"):
                    continue
                county_name = str(row.get("county_name") or "")
                dims = {
                    "disease_id": "covid_19",
                    "disease_name": "COVID-19",
                    "county_name": county_name,
                    "raw_file": raw_file,
                    "native_community_transmission_level": row.get("community_transmission_level"),
                }
                for source_col, variable_id, unit in [
                    ("cases_per_100k_7_day_count", "covid_cases_per_100k_7_day", "cases_per_100k_population"),
                    ("percent_test_results_reported", "covid_percent_test_results_reported", "percent"),
                ]:
                    value = parse_float(row.get(source_col))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="disease_us", variable_id=variable_id, value=value, unit=unit,
                        geography_level="county", geography_id=fips, geography_name=f"{county_name}, Arkansas",
                        observation_time=date, source_id="cdc_covid_county_transmission",
                        source_name="CDC United States COVID-19 County Level of Community Transmission Historical Data",
                        source_url="https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-County-Level-of-Community-T/nra9-vzzn",
                        source_timestamp=RUN_AT, release_frequency="daily",
                        coverage_start="2020-01-01", coverage_end="2022-10-20",
                        dimensions={**dims, "source_field": source_col},
                    ))
                level = str(row.get("community_transmission_level") or "").strip().lower()
                if level in level_map:
                    sink.write(feature_record(
                        domain="disease_us", variable_id="covid_community_transmission_level", value=level_map[level], unit="ordinal_level",
                        geography_level="county", geography_id=fips, geography_name=f"{county_name}, Arkansas",
                        observation_time=date, source_id="cdc_covid_county_transmission",
                        source_name="CDC United States COVID-19 County Level of Community Transmission Historical Data",
                        source_url="https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-County-Level-of-Community-T/nra9-vzzn",
                        source_timestamp=RUN_AT, release_frequency="daily",
                        coverage_start="2020-01-01", coverage_end="2022-10-20",
                        transformation="ordered_category_low_1_moderate_2_substantial_3_high_4",
                        dimensions=dims,
                    ))
    except Exception as exc:
        stats["cdc_covid_county_transmission_error"] = str(exc)
    stats["disease_us"] = stats.get("disease_us", 0) + (sink.count - before)


def ingest_disease_global(sink: FeatureSink, stats: dict[str, Any]) -> None:
    source_dir = ROOT / "disease_global/data/by_source"
    paths = sorted(source_dir.glob("*/observations.jsonl"))
    for path in paths:
        before = sink.count
        for row in jsonl_rows(path):
            value = parse_float(row.get("value"))
            if value is None:
                continue
            period = row.get("period", {})
            year = parse_float(period.get("year"))
            if year is None:
                continue
            week = period.get("week") or (int(str(period.get("epiweek"))[-2:]) if period.get("epiweek") else None)
            granularity = "week" if week else "year"
            start = week_start(int(year), int(week)) if week else f"{int(year)}-01-01"
            geo = row.get("geography", {})
            geo_id = str(geo.get("id") or "")
            geo_name = str(geo.get("name") or "")
            geo_level = str(geo.get("level") or "")
            if row.get("source_id") != "who_don" and geo_id.upper() not in {"USA", "US", "OWID_WRL"} and geo_name.lower() not in {"united states", "world", "global"}:
                continue
            if row.get("source_id") == "ecdc_surveillance_atlas" and geo_id.upper() not in {"USA", "US"}:
                continue
            prov = row.get("provenance", {})
            dims = dict(row.get("dimensions") or {})
            source_id = str(row.get("source_id") or "")
            dataset_id = str(row.get("dataset_id") or "")
            metric = str(row.get("metric") or "")
            if source_id in {"who_flunet", "who_fluid"}:
                dims["disease_id"] = "influenza"
                dims["disease_name"] = "Influenza"
            elif metric == "outbreak_event":
                dims["disease_id"] = normalize_identifier(dims.get("title") or dims.get("url_name") or "who_don_outbreak_event")
                dims["disease_name"] = dims.get("title") or "WHO Disease Outbreak News event"
            elif "immunization" in metric.lower() or "coverage" in metric.lower():
                dims["disease_id"] = normalize_identifier(metric)
                dims["disease_name"] = metric
            sink.write(feature_record(
                domain="disease_global", variable_id=str(row.get("metric")), value=value, unit=str(row.get("unit") or "source_native"),
                geography_level=geo_level or "unknown", geography_id=geo_id,
                geography_name=geo_name, observation_time=start,
                period_end=period_end_from_start(start, granularity), source_id=str(row.get("source_id")),
                source_name=str(row.get("dataset_id")), source_url=str(prov.get("source_url") or ""),
                source_timestamp=str(prov.get("retrieved_at") or RUN_AT), release_frequency="weekly_or_annual",
                coverage_start="2012-01-01", coverage_end="2022-12-31",
                dimensions={"dataset_id": dataset_id, **dims},
            ))
        stats["disease_global"] = stats.get("disease_global", 0) + (sink.count - before)


def ingest_economics_us(sink: FeatureSink, stats: dict[str, Any]) -> None:
    before = sink.count
    for path in sorted((ROOT / "economics_us/data/by_source").glob("*/*.jsonl")):
        for row in jsonl_rows(path):
            obs = row.get("observation", {})
            value = parse_float(obs.get("value"))
            if value is None:
                continue
            geo = row.get("geography", {})
            source = row.get("source", {})
            sink.write(feature_record(
                domain="economics_us", variable_id=str(obs.get("metric")), value=value, unit=str(obs.get("unit") or "source_native"),
                geography_level=str(geo.get("level") or "unknown"), geography_id=str(geo.get("code") or ""),
                geography_name=str(geo.get("name") or ""), observation_time=str(row.get("period_start")),
                period_end=str(row.get("period_end")), source_id=str(source.get("source_id")),
                source_name=str(source.get("source_name") or ""), source_url=str(source.get("source_url") or ""),
                source_timestamp=str(source.get("retrieved_at") or RUN_AT), release_frequency=str(row.get("granularity") or "source_native"),
                coverage_start="2012-01-01", coverage_end="2022-12-31",
                dimensions={"record_id": row.get("record_id"), "semantics": obs.get("semantics"), "vintage_date": source.get("vintage_date")},
            ))
    stats["economics_us"] = sink.count - before


def ingest_demand_price(sink: FeatureSink, event_rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    base = ROOT / "demand_price/data"
    before = sink.count
    source_map = {
        "medicaid_sdud_state_quarter.csv.gz": ("medicaid_sdud", "Medicaid State Drug Utilization Data", "https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data", "quarter"),
        "cms_partd_geography_drug.csv.gz": ("cms_partd", "CMS Part D Prescription Drug Profiles", "https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/prescription-drug-profiles-puf", "year"),
    }
    numeric_columns = {
        "medicaid_sdud_state_quarter.csv.gz": ["source_rows", "non_suppressed_rows", "suppressed_rows", "units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed", "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed"],
        "cms_partd_geography_drug.csv.gz": ["tot_prscrbrs", "tot_clms", "tot_30day_fills", "tot_drug_cst", "tot_benes", "ge65_tot_clms", "ge65_tot_30day_fills", "ge65_tot_drug_cst", "ge65_tot_benes", "lis_bene_cst_shr", "nonlis_bene_cst_shr"],
    }
    for filename, (source_id, source_name, source_url, freq) in source_map.items():
        path = base / filename
        if not path.exists():
            continue
        with open_csv(path) as fh:
            reader = csv.DictReader(fh)
            cols = numeric_columns[filename] or [c for c in (reader.fieldnames or []) if c.lower() not in {"ndc", "package_ndc"}]
            for row in reader:
                year = int(parse_float(row.get("source_year") or row.get("year") or "0") or 0)
                if filename == "medicaid_sdud_state_quarter.csv.gz":
                    if row.get("state") != "AR":
                        continue
                    start = q_start(year, int(parse_float(row.get("quarter")) or 1))
                    geo_level = "state"
                    geo_id = row.get("state") or ""
                    geo_name = geo_id
                    dims = {"utilization_type": row.get("utilization_type")}
                elif filename == "cms_partd_geography_drug.csv.gz":
                    lvl = str(row.get("prscrbr_geo_lvl") or "")
                    desc = str(row.get("prscrbr_geo_desc") or "")
                    code = str(row.get("prscrbr_geo_cd") or "")
                    if desc.lower() not in {"national", "arkansas"} and code.upper() != "AR":
                        continue
                    start = f"{year}-01-01"
                    geo_level = str(row.get("prscrbr_geo_lvl") or "unknown").lower()
                    geo_id = row.get("prscrbr_geo_cd") or row.get("prscrbr_geo_desc") or ""
                    geo_name = row.get("prscrbr_geo_desc") or geo_id
                    dims = {k: row.get(k) for k in ["brnd_name", "gnrc_name", "opioid_drug_flag", "opioid_la_drug_flag", "antbtc_drug_flag", "antpsyct_drug_flag"]}
                if not start[:4].isdigit():
                    continue
                for col in cols:
                    value = parse_float(row.get(col))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="demand_price", variable_id=col, value=value, unit="source_native",
                        geography_level=geo_level, geography_id=geo_id, geography_name=geo_name,
                        observation_time=start, period_end=period_end_from_start(start, freq if freq in {"week", "month", "quarter", "year"} else "year"),
                        source_id=source_id, source_name=source_name, source_url=source_url,
                        release_frequency=freq, coverage_start="2012-01-01", coverage_end="2022-12-31",
                        dimensions=dims,
                    ))
    nadac_path = base / "nadac_ndc_weekly.csv.gz"
    if nadac_path.exists():
        aggregates: dict[tuple[str, str, str], dict[str, float]] = {}
        with open_csv(nadac_path) as fh:
            for row in csv.DictReader(fh):
                date = parse_source_date(row.get("effective_date") or "")
                price = parse_float(row.get("nadac_per_unit"))
                if not date or price is None:
                    continue
                month = date[:7] + "-01"
                key = (month, row.get("classification_for_rate_setting") or "unknown", row.get("pricing_unit") or "unknown")
                agg = aggregates.setdefault(key, {"sum": 0.0, "count": 0.0, "min": price, "max": price})
                agg["sum"] += price
                agg["count"] += 1.0
                agg["min"] = min(agg["min"], price)
                agg["max"] = max(agg["max"], price)
        for (month, classification, pricing_unit), agg in sorted(aggregates.items()):
            dims = {"classification_for_rate_setting": classification, "pricing_unit": pricing_unit, "source_grain": "NDC weekly rows aggregated to month"}
            for var, value in [
                ("nadac_per_unit_mean", agg["sum"] / agg["count"]),
                ("nadac_per_unit_min", agg["min"]),
                ("nadac_per_unit_max", agg["max"]),
                ("nadac_ndc_price_row_count", agg["count"]),
            ]:
                sink.write(feature_record(
                    domain="demand_price", variable_id=var, value=value,
                    unit="usd_per_unit" if var != "nadac_ndc_price_row_count" else "rows",
                    geography_level="national", geography_id="US", geography_name="United States",
                    observation_time=month, period_end=period_end_from_start(month, "month"),
                    source_id="nadac", source_name="National Average Drug Acquisition Cost",
                    source_url="https://data.medicaid.gov/dataset/nadac-national-average-drug-acquisition-cost",
                    release_frequency="weekly", coverage_start="2013-01-01", coverage_end="2022-12-31",
                    transformation="monthly_aggregate_from_ndc_weekly_rows", dimensions=dims,
                ))
    for filename, source_id, source_name, url, date_col, status_col in [
        ("openfda_shortages_2012_2022.csv.gz", "openfda_shortages", "FDA Drug Shortages", "https://open.fda.gov/apis/drug/drugshortages/", "initial_posting_date", "status"),
        ("openfda_enforcement_2012_2022.csv.gz", "openfda_enforcement", "FDA Enforcement / Recalls", "https://open.fda.gov/apis/drug/enforcement/", "recall_initiation_date", "classification"),
    ]:
        path = base / filename
        if not path.exists():
            continue
        with open_csv(path) as fh:
            for row in csv.DictReader(fh):
                date_raw = row.get(date_col) or ""
                start = parse_source_date(date_raw)
                if not start:
                    continue
                status = row.get(status_col) or ""
                severity = recall_or_shortage_severity(status)
                event_id = stable_id([
                    source_id,
                    row.get("event_id"),
                    row.get("recall_number"),
                    row.get("package_ndc"),
                    row.get("generic_name") or row.get("product_description"),
                    row.get("company_name") or row.get("recalling_firm") or row.get("manufacturer"),
                    start,
                    status,
                ])
                event_rows.append({
                    "event_id": event_id,
                    "event_type": "drug_shortage" if source_id == "openfda_shortages" else "product_recall",
                    "entity_id": row.get("package_ndc") or row.get("recall_number") or row.get("generic_name") or "",
                    "geography": row.get("state") or row.get("country") or "US",
                    "start_time": start,
                    "reported_time": row.get("update_date") or row.get("report_date") or start,
                    "severity": severity,
                    "confidence": 1.0,
                    "expected_duration": "",
                    "source_id": source_id,
                    "affected_entities_json": json.dumps([row.get("company_name") or row.get("recalling_firm") or ""], ensure_ascii=True),
                    "affected_drugs_json": json.dumps([row.get("generic_name") or row.get("product_description") or ""], ensure_ascii=True),
                    "affected_diseases_json": "[]",
                    "details_json": json.dumps(row, ensure_ascii=True, sort_keys=True),
                })
                sink.write(feature_record(
                    domain="demand_price", variable_id=("shortage_active" if source_id == "openfda_shortages" else "recall_active"),
                    value=1.0 if severity > 0 else 0.0, unit="indicator", geography_level="national", geography_id="US", geography_name="United States",
                    observation_time=start, source_id=source_id, source_name=source_name, source_url=url,
                    release_frequency="daily_or_weekly", coverage_start="2012-01-01", coverage_end="2022-12-31",
                    transformation="categorical_status_to_indicator", dimensions={"status": status, "event_id": event_id},
                ))
    stats["demand_price"] = sink.count - before


def download_openfda_recent_supply(sink: FeatureSink, event_rows: list[dict[str, Any]], stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    added_events = 0
    sources = [
        ("openfda_shortages_recent", "FDA Drug Shortages", "https://api.fda.gov/drug/shortages.json", "initial_posting_date", "drug_shortage", "shortage_active"),
        ("openfda_enforcement_recent", "FDA Enforcement / Recalls", "https://api.fda.gov/drug/enforcement.json", "recall_initiation_date", "product_recall", "recall_active"),
    ]
    try:
        for source_id, source_name, base, date_field, event_type, variable_id in sources:
            skip = 0
            while True:
                if source_id == "openfda_enforcement_recent":
                    params = {"search": "recall_initiation_date:[20230101+TO+20261231]", "limit": "1000", "skip": str(skip)}
                else:
                    params = {"limit": "1000", "skip": str(skip)}
                url = f"{base}?{urllib.parse.urlencode(params, safe=':+[]')}"
                payload, raw_file = load_json_url(url, f"{source_id}_{skip}", refresh)
                rows = payload.get("results", [])
                if not rows:
                    break
                for row in rows:
                    start = parse_source_date(str(row.get(date_field) or ""))
                    if not start or int(start[:4]) < 2023:
                        continue
                    status = str(row.get("status") or row.get("classification") or "")
                    severity = recall_or_shortage_severity(status)
                    event_id = stable_id([source_id, row.get("event_id"), row.get("recall_number"), row.get("package_ndc"), row.get("generic_name"), start])
                    event_rows.append({
                        "event_id": event_id,
                        "event_type": event_type,
                        "entity_id": str(row.get("package_ndc") or row.get("recall_number") or row.get("generic_name") or ""),
                        "geography": str(row.get("state") or row.get("country") or "US"),
                        "start_time": start,
                        "reported_time": str(row.get("update_date") or row.get("report_date") or start),
                        "severity": severity,
                        "confidence": 1.0,
                        "expected_duration": "",
                        "source_id": source_id,
                        "affected_entities_json": json.dumps([row.get("company_name") or row.get("recalling_firm") or ""], ensure_ascii=True),
                        "affected_drugs_json": json.dumps([row.get("generic_name") or row.get("product_description") or ""], ensure_ascii=True),
                        "affected_diseases_json": "[]",
                        "details_json": json.dumps(row, ensure_ascii=True, sort_keys=True),
                    })
                    added_events += 1
                    sink.write(feature_record(
                        domain="demand_price", variable_id=variable_id, value=1.0 if severity > 0 else 0.0,
                        unit="indicator", geography_level="national", geography_id="US", geography_name="United States",
                        observation_time=start, source_id=source_id, source_name=source_name, source_url=base,
                        source_timestamp=str(payload.get("meta", {}).get("last_updated") or RUN_AT), release_frequency="daily_or_weekly",
                        coverage_start="2023-01-01", coverage_end=f"{CURRENT_YEAR}-12-31",
                        transformation="categorical_status_to_indicator", dimensions={"status": status, "event_id": event_id, "raw_file": raw_file},
                    ))
                if len(rows) < 1000:
                    break
                skip += 1000
    except Exception as exc:
        stats["openfda_recent_error"] = str(exc)
    stats["openfda_recent_events_added"] = added_events
    stats["demand_price"] = stats.get("demand_price", 0) + (sink.count - before)


def parse_source_date(text: str) -> str | None:
    text = str(text or "").strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text[:10], fmt).date().isoformat()
        except ValueError:
            pass
    return None


def recall_or_shortage_severity(status: str) -> float:
    text = status.lower()
    if "class i" in text and "class ii" not in text and "class iii" not in text:
        return 1.0
    if "class ii" in text:
        return 0.66
    if "class iii" in text:
        return 0.33
    if "current" in text or "ongoing" in text:
        return 0.75
    if "resolved" in text or "terminated" in text or "discontinued" in text:
        return 0.0
    return 0.5


def ingest_supply_chain(sink: FeatureSink, stats: dict[str, Any]) -> None:
    before = sink.count
    for path in sorted((ROOT / "supply_chain/data").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                period = row.get("period") or row.get("date") or row.get("month") or row.get("year") or ""
                if len(period) == 7:
                    start = f"{period}-01"
                    gran = "month"
                elif len(period) == 4 and period.isdigit():
                    start = f"{period}-01-01"
                    gran = "year"
                else:
                    continue
                for col, raw in row.items():
                    if col in {"period", "date", "month", "year", "latest_vintage"}:
                        continue
                    value = parse_float(raw)
                    if value is None:
                        continue
                    dims = {
                        "source_file": str(path.relative_to(ROOT)),
                        "latest_vintage": row.get("latest_vintage"),
                        "source_column": col,
                        "row_context": {k: v for k, v in row.items() if k not in {"period", "date", "month", "year", col}},
                    }
                    sink.write(feature_record(
                        domain="supply_chain", variable_id=col, value=value, unit="source_native",
                        geography_level="global", geography_id="GLOBAL", geography_name="Global",
                        observation_time=start, period_end=period_end_from_start(start, gran),
                        source_id=path.stem, source_name=path.stem.replace("_", " "), source_url="see supply_chain/sources.md",
                        release_frequency=gran, coverage_start="2012-01-01", coverage_end="2022-12-31",
                        dimensions=dims,
                    ))
    stats["supply_chain"] = sink.count - before


def ingest_trade(sink: FeatureSink, stats: dict[str, Any]) -> None:
    before = sink.count
    us_code = "842"  # CEPII BACI country_codes_V202601.csv maps 842 -> USA.
    country_names: dict[str, str] = {}
    product_names: dict[str, str] = {}
    zpath = ROOT / "trade/cache/BACI_HS92_V202601.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            if "country_codes_V202601.csv" in zf.namelist():
                with zf.open("country_codes_V202601.csv") as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                        country_names[row.get("country_code") or ""] = row.get("country_name") or row.get("country_iso3") or ""
            if "product_codes_HS92_V202601.csv" in zf.namelist():
                with zf.open("product_codes_HS92_V202601.csv") as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                        product_names[row.get("code") or ""] = row.get("description") or ""
    baci_paths = sorted((ROOT / "trade/data/baci").glob("baci_hs92_*.csv"))
    available_years = {int(path.stem.split("_")[-1]): path for path in baci_paths}

    def trade_rows() -> Iterator[dict[str, str]]:
        for year in range(2000, 2025):
            path = available_years.get(year)
            if path:
                with path.open(encoding="utf-8", newline="") as fh:
                    yield from csv.DictReader(fh)
            elif zpath.exists():
                member = f"BACI_HS92_Y{year}_V202601.csv"
                with zipfile.ZipFile(zpath) as zf:
                    if member not in zf.namelist():
                        continue
                    with zf.open(member) as raw:
                        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))

    partner_import_totals: dict[tuple[int, str], float] = {}
    partner_export_totals: dict[tuple[int, str], float] = {}
    retained: list[dict[str, str]] = []
    for row in trade_rows():
        hs = str(row.get("k") or "")
        if not hs.startswith(PHARMA_HS_PREFIXES) and hs[:4] not in PHARMA_HS_CODES:
            continue
        if row.get("i") != us_code and row.get("j") != us_code:
            continue
        retained.append(row)
        year = int(row.get("t") or 0)
        value = parse_float(row.get("v")) or 0.0
        if row.get("j") == us_code:
            partner_import_totals[(year, row.get("i") or "")] = partner_import_totals.get((year, row.get("i") or ""), 0.0) + value
        if row.get("i") == us_code:
            partner_export_totals[(year, row.get("j") or "")] = partner_export_totals.get((year, row.get("j") or ""), 0.0) + value

    total_imports = {year: sum(value for (y, _), value in partner_import_totals.items() if y == year) for year in range(2012, 2025)}
    total_exports = {year: sum(value for (y, _), value in partner_export_totals.items() if y == year) for year in range(2012, 2025)}

    for row in retained:
                hs = str(row.get("k") or "")
                year = int(row.get("t") or 0)
                start = f"{year}-01-01"
                is_import = row.get("j") == us_code
                partner = row.get("i") if is_import else row.get("j")
                partner_name = country_names.get(partner or "", partner or "")
                flow = "import" if is_import else "export"
                value_total = total_imports.get(year, 0.0) if is_import else total_exports.get(year, 0.0)
                dimensions = {
                    "exporter_code": row.get("i"),
                    "importer_code": row.get("j"),
                    "origin_country": country_names.get(row.get("i") or "", row.get("i") or ""),
                    "destination_country": country_names.get(row.get("j") or "", row.get("j") or ""),
                    "partner_country_code": partner,
                    "partner_country": partner_name,
                    "hs_code": hs,
                    "hs_description": product_names.get(hs, ""),
                    "flow": flow,
                    "geographic_resolution": "national_us_partner_country",
                }
                for col, var, unit in [
                    ("v", f"pharmaceutical_{flow}_value", "thousand_usd"),
                    ("q", f"pharmaceutical_{flow}_volume", "tonnes"),
                ]:
                    value = parse_float(row.get(col))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="trade", variable_id=var, value=value, unit=unit, geography_level="country",
                        geography_id=partner or "", geography_name=partner_name,
                        observation_time=start, period_end=f"{year}-12-31", source_id="cepii_baci",
                        source_name="CEPII BACI HS92", source_url="https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37",
                        release_frequency="annual", coverage_start="2000-01-01", coverage_end="2024-12-31",
                        dimensions=dimensions,
                    ))
                share_value = (parse_float(row.get("v")) or 0.0) / value_total if value_total else None
                if share_value is not None:
                    sink.write(feature_record(
                        domain="trade", variable_id=f"country_{flow}_share", value=share_value, unit="share",
                        geography_level="country", geography_id=partner or "", geography_name=partner_name,
                        observation_time=start, period_end=f"{year}-12-31", source_id="cepii_baci",
                        source_name="CEPII BACI HS92", source_url="https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37",
                        release_frequency="annual", coverage_start="2000-01-01", coverage_end="2024-12-31",
                        transformation="partner_hs_value_divided_by_total_us_pharmaceutical_flow_value",
                        dimensions=dimensions,
                    ))
    stats["trade"] = sink.count - before
    if stats["trade"] == 0:
        stats["trade_status"] = "BACI files were available, but no CEPII USA code 842 pharmaceutical HS rows were retained."


def ingest_policy(sink: FeatureSink, stats: dict[str, Any]) -> None:
    before = sink.count
    path = ROOT / "tariff_policy/data/wits_trains_tariffs.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                value = parse_float(row.get("measure_rate"))
                if value is None:
                    continue
                sink.write(feature_record(
                    domain="policy", variable_id="pharmaceutical_tariff_rate", value=value, unit=row.get("measure_unit") or "percent_or_source_native",
                    geography_level=row.get("reporter_level") or "country", geography_id=row.get("reporter_code") or "",
                    geography_name=row.get("reporter_name") or row.get("reporter_code") or "",
                    observation_time=row.get("period_start") or "", period_end=row.get("period_end") or "",
                    source_id=row.get("source_id") or "wits_trains", source_name=row.get("source_name") or "WITS / UNCTAD TRAINS",
                    source_url=row.get("source_url") or "https://wits.worldbank.org/",
                    source_timestamp=row.get("retrieved_at") or RUN_AT, release_frequency="annual",
                    coverage_start="2012-01-01", coverage_end="2022-12-31",
                    dimensions={k: row.get(k) for k in ["partner_code", "product_code", "product_revision", "measure_subtype", "product_description"]},
                ))
    stats["policy"] = sink.count - before


def ingest_crosswalk_entities(stats: dict[str, Any]) -> None:
    copied = 0
    for src, dest in [
        (ROOT / "crosswalks/data/dartmouth_zip_hsa_hrr.csv", ENTITIES / "dartmouth_zip_hsa_hrr.csv.gz"),
        (ROOT / "crosswalks/data/rxnav_ndc_catalog.csv", ENTITIES / "rxnav_ndc_catalog.csv.gz"),
        (ROOT / "crosswalks/data/gleif_lei_records.jsonl", ENTITIES / "gleif_lei_records.jsonl"),
        (ROOT / "S_D/data/drug_dictionary.json", ENTITIES / "drug_dictionary.json"),
    ]:
        if src.exists():
            if dest.suffix == ".gz":
                with src.open("rb") as inf, gzip.open(dest, "wb") as outf:
                    shutil.copyfileobj(inf, outf)
            else:
                shutil.copy2(src, dest)
            copied += 1
    stats["entity_files_copied"] = copied


def download_acs_arkansas(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    fields = {
        "DP05_0001E": ("population_total", "persons"),
        "DP05_0024PE": ("population_age_65_plus_pct", "percent"),
        "DP03_0062E": ("median_household_income", "usd"),
        "DP03_0128PE": ("poverty_rate", "percent"),
        "DP03_0096PE": ("uninsured_rate", "percent"),
    }
    get_fields = "NAME," + ",".join(fields)
    url = f"https://api.census.gov/data/2024/acs/acs5/profile?get={get_fields}&for=county:*&in=state:05"
    before = sink.count
    try:
        payload, raw_file = load_json_url(url, "census_acs_2024_ar_counties", refresh)
        header = payload[0]
        for values in payload[1:]:
            row = dict(zip(header, values))
            county = row.get("county", "")
            for field, (var, unit) in fields.items():
                value = parse_float(row.get(field))
                if value is None:
                    continue
                sink.write(feature_record(
                    domain="population", variable_id=var, value=value, unit=unit, geography_level="county",
                    geography_id=f"05{county}", geography_name=row.get("NAME", ""),
                    observation_time="2024-01-01", period_end="2024-12-31", source_id="census_acs_2024",
                    source_name="Census ACS 5-year profile 2024", source_url=url, source_timestamp=RUN_AT,
                    release_frequency="annual", coverage_start="2024-01-01", coverage_end="2024-12-31",
                    dimensions={"raw_file": raw_file, "field": field},
                ))
    except Exception as exc:
        stats["census_acs_error"] = str(exc)
    stats["population"] = stats.get("population", 0) + (sink.count - before)


def download_census_county_population_estimates(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    urls = [
        ("2010_2020", "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv", range(2010, 2020)),
        ("2020_2025", "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv", range(2020, 2026)),
    ]
    variable_columns = [
        ("POPESTIMATE", "population_total", "persons"),
        ("NPOPCHG", "population_growth_abs", "persons"),
        ("BIRTHS", "births", "persons"),
        ("DEATHS", "deaths", "persons"),
        ("NATURALINC", "natural_change", "persons"),
        ("NATURALCHG", "natural_change", "persons"),
        ("INTERNATIONALMIG", "international_migration", "persons"),
        ("DOMESTICMIG", "domestic_migration", "persons"),
        ("NETMIG", "net_migration", "persons"),
        ("RBIRTH", "birth_rate", "per_1000_population"),
        ("RDEATH", "death_rate", "per_1000_population"),
        ("RNATURALCHG", "natural_change_rate", "per_1000_population"),
        ("RNETMIG", "net_migration_rate", "per_1000_population"),
    ]
    region_totals: dict[tuple[str, int, str], float] = collections.defaultdict(float)
    region_sources: dict[tuple[str, int, str], set[str]] = collections.defaultdict(set)
    county_pops: dict[tuple[str, int], float] = {}
    try:
        for vintage, url, years in urls:
            body, raw_file = fetch_bytes_url(url, f"census_pep_county_totals_{vintage}", ".csv", refresh)
            reader = csv.DictReader(io.StringIO(body.decode("latin1")))
            for row in reader:
                if row.get("SUMLEV") != "050" or row.get("STATE") != "05":
                    continue
                county_fips = f"05{str(row.get('COUNTY') or '').zfill(3)}"
                county_name = str(row.get("CTYNAME") or "")
                region_id = COUNTY_TO_REGION.get(county_name.lower())
                for year in years:
                    date = f"{year}-01-01"
                    pop = parse_float(row.get(f"POPESTIMATE{year}"))
                    if pop is not None:
                        county_pops[(county_fips, year)] = pop
                    for prefix, variable_id, unit in variable_columns:
                        value = parse_float(row.get(f"{prefix}{year}") or row.get(f"{prefix}_{year}"))
                        if value is None:
                            continue
                        sink.write(feature_record(
                            domain="population", variable_id=variable_id, value=value, unit=unit,
                            geography_level="county", geography_id=county_fips, geography_name=f"{county_name}, AR",
                            observation_time=date, period_end=f"{year}-12-31", source_id="census_county_population_estimates",
                            source_name="U.S. Census Bureau County Population Estimates",
                            source_url=url, source_timestamp=RUN_AT, release_frequency="annual",
                            coverage_start="2010-01-01", coverage_end="2025-12-31",
                            dimensions={"vintage": vintage, "raw_file": raw_file, "county_name": county_name},
                        ))
                        if region_id and unit == "persons":
                            region_totals[(region_id, year, variable_id)] += value
                            region_sources[(region_id, year, variable_id)].add(county_fips)
        for (region_id, year, variable_id), value in sorted(region_totals.items()):
            sink.write(feature_record(
                domain="population", variable_id=variable_id, value=value, unit="persons",
                geography_level="region", geography_id=region_id, geography_name=REGION_NAMES.get(region_id, region_id),
                observation_time=f"{year}-01-01", period_end=f"{year}-12-31",
                source_id="census_county_population_estimates",
                source_name="U.S. Census Bureau County Population Estimates",
                source_url="https://www.census.gov/programs-surveys/popest.html",
                source_timestamp=RUN_AT, release_frequency="annual",
                coverage_start="2010-01-01", coverage_end="2025-12-31",
                transformation="sum_of_county_observations_in_arkansas_edd_region",
                dimensions={
                    "region_definition": "Arkansas planning and development districts recognized in AR Code section 14-166-202",
                    "county_fips": sorted(region_sources[(region_id, year, variable_id)]),
                },
            ))
    except Exception as exc:
        stats["census_pep_error"] = str(exc)
    stats["population"] = stats.get("population", 0) + (sink.count - before)


def download_places_2024_county(sink: FeatureSink, population_sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    measures = {
        "diabetes": "diabetes_prevalence",
        "bphigh": "hypertension_prevalence",
        "highchol": "hyperlipidemia_prevalence",
        "chd": "coronary_heart_disease_prevalence",
        "stroke": "stroke_prevalence",
        "casthma": "asthma_prevalence",
        "copd": "copd_prevalence",
        "depression": "depression_prevalence",
        "kidney": "chronic_kidney_disease_prevalence",
        "arthritis": "arthritis_prevalence",
        "obesity": "obesity_prevalence",
        "csmoking": "tobacco_use_prevalence",
    }
    before = sink.count
    base = "https://data.cdc.gov/resource/d3i6-k6z5.json"
    where = "stateabbr='AR'"
    try:
        for payload, raw_file in socrata_pages(base, "cdc_places_2024_ar_county", {"$where": where}, refresh):
            for row in payload:
                county_fips = str(row.get("countyfips") or "")
                county_name = str(row.get("countyname") or "")
                pop = parse_float(row.get("totalpopulation"))
                if pop is not None:
                    population_sink.write(feature_record(
                        domain="population", variable_id="population_total", value=pop, unit="persons",
                        geography_level="county", geography_id=county_fips, geography_name=county_name,
                        observation_time="2022-01-01", period_end="2022-12-31",
                        source_id="cdc_places_2024", source_name="CDC PLACES county release 2024",
                        source_url="https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2024-releas/d3i6-k6z5",
                        source_timestamp=RUN_AT, release_frequency="annual", coverage_start="2022-01-01", coverage_end="2022-12-31",
                        dimensions={"raw_file": raw_file, "source_field": "totalpopulation"},
                    ))
                for prefix, variable_id in measures.items():
                    value = parse_float(row.get(f"{prefix}_crudeprev"))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="chronic_health", variable_id=variable_id, value=value, unit="percent",
                        geography_level="county", geography_id=county_fips, geography_name=county_name,
                        observation_time="2022-01-01", period_end="2022-12-31",
                        source_id="cdc_places_2024", source_name="CDC PLACES county release 2024",
                        source_url="https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2024-releas/d3i6-k6z5",
                        source_timestamp=RUN_AT, release_frequency="annual", coverage_start="2022-01-01", coverage_end="2022-12-31",
                        dimensions={"raw_file": raw_file, "source_field": f"{prefix}_crudeprev", "confidence_interval": row.get(f"{prefix}_crude95ci")},
                    ))
    except Exception as exc:
        stats["cdc_places_2024_error"] = str(exc)
    stats["chronic_health"] = stats.get("chronic_health", 0) + (sink.count - before)
    stats["population"] = population_sink.count


def download_fema_arkansas(sink: FeatureSink, event_rows: list[dict[str, Any]], stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    base = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
    params = {
        "$filter": "state eq 'AR' and declarationDate ge '2012-01-01T00:00:00.000z'",
        "$top": "5000",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    try:
        payload, raw_file = load_json_url(url, "fema_disaster_declarations_ar_2012_present", refresh)
        for row in payload.get("DisasterDeclarationsSummaries", []):
            date = str(row.get("declarationDate") or "")[:10]
            if not date:
                continue
            event_id = stable_id(["fema", row.get("disasterNumber"), row.get("designatedArea"), date])
            severity = 1.0 if row.get("declarationType") == "DR" else 0.5
            event_rows.append({
                "event_id": event_id,
                "event_type": "natural_disaster",
                "entity_id": str(row.get("disasterNumber") or ""),
                "geography": f"AR:{row.get('designatedArea') or ''}",
                "start_time": date,
                "reported_time": date,
                "severity": severity,
                "confidence": 1.0,
                "expected_duration": "",
                "source_id": "fema_openfema",
                "affected_entities_json": "[]",
                "affected_drugs_json": "[]",
                "affected_diseases_json": "[]",
                "details_json": json.dumps(row, ensure_ascii=True, sort_keys=True),
            })
            sink.write(feature_record(
                domain="disasters", variable_id="disaster_active", value=1.0, unit="indicator",
                geography_level="county_or_area", geography_id=str(row.get("fipsCountyCode") or row.get("designatedArea") or ""),
                geography_name=str(row.get("designatedArea") or ""), observation_time=date,
                source_id="fema_openfema", source_name="FEMA Disaster Declarations Summaries",
                source_url="https://www.fema.gov/about/openfema/disaster-declarations-summaries",
                source_timestamp=RUN_AT, release_frequency="event", coverage_start="2012-01-01", coverage_end=RUN_AT[:10],
                transformation="declaration_to_indicator", dimensions={"raw_file": raw_file, "event_id": event_id,
                "incident_type": row.get("incidentType"), "declaration_type": row.get("declarationType")},
            ))
            sink.write(feature_record(
                domain="disasters", variable_id="disaster_severity", value=severity, unit="index_0_1",
                geography_level="county_or_area", geography_id=str(row.get("fipsCountyCode") or row.get("designatedArea") or ""),
                geography_name=str(row.get("designatedArea") or ""), observation_time=date,
                source_id="fema_openfema", source_name="FEMA Disaster Declarations Summaries",
                source_url="https://www.fema.gov/about/openfema/disaster-declarations-summaries",
                source_timestamp=RUN_AT, release_frequency="event", coverage_start="2012-01-01", coverage_end=RUN_AT[:10],
                transformation="declaration_type_to_severity", dimensions={"raw_file": raw_file, "event_id": event_id,
                "incident_type": row.get("incidentType"), "declaration_type": row.get("declarationType")},
            ))
    except Exception as exc:
        stats["fema_error"] = str(exc)
    stats["disasters"] = stats.get("disasters", 0) + (sink.count - before)


def download_noaa_weather(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    stations = {
        "USW00013963": "Little Rock / Central Arkansas",
        "USW00013964": "Fort Smith / West Arkansas",
        "USW00003952": "North Little Rock / Central Arkansas",
        "USW00013971": "Harrison / North Arkansas",
        "USW00093992": "El Dorado / South Arkansas",
        "USW00053869": "Blytheville / Northeast Arkansas",
        "USC00034666": "Marshall / Ozark region",
        "USC00032444": "Fayetteville / Northwest Arkansas",
    }
    fields = {
        "TMAX": ("temperature_max", "degrees_fahrenheit"),
        "TMIN": ("temperature_min", "degrees_fahrenheit"),
        "PRCP": ("precipitation", "inches"),
        "SNOW": ("snowfall", "inches"),
        "SNWD": ("snow_depth", "inches"),
        "AWND": ("wind", "miles_per_hour"),
        "RHAV": ("humidity", "percent"),
    }
    before_daily = sink.count
    try:
        for station, station_region in stations.items():
            for year in range(EXTENDED_START_YEAR, CURRENT_YEAR + 1):
                params = urllib.parse.urlencode({
                    "dataset": "daily-summaries",
                    "stations": station,
                    "startDate": f"{year}-01-01",
                    "endDate": f"{year}-12-31",
                    "format": "json",
                    "units": "standard",
                    "includeAttributes": "false",
                    "includeStationName": "true",
                    "includeStationLocation": "true",
                })
                url = f"https://www.ncei.noaa.gov/access/services/data/v1?{params}"
                payload, raw_file = load_json_url(url, f"noaa_daily_summaries_{station.lower()}_{year}", refresh)
                for row in payload:
                    date = row.get("DATE")
                    if not date:
                        continue
                    dims = {"station": station, "station_region": station_region, "station_name": row.get("NAME"), "latitude": row.get("LATITUDE"), "longitude": row.get("LONGITUDE"), "raw_file": raw_file}
                    tmax = parse_float(row.get("TMAX"))
                    tmin = parse_float(row.get("TMIN"))
                    if tmax is not None and tmin is not None:
                        for variable_id, value, unit, transformation in [
                            ("temperature_mean", (tmax + tmin) / 2.0, "degrees_fahrenheit", "mean_of_tmax_tmin"),
                            ("extreme_heat_day", 1.0 if tmax >= 95.0 else 0.0, "indicator", "threshold_tmax_ge_95f"),
                            ("extreme_cold_day", 1.0 if tmin <= 20.0 else 0.0, "indicator", "threshold_tmin_le_20f"),
                        ]:
                            sink.write(feature_record(
                                domain="environment", variable_id=variable_id, value=value, unit=unit,
                                geography_level="station", geography_id=station, geography_name=str(row.get("NAME") or station),
                                observation_time=date, source_id="noaa_ncei_daily_summaries", source_name="NOAA NCEI Daily Summaries",
                                source_url="https://www.ncei.noaa.gov/access/services/data/v1",
                                source_timestamp=RUN_AT, release_frequency="daily", coverage_start=f"{EXTENDED_START_YEAR}-01-01", coverage_end=f"{CURRENT_YEAR}-12-31",
                                transformation=transformation, dimensions=dims,
                            ))
                    for native, (var, unit) in fields.items():
                        value = parse_float(row.get(native))
                        if value is None:
                            continue
                        sink.write(feature_record(
                            domain="environment", variable_id=var, value=value, unit=unit,
                            geography_level="station", geography_id=station, geography_name=str(row.get("NAME") or station),
                            observation_time=date, source_id="noaa_ncei_daily_summaries", source_name="NOAA NCEI Daily Summaries",
                            source_url="https://www.ncei.noaa.gov/access/services/data/v1", source_timestamp=RUN_AT,
                            release_frequency="daily", coverage_start=f"{EXTENDED_START_YEAR}-01-01", coverage_end=f"{CURRENT_YEAR}-12-31",
                            dimensions=dims,
                        ))
    except Exception as exc:
        stats["noaa_weather_error"] = str(exc)
    stats["environment_weather"] = sink.count - before_daily
    stats["environment"] = sink.count - before


def download_epa_air_quality(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    try:
        for year in range(EXTENDED_START_YEAR, 2026):
            url = f"https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_{year}.zip"
            body, raw_file = fetch_bytes_url(url, f"epa_annual_aqi_by_county_{year}", ".zip", refresh)
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                    for row in reader:
                        if row.get("State Code") != "05":
                            continue
                        county = f"05{str(row.get('County Code') or '').zfill(3)}"
                        date = f"{year}-01-01"
                        for col, var, unit in [
                            ("Days with AQI", "aqi_days_observed", "days"),
                            ("Good Days", "aqi_good_days", "days"),
                            ("Moderate Days", "aqi_moderate_days", "days"),
                            ("Unhealthy for Sensitive Groups Days", "aqi_unhealthy_sensitive_days", "days"),
                            ("Unhealthy Days", "aqi_unhealthy_days", "days"),
                            ("Very Unhealthy Days", "aqi_very_unhealthy_days", "days"),
                            ("Hazardous Days", "aqi_hazardous_days", "days"),
                            ("Max AQI", "AQI_max", "index"),
                            ("90th Percentile AQI", "AQI_p90", "index"),
                            ("Median AQI", "AQI_median", "index"),
                            ("Days Ozone", "ozone_days_primary_pollutant", "days"),
                            ("Days PM2.5", "PM2_5_days_primary_pollutant", "days"),
                            ("Days PM10", "PM10_days_primary_pollutant", "days"),
                        ]:
                            value = parse_float(row.get(col))
                            if value is None:
                                continue
                            sink.write(feature_record(
                                domain="environment", variable_id=var, value=value, unit=unit,
                                geography_level="county", geography_id=county, geography_name=f"{row.get('County')}, AR",
                                observation_time=date, period_end=f"{year}-12-31", source_id="epa_airdata_annual_aqi_by_county",
                                source_name="EPA AirData annual AQI by county", source_url=url, source_timestamp=RUN_AT,
                                release_frequency="annual", coverage_start=f"{EXTENDED_START_YEAR}-01-01", coverage_end="2025-12-31",
                                dimensions={"raw_file": raw_file, "source_field": col},
                            ))
            url = f"https://aqs.epa.gov/aqsweb/airdata/annual_conc_by_monitor_{year}.zip"
            body, raw_file = fetch_bytes_url(url, f"epa_annual_conc_by_monitor_{year}", ".zip", refresh)
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                    for row in reader:
                        if row.get("State Code") != "05":
                            continue
                        pname = str(row.get("Parameter Name") or "")
                        if pname not in {"PM2.5 - Local Conditions", "Ozone"}:
                            continue
                        prefix = "PM2_5" if pname.startswith("PM2.5") else "ozone"
                        county = f"05{str(row.get('County Code') or '').zfill(3)}"
                        monitor_id = f"05-{str(row.get('County Code') or '').zfill(3)}-{str(row.get('Site Num') or '').zfill(4)}-{prefix}"
                        date = f"{year}-01-01"
                        for col, suffix, unit in [
                            ("Arithmetic Mean", "mean", "source_native"),
                            ("1st Max Value", "max", "source_native"),
                            ("Observation Percent", "observation_percent", "percent"),
                        ]:
                            value = parse_float(row.get(col))
                            if value is None:
                                continue
                            sink.write(feature_record(
                                domain="environment", variable_id=f"{prefix}_{suffix}", value=value, unit=unit,
                                geography_level="monitor", geography_id=monitor_id,
                                geography_name=f"{row.get('County Name')}, AR {pname}",
                                observation_time=date, period_end=f"{year}-12-31", source_id="epa_airdata_annual_conc_by_monitor",
                                source_name="EPA AirData annual concentration by monitor", source_url=url, source_timestamp=RUN_AT,
                                release_frequency="annual", coverage_start=f"{EXTENDED_START_YEAR}-01-01", coverage_end="2025-12-31",
                                dimensions={"raw_file": raw_file, "parameter_name": pname, "county_fips": county, "site_num": row.get("Site Num"), "source_field": col},
                            ))
    except Exception as exc:
        stats["epa_air_quality_error"] = str(exc)
    stats["environment"] = sink.count - before


def download_cms_medicare_geo(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    dataset_id = "6219697b-8f6c-4164-bed4-cd9317c58ebc"
    base = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
    wanted_native = {
        "BENES_TOTAL_CNT": ("medicare_beneficiary_count", "beneficiaries"),
        "TOT_MDCR_PYMT_PC": ("medicare_per_capita_spending", "usd_per_beneficiary"),
        "IP_CVRD_STAYS_PER_1000_BENES": ("inpatient_utilization", "stays_per_1000_beneficiaries"),
        "OP_VISITS_PER_1000_BENES": ("outpatient_utilization", "visits_per_1000_beneficiaries"),
        "ER_VISITS_PER_1000_BENES": ("emergency_department_utilization", "visits_per_1000_beneficiaries"),
        "PHYS_EVENTS_PER_1000_BENES": ("physician_service_utilization", "events_per_1000_beneficiaries"),
        "HH_VISITS_PER_1000_BENES": ("home_health_utilization", "visits_per_1000_beneficiaries"),
        "SNF_CVRD_STAYS_PER_1000_BENES": ("post_acute_utilization", "stays_per_1000_beneficiaries"),
        "BENE_DUAL_PCT": ("dual_eligible_rate", "share"),
        "MA_PRTCPTN_RATE": ("medicare_advantage_participation_rate", "share"),
    }
    try:
        offset = 0
        size = 5000
        while True:
            url = f"{base}?{urllib.parse.urlencode({'size': size, 'offset': offset})}"
            rows, raw_file = load_json_url(url, f"cms_medicare_geo_{offset}", refresh)
            if not rows:
                break
            for row in rows:
                geo_desc = str(row.get("BENE_GEO_DESC") or "")
                geo_cd = str(row.get("BENE_GEO_CD") or "")
                if geo_desc != "National" and geo_desc != "Arkansas" and not geo_cd.startswith("05"):
                    continue
                if str(row.get("BENE_AGE_LVL") or "") != "All":
                    continue
                year = int(parse_float(row.get("YEAR")) or 0)
                if year < 2014 or year > 2024:
                    continue
                if geo_desc == "National":
                    level, gid, gname = "national", "US", "United States"
                elif geo_desc == "Arkansas":
                    level, gid, gname = "state", "AR", "Arkansas"
                else:
                    level, gid, gname = "county", geo_cd, geo_desc
                for native, (var, unit) in wanted_native.items():
                    value = parse_float(row.get(native))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="healthcare", variable_id=var, value=value, unit=unit,
                        geography_level=level, geography_id=gid, geography_name=gname,
                        observation_time=f"{year}-01-01", period_end=f"{year}-12-31",
                        source_id="cms_medicare_geographic_variation", source_name="CMS Medicare Geographic Variation - by National, State & County",
                        source_url=base, source_timestamp=RUN_AT, release_frequency="annual",
                        coverage_start="2014-01-01", coverage_end="2024-12-31",
                        dimensions={"raw_file": raw_file, "native_field": native},
                    ))
            if len(rows) < size:
                break
            offset += size
    except Exception as exc:
        stats["cms_medicare_geo_error"] = str(exc)
    stats["healthcare"] = stats.get("healthcare", 0) + (sink.count - before)


def download_medicaid_enrollment(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    url = "https://download.medicaid.gov/data/pi-dataset-july-2026-release.csv"
    path = RAW / "medicaid_chip_enrollment_pi_july_2026.csv"
    try:
        if refresh or not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "meditrack-final-data/1.0"})
            with urllib.request.urlopen(request, timeout=120) as resp, path.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("State Abbreviation") != "AR":
                    continue
                period = str(row.get("Reporting Period") or "")
                date = parse_source_date(period) or (period[:7] + "-01" if len(period) >= 7 and period[:4].isdigit() else None)
                if not date:
                    continue
                year = int(date[:4])
                if year < 2014 or year > 2026:
                    continue
                for col, var, unit in [
                    ("Medicaid and CHIP Child Enrollment", "medicaid_chip_child_enrollment", "persons"),
                    ("Total Medicaid and CHIP Enrollment", "medicaid_chip_total_enrollment", "persons"),
                    ("Total Medicaid Enrollment", "medicaid_enrollment", "persons"),
                    ("Total CHIP Enrollment", "chip_enrollment", "persons"),
                    ("Total Adult Medicaid Enrollment", "adult_medicaid_enrollment", "persons"),
                    ("New Applications Submitted to Medicaid and CHIP Agencies", "medicaid_chip_new_applications", "applications"),
                    ("Average Call Center Wait Time (Minutes)", "medicaid_call_center_wait_time", "minutes"),
                    ("Average Call Center Abandonment Rate", "medicaid_call_center_abandonment_rate", "rate"),
                ]:
                    value = parse_float(row.get(col))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="healthcare", variable_id=var, value=value, unit=unit,
                        geography_level="state", geography_id="AR", geography_name="Arkansas",
                        observation_time=date, period_end=period_end_from_start(date, "month"),
                        source_id="medicaid_chip_performance_indicator", source_name="State Medicaid and CHIP Applications, Eligibility Determinations, and Enrollment Data",
                        source_url=url, source_timestamp=RUN_AT, release_frequency="monthly",
                        coverage_start="2014-01-01", coverage_end="2026-07-31",
                        dimensions={"raw_file": str(path.relative_to(OUT)), "native_field": col, "preliminary_or_updated": row.get("Preliminary or Updated")},
                    ))
    except Exception as exc:
        stats["medicaid_enrollment_error"] = str(exc)
    stats["healthcare"] = stats.get("healthcare", 0) + (sink.count - before)


def download_bls_economics(sink: FeatureSink, stats: dict[str, Any], refresh: bool) -> None:
    before = sink.count
    series = {
        "LAUST050000000000003": ("arkansas_unemployment_rate", "percent", "state", "AR", "Arkansas"),
        "LAUST050000000000004": ("arkansas_unemployed_population", "persons", "state", "AR", "Arkansas"),
        "LAUST050000000000005": ("arkansas_employed_population", "persons", "state", "AR", "Arkansas"),
        "LAUST050000000000006": ("arkansas_labor_force", "persons", "state", "AR", "Arkansas"),
        "LNS14000000": ("national_unemployment_rate", "percent", "national", "US", "United States"),
        "CUUR0000SA0": ("consumer_price_index_all_items", "index", "national", "US", "United States"),
        "CUUR0000SAM": ("consumer_price_index_medical_care", "index", "national", "US", "United States"),
        "CUUR0000SAM1": ("consumer_price_index_medical_care_commodities", "index", "national", "US", "United States"),
        "CUUR0000SEMD": ("consumer_price_index_prescription_drugs", "index", "national", "US", "United States"),
        "PCU32543254": ("pharma_producer_price_index", "index", "national", "US", "United States"),
        "SMU05000006562000001": ("arkansas_healthcare_social_assistance_employment", "thousand_persons", "state", "AR", "Arkansas"),
        "SMU05000006500000001": ("arkansas_private_education_health_services_employment", "thousand_persons", "state", "AR", "Arkansas"),
    }
    try:
        for start_year in range(EXTENDED_START_YEAR, CURRENT_YEAR + 1, 10):
            end_year = min(start_year + 9, CURRENT_YEAR)
            params = urllib.parse.urlencode({
                "startyear": str(start_year),
                "endyear": str(end_year),
            })
            for series_id, (variable_id, unit, geo_level, geo_id, geo_name) in series.items():
                url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?{params}"
                payload, raw_file = load_json_url(url, f"bls_{series_id}_{start_year}_{end_year}", refresh)
                for item in payload.get("Results", {}).get("series", [{}])[0].get("data", []):
                    period = str(item.get("period") or "")
                    year = int(parse_float(item.get("year")) or 0)
                    if not period.startswith("M") or period == "M13" or year < EXTENDED_START_YEAR:
                        continue
                    month = int(period[1:])
                    date = f"{year}-{month:02d}-01"
                    value = parse_float(item.get("value"))
                    if value is None:
                        continue
                    sink.write(feature_record(
                        domain="economics_us", variable_id=variable_id, value=value, unit=unit,
                        geography_level=geo_level, geography_id=geo_id, geography_name=geo_name,
                        observation_time=date, period_end=period_end_from_start(date, "month"),
                        source_id="bls_public_api", source_name="BLS Public Data API",
                        source_url=url, source_timestamp=RUN_AT, release_frequency="monthly",
                        coverage_start=f"{EXTENDED_START_YEAR}-01-01", coverage_end=f"{CURRENT_YEAR}-12-31",
                        dimensions={"series_id": series_id, "period_name": item.get("periodName"), "raw_file": raw_file},
                    ))
    except Exception as exc:
        stats["bls_economics_error"] = str(exc)
    stats["economics_us"] = stats.get("economics_us", 0) + (sink.count - before)


def write_events(event_rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_id", "event_type", "entity_id", "geography", "start_time", "reported_time",
        "severity", "confidence", "expected_duration", "source_id", "affected_entities_json",
        "affected_drugs_json", "affected_diseases_json", "details_json",
    ]
    csv_path = EVENTS / "events.csv.gz"
    json_path = EVENTS / "events.jsonl.gz"
    with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as cf, gzip.open(json_path, "wt", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in event_rows:
            writer.writerow(row)
            jf.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_combined_indexes(domain_counts: dict[str, int]) -> None:
    combined_csv = FEATURES / "external_state_features.csv.gz"
    combined_jsonl = FEATURES / "external_state_features.jsonl.gz"
    with gzip.open(combined_csv, "wt", encoding="utf-8", newline="") as cf, gzip.open(combined_jsonl, "wt", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=FEATURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for path in sorted(FEATURES.glob("*/*_features.csv.gz")):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    writer.writerow(row)
                    jf.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    (QUALITY / "domain_counts.json").write_text(json.dumps(domain_counts, indent=2, sort_keys=True), encoding="utf-8")


def generate_derived_features(stats: dict[str, Any]) -> int:
    selected = {
        "environment": {"temperature_mean", "precipitation", "wind", "PM2_5_mean", "ozone_mean"},
        "economics_us": {
            "arkansas_unemployment_rate", "arkansas_unemployed_population", "arkansas_employed_population",
            "arkansas_labor_force", "arkansas_healthcare_social_assistance_employment",
            "arkansas_private_education_health_services_employment", "national_unemployment_rate",
            "consumer_price_index_all_items", "consumer_price_index_medical_care",
            "consumer_price_index_medical_care_commodities", "consumer_price_index_prescription_drugs",
            "pharma_producer_price_index",
        },
        "disease_us": {"wili", "ili", "num_ili", "covid_cases_per_100k_7_day", "covid_percent_test_results_reported", "covid_community_transmission_level"},
    }
    rows_by_series: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for domain, variables in selected.items():
        path = FEATURES / domain / f"{domain}_features.csv.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["variable_id"] not in variables:
                    continue
                key = (row["domain"], row["variable_id"], row["geography_level"], row["geography_id"], row["release_frequency"])
                rows_by_series.setdefault(key, []).append(row)

    sink = FeatureSink("derived")
    for key, rows in rows_by_series.items():
        rows.sort(key=lambda r: r["observation_time"])
        prior_by_season: dict[str, list[float]] = {}
        values: list[float] = []
        for idx, row in enumerate(rows):
            value = parse_float(row["value"])
            if value is None:
                continue
            date = dt.date.fromisoformat(row["observation_time"])
            freq = row["release_frequency"]
            if freq == "daily":
                season_key = f"{date.timetuple().tm_yday:03d}"
            elif freq in {"month", "monthly"}:
                season_key = f"{date.month:02d}"
            elif freq == "weekly_or_annual":
                season_key = f"w{date.isocalendar().week:02d}"
            else:
                season_key = "annual"
            dims = {"base_record_id": row["record_id"], "base_variable_id": row["variable_id"], "base_source_id": row["source_id"], "base_frequency": freq}
            for lag in (1, 4, 12):
                if idx >= lag:
                    lag_value = parse_float(rows[idx - lag]["value"])
                    if lag_value is not None:
                        for suffix, out_value, transform in [
                            (f"lag_{lag}", lag_value, f"lag_{lag}_periods"),
                            (f"change_{lag}", value - lag_value, f"value_minus_lag_{lag}"),
                        ]:
                            sink.write(feature_record(
                                domain="derived", variable_id=f"{row['variable_id']}_{suffix}", value=out_value, unit=row["unit"],
                                geography_level=row["geography_level"], geography_id=row["geography_id"], geography_name=row["geography_name"],
                                observation_time=row["observation_time"], period_end=row["period_end"], source_id=row["source_id"],
                                source_name=row["source_name"], source_url=row["source_url"], source_timestamp=row["source_timestamp"],
                                release_frequency=freq, coverage_start=row["coverage_start"], coverage_end=row["coverage_end"],
                                transformation=transform, dimensions=dims,
                            ))
            for window in (4, 12):
                if len(values) >= window:
                    sample = values[-window:]
                    mean = sum(sample) / window
                    variance = sum((x - mean) ** 2 for x in sample) / window
                    for suffix, out_value, transform in [
                        (f"rolling_mean_{window}", mean, f"rolling_mean_prior_{window}_periods"),
                        (f"rolling_std_{window}", math.sqrt(variance), f"rolling_std_prior_{window}_periods"),
                    ]:
                        sink.write(feature_record(
                            domain="derived", variable_id=f"{row['variable_id']}_{suffix}", value=out_value, unit=row["unit"],
                            geography_level=row["geography_level"], geography_id=row["geography_id"], geography_name=row["geography_name"],
                            observation_time=row["observation_time"], period_end=row["period_end"], source_id=row["source_id"],
                            source_name=row["source_name"], source_url=row["source_url"], source_timestamp=row["source_timestamp"],
                            release_frequency=freq, coverage_start=row["coverage_start"], coverage_end=row["coverage_end"],
                            transformation=transform, dimensions=dims,
                        ))
            historical = prior_by_season.get(season_key, [])
            if historical:
                baseline = sum(historical) / len(historical)
                sink.write(feature_record(
                    domain="derived", variable_id=f"{row['variable_id']}_seasonal_baseline", value=baseline, unit=row["unit"],
                    geography_level=row["geography_level"], geography_id=row["geography_id"], geography_name=row["geography_name"],
                    observation_time=row["observation_time"], period_end=row["period_end"], source_id=row["source_id"],
                    source_name=row["source_name"], source_url=row["source_url"], source_timestamp=row["source_timestamp"],
                    release_frequency=freq, coverage_start=row["coverage_start"], coverage_end=row["coverage_end"],
                    transformation="mean_same_calendar_period_prior_years", dimensions={**dims, "season_key": season_key, "prior_count": len(historical)},
                ))
                sink.write(feature_record(
                    domain="derived", variable_id=f"{row['variable_id']}_seasonal_anomaly", value=value - baseline, unit=row["unit"],
                    geography_level=row["geography_level"], geography_id=row["geography_id"], geography_name=row["geography_name"],
                    observation_time=row["observation_time"], period_end=row["period_end"], source_id=row["source_id"],
                    source_name=row["source_name"], source_url=row["source_url"], source_timestamp=row["source_timestamp"],
                    release_frequency=freq, coverage_start=row["coverage_start"], coverage_end=row["coverage_end"],
                    transformation="value_minus_seasonal_baseline", dimensions={**dims, "season_key": season_key, "prior_count": len(historical)},
                ))
            values.append(value)
            prior_by_season.setdefault(season_key, []).append(value)
    count = sink.count
    sink.close()
    stats["derived"] = count
    return count


def write_entities_and_relationships() -> dict[str, int]:
    entity_fields = ["entity_id", "entity_type", "name", "geography_level", "geography_id", "source_id", "attributes_json"]
    relationship_fields = ["relationship_id", "source_entity_id", "relationship_type", "target_entity_id", "source_id", "attributes_json"]
    entities: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}

    def add_entity(entity_id: str, entity_type: str, name: str, geography_level: str = "", geography_id: str = "", source_id: str = "final_data", attrs: dict[str, Any] | None = None) -> None:
        entities[entity_id] = {
            "entity_id": entity_id, "entity_type": entity_type, "name": name,
            "geography_level": geography_level, "geography_id": geography_id,
            "source_id": source_id, "attributes_json": json.dumps(attrs or {}, ensure_ascii=True, sort_keys=True),
        }

    def add_rel(src: str, rel: str, tgt: str, source_id: str = "final_data", attrs: dict[str, Any] | None = None) -> None:
        rid = stable_id([src, rel, tgt, source_id, json.dumps(attrs or {}, sort_keys=True)])
        relationships[rid] = {
            "relationship_id": rid, "source_entity_id": src, "relationship_type": rel,
            "target_entity_id": tgt, "source_id": source_id,
            "attributes_json": json.dumps(attrs or {}, ensure_ascii=True, sort_keys=True),
        }

    add_entity("GLOBAL", "global", "Global", "global", "GLOBAL")
    add_entity("US", "country", "United States", "country", "US")
    add_entity("AR", "state", "Arkansas", "state", "AR")
    add_rel("AR", "located_in", "US")
    for region_id, region_name in REGION_NAMES.items():
        add_entity(
            f"region:{region_id}", "region", region_name, "region", region_id,
            "arkansas_code_14_166_202",
            {"definition": "Arkansas planning and development district", "county_names": ARKANSAS_EDD_REGIONS.get(region_id, [])},
        )
        add_rel(f"region:{region_id}", "located_in", "AR", "arkansas_code_14_166_202")
    places = RAW / "cdc_places_2024_ar_county_0_0ad72d4fe947807b.json"
    if places.exists():
        for row in json.loads(places.read_text(encoding="utf-8")):
            county = str(row.get("countyfips") or "")
            if county:
                eid = f"county:{county}"
                county_name = str(row.get("countyname") or "")
                region_id = COUNTY_TO_REGION.get(f"{county_name.lower()} county") or COUNTY_TO_REGION.get(county_name.lower())
                add_entity(eid, "county", f"{county_name}, AR", "county", county, "cdc_places_2024", {"population": row.get("totalpopulation"), "region_id": region_id})
                add_rel(eid, "located_in", "AR", "cdc_places_2024")
                if region_id:
                    add_rel(eid, "located_in", f"region:{region_id}", "arkansas_code_14_166_202")
    drug_path = ROOT / "S_D/data/drug_dictionary.json"
    if drug_path.exists():
        for row in json.loads(drug_path.read_text(encoding="utf-8")):
            drug = str(row.get("canonical_name") or row.get("original_name") or "").strip()
            ingredient = str(row.get("ingredient") or "").strip()
            ndc = str(row.get("ndc") or "").strip()
            if drug:
                did = f"drug:{drug}"
                add_entity(did, "drug", drug, source_id="drug_dictionary", attrs={k: row.get(k) for k in ["original_name", "rxnorm_rxcui", "atc_code", "ndc"]})
                if ingredient:
                    iid = f"ingredient:{ingredient}"
                    add_entity(iid, "active_ingredient", ingredient, source_id="drug_dictionary")
                    add_rel(did, "contains", iid, "drug_dictionary")
                if ndc:
                    nid = f"ndc:{ndc}"
                    add_entity(nid, "drug_package", ndc, source_id="drug_dictionary")
                    add_rel(nid, "represents", did, "drug_dictionary")
    env_path = FEATURES / "environment/environment_features.csv.gz"
    if env_path.exists():
        with gzip.open(env_path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["geography_level"] == "station":
                    sid = f"station:{row['geography_id']}"
                    attrs = json.loads(row["dimensions_json"] or "{}")
                    add_entity(sid, "weather_station", row["geography_name"], "station", row["geography_id"], row["source_id"], attrs)
                    add_rel(sid, "located_in", "AR", row["source_id"])
    event_path = EVENTS / "events.csv.gz"
    if event_path.exists():
        with gzip.open(event_path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                eid = f"event:{row['event_id']}"
                add_entity(eid, row["event_type"], row["event_id"], source_id=row["source_id"], attrs={"start_time": row["start_time"], "geography": row["geography"], "severity": row["severity"]})
                if row["geography"].startswith("AR"):
                    add_rel(eid, "affects", "AR", row["source_id"])
    ENTITIES.mkdir(parents=True, exist_ok=True)
    with gzip.open(ENTITIES / "entities.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=entity_fields)
        writer.writeheader()
        writer.writerows(sorted(entities.values(), key=lambda r: r["entity_id"]))
    with gzip.open(ENTITIES / "relationships.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=relationship_fields)
        writer.writeheader()
        writer.writerows(sorted(relationships.values(), key=lambda r: r["relationship_id"]))
    return {"entities": len(entities), "relationships": len(relationships)}


def write_quality_reports(stats: dict[str, Any]) -> dict[str, Any]:
    coverage_rows: list[dict[str, Any]] = []
    source_rows: dict[str, dict[str, Any]] = {}
    series_dates: dict[tuple[str, str, str, str, str, str], set[str]] = collections.defaultdict(set)
    family_geography: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    disease_rows: dict[str, dict[str, Any]] = {}
    duplicate_record_ids: set[str] = set()
    seen_ids: set[str] = set()
    invalid_dates = 0
    out_of_range = 0
    total = 0
    for path in sorted(FEATURES.glob("*/*_features.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                total += 1
                if row["record_id"] in seen_ids:
                    duplicate_record_ids.add(row["record_id"])
                seen_ids.add(row["record_id"])
                try:
                    dt.date.fromisoformat(row["observation_time"])
                    value = float(row["value"])
                    if not math.isfinite(value) or abs(value) > 1e15:
                        out_of_range += 1
                except Exception:
                    invalid_dates += 1
                key = (row["domain"], row["variable_id"], row["source_id"], row["release_frequency"], row["geography_level"])
                series_key = (*key, row["geography_id"])
                series_dates[series_key].add(row["observation_time"])
                family_geography[row["domain"]][row["geography_level"]].add(row["geography_id"])
                if row["domain"].startswith("disease"):
                    try:
                        dims = json.loads(row.get("dimensions_json") or "{}")
                    except Exception:
                        dims = {}
                    disease_id = dims.get("disease_id")
                    if disease_id:
                        drow = disease_rows.setdefault(str(disease_id), {
                            "disease_id": str(disease_id),
                            "disease_name": dims.get("disease_name") or str(disease_id),
                            "first_date": row["observation_time"],
                            "last_date": row["observation_time"],
                            "number_of_observations": 0,
                            "variables": set(),
                            "sources": set(),
                            "geographies": set(),
                        })
                        drow["number_of_observations"] += 1
                        drow["first_date"] = min(drow["first_date"], row["observation_time"])
                        drow["last_date"] = max(drow["last_date"], row["observation_time"])
                        drow["variables"].add(row["variable_id"])
                        drow["sources"].add(row["source_id"])
                        drow["geographies"].add(f"{row['geography_level']}:{row['geography_id']}")
                existing = next((r for r in coverage_rows if r["_key"] == key), None)
                if existing is None:
                    existing = {"_key": key, "variable_id": row["variable_id"], "domain": row["domain"], "source": row["source_id"], "frequency": row["release_frequency"], "geography": row["geography_level"], "first_date": row["observation_time"], "last_date": row["observation_time"], "number_of_observations": 0}
                    coverage_rows.append(existing)
                existing["number_of_observations"] += 1
                existing["first_date"] = min(existing["first_date"], row["observation_time"])
                existing["last_date"] = max(existing["last_date"], row["observation_time"])
                src = source_rows.setdefault(row["source_id"], {"source_id": row["source_id"], "source_name": row["source_name"], "official_url": row["source_url"], "api_or_download": "api_or_download", "authentication_required": False, "historical_start": row["observation_time"], "latest_available": row["observation_time"], "frequency": set(), "geography": set(), "license_access": "public/government or documented local extract"})
                src["historical_start"] = min(src["historical_start"], row["observation_time"])
                src["latest_available"] = max(src["latest_available"], row["observation_time"])
                src["frequency"].add(row["release_frequency"])
                src["geography"].add(row["geography_level"])
    for row in coverage_rows:
        start = dt.date.fromisoformat(row["first_date"])
        end = dt.date.fromisoformat(row["last_date"])
        days = max(1, (end - start).days + 1)
        freq = row["frequency"]
        if freq == "daily":
            expected = days
        elif freq in {"month", "monthly"}:
            expected = max(1, (end.year - start.year) * 12 + end.month - start.month + 1)
        elif freq in {"annual", "year", "fiscal_year"}:
            expected = max(1, end.year - start.year + 1)
        elif freq == "weekly_or_annual":
            expected = max(1, days // 7)
        else:
            expected = row["number_of_observations"]
        row["expected_observations"] = expected
        row["coverage_percentage"] = min(100.0, 100.0 * row["number_of_observations"] / expected) if expected else 100.0
        row["missing_percentage"] = max(0.0, 100.0 - row["coverage_percentage"])
        del row["_key"]
    with (QUALITY / "variable_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump({"variables": sorted(coverage_rows, key=lambda r: (r["domain"], r["variable_id"], r["source"], r["geography"]))}, fh, indent=2, sort_keys=True)
    for row in source_rows.values():
        row["frequency"] = sorted(row["frequency"])
        row["geography"] = sorted(row["geography"])
    with (QUALITY / "source_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump(sorted(source_rows.values(), key=lambda r: r["source_id"]), fh, indent=2, sort_keys=True)
    historical_rows = []
    for row in coverage_rows:
        source = row["source"]
        action = "retained_current_extraction"
        if source == "census_county_population_estimates":
            action = "expanded_with_census_bulk_county_estimates"
        elif source == "cepii_baci" and row["first_date"] <= "2000-01-01":
            action = "expanded_baci_trade_to_2000"
        elif source == "bls_public_api" and row["first_date"] <= "2000-01-01":
            action = "expanded_bls_public_series_to_2000"
        historical_rows.append({
            "variable_id": row["variable_id"],
            "variable_name": row["variable_id"],
            "family": row["domain"],
            "current_start": row["first_date"],
            "current_end": row["last_date"],
            "verified_public_start": row["first_date"],
            "verified_public_end": row["last_date"],
            "native_frequency": row["frequency"],
            "geography": row["geography"],
            "historical_gap": None,
            "expansion_possible": "unknown_or_requires_source_specific_research",
            "source": source,
            "action_taken": action,
            "quality_rank": {
                "authoritative_source": source not in {"see supply_chain/sources.md"},
                "historical_depth_years": max(1, dt.date.fromisoformat(row["last_date"]).year - dt.date.fromisoformat(row["first_date"]).year + 1),
                "geographic_resolution": row["geography"],
                "pharmaceutical_relevance": "direct" if row["domain"] in {"demand_price", "trade", "policy", "supply_chain"} else "contextual",
            },
        })
    with (QUALITY / "historical_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump({"variables": sorted(historical_rows, key=lambda r: (r["family"], r["variable_id"], r["source"], r["geography"]))}, fh, indent=2, sort_keys=True)

    geographic_rows = []
    for family, levels in sorted(family_geography.items()):
        geographic_rows.append({
            "family": family,
            "arkansas_statewide": int("AR" in levels.get("state", set())),
            "arkansas_counties": len({gid for gid in levels.get("county", set()) if gid.startswith("05")}),
            "arkansas_regions": len(levels.get("region", set())),
            "us_states": len(levels.get("state", set())),
            "countries": len(levels.get("country", set())),
            "national": len(levels.get("national", set())),
            "global": len(levels.get("global", set())),
            "stations": len(levels.get("station", set())),
            "monitors": len(levels.get("monitor", set())),
        })
    with (QUALITY / "geographic_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump({"families": geographic_rows}, fh, indent=2, sort_keys=True)

    def expected_periods(start: dt.date, end: dt.date, freq: str) -> int:
        if freq == "daily":
            return max(1, (end - start).days + 1)
        if freq in {"month", "monthly"}:
            return max(1, (end.year - start.year) * 12 + end.month - start.month + 1)
        if freq in {"annual", "year", "fiscal_year"}:
            return max(1, end.year - start.year + 1)
        if freq == "weekly_or_annual":
            return max(1, ((end - start).days // 7) + 1)
        return 1

    def is_next(prev: dt.date, cur: dt.date, freq: str) -> bool:
        if freq == "daily":
            return (cur - prev).days == 1
        if freq == "weekly_or_annual":
            return 1 <= (cur - prev).days <= 8
        if freq in {"month", "monthly"}:
            return (cur.year - prev.year) * 12 + cur.month - prev.month == 1
        if freq in {"annual", "year", "fiscal_year"}:
            return cur.year - prev.year == 1
        return True

    continuity_rows = []
    family_continuity: dict[str, list[float]] = collections.defaultdict(list)
    for (domain, variable_id, source_id, freq, geo_level, geo_id), dates in series_dates.items():
        ordered = sorted(dt.date.fromisoformat(d) for d in dates)
        if not ordered:
            continue
        expected = expected_periods(ordered[0], ordered[-1], freq)
        longest = cur_run = 1
        gaps = 0
        longest_gap = 0
        for prev, cur in zip(ordered, ordered[1:]):
            if is_next(prev, cur, freq):
                cur_run += 1
            else:
                gaps += 1
                longest = max(longest, cur_run)
                cur_run = 1
                longest_gap = max(longest_gap, (cur - prev).days)
        longest = max(longest, cur_run)
        pct = min(100.0, 100.0 * len(ordered) / expected) if expected else 100.0
        family_continuity[domain].append(pct)
        continuity_rows.append({
            "family": domain,
            "variable_id": variable_id,
            "source": source_id,
            "frequency": freq,
            "geography_type": geo_level,
            "geography_id": geo_id,
            "first_date": ordered[0].isoformat(),
            "last_date": ordered[-1].isoformat(),
            "actual_observations": len(ordered),
            "expected_observations": expected,
            "continuity_percentage": round(pct, 4),
            "longest_continuous_run": longest,
            "number_of_gaps": gaps,
            "longest_gap_days": longest_gap,
        })
    with (QUALITY / "temporal_continuity.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "series": sorted(continuity_rows, key=lambda r: (r["family"], r["variable_id"], r["geography_type"], r["geography_id"]))[:200000],
            "family_summary": {
                family: {
                    "series_count": len(values),
                    "mean_continuity_percentage": round(sum(values) / len(values), 4) if values else None,
                }
                for family, values in sorted(family_continuity.items())
            },
        }, fh, indent=2, sort_keys=True)

    for row in disease_rows.values():
        row["variables"] = sorted(row["variables"])
        row["sources"] = sorted(row["sources"])
        row["geographies"] = sorted(row["geographies"])
    with (QUALITY / "disease_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump({"diseases": sorted(disease_rows.values(), key=lambda r: r["disease_id"])}, fh, indent=2, sort_keys=True)

    issues = {
        "generated_at": RUN_AT,
        "source_errors": {k: v for k, v in stats.items() if k.endswith("_error") or k.endswith("_status")},
        "blocked_or_limited_sources": {
            "census_acs_profile": "api.census.gov returned key-gated HTML Missing Key response; fallback CDC PLACES totalpopulation retained for county population.",
            "census_international_trade_api": "Census International Trade API returned key-gated HTML Missing Key response; official CEPII BACI HS92 annual U.S. pharmaceutical trade by partner/product was used as the accessible fallback.",
            "fda_inspection_facility_compliance": "HHS/FDA catalog metadata reachable; HHS JSON returned 403; original FDA Excel distribution returned 404.",
            "licensed_sources": "ASHP, IQVIA, and several trade policy portals require credentials/license/manual export.",
        },
    }
    (QUALITY / "extraction_issues.json").write_text(json.dumps(issues, indent=2, sort_keys=True), encoding="utf-8")
    validation = {
        "validated_at": RUN_AT,
        "feature_rows_total": total,
        "duplicate_record_id_count": len(duplicate_record_ids),
        "invalid_date_count": invalid_dates,
        "implausible_value_count": out_of_range,
        "checks_passed": len(duplicate_record_ids) == 0 and invalid_dates == 0 and out_of_range == 0,
        "domain_counts": stats.get("domain_counts", {}),
        "events": stats.get("events", 0),
    }
    (QUALITY / "validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return validation


def collect_source_docs(stats: dict[str, Any]) -> None:
    docs = []
    for path in sorted(ROOT.glob("*/sources.md")) + sorted(ROOT.glob("*/README.md")) + [ROOT / "data_infra.md"]:
        if path.exists():
            dest = SOURCES / path.parent.name / path.name if path.parent != ROOT else SOURCES / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            docs.append(str(dest.relative_to(OUT)))
    for path in sorted(ROOT.glob("*/source_manifest.*")) + sorted(ROOT.glob("*/metadata/extraction_status.json")) + sorted(ROOT.glob("*/metadata/validation_errors.json")):
        if path.exists():
            dest = SOURCES / path.parent.parent.name / path.parent.name / path.name if path.parent.name == "metadata" else SOURCES / path.parent.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            docs.append(str(dest.relative_to(OUT)))
    stats["source_documents"] = docs


def write_readme(stats: dict[str, Any], events_count: int) -> None:
    readme = f"""# final_data

Built at `{RUN_AT}` from `data_infra.md`, local normalized extracts, and new official public downloads.

## Structure

- `features/<domain>/<domain>_features.csv.gz` and `.jsonl.gz`: canonical long-format numeric feature rows.
- `features/external_state_features.csv.gz` and `.jsonl.gz`: combined feature store across domains.
- `events/events.csv.gz` and `.jsonl.gz`: structured event records for shortages, recalls, and FEMA disaster declarations.
	- `entities/`: entity graph and copied entity/crosswalk reference files used for drug, NDC, HRR, LEI, and related joins.
- `sources/`: copied source manifests, READMEs, source notes, extraction statuses, and validation reports.
- `raw_downloads/`: newly downloaded official API responses used only by this final assembly.
	- `quality/`: validation, coverage, source coverage, extraction issues, run counts, and build manifest.

## Feature Contract

Every feature row includes `variable_id`, `value`, `geography_*`, `observation_time`, `forecast_horizon`,
and source/quality metadata required by `data_infra.md`. Values are numeric floats wherever the source
provides a numeric observation. Categorical observations are either transformed into deterministic numeric
indicators with the native category retained in `dimensions_json`, or stored in `events/`.

No imputation, fake county disease estimates, or fake daily values are produced.

## Counts

- Feature rows by domain: `{json.dumps(stats.get("domain_counts", {}), sort_keys=True)}`
- Event rows: `{events_count}`

## Newly Downloaded Sources

	- CDC PLACES 2024 county release, Arkansas county population and chronic-disease measures.
	- FEMA OpenFEMA Disaster Declarations Summaries for Arkansas declarations from 2012 onward.
	- NOAA NCEI Daily Summaries for geographically distributed Arkansas weather stations from 2000 onward.
	- EPA AirData annual AQI and monitor concentration files for Arkansas from 2000 onward.
	- CMS Medicare Geographic Variation and Medicaid/CHIP enrollment public files.
	- BLS Public Data API unemployment, CPI, and pharmaceutical PPI series from 2000 onward.
	- openFDA drug shortages and enforcement/recall APIs for recent FDA supply events.
	- CEPII BACI HS92 U.S. pharmaceutical import/export exposure by partner country and HS product.
	- Derived lag, change, rolling, and seasonal anomaly features for appropriate historical series.

	Attempted but limited: Census ACS 2024 county profile and Census International Trade API returned key-gated
	HTML responses in this environment. CDC PLACES was retained for county population/chronic health, and CEPII
	BACI was used for official public U.S. pharmaceutical trade fallback. FDA inspection metadata was reachable
	through HHS, but its JSON resource returned 403 and the archived Excel distribution returned 404; openFDA
	enforcement/recall events were retained as the official accessible FDA supply-chain substitute.

See `quality/build_manifest.json` for detailed statuses and source paths.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    refresh = "--refresh" in sys.argv
    clean_previous_outputs()
    stats: dict[str, Any] = {"run_at": RUN_AT, "schema_version": SCHEMA_VERSION}
    events: list[dict[str, Any]] = []
    sinks: dict[str, FeatureSink] = {}

    def sink(domain: str) -> FeatureSink:
        if domain not in sinks:
            sinks[domain] = FeatureSink(domain)
        return sinks[domain]

    try:
        ingest_disease_us(sink("disease_us"), stats)
        download_cdc_covid_county_transmission(sink("disease_us"), stats, refresh)
        ingest_disease_global(sink("disease_global"), stats)
        ingest_economics_us(sink("economics_us"), stats)
        ingest_demand_price(sink("demand_price"), events, stats)
        download_openfda_recent_supply(sink("demand_price"), events, stats, refresh)
        ingest_supply_chain(sink("supply_chain"), stats)
        ingest_trade(sink("trade"), stats)
        ingest_policy(sink("policy"), stats)
        download_acs_arkansas(sink("population"), stats, refresh)
        download_census_county_population_estimates(sink("population"), stats, refresh)
        download_places_2024_county(sink("chronic_health"), sink("population"), stats, refresh)
        download_fema_arkansas(sink("disasters"), events, stats, refresh)
        download_noaa_weather(sink("environment"), stats, refresh)
        download_epa_air_quality(sink("environment"), stats, refresh)
        download_cms_medicare_geo(sink("healthcare"), stats, refresh)
        download_medicaid_enrollment(sink("healthcare"), stats, refresh)
        download_bls_economics(sink("economics_us"), stats, refresh)
    finally:
        for s in sinks.values():
            s.close()

    domain_counts = {domain: s.count for domain, s in sorted(sinks.items())}
    derived_count = generate_derived_features(stats)
    if derived_count:
        domain_counts["derived"] = derived_count
    stats["domain_counts"] = domain_counts
    write_events(events)
    entity_counts = write_entities_and_relationships()
    stats.update(entity_counts)
    ingest_crosswalk_entities(stats)
    write_combined_indexes(domain_counts)
    collect_source_docs(stats)
    stats["events"] = len(events)
    stats["feature_rows_total"] = sum(domain_counts.values())
    (QUALITY / "build_manifest.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    validation = write_quality_reports(stats)
    write_readme(stats, len(events))
    print(json.dumps({"feature_rows_total": stats["feature_rows_total"], "events": len(events), "domains": domain_counts, "entities": entity_counts, "validation": validation["checks_passed"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
