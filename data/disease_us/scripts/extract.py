#!/usr/bin/env python3
"""Extract public US disease data and normalize it to long-format JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from common import DATA_DIR, METADATA_DIR, as_number, fetch_json, socrata_pages, utc_now, write_jsonl

FLUVIEW_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
NNDSS_URL = "https://data.cdc.gov/resource/x9gk-5huc.json"
PLACES_URL = "https://data.cdc.gov/resource/epbn-9bv3.json"

STATE_REGIONS = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id", "il", "in",
    "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh",
    "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut",
    "vt", "va", "wa", "wv", "wi", "wy",
]


def base_record(source_id: str, dataset_id: str, period: dict[str, Any], geography: dict[str, str], metric: str,
                value: int | float, unit: str, dimensions: dict[str, Any], source_url: str,
                raw_file: str, retrieved_at: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dataset_id": dataset_id,
        "period": period,
        "geography": geography,
        "metric": metric,
        "value": value,
        "unit": unit,
        "dimensions": dimensions,
        "provenance": {
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "raw_file": raw_file,
        },
    }


def extract_delphi(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "delphi_fluview"
    output = DATA_DIR / source_id / "observations.jsonl"
    retrieved_at = utc_now()
    metric_units = {
        "num_ili": "cases",
        "num_patients": "patients",
        "num_providers": "providers",
        "num_age_0": "cases",
        "num_age_1": "cases",
        "num_age_2": "cases",
        "num_age_3": "cases",
        "num_age_4": "cases",
        "num_age_5": "cases",
        "wili": "percent",
        "ili": "percent",
    }
    metric_names = list(metric_units)
    batches: list[tuple[list[dict[str, Any]], str]] = []
    for year in range(start_year, end_year + 1):
        regions = "nat," + ",".join(STATE_REGIONS)
        url = f"{FLUVIEW_URL}?regions={quote(regions, safe=',')}&epiweeks={year}01-{year}53"
        payload, raw_file, _ = fetch_json(source_id, url, refresh=refresh)
        if payload.get("result") not in (1, -2):
            raise RuntimeError(f"Delphi FluView failed for {year}: {payload.get('message')}")
        batches.append((payload.get("epidata", []), raw_file))

    def rows() -> Iterator[dict[str, Any]]:
        for records, raw_file in batches:
            for record in records:
                epiweek = as_number(record.get("epiweek"))
                if epiweek is None:
                    continue
                region = str(record.get("region", ""))
                level = "national" if region == "nat" else "state"
                period = {
                    "year": int(epiweek) // 100,
                    "week": int(epiweek) % 100,
                    "epiweek": int(epiweek),
                    "issue": as_number(record.get("issue")),
                }
                dimensions = {
                    "lag": as_number(record.get("lag")),
                    "release_date": record.get("release_date"),
                }
                for metric in metric_names:
                    value = as_number(record.get(metric))
                    if value is None:
                        continue
                    yield base_record(
                        source_id, "fluview", period,
                        {"level": level, "id": region, "name": region}, metric, value,
                        metric_units[metric], dimensions, FLUVIEW_URL, raw_file, retrieved_at,
                    )

    count = write_jsonl(output, rows())
    return {"status": "success", "records": count, "output": str(output.relative_to(DATA_DIR.parent.parent)),
            "years": [start_year, end_year], "regions": len(STATE_REGIONS) + 1}


def extract_nndss(start_year: int, end_year: int, refresh: bool, all_geographies: bool) -> dict[str, Any]:
    source_id = "nndss_weekly"
    output = DATA_DIR / source_id / "observations.jsonl"
    retrieved_at = utc_now()
    where = f"year between '{start_year}' and '{end_year}'"
    if not all_geographies:
        where += " AND states='US RESIDENTS'"
    pages: list[tuple[list[dict[str, Any]], str]] = []
    for payload, raw_file, _ in socrata_pages(source_id, NNDSS_URL, where, refresh=refresh):
        pages.append((payload, raw_file))

    def rows() -> Iterator[dict[str, Any]]:
        metric_names = {
            "m1": ("current_week_cases", "cases"),
            "m2": ("previous_52_week_max", "cases"),
            "m3": ("cumulative_ytd_current_year", "cases"),
            "m4": ("cumulative_ytd_previous_year", "cases"),
        }
        for records, raw_file in pages:
            for record in records:
                year = as_number(record.get("year"))
                week = as_number(record.get("week"))
                if year is None or week is None:
                    continue
                area = str(record.get("states") or record.get("location2") or "")
                level = "national" if area == "US RESIDENTS" else "reporting_area"
                dimensions = {
                    "condition": record.get("label"),
                    "reporting_area": area,
                    "location1": record.get("location1"),
                    "m1_flag": record.get("m1_flag"),
                    "m2_flag": record.get("m2_flag"),
                    "m3_flag": record.get("m3_flag"),
                    "m4_flag": record.get("m4_flag"),
                }
                for native, (metric, unit) in metric_names.items():
                    value = as_number(record.get(native))
                    if value is None:
                        continue
                    yield base_record(
                        source_id, "x9gk-5huc", {"year": int(year), "week": int(week)},
                        {"level": level, "id": area, "name": area}, metric, value, unit,
                        dimensions, "https://data.cdc.gov/NNDSS/NNDSS-Weekly-Data/x9gk-5huc", raw_file, retrieved_at,
                    )

    count = write_jsonl(output, rows())
    return {"status": "success", "records": count, "output": str(output.relative_to(DATA_DIR.parent.parent)),
            "geography": "all" if all_geographies else "US RESIDENTS", "years": [start_year, end_year],
            "pages": len(pages)}


def extract_places(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "places_2022"
    output = DATA_DIR / source_id / "observations.jsonl"
    retrieved_at = utc_now()
    page_count = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal page_count
        for records, raw_file, _ in socrata_pages(source_id, PLACES_URL, "", refresh=refresh):
            page_count += 1
            for record in records:
                value = as_number(record.get("data_value"))
                if value is None:
                    continue
                year = as_number(record.get("year"))
                if year is None:
                    continue
                dimensions = {
                    "state_abbr": record.get("stateabbr"),
                    "state_name": record.get("statedesc"),
                    "location_name": record.get("locationname"),
                    "category": record.get("category"),
                    "measure": record.get("measure"),
                    "data_value_type": record.get("data_value_type"),
                    "low_confidence_limit": as_number(record.get("low_confidence_limit")),
                    "high_confidence_limit": as_number(record.get("high_confidence_limit")),
                    "total_population": as_number(record.get("totalpopulation")),
                    "category_id": record.get("categoryid"),
                    "measure_id": record.get("measureid"),
                    "short_question_text": record.get("short_question_text"),
                    "footnote_symbol": record.get("data_value_footnote_symbol"),
                    "footnote": record.get("data_value_footnote"),
                }
                yield base_record(
                    source_id, "epbn-9bv3", {"year": int(year)},
                    {"level": "place", "id": str(record.get("locationid", "")),
                     "name": str(record.get("locationname", ""))},
                    str(record.get("measureid") or record.get("measure") or "unknown"), value,
                    str(record.get("data_value_unit") or "source_native"), dimensions,
                    "https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-Place-Data-202/epbn-9bv3",
                    raw_file, retrieved_at,
                )

    count = write_jsonl(output, rows())
    return {"status": "success", "records": count, "output": str(output.relative_to(DATA_DIR.parent.parent)),
            "release": "2022", "pages": page_count, "source_years": [start_year, end_year]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.sources
    if "all_public" in requested:
        requested = ["delphi_fluview", "nndss_weekly", "places_2022"]
    results: dict[str, Any] = {}
    for source in requested:
        try:
            if source == "delphi_fluview":
                results[source] = extract_delphi(args.start_year, args.end_year, args.refresh)
            elif source == "nndss_weekly":
                results[source] = extract_nndss(args.start_year, args.end_year, args.refresh, args.nndss_all_geographies)
            elif source == "places_2022":
                results[source] = extract_places(args.start_year, args.end_year, args.refresh)
            else:
                results[source] = {"status": "not_implemented"}
        except Exception as exc:  # continue so one source does not hide other results
            results[source] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "run_at": utc_now(),
        "focus_window": {"start_year": args.start_year, "end_year": args.end_year},
        "refresh": args.refresh,
        "results": results,
    }
    (METADATA_DIR / "extraction_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", dest="sources", action="append", choices=["all_public", "delphi_fluview", "nndss_weekly", "places_2022"], help="source to extract; repeatable")
    parser.add_argument("--list", action="store_true", help="list implemented extractors")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--refresh", action="store_true", help="ignore cached HTTP responses")
    parser.add_argument("--nndss-all-geographies", action="store_true", help="extract all NNDSS reporting areas instead of national only")
    args = parser.parse_args()
    if args.list:
        print("all_public\ndelphi_fluview\nnndss_weekly\nplaces_2022")
        return 0
    args.sources = args.sources or ["all_public"]
    results = run(args)
    print(json.dumps(results, indent=2))
    return 0 if all(item.get("status") == "success" for item in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
