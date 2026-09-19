#!/usr/bin/env python3
"""Extract tariff-policy sources into normalized CSV.

The default public run is intentionally conservative. It does not guess bulk
endpoints for gated or interactive sources and never creates synthetic rows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
import urllib.parse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE, DATA, METADATA, ROOT, START, END, canonical_record, clamp_interval, code,
    fetch_json, parse_date, parse_number, read_csv, write_records, year_date,
)

SOURCE_INFO: dict[str, tuple[str, str, str]] = {
    "wits_trains": ("WITS / UNCTAD TRAINS", "https://wits.worldbank.org/", "wits_trains_tariffs.csv"),
    "wto_idb_cts": ("WTO Integrated Database / Consolidated Tariff Schedules", "https://ttd.wto.org/en/data/idb", "wto_idb_cts.csv"),
    "global_trade_alert": ("Global Trade Alert", "https://globaltradealert.org/data-center", "global_trade_alert.csv"),
    "eui_covid": ("EUI/GTA/World Bank COVID-19 Trade Policy Database", "https://globalgovernanceprogramme.eui.eu/covid-19-trade-policy-database-food-and-medical-products/", "eui_covid_trade_measures.csv"),
    "macmap_covid": ("ITC Market Access Map COVID-19 Temporary Trade Measures", "https://www.macmap.org/covid19", "macmap_covid_measures.csv"),
    "us_hts": ("US Harmonized Tariff Schedule / USITC", "https://hts.usitc.gov/", "us_hts.csv"),
}
PENDING = {
    "wto_idb_cts": "WTO IDB/CTS requires a pinned portal export or registered API subscription. Place the source-native CSV under a local path and run: python3 scripts/extract.py --source wto_idb_cts --input /path/to/export.csv",
    "global_trade_alert": "GTA data-center/API access is subject to non-commercial/license terms. Obtain an authorized export and run the explicit --input command documented in README.md.",
    "eui_covid": "The official EUI project page is a release archive, but a stable current machine-readable file URL was not pinned. Obtain the release file from the project page and run the explicit --input command documented in README.md.",
    "macmap_covid": "MACMap COVID tracker is interactive and download access depends on registration/account tier. Obtain an authorized export and run the explicit --input command documented in README.md.",
    "us_hts": "US HTS historical annual workbooks are release-dependent. Pin the required 2012-2022 annual files from the official USITC portal, convert/export to CSV, then run the explicit --input command documented in README.md.",
}


def geo(value: Any, name: Any = None, level: str = "country") -> dict[str, Any]:
    return {"code": code(value), "name": str(name).strip() if name else None, "level": level}


def product(value: Any, revision: Any = None, description: Any = None, level: str = "HS6") -> dict[str, Any]:
    raw = code(value)
    return {"code": raw.zfill(6) if raw and raw.isdigit() and len(raw) <= 6 else raw,
            "revision": code(revision), "level": level,
            "description": str(description).strip() if description else None}


def first(row: dict[str, Any], *names: str) -> Any:
    normalized = {str(key).strip().lower().replace(" ", "_"): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.lower().replace(" ", "_"))
        if value not in (None, ""):
            return value
    return None


def record_from_row(source_id: str, row: dict[str, Any], source_url: str, line_no: int) -> dict[str, Any] | None:
    source_name = SOURCE_INFO[source_id][0]
    start = parse_date(first(row, "period_start", "start_date", "effective_date", "date", "begin_date"))
    end = parse_date(first(row, "period_end", "end_date", "expiry_date", "termination_date"))
    if start is None:
        start = year_date(first(row, "year", "schedule_year", "tariff_year", "period"))
    if end is None and start is not None:
        # A tariff schedule year ends in December; an undated policy event is
        # open-ended as far as the source extract can establish and remains
        # active through the focus-window boundary.
        end = dt.date(start.year, 12, 31) if source_id in {"wits_trains", "wto_idb_cts", "us_hts"} else END
    interval = clamp_interval(start, end)
    if interval is None:
        return None
    start, end = interval
    reporter = geo(first(row, "reporter_iso3", "reporter", "reporter_code", "country", "economy", "reporting_country"), first(row, "reporter_name", "country_name", "economy_name"))
    partner_value = first(row, "partner_iso3", "partner", "partner_code", "destination", "importer")
    partner = geo(partner_value, first(row, "partner_name", "destination_name", "partner_country_name"), "country_or_world")
    if partner["code"] in {None, "", "WLD", "WORLD", "000"}:
        partner["level"] = "world_or_group"
    product_code = first(row, "hs6", "hs_code", "product", "product_code", "hts_code", "tariff_line")
    revision = first(row, "hs_revision", "nomenclature", "nomencode", "hs_year", "revision")
    product_level = "HTS tariff line" if source_id == "us_hts" else "HS6"
    prod = product(product_code, revision, first(row, "product_description", "description", "hs_description"), product_level)
    rate = parse_number(first(row, "tariff_rate", "rate", "duty_rate", "simple_average", "ad_valorem_rate", "duty"))
    unit = first(row, "rate_unit", "unit", "duty_unit")
    category = first(row, "measure_category", "measure_type", "category", "policy_type", "instrument")
    if not category:
        category = "tariff" if source_id in {"wits_trains", "wto_idb_cts", "us_hts"} else "trade_measure"
    record_type = "tariff" if source_id in {"wits_trains", "wto_idb_cts", "us_hts"} or category.lower() in {"tariff", "duty", "mfn", "preferential"} else "measure"
    subtype = first(row, "tariff_type", "rate_type", "measure_subtype", "subtype", "status", "action")
    medical_raw = first(row, "medical_flag", "medical", "medical_product", "medical_goods")
    medical_flag = None if medical_raw is None else str(medical_raw).strip().lower() in {"1", "true", "yes", "y", "medical"}
    source_record_id = first(row, "source_record_id", "id", "measure_id", "gta_id", "record_id") or f"line:{line_no}"
    return canonical_record(
        record_type=record_type, period_start=start, period_end=end,
        granularity="year" if record_type == "tariff" else "event",
        reporter=reporter, partner=partner, product=prod, category=str(category),
        subtype=str(subtype) if subtype is not None else None, rate=rate,
        unit=str(unit) if unit is not None else None, medical_flag=medical_flag,
        source_id=source_id, source_name=source_name, source_url=source_url,
        source_record_id=str(source_record_id), payload=row,
        release=first(row, "release", "dataset_version", "vintage"),
        legal_text=first(row, "legal_text", "measure_description", "notes"),
        notes="Source-native row normalized; fields not mapped to canonical columns are retained in source_payload_json.",
    )


def extract_manual(source_id: str, input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise RuntimeError(f"input does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise RuntimeError("manual normalization currently accepts CSV only; preserve the original workbook/archive in cache and export a source-native CSV")
    rows = read_csv(input_path)
    records = [record for n, row in enumerate(rows, 2) if (record := record_from_row(source_id, row, SOURCE_INFO[source_id][1], n)) is not None]
    output = SOURCE_INFO[source_id][2]
    count = write_records(output, records)
    # Preserve a content-addressed copy of manually supplied input inside the
    # package so the normalized output remains auditable across machines.
    raw = input_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    cached = CACHE / "manual_inputs" / f"{source_id}_{digest[:16]}.csv"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        shutil.copy2(input_path, cached)
    return {
        "status": "success", "records": count,
        "input": str(input_path.resolve()),
        "cache": str(cached.relative_to(ROOT)), "sha256": digest,
        "output": str((DATA / output).relative_to(ROOT)),
    }


def parse_wits_sdmx(payload: Any, reporter: str, partner: str, product_code: str, source_url: str) -> list[dict[str, Any]]:
    """Flatten common SDMX-JSON 2.0/2.1 WITS responses.

    WITS documentation exposes TRAINS fields such as SimpleAverage, TARIFFTYPE,
    NOMENCODE, and line counts. The parser supports both dimension metadata
    layouts commonly returned by SDMX endpoints and retains the observation array.
    """
    if not isinstance(payload, dict):
        raise RuntimeError("WITS response is not a JSON object")
    structure = payload.get("structure") or {}
    dimensions = structure.get("dimensions") or {}
    series_dims = dimensions.get("series") or []
    observation_dims = dimensions.get("observation") or []
    datasets = payload.get("dataSets") or payload.get("data", {}).get("dataSets") or []
    if not datasets:
        raise RuntimeError("WITS response has no dataSets; response may be an API error or unsupported format")
    times: list[str] = []
    for dim in observation_dims:
        dimension_id = str(dim.get("id", "")).lower().replace("-", "_")
        if dimension_id in {"timeperiod", "time_period", "time"}:
            times = [str(item.get("id", item.get("name", ""))) for item in dim.get("values", [])]
    if not series_dims or not times:
        raise RuntimeError("WITS response is missing series or time-period dimensions")
    observation_attributes = ((structure.get("attributes") or {}).get("observation") or [])
    attribute_names = [str(item.get("id", "")) for item in observation_attributes]
    attribute_values = [item.get("values", []) for item in observation_attributes]
    records: list[dict[str, Any]] = []
    series = datasets[0].get("series", {})
    if not isinstance(series, dict):
        raise RuntimeError("WITS response series is not an object")
    for key, series_value in series.items():
        indexes = [int(part) for part in str(key).split(":") if part != ""]
        resolved: dict[str, str] = {}
        for index, dimension in enumerate(series_dims):
            values = dimension.get("values", [])
            if index < len(indexes) and indexes[index] < len(values):
                item = values[indexes[index]]
                resolved[str(dimension.get("id", "")).lower()] = str(item.get("id", item.get("name", "")))
        observations = series_value.get("observations", {})
        for obs_index, values in observations.items():
            index = int(obs_index)
            year = times[index] if index < len(times) else ""
            start = year_date(year)
            end = year_date(year, end=True)
            if not start or not end or not (START.year <= start.year <= END.year):
                continue
            attrs: dict[str, Any] = {"observation_values": values}
            if isinstance(values, list):
                # SDMX stores the observation value first; the remaining
                # positions correspond to structure.attributes.observation.
                attrs["tariff_rate"] = values[0] if values else None
                for attr_index, attr_name in enumerate(attribute_names):
                    value_index = attr_index + 1
                    if value_index >= len(values) or not attr_name:
                        continue
                    attr_index_value = values[value_index]
                    choices = attribute_values[attr_index] if attr_index < len(attribute_values) else []
                    if attr_index_value is None:
                        attrs[attr_name.lower()] = None
                    elif isinstance(attr_index_value, int) and 0 <= attr_index_value < len(choices):
                        choice = choices[attr_index_value]
                        attrs[attr_name.lower()] = choice.get("id", choice.get("name"))
                    else:
                        attrs[attr_name.lower()] = attr_index_value
            row = {
                "year": year, "reporter_iso3": reporter, "partner_iso3": partner,
                "hs6": product_code, "tariff_rate": attrs.get("tariff_rate"),
                "tariff_type": attrs.get("tarifftype") or resolved.get("datasource"),
                "hs_revision": attrs.get("nomencode"), **attrs,
            }
            record = record_from_row("wits_trains", row, source_url, f"{key}:{year}")
            if record:
                records.append(record)
    return records


def extract_wits(args: argparse.Namespace) -> dict[str, Any]:
    if not all([args.reporter, args.partner, args.product]):
        raise RuntimeError("WITS requires --reporter, --partner, and --product; unbounded global pulls are not allowed")
    if len(str(args.product)) != 6 or not str(args.product).isdigit():
        raise RuntimeError("--product must be a six-digit HS code")
    endpoint = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
    key = f"A.{args.reporter}.{args.partner}.{args.product}.reported"
    query = urllib.parse.urlencode({"startPeriod": args.start_year, "endPeriod": args.end_year})
    url = f"{endpoint}DF_WITS_Tariff_TRAINS/{key}?{query}"
    payload, path = fetch_json(url, label=f"wits_trains_{args.reporter}_{args.partner}_{args.product}_{args.start_year}_{args.end_year}", refresh=args.refresh)
    records = parse_wits_sdmx(payload, str(args.reporter), str(args.partner), str(args.product), url)
    output = SOURCE_INFO["wits_trains"][2]
    output_path = DATA / output
    if records or not output_path.exists() or output_path.stat().st_size <= 1:
        count = write_records(output, records)
    else:
        # Keep a previously populated extract when a later valid query returns
        # no observations. The status still records the no-data result.
        count = 0
    status = "success" if records else "no_data"
    return {
        "status": status,
        "records": count,
        "output": str((DATA / output).relative_to(ROOT)),
        "cache": str(path.relative_to(ROOT)),
        "query_url": url,
        "details": "Valid WITS response contained no observations in the requested window." if not count else None,
    }


def write_pending(source_id: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    DATA.joinpath(f"{source_id.upper()}_PENDING.txt").write_text(PENDING[source_id] + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    requested = [args.source] if args.source else ["wto_idb_cts", "global_trade_alert", "eui_covid", "macmap_covid", "us_hts"]
    status_path = METADATA / "extraction_status.json"
    status: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(), "focus_window": [2012, 2022], "sources": {}}
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(previous.get("sources"), dict):
                status["sources"].update(previous["sources"])
        except (OSError, json.JSONDecodeError):
            pass
    failed = False
    for source_id in requested:
        if source_id == "wits_trains":
            try:
                result = extract_wits(args)
            except Exception as exc:
                result = {"status": "failed", "records": 0, "reason": str(exc)}
                failed = True
        elif source_id in PENDING and not args.input:
            write_pending(source_id)
            result = {"status": "pending_access", "records": 0, "reason": PENDING[source_id]}
        elif source_id in PENDING:
            try:
                result = extract_manual(source_id, Path(args.input))
            except Exception as exc:
                result = {"status": "failed", "records": 0, "reason": str(exc)}
                failed = True
        else:
            result = {"status": "failed", "records": 0, "reason": f"unknown source: {source_id}"}
            failed = True
        status["sources"][source_id] = result
        print(f"[{source_id}] {result['status']}: {result.get('records', 0)} records")
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="write pending notes for non-bounded/gated public sources")
    parser.add_argument("--source", choices=[*SOURCE_INFO], help="run one source")
    parser.add_argument("--input", help="source-native CSV for a manual/pending source")
    parser.add_argument("--reporter", help="WITS reporter code")
    parser.add_argument("--partner", help="WITS partner code; 000 means world")
    parser.add_argument("--product", help="WITS six-digit HS code")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--refresh", action="store_true", help="ignore cached HTTP response")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for source_id, (name, _, _) in SOURCE_INFO.items():
            print(f"{source_id}: {name}")
        return 0
    if args.start_year < 2012 or args.end_year > 2022 or args.start_year > args.end_year:
        parser.error("requested years must be within 2012-2022")
    if args.input and not args.source:
        parser.error("--input requires --source")
    if args.public and args.source:
        parser.error("use either --public or --source")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
