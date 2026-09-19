"""Extract the economics_us source manifest into normalized JSONL.

Examples:
    python3 scripts/extract.py --list
    python3 scripts/extract.py --source public
    python3 scripts/extract.py --source fred_ppi --refresh

The command is intentionally best-effort: one unavailable source is recorded
as failed/skipped while independent sources continue. Raw response bodies are
cached by scripts.common.fetch before parsing.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    START_YEAR,
    FetchError,
    canonical_record,
    csv_rows,
    _redact_url,
    get_json,
    get_text,
    national_geo,
    parse_date_period,
    parse_number,
    post_json,
    state_geo,
    utc_now,
    write_json,
    write_partitioned,
)

SOURCE_NAMES = {
    "fred_ppi": "FRED current observations / BLS PPI",
    "fred_alfred": "ALFRED real-time vintages",
    "bea_regional": "BEA Regional",
    "bls_api": "BLS LAUS / CES / CPI / PPI",
    "bls_qcew": "BLS Quarterly Census of Employment and Wages",
    "census_cbp": "Census County Business Patterns",
    "census_asm": "Census Annual Survey of Manufactures / Economic Census",
    "census_construction": "Census Construction Spending / VIP",
    "fed_g17": "Federal Reserve Industrial Production G.17",
    "philly_state_coincident": "Philadelphia Fed State Coincident Indexes",
    "usaspending": "USAspending federal contracts",
    "census_bds": "Census Business Dynamics Statistics",
}
PUBLIC_SOURCES = ["fred_ppi", "bls_api", "bls_qcew", "census_cbp", "fed_g17", "usaspending"]
KEYS = {"fred_alfred": "FRED_API_KEY", "bea_regional": "BEA_API_KEY", "census_cbp": "CENSUS_API_KEY"}


def source_url(source_id: str) -> str:
    return {
        "fred_ppi": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCU32543254",
        "fred_alfred": "https://api.stlouisfed.org/fred/series/observations",
        "bea_regional": "https://apps.bea.gov/api/data",
        "bls_api": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        "bls_qcew": "https://www.bls.gov/cew/downloadable-data-files.htm",
        "census_cbp": "https://api.census.gov/data/{year}/cbp",
        "fed_g17": "https://www.federalreserve.gov/releases/g17/",
        "usaspending": "https://api.usaspending.gov/",
    }.get(source_id, "")


def records_for_bls_series(
    payload: dict[str, Any],
    series_info: dict[str, tuple[str, str, dict[str, Any]]],
    *,
    source_id: str = "bls_api",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for series in payload.get("Results", {}).get("series", []):
        series_id = series.get("seriesID", "")
        metric, unit, geography = series_info.get(series_id, (series_id, "source_native", national_geo()))
        for item in series.get("data", []):
            year = int(item.get("year", 0)) if str(item.get("year", "")).isdigit() else 0
            if not START_YEAR <= year <= END_YEAR:
                continue
            start, end, granularity = parse_date_period(year, item.get("period"))
            value = parse_number(item.get("value"))
            if value is None:
                continue
            records.append(canonical_record(
                source_id=source_id,
                source_name=SOURCE_NAMES[source_id],
                source_url=_redact_url(source_url(source_id)),
                source_record_id=f"{series_id}:{item.get('year')}:{item.get('period')}",
                period_start=start,
                period_end=end,
                granularity=granularity,
                geography=geography,
                metric=metric,
                value=value,
                unit=unit,
                semantics=item.get("footnotes") or item.get("periodName") or metric,
            ))
    return records


def extract_fred_ppi(refresh: bool) -> dict[str, Any]:
    url = source_url("fred_ppi")
    text, path = get_text(url, label="fred_ppi_current", refresh=refresh, content_suffix=".csv")
    records: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines()):
        date = row.get("observation_date", "")
        value = parse_number(row.get("PCU32543254"))
        if not date or value is None or not (f"{START_YEAR}-01-01" <= date <= f"{END_YEAR}-12-31"):
            continue
        month = int(date[5:7])
        next_month = dt.date(int(date[:4]) + (month == 12), (month % 12) + 1, 1)
        records.append(canonical_record(
            source_id="fred_ppi", source_name=SOURCE_NAMES["fred_ppi"],                source_url=_redact_url(url),

            source_record_id=f"PCU32543254:{date}", period_start=date,
            period_end=(next_month - dt.timedelta(days=1)).isoformat(), granularity="month",
            geography=national_geo(), metric="pharma_producer_price_index", value=value,
            unit="index", semantics="Producer Price Index: Pharmaceutical and Medicine Manufacturing",
        ))
    return {        "status": "success", "records": records, "cache": str(path), "details": {"series_id": "PCU32543254"}}



def extract_fred_alfred(refresh: bool) -> dict[str, Any]:
    key = os.getenv("FRED_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "missing FRED_API_KEY", "records": []}
    records: list[dict[str, Any]] = []
    cache_files: list[str] = []
    for year in range(START_YEAR, END_YEAR + 1):
        vintage = f"{year}-12-31"
        params = urllib.parse.urlencode({
            "series_id": "PCU32543254", "api_key": key, "file_type": "json",
            "realtime_start": vintage, "realtime_end": vintage,
            "observation_start": f"{year}-01-01", "observation_end": f"{year}-12-31",
        })
        url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
        payload, path = get_json(url, label=f"fred_alfred_{year}", refresh=refresh, content_suffix=".json")
        cache_files.append(str(path))
        for item in payload.get("observations", []):
            value = parse_number(item.get("value"))
            date = item.get("date", "")
            if value is None or not date:
                continue
            month = int(date[5:7])
            next_month = dt.date(int(date[:4]) + (month == 12), (month % 12) + 1, 1)
            records.append(canonical_record(
                source_id="fred_alfred", source_name=SOURCE_NAMES["fred_alfred"],                source_url=_redact_url(url),

                source_record_id=f"PCU32543254:{date}:{vintage}", period_start=date,
                period_end=(next_month - dt.timedelta(days=1)).isoformat(), granularity="month",
                geography=national_geo(), metric="pharma_producer_price_index", value=value,
                unit="index", semantics="ALFRED point-in-time PPI observation", vintage_date=vintage,
            ))
    return {"status": "success", "records": records, "cache": cache_files}


def extract_bea_regional(refresh: bool) -> dict[str, Any]:
    key = os.getenv("BEA_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "missing BEA_API_KEY", "records": []}
    years = ",".join(str(y) for y in range(START_YEAR, END_YEAR + 1))
    params = urllib.parse.urlencode({            "UserID": key, "method": "GetData", "DatasetName": "Regional",

        "TableName": "SAGDP9N", "LineCode": "1", "GeoFIPS": "STATE",
        "Year": years, "ResultFormat": "JSON",
    })
    url = f"https://apps.bea.gov/api/data?{params}"
    payload, path = get_json(url, label="bea_regional_sagdp9n", refresh=refresh, content_suffix=".json")
    rows = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    records: list[dict[str, Any]] = []
    for row in rows:
        year = str(row.get("TimePeriod", ""))
        value = parse_number(row.get("DataValue"))
        code = str(row.get("GeoFIPS", "")).strip().replace("'", "")
        if value is None or not year.isdigit() or not START_YEAR <= int(year) <= END_YEAR or len(code) < 2:
            continue
        records.append(canonical_record(
            source_id="bea_regional", source_name=SOURCE_NAMES["bea_regional"],                source_url=_redact_url(url),

            source_record_id=f"{code}:{year}:SAGDP9N:1", period_start=f"{year}-01-01",
            period_end=f"{year}-12-31", granularity="year", geography=state_geo(code[:2].zfill(2), row.get("GeoName")),
            metric="state_gdp", value=value, unit=str(row.get("CL_UNIT", "USD millions")),
            semantics="BEA Regional SAGDP9N line 1; source-native GDP value (nominal table)",
        ))
    return {"status": "success", "records": records, "cache": str(path), "details": {"table": "SAGDP9N", "line_code": "1"}}


def extract_bls_api(refresh: bool) -> dict[str, Any]:
    # LAUS state unemployment-rate IDs use the two-digit state FIPS in the
    # standard LAUST{FIPS}0000000000003 form. Include all states/DC.
    states = {
        "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas","06":"California","08":"Colorado","09":"Connecticut","10":"Delaware","11":"District of Columbia","12":"Florida","13":"Georgia","15":"Hawaii","16":"Idaho","17":"Illinois","18":"Indiana","19":"Iowa","20":"Kansas","21":"Kentucky","22":"Louisiana","23":"Maine","24":"Maryland","25":"Massachusetts","26":"Michigan","27":"Minnesota","28":"Mississippi","29":"Missouri","30":"Montana","31":"Nebraska","32":"Nevada","33":"New Hampshire","34":"New Jersey","35":"New Mexico","36":"New York","37":"North Carolina","38":"North Dakota","39":"Ohio","40":"Oklahoma","41":"Oregon","42":"Pennsylvania","44":"Rhode Island","45":"South Carolina","46":"South Dakota","47":"Tennessee","48":"Texas","49":"Utah","50":"Vermont","51":"Virginia","53":"Washington","54":"West Virginia","55":"Wisconsin","56":"Wyoming"
    }
    series_info: dict[str, tuple[str, str, dict[str, Any]]] = {}
    series_ids: list[str] = []
    for fips, name in states.items():
        series_id = f"LAUST{fips}0000000000003"
        series_ids.append(series_id)
        series_info[series_id] = ("state_unemployment_rate", "percent", state_geo(fips, name))
    fixed = {
        "CEU3000000001": ("manufacturing_employment", "thousands_of_jobs", national_geo()),
        "CUUR0000SA0": ("consumer_price_index_all_items", "index", national_geo()),
        "PCU32543254": ("pharma_producer_price_index", "index", national_geo()),
    }
    series_info.update(fixed)
    series_ids.extend(fixed)
    records: list[dict[str, Any]] = []
    cache_files: list[str] = []
    for offset in range(0, len(series_ids), 50):
        chunk = series_ids[offset:offset + 50]
        request = {"seriesid": chunk, "startyear": str(START_YEAR), "endyear": str(END_YEAR)}
        if os.getenv("BLS_API_KEY"):
            request["registrationkey"] = os.getenv("BLS_API_KEY")
        payload, path = post_json(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/", request,
            label=f"bls_api_{offset}", refresh=refresh, content_suffix=".json",
        )
        cache_files.append(str(path))
        if payload.get("status") not in (None, "REQUEST_SUCCEEDED"):
            raise FetchError(f"BLS API returned {payload.get('message') or payload.get('status')}")
        records.extend(records_for_bls_series(payload, series_info))
    return {"status": "success", "records": records, "cache": cache_files, "details": {"series_count": len(series_ids)}}


def extract_bls_qcew(refresh: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cache_files: list[str] = []
    state_fips = {f"{i:02d}" for i in [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56]}
    for year in range(START_YEAR, END_YEAR + 1):
        url = f"https://data.bls.gov/cew/data/api/{year}/a/industry/3254.csv"
        text, path = get_text(url, label=f"bls_qcew_{year}_3254", refresh=refresh, content_suffix=".csv")
        cache_files.append(str(path))
        for row in csv_rows(text):
            area = str(row.get("area_fips", "")).strip().zfill(5)
            if area[:2] not in state_fips or area[2:] != "000":
                continue
            name = row.get("area_title") or area[:2]
            metrics = [
                ("qcew_establishments", row.get("annual_avg_estabs"), "establishments", "annual average establishments"),
                ("qcew_employment", row.get("annual_avg_emplvl"), "employees", "annual average employment"),
                ("qcew_total_annual_wages", row.get("total_annual_wages"), "USD", "total annual wages"),
                ("qcew_avg_annual_pay", row.get("avg_annual_pay"), "USD_per_employee", "average annual pay"),
            ]
            for metric, raw, unit, semantics in metrics:
                value = parse_number(raw)
                if value is None:
                    continue
                records.append(canonical_record(
                    source_id="bls_qcew", source_name=SOURCE_NAMES["bls_qcew"],                source_url=_redact_url(url),

                    source_record_id=f"{year}:{area}:3254:{metric}", period_start=f"{year}-01-01",
                    period_end=f"{year}-12-31", granularity="year", geography=state_geo(area[:2], name),
                    metric=metric, value=value, unit=unit, semantics=semantics,
                    notes="QCEW source-native annual industry value; suppressed cells omitted",
                ))
    return {"status": "success", "records": records, "cache": cache_files, "details": {"naics": "3254"}}


def extract_census_cbp(refresh: bool) -> dict[str, Any]:
    key = os.getenv("CENSUS_API_KEY")
    records: list[dict[str, Any]] = []
    cache_files: list[str] = []
    for year in range(START_YEAR, END_YEAR + 1):
        naics_field = "NAICS2012" if year <= 2016 else "NAICS2017"
        params = {"get": "NAME,ESTAB,EMP,PAYANN", "for": "state:*", naics_field: "3254"}
        if key:
            params["key"] = key
        url = f"https://api.census.gov/data/{year}/cbp?{urllib.parse.urlencode(params)}"
        payload, path = get_json(url, label=f"census_cbp_{year}_3254", refresh=refresh, content_suffix=".json")
        cache_files.append(str(path))
        if not isinstance(payload, list) or len(payload) < 2:
            raise FetchError(f"Census CBP {year} returned unexpected shape")
        headers = payload[0]
        for values in payload[1:]:
            row = dict(zip(headers, values))
            fips, name = row.get("state", ""), row.get("NAME")
            for metric, raw, unit, semantics in [
                ("cbp_establishments", row.get("ESTAB"), "establishments", "County Business Patterns establishments"),
                ("cbp_employment", row.get("EMP"), "employees", "County Business Patterns employment"),
                ("cbp_annual_payroll", row.get("PAYANN"), "USD_thousands", "County Business Patterns annual payroll"),
            ]:
                value = parse_number(raw)
                if value is None:
                    continue
                records.append(canonical_record(
                    source_id="census_cbp", source_name=SOURCE_NAMES["census_cbp"],                source_url=_redact_url(url),

                    source_record_id=f"{year}:{fips}:3254:{metric}", period_start=f"{year}-01-01",
                    period_end=f"{year}-12-31", granularity="year", geography=state_geo(fips, name),
                    metric=metric, value=value, unit=unit, semantics=semantics,
                    notes=f"CBP {naics_field}=3254; source-native payroll units",
                ))
    return {"status": "success", "records": records, "cache": cache_files, "details": {"naics": "3254"}}


def extract_fed_g17(refresh: bool) -> dict[str, Any]:
    # G.17/FRED identifiers have changed across revisions; try only explicit
    # candidates and record which one is found rather than claiming a proxy.
    candidates = ["IPUAN3254T011000000", "IPUAN3254T010000000", "IPUAN3254T001000000"]
    errors: list[str] = []
    for series_id in candidates:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            text, path = get_text(url, label=f"fed_g17_{series_id}", refresh=refresh, content_suffix=".csv")
        except FetchError as exc:
            errors.append(str(exc))
            continue
        rows = list(csv.DictReader(text.splitlines()))
        if not rows or series_id not in rows[0]:
            errors.append(f"{series_id}: no expected CSV column")
            continue
        records: list[dict[str, Any]] = []
        for row in rows:
            date = row.get("observation_date", "")
            value = parse_number(row.get(series_id))
            if not date or value is None or not (f"{START_YEAR}-01-01" <= date <= f"{END_YEAR}-12-31"):
                continue
            month = int(date[5:7])
            end = dt.date(int(date[:4]) + (month == 12), (month % 12) + 1, 1) - dt.timedelta(days=1)
            records.append(canonical_record(
                source_id="fed_g17", source_name=SOURCE_NAMES["fed_g17"],                source_url=_redact_url(url),

                source_record_id=f"{series_id}:{date}", period_start=date, period_end=end.isoformat(),
                granularity="month", geography=national_geo(), metric="pharma_industrial_production_index",
                value=value, unit="index", semantics="G.17/FRED candidate series; identifier recorded",
                notes=f"selected candidate {series_id}",
            ))
        if records:
            return {"status": "success", "records": records, "cache": str(path), "details": {"series_id": series_id, "candidate_errors": errors}}
    return {"status": "failed", "reason": "no candidate G.17 pharma series resolved", "errors": errors, "records": []}


def extract_usaspending(refresh: bool) -> dict[str, Any]:
    url =        "https://api.usaspending.gov/api/v2/search/spending_over_time/"

    payload = {
        "group": "fiscal_year",
        "filters": {
            "time_period": [{"start_date": "2011-10-01", "end_date": "2022-09-30"}],
            "naics_codes": ["3254"],
        },
    }
    response, path = post_json(url, payload, label="usaspending_naics_3254_fy2012_2022", refresh=refresh, content_suffix=".json")
    records: list[dict[str, Any]] = []
    # Current responses commonly return one object per fiscal year with a
    # `time_period` list and `aggregated_amount`. Preserve that source-native
    # amount as obligations; do not invent outlays when they are absent.
    result_rows = response.get("results", []) if isinstance(response, dict) else []
    flattened: list[dict[str, Any]] = []
    for row in result_rows:
        if not isinstance(row, dict):
            continue
        periods = row.get("time_period") or row.get("timePeriod")
        if isinstance(periods, list):
            for period in periods:
                if isinstance(period, dict):
                    flattened.append({**row, **period})
        else:
            flattened.append(row)
    for row in flattened:
        fy_raw = row.get("fiscal_year") or row.get("fiscalYear") or row.get("time_period") or row.get("timePeriod")
        text = str(fy_raw or "")
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 4:
            continue
        fy = int(digits[-4:])
        if not START_YEAR <= fy <= END_YEAR:
            continue
        obligations = parse_number(row.get("obligations") or row.get("Obligations") or row.get("aggregated_amount"))
        outlays = parse_number(row.get("outlays") or row.get("Outlays"))
        for metric, value, semantics in [
            ("contract_obligations", obligations, "USAspending obligations for NAICS 3254"),
            ("contract_outlays", outlays, "USAspending outlays for NAICS 3254"),
        ]:
            if value is None:
                continue
            records.append(canonical_record(
                source_id="usaspending", source_name=SOURCE_NAMES["usaspending"],                source_url=_redact_url(url),

                source_record_id=f"FY{fy}:3254:{metric}", period_start=f"{fy-1}-10-01",
                period_end=f"{fy}-09-30", granularity="fiscal_year", geography=national_geo(),
                metric=metric, value=value, unit="USD", semantics=semantics,
                notes="Federal fiscal year; NAICS filter is source query parameter",
            ))
    return {"status": "success", "records": records, "cache": str(path), "details": {"naics": "3254", "fiscal_years": "2012-2022"}}


def not_automated(source_id: str) -> dict[str, Any]:
    return {"status": "not_automated", "reason": "publication schema/table selection requires explicit source file pinning", "records": [], "details": {"source_id": source_id}}


EXTRACTORS: dict[str, Callable[[bool], dict[str, Any]]] = {
    "fred_ppi": extract_fred_ppi,
    "fred_alfred": extract_fred_alfred,
    "bea_regional": extract_bea_regional,
    "bls_api": extract_bls_api,
    "bls_qcew": extract_bls_qcew,
    "census_cbp": extract_census_cbp,
    "census_asm": lambda refresh: not_automated("census_asm"),
    "census_construction": lambda refresh: not_automated("census_construction"),
    "fed_g17": extract_fed_g17,
    "philly_state_coincident": lambda refresh: not_automated("philly_state_coincident"),
    "usaspending": extract_usaspending,
    "census_bds": lambda refresh: not_automated("census_bds"),
}


def run(source_ids: list[str], refresh: bool) -> int:
    status: dict[str, Any] = {"generated_at": utc_now(), "focus_window": [START_YEAR, END_YEAR], "sources": {}}
    overall_failure = False
    for source_id in source_ids:
        if source_id not in EXTRACTORS:
            status["sources"][source_id] = {"status": "unknown", "records": 0}
            overall_failure = True
            continue
        print(f"[{source_id}] extracting...")
        try:
            result = EXTRACTORS[source_id](refresh)
            records = result.pop("records", [])
            if result.get("status") == "success":
                # Replace old partitions even when a successful response now
                # contains zero rows; failed jobs must not destroy good data.
                result["partitions"] = write_partitioned(source_id, records, clear=True)
            result["records"] = len(records)
            status["sources"][source_id] = result
            print(f"[{source_id}] {result.get('status')}: {len(records)} records")
            if result.get("status") == "failed":
                overall_failure = True
        except Exception as exc:  # keep independent source jobs running
            status["sources"][source_id] = {"status": "failed", "reason": str(exc), "records": 0}
            overall_failure = True
            print(f"[{source_id}] failed: {exc}", file=sys.stderr)
    write_json(Path(__file__).resolve().parents[1] / "metadata" / "extraction_status.json", status)
    return 1 if overall_failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=[*EXTRACTORS, "public", "all"], help="source ID; repeatable")
    parser.add_argument("--refresh", action="store_true", help="ignore cached HTTP responses")
    parser.add_argument("--list", action="store_true", help="list source IDs and exit")
    args = parser.parse_args()
    if args.list:
        for source_id in EXTRACTORS:
            auth = f" (requires {KEYS[source_id]})" if source_id in KEYS else ""
            print(f"{source_id}{auth}: {SOURCE_NAMES[source_id]}")
        return 0
    requested = args.source or ["public"]
    source_ids: list[str] = []
    for item in requested:
        expanded = list(EXTRACTORS) if item == "all" else PUBLIC_SOURCES if item == "public" else [item]
        for source_id in expanded:
            if source_id not in source_ids:
                source_ids.append(source_id)
    return run(source_ids, args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
