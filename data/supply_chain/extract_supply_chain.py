#!/usr/bin/env python3
"""Extract tidy 2012-2022 supply-chain data from the public source endpoints."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "supply-chain-data-extractor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def query_json(base: str, params: dict[str, object], timeout: int = 180) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    return json.loads(fetch(base + "?" + query, timeout).decode("utf-8"))


def post_json(base: str, params: dict[str, object], timeout: int = 180) -> dict:
    body = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
    req = urllib.request.Request(
        base,
        data=body,
        headers={"User-Agent": "supply-chain-data-extractor/1.0", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_gscpi() -> None:
    url = "https://www.newyorkfed.org/medialibrary/research/interactives/data/gscpi/gscpi_interactive_data.csv"
    rows = list(csv.reader(io.StringIO(fetch(url).decode("utf-8-sig"))))
    latest_col = len(rows[0]) - 1
    latest_vintage = rows[0][latest_col]
    out = []
    for row in rows[1:]:
        if not row or not row[0] or latest_col >= len(row):
            continue
        try:
            dt = datetime.strptime(row[0], "%d-%b-%Y")
        except ValueError:
            continue
        if date(2012, 1, 1) <= dt.date() <= date(2022, 12, 31):
            value = row[latest_col].strip()
            out.append([dt.strftime("%Y-%m"), float(value) if value else "", latest_vintage])
    write_csv(OUT / "nyfed_gscpi_monthly_2012_2022.csv", ["period", "gscpi", "latest_vintage"], out)


def extract_eia() -> None:
    base = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    rows = []
    for series in ("RBRTE", "RWTC"):
        payload = query_json(
            base,
            {
                "api_key": "DEMO_KEY",
                "frequency": "monthly",
                "data[0]": "value",
                "facets[series][]": series,
                "start": "2012-01",
                "end": "2022-12",
                "length": 5000,
            },
        )
        rows.extend(
            [
                [x.get("period"), x.get("series"), x.get("series-description"), x.get("value"), x.get("units")]
                for x in payload["response"]["data"]
            ]
        )
    rows.sort(key=lambda x: (x[0], x[1]))
    write_csv(OUT / "eia_crude_spot_prices_monthly_2012_2022.csv", ["period", "series", "series_description", "value", "units"], rows)


def extract_bts() -> None:
    base = "https://data.bts.gov/resource/y5ut-ibwt.json"
    payload = query_json(
        base,
        {
            "$select": "date,year,indicator,value1,units,source",
            "$where": "year between '2019' and '2022'",
            "$order": "date,indicator",
            "$limit": 50000,
        },
    )
    rows = [[x.get("date", "")[:10], x.get("year"), x.get("indicator"), x.get("value1"), x.get("units"), x.get("source")] for x in payload]
    write_csv(OUT / "bts_freight_indicators_2019_2022.csv", ["date", "year", "indicator", "value", "units", "source"], rows)


def extract_pink_sheet() -> None:
    url = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
    workbook = zipfile.ZipFile(io.BytesIO(fetch(url)))
    shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    shared = ["".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")) for si in shared_root.findall("m:si", NS)]
    sheet = ET.fromstring(workbook.read("xl/worksheets/sheet2.xml"))

    def cell_value(cell: ET.Element) -> str:
        value = cell.find("m:v", NS)
        text = value.text if value is not None else ""
        if cell.get("t") == "s" and text:
            return shared[int(text)]
        return text

    parsed = {}
    for row in sheet.findall(".//m:row", NS):
        row_values = {cell.get("r"): cell_value(cell) for cell in row.findall("m:c", NS)}
        parsed[int(row.get("r"))] = row_values
    headers = parsed[5]
    units = parsed[6]
    commodities = []
    for col in range(2, 200):
        letter = ""
        n = col
        while n:
            n, rem = divmod(n - 1, 26)
            letter = chr(65 + rem) + letter
        name = headers.get(f"{letter}5", "").strip()
        if name:
            commodities.append((letter, name, units.get(f"{letter}6", "").strip()))

    out = []
    for rownum, row in parsed.items():
        period = row.get(f"A{rownum}", "")
        if not re.fullmatch(r"20(?:12|1[3-9]|2[0-2])M(?:0[1-9]|1[0-2])", period):
            continue
        for letter, commodity, unit in commodities:
            raw = row.get(f"{letter}{rownum}", "")
            value = "" if raw in ("", "…", "..", "NA", "N/A") else raw
            out.append([period[:4] + "-" + period[-2:], commodity, unit, value])
    write_csv(OUT / "worldbank_pink_sheet_monthly_2012_2022.csv", ["period", "commodity", "unit", "value"], out)


def excel_serial_to_date(serial: str) -> date:
    return (datetime(1899, 12, 30) + timedelta(days=float(serial))).date()


def extract_rwi() -> None:
    url = "https://www.rwi-essen.de/fileadmin/user_upload/RWI/Presse/containerumschlag-Index_260629.xlsx"
    workbook = zipfile.ZipFile(io.BytesIO(fetch(url)))
    shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    shared = ["".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")) for si in shared_root.findall("m:si", NS)]
    sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    def cell_value(cell: ET.Element) -> str:
        value = cell.find("m:v", NS)
        text = value.text if value is not None else ""
        if cell.get("t") == "s" and text:
            return shared[int(text)]
        return text

    out = []
    for row in sheet.findall(".//m:row", NS):
        vals = {cell.get("r")[0]: cell_value(cell) for cell in row.findall("m:c", NS)}
        if "A" not in vals or not vals["A"]:
            continue
        try:
            dt = excel_serial_to_date(vals["A"])
        except (TypeError, ValueError):
            continue
        if date(2012, 1, 1) <= dt <= date(2022, 12, 31):
            out.append([dt.strftime("%Y-%m"), *[vals.get(c, "") for c in "BCDEFG"]])
    write_csv(
        OUT / "rwi_isl_container_throughput_monthly_2012_2022.csv",
        ["period", "total_original", "total_seasonally_adjusted", "total_trend_cycle", "north_range_original", "north_range_seasonally_adjusted", "north_range_trend_cycle"],
        out,
    )


def extract_portwatch() -> None:
    base = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0/query"
    fields = [
        "portcalls_container", "portcalls_dry_bulk", "portcalls_general_cargo", "portcalls_roro", "portcalls_tanker", "portcalls_cargo", "portcalls",
        "import_container", "import_dry_bulk", "import_general_cargo", "import_roro", "import_tanker", "import_cargo", "import",
        "export_container", "export_dry_bulk", "export_general_cargo", "export_roro", "export_tanker", "export_cargo", "export",
    ]
    stats = [{"statisticType": "sum", "onStatisticField": field, "outStatisticFieldName": f"sum_{field}"} for field in fields]
    rows = []
    for year in range(2019, 2023):
        offset = 0
        while True:
            payload = post_json(
                base,
                {
                    "where": f"year = {year}",
                    "outStatistics": json.dumps(stats, separators=(",", ":")),
                    "groupByFieldsForStatistics": "year,month,ISO3,country",
                    "outFields": "year,month,ISO3,country",
                    "orderByFields": "year,month,ISO3",
                    "returnGeometry": "false",
                    "resultOffset": offset,
                    "resultRecordCount": 1000,
                    "f": "json",
                },
                timeout=300,
            )
            if "error" in payload:
                raise RuntimeError(f"PortWatch query failed for {year}: {payload['error']}")
            features = payload.get("features", [])
            for feature in features:
                a = feature["attributes"]
                rows.append([a.get("year"), a.get("month"), a.get("ISO3"), a.get("country"), *[a.get(f"sum_{field}") for field in fields]])
            if not payload.get("exceededTransferLimit") or not features:
                break
            offset += len(features)
    rows.sort(key=lambda x: (x[0], x[1], x[2] or ""))
    write_csv(OUT / "imf_portwatch_country_monthly_2019_2022.csv", ["year", "month", "ISO3", "country", *fields], rows)

def main() -> None:
    extract_gscpi()
    extract_eia()
    extract_bts()
    extract_pink_sheet()
    extract_rwi()
    extract_portwatch()
    print(f"Wrote tidy extracts to {OUT}")


if __name__ == "__main__":
    main()
