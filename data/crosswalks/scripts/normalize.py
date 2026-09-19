#!/usr/bin/env python3
"""Normalize raw crosswalk downloads using Python's standard library."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
import urllib.error
import urllib.parse
import urllib.request
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data"
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNAV_URL = f"{RXNAV_BASE}/allNDCstatus.json"
GLEIF_URL = "https://api.gleif.org/api/v1/lei-records"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> int:
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    temporary.replace(path)
    return count


def rxnav_json(path: str) -> dict[str, object]:
    url = f"{RXNAV_BASE}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "meditrack-crosswalk-extractor/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_rxnav() -> dict[str, object]:
    source = CACHE / "rxnav_allNDCstatus.json"
    out = DATA / "rxnav_ndc_catalog.csv"
    version = CACHE / "rxnav_version.json"
    if not source.exists():
        return {"status": "pending", "reason": f"missing {source.relative_to(ROOT)}"}
    payload = json.loads(source.read_text(encoding="utf-8"))
    ndcs = ((payload.get("ndcList") or {}).get("ndc") or [])
    count = write_csv(out, ["ndc", "source_id"], ({"ndc": str(ndc), "source_id": "RXNORM"} for ndc in ndcs))
    if version.exists():
        (DATA / "rxnav_version.json").write_text(version.read_text(encoding="utf-8"), encoding="utf-8")
    return {"status": "ok", "rows": count, "output": str(out.relative_to(ROOT)), "source_url": RXNAV_URL}


def lookup_rxnav_ndcs(input_path: Path, output_path: Path, delay: float = 0.1) -> dict[str, object]:
    """Resolve caller-supplied NDCs to RxCUIs and ingredient concepts.

    RxNav exposes reliable per-identifier lookup but no practical public bulk
    NDC→RxCUI export. This deliberately requires an input file and never
    pretends that the all-NDC codelist is a mapping.
    """
    ndcs: list[str] = []
    with input_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "ndc" not in (reader.fieldnames or []):
            raise ValueError("input CSV must contain an ndc column")
        for row in reader:
            ndc = clean(row.get("ndc", ""))
            if ndc:
                ndcs.append(ndc)
    fields = ["ndc", "rxcui", "rxcui_name", "rxcui_tty", "ingredient_rxcui", "ingredient_name", "status"]
    temporary = output_path.with_name(output_path.name + ".tmp")
    count = 0
    error_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ndc in dict.fromkeys(ndcs):
            try:
                payload = rxnav_json("rxcui.json?" + urllib.parse.urlencode({"idtype": "NDC", "id": ndc}))
                rxcuis = ((payload.get("idGroup") or {}).get("rxnormId") or [])
                if not rxcuis:
                    writer.writerow({"ndc": ndc, "status": "not_found"})
                    count += 1
                    continue
                for rxcui in rxcuis:
                    props = rxnav_json(f"rxcui/{rxcui}/properties.json").get("properties") or {}
                    related = rxnav_json(f"rxcui/{rxcui}/related.json?tty=IN")
                    ingredients = []
                    for group in ((related.get("relatedGroup") or {}).get("conceptGroup") or []):
                        ingredients.extend(group.get("conceptProperties") or [])
                    if not ingredients:
                        ingredients = [{}]
                    for ingredient in ingredients:
                        writer.writerow({"ndc": ndc, "rxcui": rxcui, "rxcui_name": props.get("name", ""), "rxcui_tty": props.get("tty", ""), "ingredient_rxcui": ingredient.get("rxcui", ""), "ingredient_name": ingredient.get("name", ""), "status": "ok"})
                        count += 1
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                writer.writerow({"ndc": ndc, "status": f"error:{type(exc).__name__}"})
                count += 1
                error_count += 1
            time.sleep(delay)
    temporary.replace(output_path)
    try:
        output_name = str(output_path.resolve().relative_to(ROOT))
    except ValueError:
        output_name = str(output_path)
    return {"status": "partial" if error_count else "ok", "rows": count, "errors": error_count, "output": output_name, "source_url": RXNAV_BASE}


def flatten_gleif(record: dict[str, object]) -> dict[str, object]:
    attrs = record.get("attributes") or {}
    entity = attrs.get("entity") or {}
    registration = attrs.get("registration") or {}
    relationships = attrs.get("relationships") or {}
    direct_parent = relationships.get("direct_parent") or {}
    ultimate_parent = relationships.get("ultimate_parent") or {}
    def rel_id(item: object) -> str:
        return str(((item if isinstance(item, dict) else {}).get("data") or {}).get("id") or "")
    names = entity.get("legalName") or {}
    legal_name = names.get("name") if isinstance(names, dict) else names
    return {
        "lei": record.get("id", ""),
        "legal_name": legal_name or "",
        "entity_status": entity.get("status", ""),
        "legal_form_id": ((entity.get("legalForm") or {}).get("id", "") if isinstance(entity.get("legalForm"), dict) else ""),
        "legal_address_country": ((entity.get("legalAddress") or {}).get("country", "") if isinstance(entity.get("legalAddress"), dict) else ""),
        "headquarters_country": ((entity.get("headquartersAddress") or {}).get("country", "") if isinstance(entity.get("headquartersAddress"), dict) else ""),
        "registration_status": registration.get("status", ""),
        "initial_registration_date": registration.get("initialRegistrationDate", ""),
        "last_update_date": registration.get("lastUpdateDate", ""),
        "next_renewal_date": registration.get("nextRenewalDate", ""),
        "direct_parent_lei": rel_id(direct_parent),
        "ultimate_parent_lei": rel_id(ultimate_parent),
        "raw_json": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    }


def normalize_gleif() -> dict[str, object]:
    files = sorted(CACHE.glob("gleif_lei_page_*.json"))
    out = DATA / "gleif_lei_records.jsonl"
    if not files:
        return {"status": "pending", "reason": "no GLEIF API page downloaded"}
    fields = ["lei", "legal_name", "entity_status", "legal_form_id", "legal_address_country", "headquarters_country", "registration_status", "initial_registration_date", "last_update_date", "next_renewal_date", "direct_parent_lei", "ultimate_parent_lei", "raw_json"]
    temporary = out.with_name(out.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as fh:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("data") or []:
                fh.write(json.dumps(flatten_gleif(record), ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    temporary.replace(out)
    return {"status": "ok", "rows": count, "output": str(out.relative_to(ROOT)), "source_url": GLEIF_URL}


def xlsx_rows(path: Path) -> list[list[str]]:
    """Read simple shared-string/inline-string XLSX sheets without openpyxl."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            for si in root.findall(f"{ns}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(archive.read(sheet_name))
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        rows: list[list[str]] = []
        for row in root.findall(f".//{ns}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{ns}c"):
                ref = cell.attrib.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                col = 0
                for ch in letters:
                    col = col * 26 + ord(ch.upper()) - 64
                value = ""
                if cell.attrib.get("t") == "s":
                    node = cell.find(f"{ns}v")
                    if node is not None and node.text:
                        value = shared[int(node.text)]
                elif cell.attrib.get("t") == "inlineStr":
                    node = cell.find(f".{ns}is/{ns}t")
                    value = node.text if node is not None and node.text else ""
                else:
                    node = cell.find(f"{ns}v")
                    value = node.text if node is not None and node.text else ""
                if col:
                    values[col] = value
            if values:
                rows.append([values.get(i, "") for i in range(1, max(values) + 1)])
        return rows


def normalize_census() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in sorted(CACHE.glob("census_*.xlsx")):
        try:
            rows = xlsx_rows(path)
            if not rows:
                raise ValueError("workbook has no rows")
            header = [clean(x) or f"column_{i + 1}" for i, x in enumerate(rows[0])]
            out = DATA / (path.stem + ".csv")
            count = write_csv(out, header, ({header[i]: clean(value) for i, value in enumerate(row)} for row in rows[1:]))
            results.append({"status": "ok", "rows": count, "output": str(out.relative_to(ROOT)), "raw_file": str(path.relative_to(ROOT))})
        except Exception as exc:
            results.append({"status": "failed", "raw_file": str(path.relative_to(ROOT)), "reason": f"{type(exc).__name__}: {exc}"})
    for path in sorted(CACHE.glob("census_*.xls")):
        results.append({"status": "pending_parser", "raw_file": str(path.relative_to(ROOT)), "reason": "legacy XLS requires an external parser; retained without lossy conversion"})
    if not results:
        results.append({"status": "pending", "reason": "no Census spreadsheets in cache"})
    return results


def normalize_dartmouth() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    out = DATA / "dartmouth_zip_hsa_hrr.csv"
    # Rebuild into a temporary file and replace the prior valid extract only
    # after at least one source file has parsed successfully. This prevents a
    # failed/empty rerun from destroying a usable output.
    temporary_out = out.with_name(out.name + ".tmp")
    temporary_out.unlink(missing_ok=True)
    successful_files = 0
    for path in sorted(CACHE.glob("dartmouth_*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith((".csv", ".txt"))]
                if not members:
                    raise ValueError("archive contains no CSV/TXT member")
                member = members[0]
                raw = archive.read(member).decode("utf-8-sig", "replace")
            lines = raw.splitlines()
            if not lines:
                raise ValueError("empty CSV")
            dialect = csv.Sniffer().sniff("\n".join(lines[:5]), delimiters=",\t;|")
            reader = csv.reader(lines, dialect)
            rows = list(reader)
            header = [clean(x) or f"column_{i + 1}" for i, x in enumerate(rows[0])]
            # Multiple annual files share a single normalized output; append via
            # a source_vintage column while preserving all source-native columns.
            source_vintage = re.search(r"ZipHsaHrr(\d{2})", path.name, flags=re.I)
            vintage = ("20" + source_vintage.group(1)) if source_vintage else ""
            fields = ["source_vintage"] + header
            mode = "w" if not temporary_out.exists() else "a"
            with temporary_out.open(mode, encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                if mode == "w":
                    writer.writeheader()
                count = 0
                for row in rows[1:]:
                    writer.writerow({"source_vintage": vintage, **{header[i]: clean(value) for i, value in enumerate(row)}})
                    count += 1
            successful_files += 1
            results.append({"status": "ok", "rows": count, "output": str(out.relative_to(ROOT)), "raw_file": str(path.relative_to(ROOT)), "member": member, "vintage": vintage})
        except Exception as exc:
            results.append({"status": "failed", "raw_file": str(path.relative_to(ROOT)), "reason": f"{type(exc).__name__}: {exc}"})
    if successful_files:
        temporary_out.replace(out)
    else:
        temporary_out.unlink(missing_ok=True)
    if not results:
        results.append({"status": "pending", "reason": "no Dartmouth ZIP crosswalk in cache"})
    return results


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxnav-ndc-input", type=Path, help="CSV with an ndc column for per-NDC RxCUI/ingredient lookup")
    ap.add_argument("--rxnav-ndc-output", type=Path, default=DATA / "rxnav_ndc_rxcui_ingredient.csv")
    args = ap.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"normalized_at": now(), "rxnav": normalize_rxnav(), "gleif": normalize_gleif(), "census": normalize_census(), "dartmouth": normalize_dartmouth()}
    if args.rxnav_ndc_input:
        status["rxnav_lookup"] = lookup_rxnav_ndcs(args.rxnav_ndc_input, args.rxnav_ndc_output)
    status["pending"] = ["who_atc_ddd", "wits_product_concordances", "hud_usps_crosswalk"]
    (DATA / "normalization_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
