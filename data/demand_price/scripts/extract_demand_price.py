#!/usr/bin/env python3
"""Extract and normalize the demand/price source set.

The extractor keeps the outputs reproducible and documents any source-specific
transformations in a machine-readable manifest.
"""

from __future__ import annotations

import csv
import datetime as dt
import calendar
import gzip
import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "source_manifest.csv"
README_PATH = ROOT / "README.md"

NOW_UTC = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

MEDICAID_URLS = {
    2012: "https://download.medicaid.gov/data/StateDrugUtilizationData-2012.csv",
    2013: "https://download.medicaid.gov/data/StateDrugUtilizationData-2013.csv",
    2014: "https://download.medicaid.gov/data/StateDrugUtilizationData-2014.csv",
    2015: "https://download.medicaid.gov/data/StateDrugUtilizationData-2015.csv",
    2016: "https://download.medicaid.gov/data/StateDrugUtilizationData-2016.csv",
    2017: "https://download.medicaid.gov/data/StateDrugUtilizationData-2017.csv",
    2018: "https://download.medicaid.gov/data/StateDrugUtilizationData-2018.csv",
    2019: "https://download.medicaid.gov/data/StateDrugUtilizationData-2019.csv",
    2020: "https://download.medicaid.gov/data/sdud2020_updatedJuly2026.csv",
    2021: "https://download.medicaid.gov/data/sdud2021_updatedJuly2026.csv",
    2022: "https://download.medicaid.gov/data/sdud2022_updatedJuly2026.csv",
}

NADAC_URLS = {
    2013: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.1fe73992-cbfd-5109-97bc-dee8b33fdcff.csv",
    2014: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.ba0c3734-8012-549a-8f50-2ff389d0e0ef.csv",
    2015: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.4d7af295-2132-55a8-b40c-d6630061f3e8.csv",
    2016: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.7656fc17-f1b4-566b-9a2d-c4a4f2ac7ae1.csv",
    2017: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.1c5d0fc9-693a-534a-8240-4627d9362b0d.csv",
    2018: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.8de1b213-73c5-552b-b84e-ac795f34d056.csv",
    2019: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.76a1984a-6d69-5e4d-86c8-65eb31f0506d.csv",
    2020: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.c933dc16-7de9-52b6-8971-4b75992673e0.csv",
    2021: "https://download.medicaid.gov/data/national-average-drug-acquisition-cost-12-29-2021.csv",
    2022: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-2022.csv",
}

CMS_DATASET_UUIDS = {
    2013: "9aec2a70-4ef2-438a-bc35-1747fe2492c9",
    2014: "63099eb4-6c9e-4fec-8c9e-b6b132f1b7f4",
    2015: "73748d0b-2acb-420a-ac66-27d8e238fdc8",
    2016: "c5b3c840-5eb2-4e4c-ac35-c30995dcb051",
    2017: "7863e41c-3f6f-4889-93a8-be65e9f3d073",
    2018: "b083c9b6-b841-4676-8d0a-fb03ee1431a1",
    2019: "73a6335e-f16f-4c81-a84b-6b5a986e2bf8",
    2020: "83891e77-99cf-4865-b60a-97703b916e09",
    2021: "7dda2a9d-034a-446a-b4b3-e1254e0127b2",
    2022: "1fc57194-a51d-4864-aee6-de0889488151",
}

OPENFDA_SHORTAGES_URL = "https://api.fda.gov/drug/shortages.json"
OPENFDA_ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
OPENFDA_NDC_URL = "https://api.fda.gov/drug/ndc.json"
EMA_SHORTAGES_URL = "https://www.ema.europa.eu/en/documents/report/shortages-output-json-report_en.json"
WHO_GHED_URL = "https://apps.who.int/nha/database/Home/IndicatorsDownload/en"
OECD_CONSUMPTION_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ELS.HD,HEALTH_PHMC@DF_PHMC_CONSUM,1.0/.?"
    "startPeriod=2012&endPeriod=2022&dimensionAtObservation=AllDimensions"
)
OECD_KEY_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ELS.HD,HEALTH_PHMC@DF_KEY_INDIC,1.0/.?"
    "startPeriod=2012&endPeriod=2022&dimensionAtObservation=AllDimensions"
)
MSH_PDF_URL = (
    "https://msh.org/wp-content/uploads/2020/03/"
    "msh-2015-international-medical-products-price-guide.pdf"
)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def utc_stamp() -> str:
    return NOW_UTC


def _open_url_with_retries(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
):
    request = urllib.request.Request(url, headers=headers or {})
    last_error: Optional[Exception] = None
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def http_get(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    with _open_url_with_retries(url, headers=headers) as resp:
        return resp.read()


def open_text_url(url: str, headers: Optional[Dict[str, str]] = None) -> io.TextIOBase:
    resp = _open_url_with_retries(url, headers=headers)
    return io.TextIOWrapper(resp, encoding="utf-8", newline="")


def write_gz_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> int:
    # Never replace a previously valid extract with a partial file if a network
    # stream fails midway through generation.
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    count = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: serialize_csv_value(row.get(k)) for k in fieldnames})
                count += 1
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def serialize_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    try:
        return int(Decimal(value))
    except Exception:
        return None


def is_suppressed(value: Optional[str]) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"true", "t", "1", "y", "yes", "*"}


def stream_csv(url: str) -> Iterator[Dict[str, str]]:
    with open_text_url(url) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def fetch_json(url: str, params: Optional[Dict[str, str]] = None, headers: Optional[Dict[str, str]] = None) -> Dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    raw = http_get(url, headers=headers)
    return json.loads(raw.decode("utf-8"))


def flatten_json(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def year_from_date(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    cleaned = str(text).strip()
    if len(cleaned) >= 4 and cleaned[:4].isdigit():
        return int(cleaned[:4])
    return None


def normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def extract_medicaid() -> Tuple[Path, int]:
    out = DATA_DIR / "medicaid_sdud_state_quarter.csv.gz"
    fieldnames = [
        "source_year",
        "state",
        "quarter",
        "utilization_type",
        "source_rows",
        "non_suppressed_rows",
        "suppressed_rows",
        "units_reimbursed",
        "number_of_prescriptions",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed",
    ]
    aggregate: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for year, url in sorted(MEDICAID_URLS.items()):
        for row in stream_csv(url):
            key = (
                row.get("State", "").strip(),
                str(row.get("Year", "")).strip(),
                str(row.get("Quarter", "")).strip(),
                row.get("Utilization Type", "").strip(),
            )
            bucket = aggregate.setdefault(
                key,
                {
                    "source_year": year,
                    "state": key[0],
                    "quarter": key[2],
                    "utilization_type": key[3],
                    "source_rows": 0,
                    "non_suppressed_rows": 0,
                    "suppressed_rows": 0,
                    "units_reimbursed": Decimal("0"),
                    "number_of_prescriptions": Decimal("0"),
                    "total_amount_reimbursed": Decimal("0"),
                    "medicaid_amount_reimbursed": Decimal("0"),
                    "non_medicaid_amount_reimbursed": Decimal("0"),
                },
            )
            bucket["source_rows"] = int(bucket["source_rows"]) + 1
            if is_suppressed(row.get("Suppression Used")):
                bucket["suppressed_rows"] = int(bucket["suppressed_rows"]) + 1
                continue
            bucket["non_suppressed_rows"] = int(bucket["non_suppressed_rows"]) + 1
            for csv_name, bucket_name in [
                ("Units Reimbursed", "units_reimbursed"),
                ("Number of Prescriptions", "number_of_prescriptions"),
                ("Total Amount Reimbursed", "total_amount_reimbursed"),
                ("Medicaid Amount Reimbursed", "medicaid_amount_reimbursed"),
                ("Non Medicaid Amount Reimbursed", "non_medicaid_amount_reimbursed"),
            ]:
                amount = parse_decimal(row.get(csv_name))
                if amount is not None:
                    bucket[bucket_name] = Decimal(bucket[bucket_name]) + amount

    def rows() -> Iterator[Dict[str, object]]:
        for key in sorted(aggregate):
            yield aggregate[key]

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def extract_nadac() -> Tuple[Path, int]:
    out = DATA_DIR / "nadac_ndc_weekly.csv.gz"
    fieldnames = [
        "source_year",
        "ndc_description",
        "ndc",
        "nadac_per_unit",
        "effective_date",
        "pricing_unit",
        "pharmacy_type_indicator",
        "otc",
        "explanation_code",
        "classification_for_rate_setting",
        "corresponding_generic_drug_nadac_per_unit",
        "corresponding_generic_drug_effective_date",
        "as_of_date",
    ]

    failed_years: List[Tuple[int, str]] = []

    def rows() -> Iterator[Dict[str, object]]:
        for year, url in sorted(NADAC_URLS.items()):
            try:
                for row in stream_csv(url):
                    yield {
                        "source_year": year,
                        "ndc_description": row.get("NDC Description"),
                        "ndc": row.get("NDC"),
                        "nadac_per_unit": parse_decimal(row.get("NADAC_Per_Unit")),
                        "effective_date": row.get("Effective_Date"),
                        "pricing_unit": row.get("Pricing_Unit"),
                        "pharmacy_type_indicator": row.get("Pharmacy_Type_Indicator"),
                        "otc": row.get("OTC"),
                        "explanation_code": row.get("Explanation_Code"),
                        "classification_for_rate_setting": row.get("Classification_for_Rate_Setting"),
                        "corresponding_generic_drug_nadac_per_unit": parse_decimal(
                            row.get("Corresponding_Generic_Drug_NADAC_Per_Unit")
                        ),
                        "corresponding_generic_drug_effective_date": row.get(
                            "Corresponding_Generic_Drug_Effective_Date"
                        ),
                        "as_of_date": row.get("As of Date"),
                    }
            except Exception as exc:
                failed_years.append((year, f"{type(exc).__name__}: {exc}"))
                continue

    count = write_gz_csv(out, fieldnames, rows())
    if failed_years:
        details = "\n".join(f"{year}: {message}" for year, message in failed_years) + "\n"
        (DATA_DIR / "nadac_FAILED.txt").write_text(details, encoding="utf-8")
        raise RuntimeError(
            "NADAC extraction incomplete for years: "
            + ", ".join(str(year) for year, _ in failed_years)
        )
    return out, count


def cms_resource_url(dataset_uuid: str) -> str:
    api_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_uuid}/resources"
    data = fetch_json(api_url)
    resources = data.get("data") or []
    for item in resources:
        file_url = item.get("file_url")
        if file_url and file_url.endswith(".csv"):
            return file_url
    raise RuntimeError(f"Could not resolve CSV resource for dataset {dataset_uuid}")


def extract_cms_partd() -> Tuple[Path, int]:
    out = DATA_DIR / "cms_partd_geography_drug.csv.gz"
    fieldnames = [
        "source_year",
        "prscrbr_geo_lvl",
        "prscrbr_geo_cd",
        "prscrbr_geo_desc",
        "brnd_name",
        "gnrc_name",
        "tot_prscrbrs",
        "tot_clms",
        "tot_30day_fills",
        "tot_drug_cst",
        "tot_benes",
        "ge65_sprsn_flag",
        "ge65_tot_clms",
        "ge65_tot_30day_fills",
        "ge65_tot_drug_cst",
        "ge65_bene_sprsn_flag",
        "ge65_tot_benes",
        "lis_bene_cst_shr",
        "nonlis_bene_cst_shr",
        "opioid_drug_flag",
        "opioid_la_drug_flag",
        "antbtc_drug_flag",
        "antpsyct_drug_flag",
    ]

    def rows() -> Iterator[Dict[str, object]]:
        for year, uuid in sorted(CMS_DATASET_UUIDS.items()):
            url = cms_resource_url(uuid)
            with open_text_url(url) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield {
                        "source_year": year,
                        "prscrbr_geo_lvl": row.get("Prscrbr_Geo_Lvl"),
                        "prscrbr_geo_cd": row.get("Prscrbr_Geo_Cd"),
                        "prscrbr_geo_desc": row.get("Prscrbr_Geo_Desc"),
                        "brnd_name": row.get("Brnd_Name"),
                        "gnrc_name": row.get("Gnrc_Name"),
                        "tot_prscrbrs": row.get("Tot_Prscrbrs"),
                        "tot_clms": row.get("Tot_Clms"),
                        "tot_30day_fills": row.get("Tot_30day_Fills"),
                        "tot_drug_cst": row.get("Tot_Drug_Cst"),
                        "tot_benes": row.get("Tot_Benes"),
                        "ge65_sprsn_flag": row.get("GE65_Sprsn_Flag"),
                        "ge65_tot_clms": row.get("GE65_Tot_Clms"),
                        "ge65_tot_30day_fills": row.get("GE65_Tot_30day_Fills"),
                        "ge65_tot_drug_cst": row.get("GE65_Tot_Drug_Cst"),
                        "ge65_bene_sprsn_flag": row.get("GE65_Bene_Sprsn_Flag"),
                        "ge65_tot_benes": row.get("GE65_Tot_Benes"),
                        "lis_bene_cst_shr": row.get("LIS_Bene_Cst_Shr"),
                        "nonlis_bene_cst_shr": row.get("NonLIS_Bene_Cst_Shr"),
                        "opioid_drug_flag": row.get("Opioid_Drug_Flag"),
                        "opioid_la_drug_flag": row.get("Opioid_LA_Drug_Flag"),
                        "antbtc_drug_flag": row.get("Antbtc_Drug_Flag"),
                        "antpsyct_drug_flag": row.get("Antpsyct_Drug_Flag"),
                    }

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def paged_openfda(url: str, search: str, limit: int = 100) -> Iterator[Dict]:
    skip = 0
    total: Optional[int] = None
    while total is None or skip < total:
        try:
            payload = fetch_json(
                url,
                params={"search": search, "limit": str(limit), "skip": str(skip)},
                headers={"User-Agent": "demand-price-extractor/1.0"},
            )
        except urllib.error.HTTPError as exc:
            # openFDA intermittently returns gateway errors for broad historical
            # shards. Retrying the same page is safe because pages are immutable
            # within a single extraction run and avoids emitting partial output.
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(5)
                payload = fetch_json(
                    url,
                    params={"search": search, "limit": str(limit), "skip": str(skip)},
                    headers={"User-Agent": "demand-price-extractor/1.0"},
                )
            else:
                raise
        meta = payload.get("meta") or {}
        results = (meta.get("results") or {})
        total = int(results.get("total", 0))
        rows = payload.get("results") or []
        if not rows:
            break
        for row in rows:
            yield row
        skip += len(rows)


def extract_openfda_shortages() -> Tuple[Path, int]:
    out = DATA_DIR / "openfda_shortages_2012_2022.csv.gz"
    fieldnames = [
        "initial_posting_date",
        "discontinued_date",
        "update_type",
        "update_date",
        "generic_name",
        "package_ndc",
        "therapeutic_category",
        "dosage_form",
        "presentation",
        "company_name",
        "status",
        "availability",
        "openfda_json",
        "contact_info_json",
        "related_info_json",
    ]

    def rows() -> Iterator[Dict[str, object]]:
        search = "initial_posting_date:[2012-01-01 TO 2022-12-31]"
        for row in paged_openfda(OPENFDA_SHORTAGES_URL, search):
            yield {
                "initial_posting_date": row.get("initial_posting_date"),
                "discontinued_date": row.get("discontinued_date"),
                "update_type": row.get("update_type"),
                "update_date": row.get("update_date"),
                "generic_name": row.get("generic_name"),
                "package_ndc": row.get("package_ndc"),
                "therapeutic_category": row.get("therapeutic_category"),
                "dosage_form": row.get("dosage_form"),
                "presentation": row.get("presentation"),
                "company_name": row.get("company_name"),
                "status": row.get("status"),
                "availability": row.get("availability"),
                "openfda_json": row.get("openfda"),
                "contact_info_json": row.get("contact_info"),
                "related_info_json": row.get("related_info"),
            }

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def extract_openfda_enforcement() -> Tuple[Path, int]:
    out = DATA_DIR / "openfda_enforcement_2012_2022.csv.gz"
    fieldnames = [
        "recall_initiation_date",
        "status",
        "city",
        "state",
        "country",
        "classification",
        "event_id",
        "recalling_firm",
        "voluntary_mandated",
        "distribution_pattern",
        "recall_number",
        "product_description",
        "reason_for_recall",
        "center_classification_date",
        "termination_date",
        "report_date",
        "code_info",
    ]

    def rows() -> Iterator[Dict[str, object]]:
        search = "recall_initiation_date:[20120101 TO 20221231]"
        for row in paged_openfda(OPENFDA_ENFORCEMENT_URL, search):
            yield {name: row.get(name) for name in fieldnames}

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def extract_openfda_ndc() -> Tuple[Path, int]:
    out = DATA_DIR / "openfda_ndc_2012_2022.csv.gz"
    fieldnames = [
        "marketing_start_date",
        "listing_expiration_date",
        "product_ndc",
        "generic_name",
        "brand_name",
        "labeler_name",
        "dosage_form",
        "route",
        "marketing_category",
        "product_type",
        "package_ndc",
        "package_description",
        "sample",
        "active_ingredients_json",
        "packaging_json",
    ]

    def rows() -> Iterator[Dict[str, object]]:
        for year in range(2012, 2023):
            for month in range(1, 13):
                last_day = calendar.monthrange(year, month)[1]
                search = f"marketing_start_date:[{year}-{month:02d}-01 TO {year}-{month:02d}-{last_day:02d}]"
                for row in paged_openfda(OPENFDA_NDC_URL, search):
                    active = row.get("active_ingredients")
                    packaging = row.get("packaging")
                    yield {
                        "marketing_start_date": row.get("marketing_start_date"),
                        "listing_expiration_date": row.get("listing_expiration_date"),
                        "product_ndc": row.get("product_ndc"),
                        "generic_name": row.get("generic_name"),
                        "brand_name": row.get("brand_name"),
                        "labeler_name": row.get("labeler_name"),
                        "dosage_form": row.get("dosage_form"),
                        "route": row.get("route"),
                        "marketing_category": row.get("marketing_category"),
                        "product_type": row.get("product_type"),
                        "package_ndc": packaging[0].get("package_ndc") if packaging else "",
                        "package_description": packaging[0].get("description") if packaging else "",
                        "sample": packaging[0].get("sample") if packaging else "",
                        "active_ingredients_json": active,
                        "packaging_json": packaging,
                    }

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def extract_ema() -> Tuple[Path, int]:
    out = DATA_DIR / "ema_shortages_2012_2022.csv.gz"
    fieldnames = [
        "category",
        "medicine_affected",
        "supply_shortage_status",
        "international_non_proprietary_name_inn_or_common_name",
        "therapeutic_area_mesh",
        "pharmaceutical_forms_affected",
        "strengths_affected",
        "availability_of_alternatives",
        "start_of_shortage_date",
        "expected_resolution_date",
        "first_published_date",
        "last_updated_date",
    ]

    payload = fetch_json(EMA_SHORTAGES_URL)
    records = payload.get("data") or payload.get("results") or []

    def rows() -> Iterator[Dict[str, object]]:
        for record in records:
            yield {name: record.get(name) for name in fieldnames}

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def fetch_oecd_csv(url: str) -> List[Dict[str, str]]:
    raw = http_get(url, headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0"})
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def extract_oecd() -> Tuple[Path, int]:
    out = DATA_DIR / "oecd_pharma_2012_2022.csv.gz"
    fieldnames = [
        "dataset",
        "ref_area",
        "measure",
        "unit_measure",
        "market_type",
        "pharmaceutical",
        "time_period",
        "obs_value",
        "obs_status",
        "obs_status2",
        "obs_status3",
        "unit_mult",
        "decimals",
    ]

    def filter_rows(rows: Iterable[Dict[str, str]], dataset_name: str) -> Iterator[Dict[str, object]]:
        for row in rows:
            if row.get("PHARMACEUTICAL") not in {"_T", "T", "_T "}:
                continue
            try:
                year = int(row.get("TIME_PERIOD", ""))
            except Exception:
                continue
            if year < 2012 or year > 2022:
                continue
            yield {
                "dataset": dataset_name,
                "ref_area": row.get("REF_AREA"),
                "measure": row.get("MEASURE"),
                "unit_measure": row.get("UNIT_MEASURE"),
                "market_type": row.get("MARKET_TYPE"),
                "pharmaceutical": row.get("PHARMACEUTICAL"),
                "time_period": row.get("TIME_PERIOD"),
                "obs_value": row.get("OBS_VALUE"),
                "obs_status": row.get("OBS_STATUS"),
                "obs_status2": row.get("OBS_STATUS2"),
                "obs_status3": row.get("OBS_STATUS3"),
                "unit_mult": row.get("UNIT_MULT"),
                "decimals": row.get("DECIMALS"),
            }

    consumption = fetch_oecd_csv(OECD_CONSUMPTION_URL)
    key_indic = fetch_oecd_csv(OECD_KEY_URL)

    def rows() -> Iterator[Dict[str, object]]:
        yield from filter_rows(consumption, "PH_PHMC_CONSUM")
        yield from filter_rows(key_indic, "PH_KEY_INDIC")

    count = write_gz_csv(out, fieldnames, rows())
    return out, count


def xlsx_column_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def extract_who() -> Tuple[Path, int]:
    out = DATA_DIR / "who_ghed_2012_2022.csv.gz"
    fieldnames: Optional[List[str]] = None
    header_refs: Dict[str, str] = {}
    year_ref: Optional[str] = None
    count = 0

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    try:
        with open(tmp.name, "wb") as fh:
            fh.write(http_get(WHO_GHED_URL))

        with zipfile.ZipFile(tmp.name) as zf:
            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in root.findall("a:si", ns):
                    shared_strings.append("".join(t.text or "" for t in si.iterfind(".//a:t", ns)))

            sheet = zf.open("xl/worksheets/sheet1.xml")
            temporary = out.with_name(out.name + ".tmp")
            temporary.unlink(missing_ok=True)
            writer_file = gzip.open(temporary, "wt", encoding="utf-8", newline="")
            try:
                writer: Optional[csv.DictWriter] = None
                ns_uri = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                ns = {"a": ns_uri}
                # Stream rows so the 38MB workbook does not have to live in memory.
                for event, elem in ET.iterparse(sheet, events=("end",)):
                    if elem.tag != f"{{{ns_uri}}}row":
                        continue
                    row_values: Dict[str, str] = {}
                    max_col = 0
                    for cell in elem.findall("a:c", ns):
                        ref = cell.attrib.get("r", "")
                        letters = "".join(ch for ch in ref if ch.isalpha())
                        if not letters:
                            continue
                        col_idx = 0
                        for ch in letters:
                            col_idx = col_idx * 26 + (ord(ch.upper()) - 64)
                        max_col = max(max_col, col_idx)
                        cell_type = cell.attrib.get("t")
                        value = ""
                        if cell_type == "s":
                            v = cell.find("a:v", ns)
                            if v is not None and v.text is not None:
                                value = shared_strings[int(v.text)]
                        elif cell_type == "inlineStr":
                            t = cell.find(".//a:t", ns)
                            value = t.text if t is not None and t.text is not None else ""
                        else:
                            v = cell.find("a:v", ns)
                            value = v.text if v is not None and v.text is not None else ""
                        row_values[xlsx_column_name(col_idx)] = value
                    if not row_values:
                        elem.clear()
                        continue
                    if fieldnames is None:
                        # The first sheet row is the header row. Keep both the
                        # human-readable names and their XML cell references so
                        # data rows are mapped correctly.
                        fieldnames = []
                        for i in range(1, max_col + 1):
                            ref = xlsx_column_name(i)
                            name = row_values.get(ref, "").strip() or ref.lower()
                            fieldnames.append(name)
                            header_refs[name] = ref
                            if name.strip().lower() == "year":
                                year_ref = ref
                        writer = csv.DictWriter(writer_file, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                    else:
                        if not year_ref:
                            raise RuntimeError("WHO workbook has no year column")
                        year_text = row_values.get(year_ref, "").strip()
                        try:
                            year = int(float(year_text))
                        except (TypeError, ValueError):
                            elem.clear()
                            continue
                        if year < 2012 or year > 2022:
                            elem.clear()
                            continue
                        assert writer is not None
                        writer.writerow({name: row_values.get(ref, "") for name, ref in header_refs.items()})
                        count += 1
                    elem.clear()
            finally:
                writer_file.close()
            os.replace(temporary, out)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if fieldnames is None:
        raise RuntimeError("WHO workbook did not yield any rows")
    return out, count


def extract_msh_note() -> Tuple[Path, int]:
    pdf_out = DATA_DIR / "msh_price_guide_2015.pdf"
    note_out = DATA_DIR / "msh_price_guide_2015_NOT_EXTRACTED.txt"
    raw = http_get(MSH_PDF_URL, headers={"User-Agent": "demand-price-extractor/1.0"})
    if not raw.startswith(b"%PDF-"):
        raise RuntimeError("MSH endpoint did not return a PDF")
    temporary = pdf_out.with_name(pdf_out.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, pdf_out)
    text = (
        "MSH International Medical Products Price Guide (2015) was downloaded to "
        "data/msh_price_guide_2015.pdf. The PDF is image-only in this environment "
        "(no embedded PDF text operators were found), and no OCR executable or "
        "library is installed. The table observations therefore remain a documented "
        "blocked extraction rather than an invented or lossy parse."
    )
    note_out.write_text(text, encoding="utf-8")
    # The note is provenance metadata, not an extracted observation.
    return note_out, 0


SOURCE_ORDER = [
    "Medicaid State Drug Utilization Data",
    "NADAC",
    "Medicare Part D PUFs",
    "openFDA Drug Shortages",
    "ASHP Shortages API",
    "EMA Shortages Catalogue",
    "openFDA Enforcement/Recalls",
    "openFDA NDC Directory",
    "MSH Intl Medical Products Price Guide",
    "OECD pharma consumption + key indicators",
    "WHO GHED",
    "IQVIA MIDAS / Xponent",
]


def _count_gz_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _write_access_note(filename: str, text: str) -> Path:
    path = DATA_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


def catalog_defaults() -> List[Dict[str, object]]:
    """Recover a complete catalog even after a single-job run.

    The previous implementation rewrote the manifest with only the requested
    job, which made a targeted retry erase unrelated successful entries.
    """
    _write_access_note(
        "ashp_shortages_RESTRICTED.txt",
        "ASHP Drug Shortages is catalogued but not extracted: no public, no-credential bulk dataset or API is available. A commercial license/API key is required. Source: https://github.com/ASHP-Software/drugShortagesDoc\n",
    )
    _write_access_note(
        "iqvia_MIDAS_Xponent_PROPRIETARY.txt",
        "IQVIA MIDAS/Xponent is catalogued but not extracted: it is proprietary commercial data requiring a paid subscription or research agreement. Source: https://www.iqvia.com/\n",
    )
    definitions = [
        ("Medicaid State Drug Utilization Data", "2012-2022", MEDICAID_URLS, "state/year/quarter aggregate", "data/medicaid_sdud_state_quarter.csv.gz"),
        ("NADAC", "2013-2022", NADAC_URLS, "row-level cleaned CSV", "data/nadac_ndc_weekly.csv.gz"),
        ("Medicare Part D PUFs", "2013-2022", CMS_DATASET_UUIDS, "row-level cleaned CSV", "data/cms_partd_geography_drug.csv.gz"),
        ("openFDA Drug Shortages", "2012-2022", OPENFDA_SHORTAGES_URL, "row-level cleaned CSV", "data/openfda_shortages_2012_2022.csv.gz"),
        ("ASHP Shortages API", "US; historical/current", "https://github.com/ASHP-Software/drugShortagesDoc", "access status documented; no public bulk extract", "data/ashp_shortages_RESTRICTED.txt"),
        ("EMA Shortages Catalogue", "current live catalogue", EMA_SHORTAGES_URL, "row-level cleaned CSV; current catalogue is not historical 2012-2022", "data/ema_shortages_2012_2022.csv.gz"),
        ("openFDA Enforcement/Recalls", "2012-2022", OPENFDA_ENFORCEMENT_URL, "row-level cleaned CSV", "data/openfda_enforcement_2012_2022.csv.gz"),
        ("openFDA NDC Directory", "2012-2022 focus window", OPENFDA_NDC_URL, "row-level cleaned CSV with marketing-start-date filter", "data/openfda_ndc_2012_2022.csv.gz"),
        ("MSH Intl Medical Products Price Guide", "2015 PDF edition", MSH_PDF_URL, "PDF downloaded; image-only extraction blocked without OCR", "data/msh_price_guide_2015_NOT_EXTRACTED.txt"),
        ("OECD pharma consumption + key indicators", "2012-2022", {"consumption": OECD_CONSUMPTION_URL, "key": OECD_KEY_URL}, "row-level cleaned CSV", "data/oecd_pharma_2012_2022.csv.gz"),
        ("WHO GHED", "2012-2022", WHO_GHED_URL, "xlsx sheet1 -> CSV filtered to years", "data/who_ghed_2012_2022.csv.gz"),
        ("IQVIA MIDAS / Xponent", "global; proprietary", "https://www.iqvia.com/", "access status documented; paid subscription required", "data/iqvia_MIDAS_Xponent_PROPRIETARY.txt"),
    ]
    entries: List[Dict[str, object]] = []
    for source, coverage, url, transform, output in definitions:
        output_path = ROOT / output
        status = "pending"
        row_count = 0
        notes = ""
        if source == "ASHP Shortages API":
            status = "restricted"
            notes = "No public no-credential bulk dataset/API; commercial license or key required."
        elif source == "IQVIA MIDAS / Xponent":
            status = "proprietary"
            notes = "No public extract; paid subscription or research agreement required."
        elif source == "MSH Intl Medical Products Price Guide":
            status = "blocked"
            notes = "PDF is downloaded; image-only and OCR tooling is unavailable."
        elif output_path.exists():
            if source == "NADAC" and (DATA_DIR / "nadac_FAILED.txt").exists():
                status = "partial"
                notes = "Existing output is readable but historical years are missing; see data/nadac_FAILED.txt."
                try:
                    row_count = _count_gz_rows(output_path)
                except (OSError, EOFError, gzip.BadGzipFile) as exc:
                    status = "failed"
                    notes = f"Invalid gzip output: {type(exc).__name__}: {exc}"
            elif output_path.suffix == ".gz":
                try:
                    row_count = _count_gz_rows(output_path)
                    status = "ok"
                except (OSError, EOFError, gzip.BadGzipFile) as exc:
                    status = "failed"
                    notes = f"Invalid gzip output: {type(exc).__name__}: {exc}"
            else:
                status = "blocked"
                row_count = 1
        entries.append({
            "source": source,
            "coverage": coverage,
            "url_or_endpoint": url,
            "retrieved_utc": utc_stamp(),
            "transformation": transform,
            "output_path": output,
            "row_count": row_count,
            "status": status,
            "notes": notes,
        })
    return entries


def write_manifest(entries: List[Dict[str, object]]) -> None:
    fieldnames = [
        "source",
        "coverage",
        "url_or_endpoint",
        "retrieved_utc",
        "transformation",
        "output_path",
        "row_count",
        "status",
        "notes",
    ]
    by_source = {str(entry["source"]): entry for entry in catalog_defaults()}
    for entry in entries:
        by_source[str(entry["source"])] = entry
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for source in SOURCE_ORDER:
            entry = by_source[source]
            writer.writerow({k: serialize_csv_value(entry.get(k)) for k in fieldnames})


def write_readme(entries: List[Dict[str, object]]) -> None:
    lines = [
        "# Demand & Price Extraction",
        "",
        "This directory now contains cleaned, source-linked extracts for the demand and price source set.",
        "",
        "## What was extracted",
        "",
        "| Source | Output | Status | Transform |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['source']} | `{entry['output_path']}` | {entry['status']} | {entry['transformation']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Medicaid SDUD is aggregated to state/year/quarter/utilization because the raw annual files are too large to keep verbatim in the current workspace.",
        "- Other sources are preserved at row level with selected columns normalized into CSV/GZ outputs.",
        "- ASHP and IQVIA are explicitly catalogued as access-controlled sources: ASHP requires a key/commercial license and IQVIA is proprietary/paid.",
        "- The MSH PDF is downloaded and retained, but is image-only and cannot be cleanly tabulated without OCR tooling; no fabricated rows are emitted.",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    ensure_dirs()
    requested = set(argv[1:]) if len(argv) > 1 else {
        "nadac",
        "cms",
        "openfda_shortages",
        "openfda_enforcement",
        "openfda_ndc",
        "ema",
        "oecd",
        "who",
        "msh",
        "medicaid",
    }
    entries: List[Dict[str, object]] = catalog_defaults()

    jobs = [
        ("nadac", "NADAC", "2013-2022", NADAC_URLS, "row-level cleaned CSV", extract_nadac),
        ("cms", "Medicare Part D PUFs", "2013-2022", CMS_DATASET_UUIDS, "row-level cleaned CSV", extract_cms_partd),
        ("openfda_shortages", "openFDA Drug Shortages", "2012-2022", OPENFDA_SHORTAGES_URL, "row-level cleaned CSV", extract_openfda_shortages),
        ("openfda_enforcement", "openFDA Enforcement/Recalls", "2012-2022", OPENFDA_ENFORCEMENT_URL, "row-level cleaned CSV", extract_openfda_enforcement),
        ("openfda_ndc", "openFDA NDC Directory", "2012-2022 focus window", OPENFDA_NDC_URL, "row-level cleaned CSV with start-date filter", extract_openfda_ndc),
        ("ema", "EMA Shortages Catalogue", "2012-2022 overlap", EMA_SHORTAGES_URL, "row-level cleaned CSV", extract_ema),
        ("oecd", "OECD pharma consumption + key indicators", "2012-2022", {"consumption": OECD_CONSUMPTION_URL, "key": OECD_KEY_URL}, "row-level cleaned CSV", extract_oecd),
        ("who", "WHO GHED", "2012-2022", WHO_GHED_URL, "xlsx sheet1 -> CSV filtered to years", extract_who),
        ("msh", "MSH Intl Medical Products Price Guide", "2015 PDF edition", MSH_PDF_URL, "PDF downloaded; image-only extraction blocked without OCR", extract_msh_note),
        ("medicaid", "Medicaid State Drug Utilization Data", "2012-2022", MEDICAID_URLS, "state/year/quarter aggregate", extract_medicaid),
    ]

    for key, source_name, coverage, source_url, transform, fn in jobs:
        if key not in requested:
            continue
        print(f"[start] {source_name}", flush=True)
        try:
            output_path, row_count = fn()
            status = "blocked" if key == "msh" else ("partial" if key == "nadac" and (DATA_DIR / "nadac_FAILED.txt").exists() else "ok")
            notes = (
                "PDF downloaded; image-only and OCR tooling is unavailable."
                if key == "msh"
                else ("Historical NADAC URLs failed for one or more years; see data/nadac_FAILED.txt." if key == "nadac" else "")
            )
        except Exception as exc:  # pragma: no cover - surfaced to user
            failure_path = DATA_DIR / f"{key}_FAILED.txt"
            failure_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            # Keep pointing at a previously valid extract when a retry fails;
            # the failure note records the incomplete refresh separately.
            if key == "nadac" and (DATA_DIR / "nadac_ndc_weekly.csv.gz").exists():
                output_path = DATA_DIR / "nadac_ndc_weekly.csv.gz"
                row_count = _count_gz_rows(output_path)
                status = "partial"
                notes = f"{type(exc).__name__}: {exc}; prior readable extract retained; see data/nadac_FAILED.txt"
            else:
                output_path = failure_path
                row_count = 0
                status = "failed"
                notes = f"{type(exc).__name__}: {exc}"
        entries = [entry for entry in entries if entry["source"] != source_name]
        entries.append(
            {
                "source": source_name,
                "coverage": coverage,
                "url_or_endpoint": source_url,
                "retrieved_utc": utc_stamp(),
                "transformation": transform,
                "output_path": str(output_path.relative_to(ROOT)),
                "row_count": row_count,
                "status": status,
                "notes": notes,
            }
        )
        print(f"[done] {source_name}: {status}, rows={row_count}, output={output_path}", flush=True)

    ordered_entries = [
        next(entry for entry in entries if entry["source"] == source)
        for source in SOURCE_ORDER
    ]
    write_manifest(ordered_entries)
    write_readme(ordered_entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
