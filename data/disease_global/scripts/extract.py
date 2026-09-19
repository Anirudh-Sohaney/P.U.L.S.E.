#!/usr/bin/env python3
"""Download reachable global health sources and normalize them to JSONL."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from common import (
    DATA_DIR,
    METADATA_DIR,
    as_number,
    fetch_bytes,
    fetch_json,
    fetch_json_post,
    utc_now,
    write_jsonl,
    xlsx_matrix,
    xlsx_rows,
)

START_YEAR = 2012
END_YEAR = 2022
OWID_CSV = "https://ourworldindata.org/grapher/{slug}.csv"
WHO_FLUNET_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
WHO_FLUID_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI?$format=csv"
WHO_FLU_METADATA_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FLU_METADATA?$format=csv"
WHO_GHO_BASE = "https://ghoapi.azureedge.net/api/"
WHO_DON_URL = "https://www.who.int/api/news/diseaseoutbreaknews"
WUENIC_URL = "https://cdn.who.int/media/docs/default-source/immunization/wuenic_input_to_pdf.xlsx?sfvrsn=a067f2ad_33&download=true"
DHS_DATA_URL = "https://api.dhsprogram.com/rest/dhs/data"

OWID_DATASETS: dict[str, dict[str, str]] = {
    "life-expectancy": {
        "name": "Life expectancy",
        "unit": "years",
        "metadata_url": "https://ourworldindata.org/grapher/life-expectancy.metadata.json",
    },
    "child-mortality": {
        "name": "Under-five mortality rate (selected)",
        "unit": "deaths per 1,000 live births",
        "metadata_url": "https://ourworldindata.org/grapher/child-mortality.metadata.json",
    },
}

GHO_INDICATORS = {
    "WHOSIS_000001": "Life expectancy at birth",
    "WHS4_100": "DTP3 immunization coverage among one-year-olds",
    "WHS4_117": "HepB3 immunization coverage among one-year-olds",
    "WHS4_129": "Hib3 immunization coverage among one-year-olds",
    "WHS4_543": "BCG immunization coverage among one-year-olds",
    "WHS4_544": "IPV coverage by locally recommended age",
    "WHS8_110": "MCV1 immunization coverage among one-year-olds",
    "MCV2": "MCV2 immunization coverage by locally recommended age",
    "PCV3": "PCV coverage by locally recommended age",
    "ROTAC": "Rotavirus completed-dose coverage",
    "SDGHPVRECEIVED": "HPV immunization coverage",
    "VACCINECOVERAGE_DTP1": "DTP1 immunization coverage",
    "VACCINECOVERAGE_IPV1": "IPV1 immunization coverage",
    "VACCINECOVERAGE_MENA_C": "Meningococcal A coverage",
    "VACCINECOVERAGE_YFV": "Yellow fever coverage",
}


def geography(name: str, code: str, default_level: str = "aggregate") -> dict[str, str]:
    code = code.strip()
    if len(code) == 3 and code.isalpha() and not code.startswith("OWID_"):
        return {"level": "country", "id": code.upper(), "name": name}
    if code:
        return {"level": default_level, "id": code, "name": name}
    identifier = name.lower().replace(" ", "_") or "unknown"
    return {"level": "region", "id": identifier, "name": name or "Unknown"}


def base_record(
    source_id: str,
    dataset_id: str,
    period: dict[str, Any],
    geo: dict[str, str],
    metric: str,
    value: int | float,
    unit: str,
    dimensions: dict[str, Any],
    source_url: str,
    raw_file: str,
    retrieved_at: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dataset_id": dataset_id,
        "period": period,
        "geography": geo,
        "metric": metric,
        "value": value,
        "unit": unit,
        "dimensions": dimensions,
        "provenance": {"source_url": source_url, "retrieved_at": retrieved_at, "raw_file": raw_file},
    }


def extract_owid(slugs: list[str], start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    output = DATA_DIR / "owid" / "observations.jsonl"
    summaries: list[dict[str, Any]] = []

    def rows() -> Iterator[dict[str, Any]]:
        for slug in slugs:
            config = OWID_DATASETS[slug]
            url = OWID_CSV.format(slug=slug)
            body, raw_file, cache_hit = fetch_bytes("owid_" + slug, url, "csv", refresh=refresh)
            reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
            fields = reader.fieldnames or []
            missing = {"Entity", "Code", "Year"}.difference(fields)
            if missing:
                raise RuntimeError(f"OWID {slug} is missing required columns: {sorted(missing)}")
            metric_fields = [field for field in fields if field not in {"Entity", "Code", "Year"}]
            count = 0
            skipped = 0
            retrieved_at = utc_now()
            for record in reader:
                year_value = as_number(record.get("Year"))
                if year_value is None or int(year_value) != year_value:
                    continue
                year = int(year_value)
                if year < start_year or year > end_year:
                    skipped += 1
                    continue
                entity = str(record.get("Entity") or "").strip()
                if not entity:
                    continue
                geo = geography(entity, str(record.get("Code") or ""), "aggregate")
                for metric_field in metric_fields:
                    value = as_number(record.get(metric_field))
                    if value is None:
                        continue
                    count += 1
                    yield base_record("owid", slug, {"year": year}, geo, metric_field, value, config["unit"],
                                      {"entity_code": record.get("Code") or None, "source_column": metric_field},
                                      url, raw_file, retrieved_at)
            summaries.append({"status": "success", "dataset_id": slug, "name": config["name"],
                              "source_url": url, "metadata_url": config["metadata_url"], "raw_file": raw_file,
                              "cache_hit": cache_hit, "records": count, "skipped_rows_outside_window": skipped,
                              "years": [start_year, end_year]})

    count = write_jsonl(output, rows())
    return {"status": "success", "source_id": "owid", "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "datasets": summaries}


def extract_who_csv(source_id: str, dataset_id: str, url: str, start_year: int, end_year: int,
                    refresh: bool, date_field: str, dimensions_fields: set[str]) -> dict[str, Any]:
    output = DATA_DIR / source_id / "observations.jsonl"
    body, raw_file, cache_hit = fetch_bytes(source_id, url, "csv", refresh=refresh)
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    if date_field not in fields:
        raise RuntimeError(f"{dataset_id} missing {date_field}; fields={fields[:20]}")
    retrieved_at = utc_now()
    rows_written = 0
    skipped = 0
    known_dimensions = set(dimensions_fields) | {date_field}
    metric_fields = [field for field in fields if field not in known_dimensions]

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal rows_written, skipped
        for record in reader:
            year_value = as_number(record.get(date_field))
            if year_value is None or int(year_value) != year_value:
                continue
            year = int(year_value)
            if year < start_year or year > end_year:
                skipped += 1
                continue
            country = str(record.get("COUNTRY_AREA_TERRITORY") or record.get("Country") or "").strip()
            code = str(record.get("COUNTRY_CODE") or record.get("ISO3") or "").strip()
            geo = geography(country, code, "country")
            dimensions = {key.lower(): record.get(key) for key in dimensions_fields if record.get(key) not in (None, "")}
            for field in metric_fields:
                value = as_number(record.get(field))
                if value is None:
                    continue
                rows_written += 1
                yield base_record(source_id, dataset_id, {"year": year}, geo, field, value, "source_native",
                                  dimensions, url, raw_file, retrieved_at)

    count = write_jsonl(output, rows())
    return {"status": "success", "source_id": source_id, "dataset_id": dataset_id, "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "raw_file": raw_file,
            "cache_hit": cache_hit, "skipped_rows_outside_window": skipped, "years": [start_year, end_year]}


def extract_gho(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "who_gho"
    output = DATA_DIR / source_id / "observations.jsonl"
    summaries: list[dict[str, Any]] = []
    retrieved_at = utc_now()

    def rows() -> Iterator[dict[str, Any]]:
        for code, name in GHO_INDICATORS.items():
            filter_text = f"TimeDim ge {start_year} and TimeDim le {end_year}"
            url = WHO_GHO_BASE + code + "?" + urllib.parse.urlencode({"$filter": filter_text})
            try:
                payload, raw_file, cache_hit = fetch_json(source_id + "_" + code, url, refresh=refresh)
            except Exception as exc:
                summaries.append({"status": "error", "indicator": code, "name": name,
                                  "error": f"{type(exc).__name__}: {exc}", "source_url": url})
                continue
            values = payload.get("value", []) if isinstance(payload, dict) else []
            pages = 1
            next_url = payload.get("@odata.nextLink") if isinstance(payload, dict) else None
            while next_url:
                page_payload, page_raw_file, _ = fetch_json(source_id + "_" + code + f"_page{pages + 1}", next_url, refresh=refresh)
                values.extend(page_payload.get("value", []))
                raw_file = page_raw_file
                pages += 1
                next_url = page_payload.get("@odata.nextLink")
            count = 0
            for record in values:
                year = as_number(record.get("TimeDim"))
                value = as_number(record.get("NumericValue"))
                if year is None or value is None:
                    continue
                country_name = str(record.get("SpatialDim") or "").strip()
                geo = geography(country_name, country_name, "country")
                dimensions = {key.lower(): record.get(key) for key in (
                    "IndicatorCode", "Dim1Type", "Dim1", "Dim2Type", "Dim2", "ParentLocationCode", "ParentLocation",
                    "Low", "High", "Value", "Comments",
                ) if record.get(key) not in (None, "")}
                count += 1
                yield base_record(source_id, code, {"year": int(year)}, geo, name, value, "percent" if "coverage" in name.lower() else "years",
                                  dimensions, url, raw_file, retrieved_at)
            summaries.append({"status": "success", "indicator": code, "name": name, "records": count,
                              "pages": pages, "raw_file": raw_file, "cache_hit": cache_hit, "source_url": url})

    count = write_jsonl(output, rows())
    indicator_errors = [item for item in summaries if item.get("status") == "error"]
    empty_indicators = [item for item in summaries if item.get("status") == "success" and item.get("records", 0) == 0]
    return {"status": "partial" if indicator_errors or empty_indicators else "success", "source_id": source_id, "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "indicators": summaries,
            "indicator_errors": len(indicator_errors), "empty_indicators": len(empty_indicators),
            "years": [start_year, end_year]}


def extract_don(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "who_don"
    output = DATA_DIR / source_id / "observations.jsonl"
    filter_text = f"PublicationDate ge {start_year}-01-01T00:00:00Z and PublicationDate le {end_year}-12-31T23:59:59Z"
    pages: list[tuple[list[dict[str, Any]], str]] = []
    skip = 0
    page_size = 100
    while True:
        url = WHO_DON_URL + "?" + urllib.parse.urlencode({"$filter": filter_text, "$top": page_size, "$skip": skip})
        payload, raw_file, _ = fetch_json(source_id, url, refresh=refresh)
        values = payload.get("value", []) if isinstance(payload, dict) else []
        pages.append((values, raw_file))
        if len(values) < page_size:
            break
        skip += page_size

    retrieved_at = utc_now()

    def rows() -> Iterator[dict[str, Any]]:
        for values, raw_file in pages:
            for item in values:
                date = str(item.get("PublicationDate") or "")
                year_value = as_number(date[:4])
                if year_value is None:
                    continue
                text = " ".join(str(item.get(key) or "") for key in ("Title", "Summary", "Overview"))
                yield base_record(source_id, "diseaseoutbreaknews", {"year": int(year_value)},
                                  {"level": "global_event", "id": str(item.get("Id") or item.get("UrlName") or "unknown"),
                                   "name": str(item.get("Title") or "Untitled")},
                                  "outbreak_event", 1, "event", {
                                      "title": item.get("Title"), "publication_date": date, "url_name": item.get("UrlName"),
                                      "summary": item.get("Summary"), "text": text, "item_url": item.get("ItemDefaultUrl"),
                                  }, source_url=WHO_DON_URL, raw_file=raw_file, retrieved_at=retrieved_at)

    count = write_jsonl(output, rows())
    return {"status": "success", "source_id": source_id, "dataset_id": "diseaseoutbreaknews", "records": count,
            "pages": len(pages), "output": str(output.relative_to(DATA_DIR.parent.parent)), "years": [start_year, end_year]}


def extract_wuenic(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "wuenic"
    output = DATA_DIR / source_id / "observations.jsonl"
    body, raw_file, cache_hit = fetch_bytes(source_id, WUENIC_URL, "xlsx", refresh=refresh)
    retrieved_at = utc_now()
    numeric_fields = {
        "WUENIC": "percent", "WUENICPreviousRevision": "percent", "AdministrativeCoverage": "percent",
        "GovernmentEstimate": "percent", "ChildrenVaccinated": "children", "ChildrenInTarget": "children",
        "BirthsUNPD": "children", "SurvivingInfantsUNPD": "children",
    }

    def rows() -> Iterator[dict[str, Any]]:
        for record in xlsx_rows(body, "wuenic_master", expected_headers={"Country", "ISOCountryCode", "Vaccine", "Year", "WUENIC"}):
            year = as_number(record.get("Year"))
            if year is None or int(year) != year or not start_year <= int(year) <= end_year:
                continue
            country = str(record.get("Country") or "").strip()
            code = str(record.get("ISOCountryCode") or "").strip()
            geo = geography(country, code, "country")
            for metric, unit in numeric_fields.items():
                value = as_number(record.get(metric))
                if value is None:
                    continue
                dimensions = {key.lower(): record.get(key) for key in ("Vaccine", "GradeOfConfidence", "Comment")
                              if record.get(key) not in (None, "")}
                yield base_record(source_id, "wuenic_master", {"year": int(year)}, geo, metric, value, unit,
                                  dimensions, WUENIC_URL, raw_file, retrieved_at)

    count = write_jsonl(output, rows())
    return {"status": "success", "source_id": source_id, "dataset_id": "wuenic_master", "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "raw_file": raw_file, "cache_hit": cache_hit,
            "years": [start_year, end_year]}


def extract_ecdc(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "ecdc_surveillance_atlas"
    output = DATA_DIR / source_id / "observations.jsonl"
    service = "https://atlas.ecdc.europa.eu/public/AtlasService/rest/"
    topics_url = service + "GetHealthTopics"
    datasets_url = service + "GetDatasets"
    topics, topics_raw, _ = fetch_json(source_id + "_healthtopics", topics_url, refresh=refresh)
    datasets, datasets_raw, _ = fetch_json(source_id + "_datasets", datasets_url, refresh=refresh)
    topic_rows = topics.get("HealthTopics", [])
    dataset_rows = datasets.get("Datasets", [])
    dataset = next((item for item in dataset_rows if item.get("Code") == "CURRENT.GENERAL"), None)
    if not dataset:
        raise RuntimeError("ECDC CURRENT.GENERAL dataset was not present")
    dataset_id = int(dataset["Id"])
    time_codes = ",".join(str(year) for year in range(start_year, end_year + 1))
    retrieved_at = utc_now()
    results: list[dict[str, Any]] = []
    topic_summaries: list[dict[str, Any]] = []
    for topic in topic_rows:
        topic_id = topic.get("Id")
        if topic_id is None:
            continue
        population_url = service + "GetPopulationsForHealthTopicAndDataset?" + urllib.parse.urlencode({"datasetId": dataset_id, "healthtopicId": topic_id})
        try:
            populations, population_raw, _ = fetch_json(source_id + f"_populations_{topic_id}", population_url, refresh=refresh)
            population_rows = populations.get("Populations", [])
            if not population_rows:
                topic_summaries.append({"health_topic_id": topic_id, "code": topic.get("Code"),
                                        "status": "empty", "population": "none", "measures": 0, "records": 0,
                                        "population_raw_file": population_raw})
                continue
            population = population_rows[0]
            measure_population = ""
            measures_url = service + "GetIndicatorMeasuresForHealthTopicDatasetAndPopulation?" + urllib.parse.urlencode({"datasetId": dataset_id, "healthtopicId": topic_id, "measurePopulation": str(population.get("Label") or "")})
            measure_meta, measure_raw, _ = fetch_json(source_id + f"_measures_{topic_id}", measures_url, refresh=refresh)
            measures = measure_meta.get("Measures", [])
            payload = {"healthTopicId": topic_id, "datasetId": dataset_id, "measurePopulation": measure_population,
                       "measureIds": "", "measureTypes": "I,Q", "timeCodes": time_codes, "geoCodes": ""}
            result_url = service + "post/GetMeasuresResults"
            response, result_raw, _ = fetch_json_post(source_id + f"_results_{topic_id}", result_url, payload, refresh=refresh)
            result_block = response.get("GetMeasuresResultsResult", {})
            measure_results = result_block.get("MeasureResults", []) if isinstance(result_block, dict) else []
            results.extend({"item": item, "topic": topic, "raw_file": result_raw, "source_url": result_url} for item in measure_results)
            topic_summaries.append({"health_topic_id": topic_id, "code": topic.get("Code"), "population": "all",
                                    "measures": len(measures), "records": len(measure_results), "raw_file": result_raw,
                                    "population_raw_file": population_raw, "measure_raw_file": measure_raw})
        except Exception as exc:
            topic_summaries.append({"health_topic_id": topic_id, "code": topic.get("Code"), "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}"})

    def rows() -> Iterator[dict[str, Any]]:
        for entry in results:
            item = entry["item"]
            year = as_number(item.get("TimeCode"))
            value = as_number(item.get("YValue"))
            if year is None or value is None:
                continue
            geo_code = str(item.get("GeoCountry") or "")
            geo_name = str(item.get("GeoLabel") or geo_code)
            dimensions = {key.lower(): item.get(key) for key in (
                "HealthTopicName", "MeasureId", "MeasureLabel", "MeasurePopulation", "MeasureType", "MeasureUnit",
                "N", "NMissing", "UID", "XValue",
            ) if item.get(key) not in (None, "")}
            yield base_record(source_id, f"dataset_{dataset_id}", {"year": int(year)}, geography(geo_name, geo_code, "country"),
                              str(item.get("MeasureLabel") or "measure"), value, str(item.get("MeasureUnit") or "source_native"),
                              dimensions, entry["source_url"], entry["raw_file"], retrieved_at)

    count = write_jsonl(output, rows())
    topic_errors = [item for item in topic_summaries if item.get("status") == "error"]
    empty_topics = [item for item in topic_summaries if item.get("records", 0) == 0]
    return {"status": "partial" if topic_errors else "success", "source_id": source_id, "dataset_id": f"dataset_{dataset_id}", "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "health_topics": len(topic_rows),
            "topics_with_results": len([item for item in topic_summaries if item.get("records", 0) > 0]),
            "empty_topics": len(empty_topics), "topic_summaries": topic_summaries,
            "catalog_raw_files": [topics_raw, datasets_raw], "years": [start_year, end_year]}


def extract_hmd(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "hmd"
    output = DATA_DIR / source_id / "observations.jsonl"
    base = "https://www.mortality.org/File/GetDocument/Public/HMD_summary/"
    files = {
        "hmd_summary_IMR.xlsx": "infant_mortality_rate",
        "hmd_summary_ex_0_65_80.xlsx": "life_expectancy",
        "hmd_summary_pop_exposures.xlsx": "population_exposure",
        "hmd_summary_px_0_to_65.xlsx": "survival_probability_0_to_65",
    }
    retrieved_at = utc_now()
    all_rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    for filename, dataset_id in files.items():
        url = base + filename
        body, raw_file, cache_hit = fetch_bytes(source_id + "_" + filename.removesuffix(".xlsx"), url, "xlsx", refresh=refresh)
        file_count = 0
        # Use workbook XML to discover all non-introduction worksheet names.
        archive = zipfile.ZipFile(io.BytesIO(body))
        root = ET.fromstring(archive.read("xl/workbook.xml"))
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        sheets = [sheet.attrib.get("name", "") for sheet in root.findall(f"{{{ns}}}sheets/{{{ns}}}sheet") if sheet.attrib.get("name") != "Introduction"]
        for sheet in sheets:
            matrix = xlsx_matrix(body, sheet)
            if len(matrix) < 3 or not matrix[1] or str(matrix[1][0]).strip().lower() != "year":
                continue
            headers = matrix[1]
            for row in matrix[2:]:
                year_value = as_number(row[0] if row else None)
                if year_value is None or int(year_value) < start_year or int(year_value) > end_year:
                    continue
                for column_index, country in enumerate(headers[1:], 1):
                    if not country or column_index >= len(row):
                        continue
                    value = as_number(row[column_index])
                    if value is None:
                        continue
                    file_count += 1
                    all_rows.append(base_record(source_id, dataset_id, {"year": int(year_value)},
                                                 geography(country, country, "country"), sheet, value, "source_native",
                                                 {"sheet": sheet, "country_column": country}, url, raw_file, retrieved_at))
        file_summaries.append({"filename": filename, "dataset_id": dataset_id, "records": file_count,
                               "raw_file": raw_file, "cache_hit": cache_hit, "sheets": sheets})
    count = write_jsonl(output, iter(all_rows))
    return {"status": "success", "source_id": source_id, "records": count,
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "files": file_summaries,
            "years": [start_year, end_year], "public_summary_only": True}


def extract_dhs(start_year: int, end_year: int, refresh: bool) -> dict[str, Any]:
    source_id = "dhs"
    output = DATA_DIR / source_id / "observations.jsonl"
    page_size = 5000
    surveys_url = "https://api.dhsprogram.com/rest/dhs/surveys?" + urllib.parse.urlencode({"f": "json", "perpage": 1000})
    surveys, surveys_raw, _ = fetch_json(source_id + "_surveys", surveys_url, refresh=refresh)
    survey_ids = []
    for survey in surveys.get("Data", []):
        year = as_number(survey.get("SurveyYear"))
        if year is not None and start_year <= int(year) <= end_year and survey.get("SurveyId"):
            survey_ids.append(str(survey["SurveyId"]))
    if not survey_ids:
        raise RuntimeError("DHS surveys endpoint returned no surveys in the requested focus window")
    first_url = DHS_DATA_URL + "?" + urllib.parse.urlencode({"f": "json", "perpage": page_size, "page": 1, "surveyIds": ",".join(survey_ids)})
    first, first_raw, _ = fetch_json(source_id + "_page1", first_url, refresh=refresh)
    total_pages = int(first.get("TotalPages") or 0)
    pages_by_number: dict[int, tuple[list[dict[str, Any]], str]] = {1: (first.get("Data", []), first_raw)}

    def fetch_dhs_page(page: int) -> tuple[int, list[dict[str, Any]], str]:
        url = DHS_DATA_URL + "?" + urllib.parse.urlencode({"f": "json", "perpage": page_size, "page": page, "surveyIds": ",".join(survey_ids)})
        payload, raw_file, _ = fetch_json(source_id + f"_page{page}", url, refresh=refresh)
        return page, payload.get("Data", []), raw_file

    with ThreadPoolExecutor(max_workers=4) as executor:
        for page, values, raw_file in executor.map(fetch_dhs_page, range(2, total_pages + 1)):
            pages_by_number[page] = (values, raw_file)
    pages = [pages_by_number[number] for number in sorted(pages_by_number)]

    retrieved_at = utc_now()

    def rows() -> Iterator[dict[str, Any]]:
        for values, raw_file in pages:
            for record in values:
                year = as_number(record.get("SurveyYear"))
                value = as_number(record.get("Value"))
                if year is None or value is None or int(year) < start_year or int(year) > end_year:
                    continue
                country = str(record.get("CountryName") or record.get("DHS_CountryCode") or "")
                code = str(record.get("ISO2_CountryCode") or record.get("DHS_CountryCode") or "")
                dimensions = {key.lower(): record.get(key) for key in (
                    "IndicatorId", "SurveyId", "IndicatorType", "CharacteristicId", "CharacteristicCategory",
                    "CharacteristicLabel", "ByVariableId", "ByVariableLabel", "Precision", "CILow", "CIHigh",
                    "DenominatorWeighted", "DenominatorUnweighted", "SurveyType",
                ) if record.get(key) not in (None, "")}
                yield base_record(source_id, str(record.get("IndicatorId") or "dhs_indicator"), {"year": int(year)},
                                  geography(country, code, "country"), str(record.get("Indicator") or "indicator"), value,
                                  "source_native", dimensions, DHS_DATA_URL, raw_file, retrieved_at)

    count = write_jsonl(output, rows())
    return {"status": "success", "source_id": source_id, "records": count, "pages": len(pages),
            "output": str(output.relative_to(DATA_DIR.parent.parent)), "years": [start_year, end_year],
            "api_record_count": first.get("RecordCount"), "surveys_raw_file": surveys_raw,
            "survey_count": len(survey_ids)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.sources or ["all_public"]
    all_public = "all_public" in requested
    results: dict[str, Any] = {}
    jobs: list[tuple[str, Any]] = []
    if all_public or "owid" in requested or any(slug in requested for slug in OWID_DATASETS):
        slugs = list(OWID_DATASETS) if all_public or "owid" in requested else [slug for slug in requested if slug in OWID_DATASETS]
        jobs.append(("owid", lambda: extract_owid(slugs, args.start_year, args.end_year, args.refresh)))
    if all_public or "who_flunet" in requested:
        jobs.append(("who_flunet", lambda: extract_who_csv("who_flunet", "VIW_FNT", WHO_FLUNET_URL, args.start_year, args.end_year, args.refresh,
                                                              "ISO_YEAR", {"WHOREGION", "FLUSEASON", "HEMISPHERE", "ITZ", "COUNTRY_CODE", "COUNTRY_AREA_TERRITORY",                                                              "ISO_YEAR", "ISO_WEEK", "ISO_WEEKSTARTDATE", "MMWR_YEAR", "MMWR_WEEK", "MMWR_WEEKSTARTDATE", "ORIGIN_SOURCE"})))

    if all_public or "who_fluid" in requested:
        jobs.append(("who_fluid", lambda: extract_who_csv("who_fluid", "VIW_FID_EPI", WHO_FLUID_URL, args.start_year, args.end_year, args.refresh,
                                                             "ISO_YEAR", {"WHOREGION", "FLUSEASON", "HEMISPHERE", "ITZ", "COUNTRY_CODE", "COUNTRY_AREA_TERRITORY", "ISO_YEAR", "ISO_WEEK", "ISOYW",                                                             "ISO_WEEKSTARTDATE", "AGEGROUP_CODE", "MMWR_YEAR", "MMWR_WEEK", "MMWR_WEEKSTARTDATE", "MMWRYW"})))

    if all_public or "who_gho" in requested:
        jobs.append(("who_gho", lambda: extract_gho(args.start_year, args.end_year, args.refresh)))
    if all_public or "who_don" in requested:
        jobs.append(("who_don", lambda: extract_don(args.start_year, args.end_year, args.refresh)))
    if all_public or "wuenic" in requested:
        jobs.append(("wuenic", lambda: extract_wuenic(args.start_year, args.end_year, args.refresh)))
    if all_public or "ecdc_surveillance_atlas" in requested:
        jobs.append(("ecdc_surveillance_atlas", lambda: extract_ecdc(args.start_year, args.end_year, args.refresh)))
    if all_public or "hmd" in requested:
        jobs.append(("hmd", lambda: extract_hmd(args.start_year, args.end_year, args.refresh)))
    if all_public or "dhs" in requested:
        jobs.append(("dhs", lambda: extract_dhs(args.start_year, args.end_year, args.refresh)))
    for source, job in jobs:
        try:
            results[source] = job()
        except Exception as exc:
            results[source] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    for source in requested:
        if source not in results and source not in {"all_public", "owid", *OWID_DATASETS}:
            results[source] = {"status": "pending", "reason": "No extractor configured"}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    current_window = {"start_year": args.start_year, "end_year": args.end_year}
    prior_status_path = METADATA_DIR / "extraction_status.json"
    prior_results: dict[str, Any] = {}
    if prior_status_path.exists():
        try:
            prior = json.loads(prior_status_path.read_text(encoding="utf-8"))
            if prior.get("focus_window") == current_window:
                prior_results = prior.get("results", {})
        except (OSError, json.JSONDecodeError):
            prior_results = {}
    merged_results = {**prior_results, **results}
    status = {
        "run_at": utc_now(), "focus_window": current_window,
        "refresh": args.refresh, "results": merged_results,
        "pending_sources": ["ihme_gbd", "ghdx", "promed"],
    }
    (METADATA_DIR / "extraction_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ["all_public", "owid", *OWID_DATASETS, "who_flunet", "who_fluid", "who_gho", "who_don", "wuenic", "ecdc_surveillance_atlas", "hmd", "dhs"]
    parser.add_argument("--source", dest="sources", action="append", choices=choices, help="source to extract; repeatable")
    parser.add_argument("--list", action="store_true", help="list configured extractors")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--refresh", action="store_true", help="ignore cached HTTP responses")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be no greater than --end-year")
    if args.list:
        print("\n".join(choices))
        return 0
    results = run(args)
    print(json.dumps(results, indent=2))
    return 0 if all(item.get("status") == "success" for item in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
