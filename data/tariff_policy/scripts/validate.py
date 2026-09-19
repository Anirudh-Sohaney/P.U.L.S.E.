#!/usr/bin/env python3
"""Validate tariff_policy normalized CSV outputs."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METADATA = ROOT / "metadata"
REQUIRED = {
    "record_id", "record_type", "period_start", "period_end", "granularity",
    "reporter_code", "partner_code", "product_code", "product_revision",
    "measure_category", "measure_rate", "source_id", "source_url",
    "source_record_id", "retrieved_at", "is_imputed", "is_suppressed",
}


def validate_file(path: Path, seen: set[str]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        return [{"file": str(path), "error": str(exc)}]
    with handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            return [{"file": str(path), "error": f"missing columns: {sorted(missing)}"}]
        for line_number, row in enumerate(reader, 2):
            for field in REQUIRED:
                if field not in row:
                    errors.append({"file": str(path), "line": str(line_number), "error": f"missing field {field}"})
            record_id = row.get("record_id", "")
            if record_id in seen:
                errors.append({"file": str(path), "line": str(line_number), "error": f"duplicate record_id {record_id}"})
            seen.add(record_id)
            try:
                start = dt.date.fromisoformat(row["period_start"])
                end = dt.date.fromisoformat(row["period_end"])
            except (KeyError, ValueError):
                errors.append({"file": str(path), "line": str(line_number), "error": "period dates must be ISO YYYY-MM-DD"})
                continue
            if start > end or start < dt.date(2012, 1, 1) or end > dt.date(2022, 12, 31):
                errors.append({"file": str(path), "line": str(line_number), "error": "period outside 2012-2022 or reversed"})
            if row.get("record_type") not in {"tariff", "measure"}:
                errors.append({"file": str(path), "line": str(line_number), "error": "record_type must be tariff or measure"})
            expected_granularity = "year" if row.get("record_type") == "tariff" else "event"
            if row.get("granularity") != expected_granularity:
                errors.append({"file": str(path), "line": str(line_number), "error": f"granularity must be {expected_granularity} for {row.get('record_type')}"})
            product_code = row.get("product_code", "")
            if row.get("record_type") == "tariff" and row.get("product_level") == "HS6" and product_code and (len(product_code) != 6 or not product_code.isdigit()):
                errors.append({"file": str(path), "line": str(line_number), "error": "HS6 tariff product_code must be six digits"})
            if row.get("medical_flag", "") not in {"", "True", "False"}:
                errors.append({"file": str(path), "line": str(line_number), "error": "medical_flag must be True, False, or empty"})
            if row.get("is_imputed", "").lower() != "false":
                errors.append({"file": str(path), "line": str(line_number), "error": "is_imputed must be false"})
            if row.get("is_suppressed", "").lower() != "false":
                errors.append({"file": str(path), "line": str(line_number), "error": "is_suppressed must be false"})
            rate = row.get("measure_rate", "")
            if rate not in {"", "None", "null"}:
                try:
                    number = float(rate)
                    if not math.isfinite(number):
                        raise ValueError
                    if row.get("record_type") == "tariff" and number < 0:
                        errors.append({"file": str(path), "line": str(line_number), "error": "tariff rate is negative"})
                except ValueError:
                    errors.append({"file": str(path), "line": str(line_number), "error": "measure_rate is not finite numeric"})
            if row.get("record_type") == "tariff" and not row.get("product_code"):
                errors.append({"file": str(path), "line": str(line_number), "error": "tariff has no product code"})
            if not row.get("source_id") or not row.get("source_record_id"):
                errors.append({"file": str(path), "line": str(line_number), "error": "source provenance identifier is empty"})
    return errors


def main() -> int:
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    files = sorted(DATA.glob("*.csv")) if DATA.exists() else []
    if not files:
        errors.append({"file": str(DATA), "error": "data directory has no normalized CSV files"})
    extraction_status: dict[str, object] = {}
    status_path = METADATA / "extraction_status.json"
    if status_path.exists():
        try:
            extraction_status = json.loads(status_path.read_text(encoding="utf-8")).get("sources", {})
        except (OSError, json.JSONDecodeError):
            extraction_status = {}
    for path in files:
        file_errors = validate_file(path, seen)
        if not file_errors:
            with path.open(newline="", encoding="utf-8") as handle:
                if next(csv.DictReader(handle), None) is None:
                    source_id = path.stem.removesuffix("_tariffs")
                    source_status = extraction_status.get(source_id, {})
                    if source_status.get("status") != "no_data":
                        file_errors.append({"file": str(path), "error": "normalized CSV contains no records"})
        errors.extend(file_errors)
    METADATA.mkdir(parents=True, exist_ok=True)
    (METADATA / "validation_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(files)} files; errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
