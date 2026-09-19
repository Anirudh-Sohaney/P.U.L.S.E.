#!/usr/bin/env python3
"""Run structural checks over generated crosswalk outputs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def check_csv(path: Path, required: list[str]) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [name for name in required if name not in fields]
        rows = 0
        blank_required = 0
        seen: set[tuple[str, ...]] = set()
        duplicates = 0
        for row in reader:
            rows += 1
            if any(not str(row.get(name, "")).strip() for name in required):
                blank_required += 1
            key = tuple(str(row.get(name, "")) for name in required)
            if key in seen:
                duplicates += 1
            seen.add(key)
    empty_output = rows == 0
    status = "ok" if not missing and blank_required == 0 and not empty_output else "failed"
    return {"path": str(path.relative_to(ROOT)), "status": status, "rows": rows, "columns": fields, "missing_columns": missing, "blank_required": blank_required, "empty_output": empty_output, "duplicate_required_keys": duplicates}


def check_jsonl(path: Path, required: list[str]) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    rows = 0
    invalid = 0
    missing = 0
    seen: set[str] = set()
    duplicates = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if any(name not in item for name in required):
                missing += 1
            key = str(item.get("lei", ""))
            if key in seen:
                duplicates += 1
            seen.add(key)
    empty_output = rows == 0
    status = "ok" if invalid == 0 and missing == 0 and not empty_output else "failed"
    return {"path": str(path.relative_to(ROOT)), "status": status, "rows": rows, "invalid_json": invalid, "missing_required": missing, "empty_output": empty_output, "duplicate_lei": duplicates}


def main() -> int:
    checks = [
        check_csv(DATA / "rxnav_ndc_catalog.csv", ["ndc", "source_id"]),
        check_csv(DATA / "dartmouth_zip_hsa_hrr.csv", ["source_vintage", "zipcode18", "hsanum", "hrrnum"]),
        check_jsonl(DATA / "gleif_lei_records.jsonl", ["lei", "legal_name", "raw_json"]),
    ]
    for path in sorted(DATA.glob("census_*.csv")):
        checks.append(check_csv(path, []))
    # RxNav and GLEIF are configured public extractors and must be present;
    # Dartmouth is allowed to be pending when no public CSV vintage is cached.
    required_checks = checks[:1] + checks[2:3]
    result = {"status": "ok" if all(item["status"] == "ok" for item in required_checks) and all(item["status"] in {"ok", "missing"} for item in checks) else "failed", "checks": checks}
    (DATA / "validation_status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
