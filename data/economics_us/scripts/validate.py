"""Validate normalized economics_us JSONL output.

This validator is deliberately dependency-free and writes a machine-readable
error list to metadata/validation_errors.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "by_source"
METADATA = ROOT / "metadata"
START = "2012-01-01"
END = "2022-12-31"
FISCAL_START = "2011-10-01"


def validate() -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    if not DATA.exists():
        errors.append({"file": str(DATA), "error": "data directory does not exist"})
        return errors
    for path in sorted(DATA.glob("**/*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"file": str(path), "line": str(line_number), "error": f"invalid JSON: {exc.msg}"})
                continue
            required = {"record_id", "period_start", "period_end", "granularity", "geography", "observation", "source", "quality"}
            missing = required - row.keys()
            if missing:
                errors.append({"file": str(path), "line": str(line_number), "error": f"missing fields: {sorted(missing)}"})
                continue
            record_id = row["record_id"]
            if record_id in seen:
                errors.append({"file": str(path), "line": str(line_number), "error": f"duplicate record_id: {record_id}"})
            seen.add(record_id)
            try:
                period_start = dt.date.fromisoformat(row["period_start"])
                period_end = dt.date.fromisoformat(row["period_end"])
            except (TypeError, ValueError):
                errors.append({"file": str(path), "line": str(line_number), "error": "period dates must be ISO YYYY-MM-DD"})
                continue
            if period_start > period_end:
                errors.append({"file": str(path), "line": str(line_number), "error": "period_start is after period_end"})
            is_fiscal = row["granularity"] == "fiscal_year"
            lower_bound = FISCAL_START if is_fiscal else START
            lower_date = dt.date.fromisoformat(lower_bound)
            upper_date = dt.date.fromisoformat(END)
            if not (lower_date <= period_start <= upper_date and lower_date <= period_end <= upper_date):
                errors.append({"file": str(path), "line": str(line_number), "error": "period outside 2012-2022 (or FY2012 start)"})
            if is_fiscal and (row["period_start"][5:] != "10-01" or row["period_end"][5:] != "09-30"):
                errors.append({"file": str(path), "line": str(line_number), "error": "fiscal-year boundaries must be Oct 1 through Sep 30"})
            nested_required = {
                "geography": {"country", "name", "code", "level"},
                "observation": {"metric", "value", "unit", "semantics"},
                "source": {"source_id", "source_name", "source_url", "retrieved_at"},
                "quality": {"is_imputed", "is_suppressed"},
            }
            for section, fields in nested_required.items():
                missing_nested = fields - row.get(section, {}).keys()
                if missing_nested:
                    errors.append({"file": str(path), "line": str(line_number), "error": f"{section} missing fields: {sorted(missing_nested)}"})
            value = row["observation"].get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append({"file": str(path), "line": str(line_number), "error": "observation.value is not finite numeric"})
            if row["quality"].get("is_imputed") is not False:
                errors.append({"file": str(path), "line": str(line_number), "error": "is_imputed must remain false"})
            if not row["source"].get("source_id"):
                errors.append({"file": str(path), "line": str(line_number), "error": "source_id is empty"})
    return errors


def main() -> int:
    errors = validate()
    METADATA.mkdir(parents=True, exist_ok=True)
    (METADATA / "validation_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"validated records; errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
