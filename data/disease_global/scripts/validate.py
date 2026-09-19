#!/usr/bin/env python3
"""Validate normalized JSONL observations and write a validation report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from common import DATA_DIR, METADATA_DIR, utc_now


def validate_file(path: Path, start_year: int = 2012, end_year: int = 2022) -> dict[str, object]:
    errors: list[str] = []
    rows = 0
    metrics: set[str] = set()
    geography_levels: set[str] = set()
    years: dict[int, int] = {}
    required = ("source_id", "dataset_id", "period", "geography", "metric", "value", "unit", "dimensions", "provenance")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            rows += 1
            for key in required:
                if key not in row:
                    errors.append(f"line {line_number}: missing {key}")
            period = row.get("period", {})
            geography = row.get("geography", {})
            provenance = row.get("provenance", {})
            dimensions = row.get("dimensions")
            value = row.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append(f"line {line_number}: non-finite numeric value")
            year = period.get("year") if isinstance(period, dict) else None
            if not isinstance(year, int) or isinstance(year, bool) or year < start_year or year > end_year:
                errors.append(f"line {line_number}: invalid period year {year!r}; expected {start_year}-{end_year}")
            else:
                years[year] = years.get(year, 0) + 1
            if not isinstance(period, dict) or set(period) - {"year", "week", "epiweek"}:
                errors.append(f"line {line_number}: invalid period shape")
            if not isinstance(geography, dict) or not all(isinstance(geography.get(key), str) for key in ("level", "id", "name")):
                errors.append(f"line {line_number}: invalid geography")
            else:
                geography_levels.add(geography["level"])
            if not isinstance(dimensions, dict):
                errors.append(f"line {line_number}: dimensions must be an object")
            if not isinstance(provenance, dict) or not all(isinstance(provenance.get(key), str) and provenance[key] for key in ("source_url", "retrieved_at", "raw_file")):
                errors.append(f"line {line_number}: invalid provenance")
            elif urlparse(provenance["source_url"]).scheme not in {"http", "https"}:
                errors.append(f"line {line_number}: invalid source URL")
            metrics.add(str(row.get("metric")))
    return {
        "rows": rows,
        "observed_years": years,
        "metrics": sorted(metrics),
        "geography_levels": sorted(geography_levels),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be no greater than --end-year")
    report: dict[str, object] = {
        "validated_at": utc_now(),
        "focus_window": {"start_year": args.start_year, "end_year": args.end_year},
        "sources": {},
    }
    errors = 0
    for path in sorted(DATA_DIR.glob("*/observations.jsonl")):
        result = validate_file(path, args.start_year, args.end_year)
        report["sources"][path.parent.name] = result  # type: ignore[index]
        errors += len(result["errors"])  # type: ignore[arg-type]
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METADATA_DIR / "validation_errors.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
