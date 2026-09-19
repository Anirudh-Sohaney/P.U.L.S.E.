#!/usr/bin/env python3
"""Validate normalized JSONL observations and write a validation report."""

from __future__ import annotations

import json
import math
from pathlib import Path

from common import DATA_DIR, METADATA_DIR, utc_now


def validate_file(path: Path) -> dict[str, object]:
    errors: list[str] = []
    rows = 0
    metrics: set[str] = set()
    geographies: set[str] = set()
    years: dict[int, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            rows += 1
            required = ("source_id", "dataset_id", "period", "geography", "metric", "value", "unit", "dimensions", "provenance")
            for key in required:
                if key not in row:
                    errors.append(f"line {line_number}: missing {key}")
            value = row.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append(f"line {line_number}: non-finite numeric value")
            year = row.get("period", {}).get("year")
            if not isinstance(year, int) or year < 1900 or year > 2100:
                errors.append(f"line {line_number}: invalid period year {year!r}")
            else:
                years[year] = years.get(year, 0) + 1
            metrics.add(str(row.get("metric")))
            geographies.add(str(row.get("geography", {}).get("level")))
    return {"rows": rows, "observed_years": years, "metrics": sorted(metrics), "geography_levels": sorted(geographies), "errors": errors}


def main() -> int:
    report: dict[str, object] = {"validated_at": utc_now(), "sources": {}}
    errors = 0
    for path in sorted(DATA_DIR.glob("*/observations.jsonl")):
        result = validate_file(path)
        report["sources"][path.parent.name] = result  # type: ignore[index]
        errors += len(result["errors"])  # type: ignore[arg-type]
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "validation_errors.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
