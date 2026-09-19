"""Write the contract-based metric qualification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arkansas_pharma_signal.metric_audit import audit_metric_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path,
                        default=Path("model/artifacts/evaluation"))
    parser.add_argument("--output", type=Path,
                        default=Path("model/artifacts/evaluation/metric_library_audit.json"))
    args = parser.parse_args()
    result = audit_metric_library(args.evaluation_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualified_metric_count": result["qualified_metric_count"],
        "part_one_gates": result["part_one_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()

