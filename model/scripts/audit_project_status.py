"""Report strict project status without conflating proxy and learned success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arkansas_pharma_signal.project_status import audit_project_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path,
                        default=Path("model/artifacts/evaluation"))
    parser.add_argument("--forecast", type=Path, default=Path(
        "model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path(
        "model/artifacts/evaluation/project_status.json"))
    args = parser.parse_args()
    result = audit_project_status(args.evaluation_dir, args.forecast)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
