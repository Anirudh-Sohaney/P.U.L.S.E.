"""Run the repository's untouched-test benchmark without writing to ``test/``.

This wrapper deliberately reuses the existing benchmark implementation and
only redirects its two generated reports to this directory.  It is a
reproducibility aid for the synthetic artifact, not a second model variant.
"""
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = ROOT / "test"
sys.path.insert(0, str(TEST))

import run_publishable_benchmark as benchmark  # noqa: E402


def main() -> None:
    benchmark.OUT_JSON = HERE / "current_benchmark_metrics.json"
    benchmark.OUT_MD = HERE / "current_benchmark_results.md"
    benchmark.main()
    # The upstream writer names its default report location.  This wrapper's
    # report is intentionally self-contained in the synthetic-data package.
    benchmark.OUT_MD.write_text(
        benchmark.OUT_MD.read_text(encoding="utf-8").replace(
            "`publishable_benchmark_metrics.json`", "`current_benchmark_metrics.json`"
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
