"""Freeze the currently audited Part D forecast rows for checkout-clean CI.

The full research forecast and audit live under ignored model/artifacts/. Run
this deliberately from a research workspace when the audit changes; CI reads
only the resulting small, tracked fixture and never assumes those ignored
files exist in a fresh checkout.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"
AUDIT = ROOT / "model/artifacts/evaluation/metric_library_audit.json"
FIXTURE = ROOT / "model/tests/fixtures/metric_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    if audit.get("target_validity", {}).get("passed") is not True:
        raise ValueError("The metric audit has not passed target validity")
    approved = sorted(row["metric"] for row in audit["candidates"]
                      if row.get("status") == "qualified_proxy" and not row.get("reasons"))
    if approved != ["arkansas_state_annual_partd_demand_five_state"]:
        raise ValueError(f"Review the qualified target set before replacing the fixture: {approved}")
    source = pd.read_csv(SOURCE, compression="gzip", low_memory=False)
    selected = source[source["target"].isin(approved)]
    if len(selected) != 1212 or len(source) != 43366:
        raise ValueError("Research forecast row counts changed; review the fixture and CI expectations")
    FIXTURE.mkdir(parents=True, exist_ok=True)
    forecast_path = FIXTURE / "qualified_partd_forecasts.csv.gz"
    audit_path = FIXTURE / "metric_library_audit.json"
    with forecast_path.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(selected.to_csv(index=False).encode("utf-8"))
    audit_path.write_bytes(AUDIT.read_bytes())
    manifest = {
        "schema": "pulse_metric_ci_fixture_v1",
        "source_forecast_sha256": sha256(SOURCE),
        "source_forecast_rows": len(source),
        "source_audit_sha256": sha256(AUDIT),
        "fixture_forecast_sha256": sha256(forecast_path),
        "fixture_audit_sha256": sha256(audit_path),
        "fixture_rows": len(selected),
        "qualified_targets": approved,
        "meaning": "Research CI fixture only; no live daily model publication claim",
    }
    (FIXTURE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
