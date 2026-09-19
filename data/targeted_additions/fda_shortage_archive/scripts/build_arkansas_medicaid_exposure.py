"""Construct an Arkansas Medicaid-exposed FDA shortage target.

The exposure universe is observed Arkansas Medicaid NDC9-quarter utilization.
The attached shortage state is national FDA evidence for the same NDC9. A zero
means no FDA shortage event was observed for that NDC9/quarter; it is not a
claim that every Arkansas pharmacy had the product available.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def _ndc9(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(11)[:9] if digits else ""


def load_arkansas_medicaid_ndc_panel(combined_dir: Path) -> pd.DataFrame:
    """Aggregate real Arkansas SDUD records to NDC9-quarter utilization."""
    aggregates: dict[tuple[str, int, int], float] = {}
    for path in sorted(combined_dir.glob("20*.json.gz")):
        with gzip.open(path, "rt") as handle:
            records = json.load(handle)
        for record in records:
            if record.get("source", {}).get("source_id") != "medicaid_sdud":
                continue
            if record.get("geography", {}).get("admin1") != "AR":
                continue
            observation = record.get("observation", {})
            if observation.get("metric") != "prescription_count":
                continue
            ndc9 = _ndc9(record.get("drug", {}).get("ndc"))
            period = str(record.get("period_start", ""))
            value = pd.to_numeric(observation.get("value"), errors="coerce")
            if not ndc9 or len(period) < 7 or pd.isna(value):
                continue
            key = (ndc9, int(period[:4]), (int(period[5:7]) - 1) // 3 + 1)
            aggregates[key] = aggregates.get(key, 0.0) + float(value)
    if not aggregates:
        raise ValueError(f"no Arkansas Medicaid NDC records under {combined_dir}")
    panel = (pd.DataFrame([
                 {"ndc9": ndc9, "year": year, "quarter": quarter,
                  "medicaid_prescriptions": value}
                 for (ndc9, year, quarter), value in aggregates.items()
             ])
             .groupby(["ndc9", "year", "quarter"], as_index=False)
             .medicaid_prescriptions.sum())
    return panel.sort_values(["ndc9", "year", "quarter"]).reset_index(drop=True)


def attach_fda_shortage_state(medicaid: pd.DataFrame,
                              archive_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach quarterly national FDA shortage evidence to Medicaid exposure."""
    archive = archive_panel.copy()
    archive["month_dt"] = pd.to_datetime(archive["month"] + "-01")
    archive["year"] = archive["month_dt"].dt.year.astype(int)
    archive["quarter"] = archive["month_dt"].dt.quarter.astype(int)
    quarterly = (archive.groupby(["ndc9", "year", "quarter"], as_index=False)
                 .agg(fda_shortage_active=("shortage_active", "max"),
                      fda_shortage_supplier_count=("supplier", "nunique"),
                      fda_resolution_observed=("resolution_observed", "max")))
    out = medicaid.merge(quarterly, on=["ndc9", "year", "quarter"], how="left")
    out["fda_archive_row_observed"] = out["fda_shortage_active"].notna().astype(int)
    for column in ("fda_shortage_active", "fda_shortage_supplier_count",
                   "fda_resolution_observed"):
        out[column] = out[column].fillna(0)
    out["fda_shortage_active"] = out["fda_shortage_active"].astype(int)
    out["fda_shortage_supplier_count"] = out[
        "fda_shortage_supplier_count"].astype(int)
    out["fda_resolution_observed"] = out["fda_resolution_observed"].astype(int)
    return out.sort_values(["ndc9", "year", "quarter"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--archive-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    medicaid = load_arkansas_medicaid_ndc_panel(args.combined_dir)
    archive = pd.read_csv(args.archive_panel, dtype={"ndc9": str})
    output = attach_fda_shortage_state(medicaid, archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata = {
        "target": "Arkansas Medicaid-exposed quarterly national FDA shortage state",
        "rows": int(len(output)),
        "unique_ndc9": int(output["ndc9"].nunique()),
        "min_period": f"{output.year.min()}-Q{output.quarter.min()}",
        "max_period": f"{output.year.max()}-Q{output.quarter.max()}",
        "fda_archive_match_fraction": float(output["fda_archive_row_observed"].mean()),
        "fda_shortage_positive_fraction": float(output["fda_shortage_active"].mean()),
        "zero_semantics": "no FDA shortage event observed; not confirmed availability",
        "geography_semantics": "Arkansas Medicaid utilization exposure; FDA shortage state is national",
        "source_inputs": [str(args.combined_dir), str(args.archive_panel)],
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
