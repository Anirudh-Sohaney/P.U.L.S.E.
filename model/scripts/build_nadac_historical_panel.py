"""Build a reproducible CMS NADAC history for the Arkansas exposure universe.

CMS publishes annual NADAC snapshots with small schema changes across years.
This script normalizes those snapshots to the project's weekly-panel schema,
filters them to NDCs observed in the Arkansas exposure bridge, and combines
them with the locally downloaded weekly history. It does not infer local
pharmacy prices or supplier allocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from arkansas_pharma_signal.nadac_target import _ndc9


def _read_cms_snapshot(path: Path, allowed: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    price_column = "NADAC_Per_Unit" if "NADAC_Per_Unit" in header else "NADAC Per Unit"
    usecols = ["NDC", price_column, "As of Date"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000,
                             dtype={"NDC": str}):
        chunk = chunk.rename(columns={
            "NDC": "ndc",
            price_column: "nadac_per_unit",
            "As of Date": "as_of_date",
        })
        chunk["ndc"] = chunk["ndc"].map(_ndc9)
        chunk["nadac_per_unit"] = pd.to_numeric(
            chunk["nadac_per_unit"], errors="coerce")
        chunk["as_of_date"] = pd.to_datetime(
            chunk["as_of_date"], errors="coerce")
        chunk = chunk[chunk["ndc"].isin(allowed)
                      & chunk["nadac_per_unit"].gt(0)
                      & chunk["as_of_date"].notna()]
        if not chunk.empty:
            chunks.append(chunk[["ndc", "nadac_per_unit", "as_of_date"]])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["ndc", "nadac_per_unit", "as_of_date"])


def _read_local_weekly(path: Path, allowed: set[str], start: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["ndc", "nadac_per_unit", "as_of_date"],
                        dtype={"ndc": str})
    frame["ndc"] = frame["ndc"].map(_ndc9)
    frame["nadac_per_unit"] = pd.to_numeric(frame["nadac_per_unit"], errors="coerce")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    return frame[frame["ndc"].isin(allowed)
                 & frame["nadac_per_unit"].gt(0)
                 & frame["as_of_date"].notna()
                 & frame["as_of_date"].ge(pd.Timestamp(start))].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--local-weekly", type=Path, required=True)
    parser.add_argument("--cms-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", default="2021-01-01")
    args = parser.parse_args()

    exposure = pd.read_csv(args.exposure, usecols=["ndc9"], dtype={"ndc9": str})
    allowed = {_ndc9(value) for value in exposure["ndc9"].dropna()}
    frames = [_read_local_weekly(args.local_weekly, allowed, args.start)]
    snapshots = sorted(args.cms_dir.glob("nadac_*.csv"))
    for path in snapshots:
        frames.append(_read_cms_snapshot(path, allowed))
    panel = pd.concat(frames, ignore_index=True)
    panel = (panel.groupby(["ndc", "as_of_date"], as_index=False)["nadac_per_unit"]
             .mean().sort_values(["ndc", "as_of_date"]))
    if panel.empty:
        raise ValueError("No verified NADAC observations match the exposure universe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False, compression="gzip", date_format="%Y-%m-%d")
    manifest = {
        "source": "CMS NADAC annual snapshots plus local weekly NADAC history",
        "official_source": "https://www.medicaid.gov/medicaid/nadac",
        "exposure_source": str(args.exposure),
        "snapshots": [str(path) for path in snapshots],
        "local_weekly_source": str(args.local_weekly),
        "start": args.start,
        "ndc_count": int(panel["ndc"].nunique()),
        "row_count": int(len(panel)),
        "first_date": panel["as_of_date"].min().strftime("%Y-%m-%d"),
        "last_date": panel["as_of_date"].max().strftime("%Y-%m-%d"),
        "years": sorted(panel["as_of_date"].dt.year.unique().tolist()),
        "filtering": "NDCs are limited to the Arkansas exposure bridge; no local allocation is inferred",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
