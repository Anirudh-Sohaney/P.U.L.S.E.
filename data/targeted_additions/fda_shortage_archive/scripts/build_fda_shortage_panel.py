"""Build a monthly FDA shortage-status panel from Wayback CSV captures.

The FDA archive contains national shortage listings, not Arkansas inventory.
This adapter therefore produces a national supplier/product supply-risk target
for training and evaluation only.  It never treats a product disappearing from
an archive snapshot as confirmed availability: unresolved products are
right-censored at the last captured month.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


NDC_RE = re.compile(r"NDC\s*[:#]?\s*([0-9]{3,5}(?:[- ]+[0-9]{1,4}){1,2})", re.I)
ACTIVE_STATUSES = {"current", "resolved"}


def normalize_ndc(value: str) -> str:
    """Convert a dashed 10/11-digit NDC to canonical 11 digits."""
    parts = [re.sub(r"\D", "", p) for p in re.split(r"[- ]+", value.strip())]
    parts = [p for p in parts if p]
    if len(parts) != 3:
        return ""
    widths = (5, 4, 2)
    # FDA NDCs use one short segment in a 10-digit representation.
    short = [i for i, part in enumerate(parts) if len(part) < widths[i]]
    if len(short) != 1:
        return "" if sum(map(len, parts)) != 11 else "".join(parts)
    i = short[0]
    if len(parts[i]) != widths[i] - 1:
        return ""
    parts[i] = parts[i].zfill(widths[i])
    result = "".join(parts)
    return result if len(result) == 11 else ""


def _read_capture(path: Path, capture_timestamp: str) -> tuple[pd.DataFrame, int]:
    bad_lines = 0

    def handle_bad_line(_: list[str]) -> None:
        nonlocal bad_lines
        bad_lines += 1
        return None

    frame = pd.read_csv(path, engine="python", on_bad_lines=handle_bad_line,
                        dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    required = {"Generic Name", "Company Name", "Presentation", "Status",
                "Date of Update", "Change Date", "Initial Posting Date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["capture_timestamp"] = capture_timestamp
    frame["capture_date"] = pd.to_datetime(capture_timestamp[:8], format="%Y%m%d")
    return frame, bad_lines


def extract_rows(raw_dir: Path, captures: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows: list[pd.DataFrame] = []
    skipped_lines = 0
    used_captures = []
    for capture in captures:
        timestamp = str(capture["timestamp"])
        path = raw_dir / f"{timestamp}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame, skipped = _read_capture(path, timestamp)
        skipped_lines += skipped
        frame["status"] = frame["Status"].str.strip().str.lower()
        frame = frame[frame["status"].isin(ACTIVE_STATUSES)].copy()
        frame["ndc11"] = frame["Presentation"].map(
            lambda value: (normalize_ndc(m.group(1)) if (m := NDC_RE.search(value)) else ""))
        frame = frame[frame["ndc11"].ne("")].copy()
        frame["ndc9"] = frame["ndc11"].str[:9]
        frame["supplier"] = frame["Company Name"].str.strip()
        frame["generic_name"] = frame["Generic Name"].str.strip()
        frame["initial_posting_date"] = pd.to_datetime(
            frame["Initial Posting Date"], errors="coerce")
        update = frame["Date of Update"].where(
            frame["Date of Update"].ne(""), frame["Change Date"])
        frame["update_date"] = pd.to_datetime(update, errors="coerce")
        rows.append(frame[["capture_timestamp", "capture_date", "ndc11", "ndc9",
                           "supplier", "generic_name", "status",
                           "initial_posting_date", "update_date"]])
        used_captures.append(timestamp)
    if not rows:
        raise ValueError("no usable archived FDA shortage captures found")
    observations = pd.concat(rows, ignore_index=True).drop_duplicates()
    metadata = {
        "capture_count": len(used_captures),
        "capture_timestamps": used_captures,
        "skipped_malformed_csv_rows": skipped_lines,
        "source_semantics": "archived FDA national shortage listings",
        "geography": "United States; no Arkansas allocation",
        "absence_semantics": "not observed after archive end is not confirmed availability",
    }
    return observations, metadata


def build_panel(observations: pd.DataFrame) -> pd.DataFrame:
    """Convert dated listing observations into right-censored monthly states."""
    observations = observations.copy()
    observations["capture_month"] = observations["capture_date"].dt.to_period("M")
    observations["initial_month"] = observations["initial_posting_date"].dt.to_period("M")
    observations["update_month"] = observations["update_date"].dt.to_period("M")
    last_month = observations["capture_month"].max()
    keys = ["ndc9", "supplier", "generic_name"]
    grouped = []
    for key, group in observations.groupby(keys, dropna=False):
        first = group["initial_month"].min()
        if pd.isna(first):
            first = group["capture_month"].min()
        resolved = group.loc[group["status"].eq("resolved"), "update_month"].min()
        end = min(last_month, resolved) if not pd.isna(resolved) else last_month
        if first > end:
            continue
        months = pd.period_range(first, end, freq="M")
        for month in months:
            is_resolved = not pd.isna(resolved) and month >= resolved
            grouped.append({
                "ndc9": key[0], "supplier": key[1], "generic_name": key[2],
                "month": str(month), "shortage_active": int(not is_resolved),
                "resolution_observed": int(is_resolved),
                "right_censored": int(pd.isna(resolved) and month == end),
                "first_posting_month": str(first),
                "last_capture_month": str(last_month),
                "source_observation_count": int(len(group)),
            })
    panel = pd.DataFrame(grouped)
    if panel.empty:
        raise ValueError("archived observations produced an empty monthly panel")
    return panel.sort_values(keys + ["month"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    captures = json.loads(args.captures.read_text())
    observations, metadata = extract_rows(args.raw_dir, captures)
    panel = build_panel(observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    metadata.update({
        "rows": int(len(panel)),
        "unique_ndc9": int(panel["ndc9"].nunique()),
        "panel_min_month": str(panel["month"].min()),
        "panel_max_month": str(panel["month"].max()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "target": "monthly national FDA-reported shortage_active by NDC9/supplier",
        "not_arkansas_inventory": True,
    })
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
