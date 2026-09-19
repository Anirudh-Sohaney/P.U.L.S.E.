"""Loader for the public CMS Medicare Quarterly Part D snapshot.

CMS publishes this product as a rolling, preliminary snapshot.  The current
download contains finalized full-year 2024 rows and a partial 2025 year, not a
historical row for every quarter.  This module therefore exposes the file as
latest periodic context only; it must not be used to manufacture historical
quarterly targets or to forward-fill missing quarters.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .entities import normalize_name


DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "targeted_additions" / "cms_medicare_quarterly_partd"
    / "data" / "medicare_quarterly_partd_spending_by_drug.csv"
)
REQUIRED_COLUMNS = {
    "Brnd_Name", "Gnrc_Name", "Mftr_Name", "Year", "Tot_Clms",
    "Tot_Spndng", "Tot_Benes",
}


def _year_period(value: object) -> Optional[Tuple[int, int]]:
    """Return the last covered year and quarter from CMS's period label."""
    match = re.search(r"(20\d{2})\s*\(Q1-Q(\d)\)", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_cms_partd_snapshot(path: Optional[Path] = None) -> pd.DataFrame:
    """Load one CMS snapshot, retaining only drug-level ``Overall`` rows.

    The returned values are current-context observations.  ``coverage_end``
    records the last quarter represented by each source row so downstream
    code can enforce an availability-time cutoff.
    """
    source = DEFAULT_PATH if path is None else Path(path)
    if not source.exists():
        return pd.DataFrame()
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = REQUIRED_COLUMNS - header
    if missing:
        raise ValueError(f"CMS Part D snapshot missing columns: {sorted(missing)}")
    raw = pd.read_csv(source, usecols=sorted(REQUIRED_COLUMNS), low_memory=False)
    raw = raw[raw["Mftr_Name"].astype(str).str.strip().eq("Overall")].copy()
    periods = raw["Year"].map(_year_period)
    raw["coverage_end_year"] = periods.map(lambda x: x[0] if x else pd.NA)
    raw["coverage_end_quarter"] = periods.map(lambda x: x[1] if x else pd.NA)
    raw = raw.dropna(subset=["coverage_end_year", "coverage_end_quarter"])
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "drug_name": raw["Gnrc_Name"].fillna(raw["Brnd_Name"]).astype(str).str.strip(),
        "brand_name": raw["Brnd_Name"].fillna("").astype(str).str.strip(),
        "claims": pd.to_numeric(raw["Tot_Clms"], errors="coerce"),
        "spending": pd.to_numeric(raw["Tot_Spndng"], errors="coerce"),
        "beneficiaries": pd.to_numeric(raw["Tot_Benes"], errors="coerce"),
        "coverage_end_year": raw["coverage_end_year"].astype(int),
        "coverage_end_quarter": raw["coverage_end_quarter"].astype(int),
        "source": "CMS Medicare Quarterly Part D Spending by Drug",
    })
    out = out[out["drug_name"].ne("")].copy()
    return out.reset_index(drop=True)


def snapshot_metadata(path: Optional[Path] = None) -> Dict:
    """Return auditable availability metadata without treating it as a label."""
    frame = load_cms_partd_snapshot(path)
    if frame.empty:
        return {"available": False, "rows": 0}
    return {
        "available": True,
        "rows": int(len(frame)),
        "drugs": int(frame["drug_name"].nunique()),
        "coverage_end": {
            "year": int(frame["coverage_end_year"].max()),
            "quarter": int(frame.loc[
                frame["coverage_end_year"].eq(frame["coverage_end_year"].max()),
                "coverage_end_quarter",
            ].max()),
        },
        "semantics": "latest periodic national Part D context; not Arkansas ground truth",
        "target_use": "context input only; never a historical target or interpolated quarter",
    }


def add_latest_context_layer(
    view: pd.DataFrame,
    path: Optional[Path] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach CMS context only at the snapshot coverage endpoint.

    A current snapshot is not a time series.  Joining it to every historical
    row would leak a future observation, so unmatched periods intentionally
    remain missing and are later imputed from the training slice if needed.
    """
    snapshot = load_cms_partd_snapshot(path)
    if snapshot.empty or not {"year", "quarter", "drug"} <= set(view.columns):
        return view, []
    end_year = int(snapshot["coverage_end_year"].max())
    end_quarter = int(snapshot.loc[
        snapshot["coverage_end_year"].eq(end_year), "coverage_end_quarter"
    ].max())
    current = snapshot[
        snapshot["coverage_end_year"].eq(end_year)
        & snapshot["coverage_end_quarter"].eq(end_quarter)
    ].copy()
    current["drug_norm"] = current["drug_name"].map(normalize_name)
    current = current.drop_duplicates("drug_norm")
    out = view.copy()
    out["drug_norm"] = out.get("ingredient", out["drug"]).fillna(out["drug"]).map(normalize_name)
    # Join the endpoint separately to keep the source's coverage timestamp
    # visible in the feature values and avoid a broad time-series merge.
    current = current.rename(columns={
        "claims": "exo_cms_partd_claims",
        "spending": "exo_cms_partd_spending",
        "beneficiaries": "exo_cms_partd_beneficiaries",
    })
    current["year"] = end_year
    current["quarter"] = end_quarter
    out = out.merge(
        current[["drug_norm", "year", "quarter", "exo_cms_partd_claims",
                 "exo_cms_partd_spending", "exo_cms_partd_beneficiaries"]],
        on=["drug_norm", "year", "quarter"], how="left",
    )
    out = out.drop(columns=["drug_norm"])
    cols = [
        "exo_cms_partd_claims", "exo_cms_partd_spending",
        "exo_cms_partd_beneficiaries",
    ]
    return out, cols
