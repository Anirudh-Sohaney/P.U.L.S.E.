"""Build the grain-preserving public evaluation suite.

The suite deliberately has separate tables for near-term Arkansas demand,
spatial Arkansas demand, and supplier/NDC shortage continuation. Public data do
not observe a county-by-supplier-week inventory label, so this module never
creates that Cartesian target.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _ndc9(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    # The public exposure bridge already uses canonical nine-digit NDC keys.
    # Padding to eleven digits before truncating silently changes the key.
    return digits.zfill(9)[:9] if digits else ""


def _qid(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return (pd.to_numeric(year, errors="coerce") - 2012) * 4 + \
        pd.to_numeric(quarter, errors="coerce") - 1


def _context_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate one source to quarter and shift it one quarter backward."""
    if frame.empty:
        return pd.DataFrame(columns=["context_qid"])
    out = frame.groupby(["year", "quarter"], as_index=False).mean(numeric_only=True)
    # A source observation at t is available as a feature only when predicting
    # the following target quarter. The +1 join key prevents same-quarter
    # target leakage while retaining the source quarter's value unchanged.
    out["context_qid"] = (_qid(out["year"], out["quarter"]) + 1).astype("Int64")
    numeric = [c for c in out.columns if c not in {"year", "quarter", "context_qid"}]
    out = out.rename(columns={c: f"{prefix}{c}" for c in numeric})
    return out[["context_qid", *[f"{prefix}{c}" for c in numeric]]]


def build_prior_flu_context(path: Path) -> pd.DataFrame:
    """Create Arkansas/national quarterly FluView context for prior quarters."""
    frame = pd.read_csv(path, usecols=["region", "issue", "ili", "wili"])
    issue = pd.to_numeric(frame["issue"], errors="coerce")
    frame["year"] = (issue // 100).astype("Int64")
    frame["week"] = (issue % 100).astype("Int64")
    frame = frame[frame["year"].notna() & frame["week"].between(1, 53)]
    frame["quarter"] = ((frame["week"].astype(int) - 1) // 13 + 1).clip(upper=4)
    frame["ili"] = pd.to_numeric(frame["ili"], errors="coerce")
    frame["wili"] = pd.to_numeric(frame["wili"], errors="coerce")
    rows = []
    for region, name in (("ar", "ar"), ("nat", "nat")):
        sub = frame[frame["region"].astype(str).str.lower().eq(region)]
        if sub.empty:
            continue
        grouped = sub.groupby(["year", "quarter"], as_index=False).agg(
            ili_mean=("ili", "mean"), wili_mean=("wili", "mean"),
            week_count=("issue", "nunique"))
        grouped = grouped.rename(columns={
            "ili_mean": f"{name}_ili_mean", "wili_mean": f"{name}_wili_mean",
            "week_count": f"{name}_flu_week_count"})
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["context_qid"])
    combined = rows[0]
    for row in rows[1:]:
        combined = combined.merge(row, on=["year", "quarter"], how="outer")
    return _context_columns(combined, "prior_")


def build_prior_news_context(path: Path) -> pd.DataFrame:
    """Aggregate Arkansas news keyword groups to the strictly prior quarter."""
    frame = pd.read_csv(path, usecols=["group", "year", "month", "article_count"])
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame["article_count"] = pd.to_numeric(frame["article_count"], errors="coerce")
    frame = frame.dropna(subset=["year", "month", "article_count"])
    frame["year"] = frame["year"].astype(int)
    frame["quarter"] = ((frame["month"].astype(int) - 1) // 3 + 1)
    frame["group_key"] = frame["group"].astype(str).map(
        lambda value: re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_"))
    wide = (frame.groupby(["year", "quarter", "group_key"], as_index=False)
            .article_count.sum()
            .pivot_table(index=["year", "quarter"], columns="group_key",
                         values="article_count", fill_value=0).reset_index())
    wide.columns.name = None
    wide.columns = [str(c) for c in wide.columns]
    return _context_columns(wide, "prior_news_")


def build_arkansas_ndc_quarter_panel(
    exposure_path: Path, flu_path: Path, news_path: Path,
    archive_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the merged Arkansas NDC9-quarter observation panel."""
    frame = pd.read_csv(exposure_path, dtype={"ndc9": str})
    required = {"ndc9", "year", "quarter", "medicaid_prescriptions",
                "fda_shortage_active", "fda_shortage_supplier_count",
                "fda_archive_row_observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"exposure panel missing columns: {sorted(missing)}")
    frame["ndc9"] = frame["ndc9"].map(_ndc9)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["quarter"] = pd.to_numeric(frame["quarter"], errors="raise").astype(int)
    frame["period_start"] = pd.to_datetime(
        frame["year"].astype(str) + "-" + ((frame["quarter"] - 1) * 3 + 1).astype(str) + "-01")
    frame["qid"] = _qid(frame["year"], frame["quarter"]).astype(int)
    if archive_path is not None:
        archive = pd.read_csv(archive_path, dtype={"ndc9": str})
        archive["ndc9"] = archive["ndc9"].map(_ndc9)
        dates = pd.to_datetime(archive["month"].astype(str) + "-01", errors="coerce")
        archive["year"] = dates.dt.year
        archive["quarter"] = dates.dt.quarter
        censor = (archive.groupby(["ndc9", "year", "quarter"], as_index=False)
                  .right_censored.max())
        frame = frame.merge(censor, on=["ndc9", "year", "quarter"], how="left")
    frame["shortage_right_censored"] = pd.to_numeric(
        frame.get("right_censored", 0), errors="coerce").fillna(0).astype(int)
    if "right_censored" in frame:
        frame = frame.drop(columns="right_censored")
    for context in (build_prior_flu_context(flu_path),
                    build_prior_news_context(news_path)):
        frame = frame.merge(context, left_on="qid", right_on="context_qid", how="left")
        frame = frame.drop(columns="context_qid", errors="ignore")
    context_cols = [c for c in frame.columns if c.startswith("prior_")]
    frame["prior_context_missing"] = frame[context_cols].isna().all(axis=1).astype(int)
    return frame.sort_values(["ndc9", "qid"]).reset_index(drop=True)


def build_next_quarter_test_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Create strict NDC9 feature-quarter -> next-quarter target rows."""
    frame = panel.sort_values(["ndc9", "qid"]).copy()
    if "period_start" not in frame:
        frame["period_start"] = pd.to_datetime(
            frame["year"].astype(str) + "-"
            + ((frame["quarter"] - 1) * 3 + 1).astype(str) + "-01")
    group = frame.groupby("ndc9", sort=False)
    previous_qid = group["qid"].shift(1)
    previous_two_qid = group["qid"].shift(2)
    frame["feature_lag1_medicaid_prescriptions"] = group[
        "medicaid_prescriptions"].shift(1).where(
            frame["qid"].sub(previous_qid).eq(1))
    frame["feature_lag2_medicaid_prescriptions"] = group[
        "medicaid_prescriptions"].shift(2).where(
            frame["qid"].sub(previous_two_qid).eq(2))
    frame["feature_ma2_medicaid_prescriptions"] = (
        frame["medicaid_prescriptions"]
        + frame["feature_lag1_medicaid_prescriptions"]
    ) / 2.0
    frame["next_qid"] = group["qid"].shift(-1)
    valid = frame["next_qid"].sub(frame["qid"]).eq(1)
    for source, target in (("medicaid_prescriptions", "target_next_medicaid_prescriptions"),
                           ("fda_shortage_active", "target_next_shortage_active"),
                           ("fda_shortage_supplier_count", "target_next_shortage_supplier_count"),
                           ("shortage_right_censored", "target_next_shortage_right_censored")):
        frame[target] = group[source].shift(-1)
    frame["target_year"] = group["year"].shift(-1)
    frame["target_quarter"] = group["quarter"].shift(-1)
    frame = frame[valid].copy()
    feature_columns = list(dict.fromkeys([
        "ndc9", "year", "quarter", "period_start", "target_year", "target_quarter",
        "medicaid_prescriptions", "fda_shortage_active", "fda_shortage_supplier_count",
        "fda_archive_row_observed", "shortage_right_censored", "prior_context_missing",
        "feature_lag1_medicaid_prescriptions", "feature_lag2_medicaid_prescriptions",
        "feature_ma2_medicaid_prescriptions",
        *[c for c in frame.columns if c.startswith("prior_")],
        "target_next_medicaid_prescriptions", "target_next_shortage_active",
        "target_next_shortage_supplier_count", "target_next_shortage_right_censored",
    ]))
    return frame[feature_columns].reset_index(drop=True)


def build_county_year_test_panel(path: Path) -> pd.DataFrame:
    """Build strict next-year county/drug demand rows from real outcomes."""
    frame = pd.read_csv(path, dtype={"county_fips": str, "drug_key": str})
    frame = frame.sort_values(["county_fips", "drug_key", "year"]).copy()
    group = frame.groupby(["county_fips", "drug_key"], sort=False)
    frame["target_year"] = group["year"].shift(-1)
    frame["target_next_demand_claims"] = group["demand_claims"].shift(-1)
    frame["target_next_demand_fills"] = group["demand_fills"].shift(-1)
    frame = frame[frame["target_year"].sub(frame["year"]).eq(1)].copy()
    columns = ["year", "target_year", "county_fips", "county_name",
               "arkansas_region", "drug_key", "ingredient", "labeler",
               "demand_claims", "demand_fills", "demand_cost",
               "mapping_confidence", "target_next_demand_claims",
               "target_next_demand_fills"]
    return frame[columns].reset_index(drop=True)


def build_supplier_month_test_panel(path: Path | pd.DataFrame) -> pd.DataFrame:
    """Build strict next-month supplier/NDC continuation rows."""
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        frame = pd.read_csv(path, dtype={"ndc9": str, "supplier": str, "month": str})
    frame["ndc9"] = frame["ndc9"].map(_ndc9)
    frame["month_period"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    frame["month_qid"] = (frame["month_period"].dt.year * 12
                           + frame["month_period"].dt.month)
    frame = frame.sort_values(["ndc9", "supplier", "month_qid"]).copy()
    group = frame.groupby(["ndc9", "supplier"], sort=False)
    frame["next_month_period"] = group["month_period"].shift(-1)
    frame["next_month_qid"] = group["month_qid"].shift(-1)
    frame["target_next_shortage_active"] = group["shortage_active"].shift(-1)
    frame["target_next_right_censored"] = group["right_censored"].shift(-1)
    valid = frame["next_month_qid"].sub(frame["month_qid"]).eq(1)
    frame = frame[valid].copy()
    frame["target_month"] = frame["next_month_period"].astype(str)
    columns = ["ndc9", "supplier", "generic_name", "month", "target_month",
               "shortage_active", "right_censored", "resolution_observed",
               "target_next_shortage_active", "target_next_right_censored",
               "first_posting_month", "last_capture_month"]
    return frame[columns].reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
