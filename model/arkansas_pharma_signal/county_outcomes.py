"""Real CMS city-demand aggregation to county outcomes."""

from __future__ import annotations

import pandas as pd

from . import io
from .geography import region_for_county

OUTCOME_COLUMNS = ["year", "county_fips", "county_name", "arkansas_region", "drug_key",
                   "ingredient", "labeler", "demand_claims", "demand_fills", "demand_cost",
                   "source_row_count", "mapped_city_count", "mapping_confidence",
                   "source", "source_timestamp"]


def build_county_outcomes(panel: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only rows with a sourced county mapping; no imputation."""
    mapping = crosswalk[crosswalk["county_fips"].fillna("").astype(str).ne("")].copy()
    mapping["city_key"] = mapping["city"].astype(str).str.lower()
    p = panel.copy()
    p["city_key"] = p["city"].astype(str).str.lower()
    p = p.merge(mapping[["city_key", "county_fips", "county_name", "arkansas_region",
                         "confidence"]], on="city_key", how="inner")
    if p.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    # County FIPS is the sourced geography. Region is a deterministic reporting
    # bucket derived from that FIPS, so a stale blank crosswalk region cannot
    # erase usable regional categorization.
    mapped_region = p["county_fips"].map(region_for_county)
    p["arkansas_region"] = p["arkansas_region"].where(
        p["arkansas_region"].fillna("").astype(str).str.strip().ne(""),
        mapped_region)
    for c in ("demand_claims", "demand_fills", "demand_cost"):
        if c not in p:
            p[c] = 0.0
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    group = ["year", "county_fips", "county_name", "arkansas_region", "drug_key"]
    result = p.groupby(group, dropna=False).agg(
        ingredient=("ingredient", "first"), labeler=("labeler", "first"),
        demand_claims=("demand_claims", "sum"), demand_fills=("demand_fills", "sum"),
        demand_cost=("demand_cost", "sum"), source_row_count=("city_key", "size"),
        mapped_city_count=("city_key", "nunique"), mapping_confidence=("confidence", "mean"),
    ).reset_index()
    result["source"] = "CMS Part D city/provider demand + Census-geocoded NPPES city mapping"
    result["source_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
    return result[OUTCOME_COLUMNS]


def validate_county_outcomes(outcomes: pd.DataFrame) -> None:
    missing = set(OUTCOME_COLUMNS) - set(outcomes.columns)
    if missing:
        raise ValueError(f"county outcome schema missing: {sorted(missing)}")
    if (pd.to_numeric(outcomes["demand_claims"], errors="coerce") < 0).any():
        raise ValueError("county demand cannot be negative")
