"""Strict next-period supervised dataset and feature-layer definitions.

The next-period view aligns features at year t with targets at year t+1 for
the same (city, drug): no same-year target leakage. Categorical drug-identity
layers are one-hot encoded against a fit mask (train years only in
evaluation), so test categories never leak into the encoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .features import (
    DRUG_IDENTITY_COLUMNS,
    EXTERNAL_FEATURE_COLUMNS,
    MEDICAID_ALL_COLUMNS,
    MEDICAID_EXACT_COLUMNS,
    MEDICAID_QUARTERLY_COLUMNS,
    SUPPLY_COLUMNS,
)
from .input_contract import is_operational_feature

HISTORY_COLUMNS = [
    "y_last_log", "demand_claims_lag1_log", "demand_claims_lag2_log",
    "demand_claims_lag3_log", "demand_claims_ma2_log", "demand_claims_ma3_log",
    "delta_log", "delta2_log",
]

PROVIDER_COLUMNS = ["n_provider_types", "demand_benes", "cost_per_fill"]

DISEASE_COLUMNS = [
    "ar_ili_mean", "ar_wili_mean", "nat_ili_mean", "nat_wili_mean",
]
DISEASE_PREFIXES = ["ww_", "nndss_"]

NEWS_PREFIXES = ["news_"]

CATEGORICAL_GROUPS = [
    ("ingredient", "ingredient::", 16),
    ("labeler", "labeler::", 16),
    ("dosage_form", "dosage_form::", 8),
    ("route", "route::", 6),
    ("market_cat", "market_cat::", 6),
    ("dea_schedule", "dea::", 4),
    ("pharm_class", "pharm_class::", 6),
    ("therapeutic_cat", "therapeutic::", 8),
]

VIEW_CORE = ["year", "city", "drug", "drug_key", "y_last", "demand_claims_lag1"]


def _slug(value: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return s[:20]


def fit_encoder_spec(panel: pd.DataFrame, fit_mask: Optional[pd.Series] = None) -> List[dict]:
    """One-hot category lists from a deterministic fit mask (default: all rows)."""
    fit = panel.loc[fit_mask] if fit_mask is not None else panel
    spec: List[dict] = []
    for col, prefix, k in CATEGORICAL_GROUPS:
        if col not in panel.columns:
            continue
        cats = fit[col].fillna("").astype(str).value_counts().head(k).index.tolist()
        cats = [c for c in cats if c]
        if cats:
            spec.append({"column": col, "prefix": prefix, "categories": cats})
    return spec


def apply_encoder(frame: pd.DataFrame, spec: List[dict]) -> pd.DataFrame:
    """One-hot encode categorical columns per spec; unknown categories -> zeros."""
    out = pd.DataFrame(index=frame.index)
    for grp in spec:
        col = grp["column"]
        if col not in frame.columns:
            continue
        series = frame[col].fillna("").astype(str)
        for cat in grp["categories"]:
            out[grp["prefix"] + _slug(cat)] = (series == cat).astype(float)
    return out


def add_interaction_layers(view: pd.DataFrame) -> pd.DataFrame:
    """Add bounded, auditable context-by-therapy interaction features.

    These are deterministic products of observed forecast-time variables. The
    context columns are deliberately capped to keep the interaction layer
    stable on the large city-drug panel; categorical columns already come from
    a train-fitted encoder, so unknown test categories remain zero.
    """
    out = view.copy()
    context = [c for c in out.columns if (
        not c.endswith("_t1")
        and c.startswith(("ar_", "nat_", "ww_", "nndss_", "news_", "recall_",
                      "shortage_", "event_", "global_", "us_"))
        and pd.api.types.is_numeric_dtype(out[c])
    )][:10]
    therapy = [c for c in out.columns if "::" in c and c.split("::", 1)[0] in {
        "ingredient", "pharm_class", "therapeutic", "dosage_form", "route",
    }][:8]
    for ctx in context:
        ctx_values = pd.to_numeric(out[ctx], errors="coerce").fillna(0.0)
        for drug in therapy:
            safe_ctx = _slug(ctx)
            safe_drug = _slug(drug.replace("::", "_"))
            out[f"interaction::{safe_ctx}::{safe_drug}"] = (
                ctx_values * pd.to_numeric(out[drug], errors="coerce").fillna(0.0)
            )
    return out


def numeric_feature_columns(panel: pd.DataFrame) -> List[str]:
    """All available numeric feature columns (no same-year demand targets)."""
    base = HISTORY_COLUMNS + EXTERNAL_FEATURE_COLUMNS + SUPPLY_COLUMNS \
        + PROVIDER_COLUMNS + DISEASE_COLUMNS + MEDICAID_ALL_COLUMNS
    base = [c for c in base if c in panel.columns]
    prefixed = [c for c in panel.columns
                if any(c.startswith(p) for p in NEWS_PREFIXES + DISEASE_PREFIXES)]
    return [c for c in base + prefixed if c in panel.columns]


def build_next_period(
    panel: pd.DataFrame,
    fit_mask: Optional[pd.Series] = None,
) -> tuple:
    """Return (view, encoder_spec).

    view: one row per (city, drug, feature-year t) with features at t and
    targets at t+1. Rows without a t+1 observation are dropped.
    """
    panel = panel.copy()
    if "y_last" not in panel.columns:
        panel["y_last"] = panel["demand_claims"]
    if "demand_claims_lag1" not in panel.columns:
        panel["demand_claims_lag1"] = (
            panel.sort_values(["drug_key", "city", "year"])
            .groupby(["drug_key", "city"])["demand_claims"]
            .shift(1)
        )
    spec = fit_encoder_spec(panel, fit_mask)

    core = panel[["year", "city", "drug", "drug_key", "y_last",
                  "demand_claims_lag1"]].copy()
    core["_panel_index"] = core.index

    tgt = panel[["drug_key", "city", "year",
                 "demand_claims", "demand_fills", "demand_cost",
                 "shortage_events", "shortage_active"]].rename(
        columns={"year": "year1",
                 "demand_claims": "demand_claims_t1",
                 "demand_fills": "demand_fills_t1",
                 "demand_cost": "demand_cost_t1",
                 "shortage_events": "shortage_events_t1",
                 "shortage_active": "shortage_active_t1"})
    core["year1"] = core["year"] + 1
    view = core.merge(tgt, on=["drug_key", "city", "year1"], how="left")
    view = view.dropna(subset=["demand_claims_t1"])
    view = view.drop(columns=["year1"])
    idx = view["_panel_index"].to_numpy()
    view = view.drop(columns=["_panel_index"]).reset_index(drop=True)

    num_cols = numeric_feature_columns(panel)
    view = pd.concat(
        [view, panel.loc[idx, num_cols].reset_index(drop=True),
         apply_encoder(panel.loc[idx], spec).reset_index(drop=True)],
        axis=1)
    return add_interaction_layers(view), spec


def layer_columns(view: pd.DataFrame) -> Dict[str, List[str]]:
    """Map feature-layer names to the columns present in the view."""
    cols = list(view.columns)
    history = [c for c in HISTORY_COLUMNS if c in cols]
    drug = [c for c in cols if "::" in c and c.split("::")[0] in
            {g[0] for g in CATEGORICAL_GROUPS}] + \
        [c for c in ("product_count", "ingredient_products", "labeler_products")
         if c in cols]
    supply = [c for c in SUPPLY_COLUMNS if c in cols]
    medicaid_exact = [c for c in MEDICAID_EXACT_COLUMNS if c in cols]
    medicaid_bridge = [c for c in MEDICAID_QUARTERLY_COLUMNS if c in cols]
    medicaid = medicaid_exact + medicaid_bridge
    disease = [c for c in cols if c in DISEASE_COLUMNS or
               c.startswith("ww_") or c.startswith("nndss_")]
    news = [c for c in cols if c.startswith("news_")]
    provider = [c for c in PROVIDER_COLUMNS if c in cols]
    interactions = [c for c in cols if c.startswith("interaction::")]
    return {
        "history": history,
        "drug": drug,
        "supply": supply,
        "medicaid_exact": medicaid_exact,
        "medicaid_bridge": medicaid_bridge,
        "medicaid": medicaid,
        "disease": disease,
        "news": news,
        "provider": provider,
        "interactions": interactions,
    }


def ablation_features(view: pd.DataFrame) -> Dict[str, List[str]]:
    """Ablation -> feature columns (each builds on the demand-history layer)."""
    layers = layer_columns(view)
    hist = layers["history"]
    extern = layers["drug"] + layers["supply"] + layers["disease"] \
        + layers["news"] + layers["provider"] + layers["medicaid"] \
        + layers["interactions"]
    return {
        "history_only": hist,
        "drug": hist + layers["drug"],
        "disease": hist + layers["disease"],
        "news": hist + layers["news"],
        "supply": hist + layers["supply"],
        "medicaid_exact": hist + layers["medicaid_exact"],
        "medicaid_bridge": hist + layers["medicaid_bridge"],
        "medicaid": hist + layers["medicaid"],
        "full_no_medicaid": [c for c in hist + layers["drug"] + layers["supply"]
                             + layers["disease"] + layers["news"] + layers["provider"]
                             if c in view.columns],
        "full_no_interactions": [c for c in hist + layers["drug"] + layers["supply"]
                                 + layers["disease"] + layers["news"] + layers["provider"]
                                 + layers["medicaid"] if c in view.columns],
        "full_medicaid_exact": [c for c in hist + layers["drug"] + layers["supply"]
                                + layers["disease"] + layers["news"] + layers["provider"]
                                + layers["medicaid_exact"] if c in view.columns],
        "full_medicaid_bridge": [c for c in hist + layers["drug"] + layers["supply"]
                                 + layers["disease"] + layers["news"] + layers["provider"]
                                 + layers["medicaid_bridge"] if c in view.columns],
        "all_external": [c for c in extern if c in view.columns],
        "full": [c for c in hist + extern if c in view.columns],
    }


def feature_columns_for_mode(view: pd.DataFrame, mode: str = "operational") -> List[str]:
    """Feature columns for a training mode.

    ``full`` returns every ablation-full column; ``operational`` drops only
    columns whose input disposition is ``PERIODIC_TRAINING_ONLY`` (annual CMS
    Part D / Medicaid / provider measures that are never live feeds). Any
    other mode raises ``ValueError``.
    """
    full = ablation_features(view)["full"]
    if mode == "full":
        return full
    if mode == "operational":
        return [c for c in full if is_operational_feature(c)]
    raise ValueError(f"unknown feature mode: {mode!r}")
