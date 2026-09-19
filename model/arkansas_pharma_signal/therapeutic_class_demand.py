"""Monthly Arkansas pharmacy demand grouped by RxNorm/ATC class.

The HHS source contains NDC keys but no therapeutic labels. NLM RxNorm/RxClass
provides a public historical NDC crosswalk and ATC relations. Each mapped
NDC's claim lines are fractionally allocated across its unique three-character
ATC therapeutic groups. This preserves total mapped demand without
arbitrarily choosing one indication for multi-class products. Unmapped NDCs
are excluded rather than treated as a class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .event_accuracy import score_event_predictions
from .monthly_pharmacy_demand import (
    build_monthly_pharmacy_demand_view, evaluate_monthly_pharmacy_demand,
)


SOURCE_URL = "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html"
CLASS_SOURCE_URL = "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html"


def _ndc11(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(11)


def load_rxnorm_atc_mapping(source: pd.DataFrame | str) -> pd.DataFrame:
    """Load the manifest-backed NDC-to-ATC relation table."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, dtype={"ndc11": str})
    required = {"ndc11", "class_id", "class_name", "mapping_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RxNorm mapping missing columns: {sorted(missing)}")
    frame["ndc11"] = _ndc11(frame["ndc11"])
    frame["class_id"] = frame["class_id"].fillna("").astype(str).str.strip()
    frame["class_name"] = frame["class_name"].fillna("").astype(str).str.strip()
    frame = frame[frame["class_id"].ne("") & frame["class_id"].str.len().ge(3)]
    frame["atc_group"] = frame["class_id"].str[:3]
    return frame.drop_duplicates(["ndc11", "atc_group"], keep="first").reset_index(drop=True)


def build_therapeutic_class_panel(
    demand: pd.DataFrame, mapping: pd.DataFrame | str,
) -> pd.DataFrame:
    """Aggregate observed HHS claim lines into monthly ATC groups."""
    required = {"month", "drug_key", "demand_claim_lines", "demand_paid",
                "pharmacy_provider_count"}
    missing = required.difference(demand.columns)
    if missing:
        raise ValueError(f"HHS demand missing columns: {sorted(missing)}")
    frame = demand.copy()
    frame["ndc11"] = _ndc11(frame["drug_key"])
    frame["month"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    for column in ("demand_claim_lines", "demand_paid", "pharmacy_provider_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["month", "ndc11", "demand_claim_lines",
                                 "demand_paid", "pharmacy_provider_count"])
    crosswalk = load_rxnorm_atc_mapping(mapping)
    class_names = (crosswalk[["atc_group", "class_name"]]
                   .sort_values(["atc_group", "class_name"])
                   .drop_duplicates("atc_group", keep="first"))
    relations = crosswalk[["ndc11", "atc_group"]].drop_duplicates()
    frame = frame.merge(relations,
                        on="ndc11", how="inner")
    frame = frame.merge(class_names, on="atc_group", how="left")
    class_counts = frame.groupby("ndc11")["atc_group"].transform("nunique")
    frame["allocation_weight"] = 1.0 / class_counts.clip(lower=1)
    frame["demand_claim_lines"] *= frame["allocation_weight"]
    frame["demand_paid"] *= frame["allocation_weight"]
    grouped = (frame.groupby(["month", "atc_group", "class_name"], as_index=False)
               .agg(demand_claim_lines=("demand_claim_lines", "sum"),
                    demand_paid=("demand_paid", "sum"),
                    pharmacy_provider_count=("pharmacy_provider_count", "max")))
    grouped["therapeutic_class"] = grouped["atc_group"]
    grouped["drug_key"] = grouped["atc_group"]
    return grouped.sort_values(["therapeutic_class", "month"]).reset_index(drop=True)


def evaluate_therapeutic_class_demand(
    demand: pd.DataFrame, mapping: pd.DataFrame | str,
) -> dict[str, Any]:
    """Evaluate next-month five-state ATC-group demand chronologically."""
    panel = build_therapeutic_class_panel(demand, mapping)
    total_claim_lines = float(pd.to_numeric(
        demand["demand_claim_lines"], errors="coerce").sum())
    mapped_claim_lines = float(panel["demand_claim_lines"].sum())
    view = build_monthly_pharmacy_demand_view(
        panel, group_columns=("therapeutic_class",))
    result = evaluate_monthly_pharmacy_demand(
        view, group_columns=("therapeutic_class",))
    return {
        **result,
        "protocol": "rolling_origin_next_month_arkansas_atc_therapeutic_class_demand",
        "target": "arkansas_monthly_atc_therapeutic_demand_state",
        "cadence": "monthly",
        "class_count": int(panel["therapeutic_class"].nunique()),
        "mapped_ndc_count": int(load_rxnorm_atc_mapping(mapping)["ndc11"].nunique()),
        "total_claim_lines": total_claim_lines,
        "mapped_claim_lines": mapped_claim_lines,
        "mapped_claim_line_fraction": (
            mapped_claim_lines / total_claim_lines if total_claim_lines else 0.0),
        "mapping_source_url": SOURCE_URL,
        "class_source_url": CLASS_SOURCE_URL,
        "state_definition": {str(i): f"fit_quantile_{i + 1}_of_5" for i in range(5)},
        "event_definition": {"kind": "state", "event_states": [3, 4]},
        "scope": (
            "Arkansas HHS Medicaid/CHIP pharmacy claim-line demand for the mapped "
            "NDC subset, fractionally allocated across three-character ATC therapeutic "
            "group; not all-payer demand or inventory"
        ),
        "publishable_candidate": bool(
            result.get("publishable_candidate")
            and result.get("class_count", result.get("group_count", 0)) >= 5
        ),
    }


def build_latest_therapeutic_class_state(
    demand: pd.DataFrame, mapping: pd.DataFrame | str,
) -> pd.DataFrame:
    """Create persistence forecasts for the latest mapped ATC groups."""
    panel = build_therapeutic_class_panel(demand, mapping)
    class_names = (panel[["therapeutic_class", "class_name"]]
                   .drop_duplicates("therapeutic_class")
                   .set_index("therapeutic_class")["class_name"]
                   .to_dict())
    view = build_monthly_pharmacy_demand_view(
        panel, group_columns=("therapeutic_class",))
    if view.empty:
        return pd.DataFrame()
    # The transition view omits the final observed month by design. Use the
    # raw panel for the live feature month, then predict its consecutive month.
    latest_month = panel["month"].max()
    latest = panel[panel["month"].eq(latest_month)]
    history = view[view["month"].lt(latest_month)]
    thresholds = history["target_demand_claims"].quantile(np.arange(1, 5) / 5).to_numpy()
    rows = []
    for _, row in latest.iterrows():
        rows.append({
            "therapeutic_class": str(row["therapeutic_class"]),
            "class_name": str(class_names.get(row["therapeutic_class"], "")),
            "forecast_period": str(latest_month + 1),
            "prediction": int(np.digitize(float(row["demand_claim_lines"]), thresholds)),
            "feature_period": str(latest_month),
            "current_claim_lines": float(row["demand_claim_lines"]),
            "thresholds": [float(value) for value in thresholds],
        })
    return pd.DataFrame(rows)
