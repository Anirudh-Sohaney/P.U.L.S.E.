"""Stress-test a two-year ATC persistence proxy against observed HHS claims.

This is a publication gap audit, not a fitted model or a live forecast. It
uses the same mapped claim-line panel as the research ATC evaluator and never
fills missing class-month observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arkansas_pharma_signal.therapeutic_class_demand import build_therapeutic_class_panel


ROOT = Path(__file__).resolve().parents[2]
DEMAND = ROOT / "data/targeted_additions/hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_ndc_monthly.csv.gz"
MAPPING = ROOT / "data/targeted_additions/rxnorm_ndc_atc/data/rxnorm_ndc_atc_mapping.csv.gz"
CLASSES = (
    "C10", "C08", "N06", "B01", "G01", "A10", "D06", "A02", "A06",
    "V03", "C02", "M03", "D01", "L01", "C03", "N07", "D10", "V04",
)


def audit() -> dict:
    demand = pd.read_csv(DEMAND)
    panel = build_therapeutic_class_panel(demand, MAPPING)
    selected = panel[panel["therapeutic_class"].isin(CLASSES)][
        ["month", "therapeutic_class", "demand_claim_lines"]]
    by_month = selected.pivot(index="month", columns="therapeutic_class",
                              values="demand_claim_lines").sort_index()
    rows = []
    for target_month in by_month.index:
        feature_month = target_month - 24
        if feature_month not in by_month.index or target_month.year < 2021:
            continue
        history = selected[selected["month"] <= feature_month]["demand_claim_lines"]
        if len(history) < 100:
            continue
        thresholds = np.quantile(history, [0.2, 0.4, 0.6, 0.8])
        for atc in CLASSES:
            if atc not in by_month:
                continue
            prior = by_month.at[feature_month, atc]
            observed = by_month.at[target_month, atc]
            if pd.isna(prior) or pd.isna(observed):
                continue
            rows.append((str(target_month), atc, float(prior), float(observed),
                         int(np.digitize(prior, thresholds)),
                         int(np.digitize(observed, thresholds))))
    scored = pd.DataFrame(rows, columns=["target_month", "atc", "predicted_claims",
                                         "observed_claims", "predicted_state", "observed_state"])
    if scored.empty:
        raise ValueError("No comparable ATC class-month pairs")
    by_state = scored.groupby("observed_state")
    recalls = [float((part["predicted_state"] == state).mean())
               for state, part in by_state]
    return {
        "purpose": "two_year_persistence_stress_test_not_live_forecast",
        "demand_sha256": hashlib.sha256(DEMAND.read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
        "source_last_month": str(by_month.index.max()),
        "target_first_month": scored["target_month"].min(),
        "target_last_month": scored["target_month"].max(),
        "horizon_months": 24,
        "class_count": int(scored["atc"].nunique()),
        "comparable_pairs": len(scored),
        "wape": float((scored["predicted_claims"] - scored["observed_claims"]).abs().sum()
                      / scored["observed_claims"].sum()),
        "five_state_accuracy": float((scored["predicted_state"] == scored["observed_state"]).mean()),
        "five_state_balanced_accuracy": float(np.mean(recalls)),
        "method": "For each target month, carry its 24-month-prior class claim lines forward; "
                  "fit global five-state quantiles only on class-month values through the "
                  "prior month. Score only pairs with observed source values at both dates.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
