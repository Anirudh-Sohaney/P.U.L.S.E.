"""County × drug × supplier forecast contract.

This is the deployable upstream intelligence surface.  It accepts the current
panel/model artifacts and keeps unavailable county, parent, factory, and API
coverage explicit rather than silently substituting a city or labeler.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .forecast import HORIZONS_DAYS, TARGETS, _feature_matrix, _restore_demand_model, _restore_risk_model
from .geography import region_for_county
from .arcos_evaluation import SOURCE_RELATIVE, build_next_quarter_view, forecast_latest_arcos

UNIVERSAL_OUTPUT_COLUMNS = [
    "forecast_timestamp", "forecast_period", "horizon", "geography_level", "geography_id",
    "county_fips", "county_name", "arkansas_region",
    "drug_key", "ingredient", "therapeutic_class", "pathogen", "dosage_form", "strength", "route", "labeler", "supplier",
    "parent_company", "factory", "api_source", "target", "prediction", "interval_low",
    "interval_high", "risk_score", "confidence", "source_freshness", "event_ids",
    "evidence_article_ids", "driver_attribution", "evidence_type", "model_version",
    "feature_window_start", "feature_window_end",
    "target_cadence", "target_geography_scope", "target_supplier_resolution",
    "target_is_direct_pharmacy_observation", "target_observation_type",
    "target_promotion_status", "target_semantics", "uncertainty_status",
    "calibration_status", "target_event_definition",
]
REGION_TARGETS = (
    "demand_claims", "demand_cost", "demand_shock_index",
    "supply_disruption_risk", "arkansas_shortage_impact",
)

# Output metadata is deliberately conservative. These fields describe what a
# row can support, not how accurate its prediction is.
TARGET_METADATA = {
    "demand_claims": {
        "cadence": "annual",
        "geography": "Arkansas county or reporting region",
        "supplier": "labeler_only",
        "direct": False,
        "observation": "forecast",
        "status": "annual_demand_research_only",
        "semantics": "forecast from annual observed demand history; not inventory",
    },
    "demand_cost": {
        "cadence": "annual",
        "geography": "Arkansas county or reporting region",
        "supplier": "labeler_only",
        "direct": False,
        "observation": "forecast",
        "status": "annual_demand_research_only",
        "semantics": "forecast from annual observed cost history; not acquisition cost",
    },
    "demand_shock_index": {
        "cadence": "annual",
        "geography": "Arkansas county or reporting region",
        "supplier": "none",
        "direct": False,
        "observation": "forecast",
        "status": "derived_research_signal",
        "semantics": "derived demand-shock signal; not a direct pharmacy observation",
    },
    "supply_disruption_risk": {
        "cadence": "monthly",
        "geography": "Arkansas projection without local supplier allocation",
        "supplier": "FDA supplier evidence",
        "direct": False,
        "observation": "forecast",
        "status": "research_only_supplier_evidence",
        "semantics": "risk projection from national shortage evidence",
    },
    "arkansas_shortage_impact": {
        "cadence": "monthly",
        "geography": "Arkansas statewide relevance context",
        "supplier": "FDA supplier evidence",
        "direct": False,
        "observation": "forecast",
        "status": "research_only_supplier_evidence",
        "semantics": "exposure-weighted shortage impact; not local inventory",
    },
    "supplier_shortage_probability": {
        "cadence": "monthly",
        "geography": "national NDC9 x FDA supplier",
        "supplier": "FDA supplier",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_supplier_evidence",
        "semantics": "continuation probability from FDA shortage observations",
    },
    "arkansas_supplier_drug_shortage_pressure": {
        "cadence": "monthly",
        "geography": "Arkansas statewide drug relevance",
        "supplier": "FDA supplier; Arkansas allocation unverified",
        "direct": False,
        "observation": "forecast",
        "status": "research_only_supplier_evidence",
        "semantics": "exact-drug supplier shortage relevance; no local allocation claim",
    },
    "regional_distribution_pressure": {
        "cadence": "quarterly",
        "geography": "Arkansas ZIP3 distribution context",
        "supplier": "none",
        "direct": False,
        "observation": "observed_proxy_context",
        "status": "proxy_only",
        "semantics": "DEA ARCOS controlled-substance distribution grams",
    },
    "regional_distribution_pressure_forecast": {
        "cadence": "quarterly",
        "geography": "Arkansas ZIP3 distribution context",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "proxy_only",
        "semantics": "forecast of DEA ARCOS distribution grams",
    },
    "arcos_zip3_drug_next_quarter_distribution_state": {
        "cadence": "quarterly",
        "geography": "Arkansas ZIP3 x controlled-substance code",
        "supplier": "none in ARCOS Report 01",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-quarter five-quantile DEA ARCOS controlled-substance distribution state; not pharmacy inventory",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "drug_key"],
            "destination_keys": ["forecast_period", "zip3", "drug_key"],
            "scope": "matching Arkansas ZIP3 and controlled-substance code",
        },
    },
    "arkansas_state_annual_partd_demand_five_state": {
        "cadence": "annual",
        "geography": "Arkansas statewide Part D claims",
        "supplier": "none available in CMS product",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-year five-quantile Medicare Part D claims demand proxy by generic drug; not all-payer demand or inventory",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "drug_key"],
            "destination_keys": ["forecast_period", "state", "drug_key"],
            "scope": "Arkansas statewide drug context; no supplier allocation",
        },
    },
    "ndc_monthly_shortage_pressure_state": {
        "cadence": "monthly",
        "geography": "national FDA evidence joined to Arkansas-exposed NDCs",
        "supplier": "supplier count, not supplier allocation",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-month FDA active-shortage supplier-count state: none, one, two, three, or four-plus suppliers",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "drug_key"],
            "destination_keys": ["forecast_period", "drug_key"],
            "scope": "national drug context; no county or supplier allocation",
        },
    },
    "arkansas_weekly_fluview_respiratory_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas statewide respiratory surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast",
        "status": "legacy_three_state_proxy",
        "semantics": "next-week low, mid, or high Arkansas FluView WILI pressure state; not pharmacy dispensing truth",
        "context_projection": {
            "source_keys": ["forecast_period"],
            "destination_keys": ["forecast_period"],
            "scope": "Arkansas statewide context",
        },
    },
    "national_weekly_fluview_respiratory_pressure_state": {
        "cadence": "weekly",
        "geography": "United States respiratory surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-week five-quantile national FluView WILI pressure state; not pharmacy dispensing truth",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period"],
            "destination_keys": ["forecast_period"],
            "scope": "national upstream respiratory context",
        },
    },
    "national_weekly_respnet_rsv_hospitalization_pressure_state": {
        "cadence": "weekly",
        "geography": "United States RESP-NET surveillance aggregate",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_five_state_proxy",
        "semantics": "next-week five-quantile national RSV hospitalization pressure; not pharmacy dispensing truth",
        "event": {"kind": "state", "event_states": [3, 4]},
    },
    "arkansas_monthly_apcd_pharmacy_claim_activity_state": {
        "cadence": "monthly",
        "geography": "Arkansas APCD reporting entity",
        "supplier": "none available",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_five_state_proxy",
        "semantics": "next-month five-quantile Arkansas APCD pharmacy-claim activity by reporting entity; not NDC demand, supplier allocation, or inventory",
        "event": {"kind": "state", "event_states": [3, 4]},
    },
    "arkansas_monthly_atc_therapeutic_demand_state": {
        "cadence": "monthly",
        "geography": "Arkansas statewide HHS pharmacy-taxonomy billing providers",
        "supplier": "none available",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_five_state_proxy",
        "semantics": "next-month five-state claim-line demand fractionally allocated across three-character ATC therapeutic groups for the mapped NDC subset; not all-payer demand or inventory",
        "event": {"kind": "state", "event_states": [3, 4]},
    },
    "arkansas_weekly_nssp_influenza_ed_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas statewide NSSP ED surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-week five-quantile Arkansas NSSP influenza ED-visit pressure state; not pharmacy dispensing truth",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "pathogen"],
            "destination_keys": ["forecast_period", "pathogen"],
            "scope": "Arkansas statewide influenza symptom/disease context",
        },
    },
    "arkansas_region_weekly_wastewater_influenza_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas DHS region inferred from CDC wastewater county served",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_five_state_proxy",
        "semantics": "next-week five-quantile regional CDC wastewater influenza activity; not pharmacy dispensing truth",
        "context_projection": {
            "source_keys": ["forecast_period", "arkansas_region"],
            "destination_keys": ["forecast_period", "arkansas_region"],
            "scope": "matching Arkansas DHS region",
        },
    },
    "arkansas_region_weekly_wastewater_sars_cov_2_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas DHS region inferred from CDC wastewater county served",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_five_state_proxy",
        "semantics": "next-week five-quantile regional CDC wastewater SARS-CoV-2 activity; not pharmacy dispensing truth",
        "context_projection": {
            "source_keys": ["forecast_period", "arkansas_region"],
            "destination_keys": ["forecast_period", "arkansas_region"],
            "scope": "matching Arkansas DHS region",
        },
    },
    "arkansas_region_weekly_wastewater_rsv_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas DHS region inferred from CDC wastewater county served",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_five_state_proxy",
        "semantics": "next-week five-quantile regional CDC wastewater RSV activity; not pharmacy dispensing truth",
        "context_projection": {
            "source_keys": ["forecast_period", "arkansas_region"],
            "destination_keys": ["forecast_period", "arkansas_region"],
            "scope": "matching Arkansas DHS region",
        },
    },
    "arkansas_region_annual_demand_state": {
        "cadence": "annual",
        "geography": "five Arkansas DHS/TEFRA regions",
        "supplier": "labeler/drug identity, not fulfillment supplier",
        "direct": False,
        "observation": "forecast",
        "status": "legacy_three_state_proxy",
        "semantics": "next-year low, mid, or high CMS Part D demand-claims state by Arkansas region and drug",
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "drug_key"],
            "destination_keys": ["forecast_period", "arkansas_region", "drug_key"],
            "scope": "matching Arkansas DHS/TEFRA region and drug",
        },
    },
    "arkansas_region_annual_demand_five_state": {
        "cadence": "annual",
        "geography": "five Arkansas DHS/TEFRA regions",
        "supplier": "labeler/drug identity, not fulfillment supplier",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-year five-quantile CMS Part D demand-claims state by Arkansas region and drug",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "drug_key"],
            "destination_keys": ["forecast_period", "arkansas_region", "drug_key"],
            "scope": "matching Arkansas DHS/TEFRA region and drug",
        },
    },
    "arkansas_county_annual_demand_five_state": {
        "cadence": "annual",
        "geography": "Arkansas county FIPS",
        "supplier": "labeler/drug identity, not fulfillment supplier",
        "direct": False,
        "observation": "forecast",
        "status": "qualified_five_state_proxy",
        "semantics": "next-year five-quantile CMS Part D demand-claims state by Arkansas county and drug",
        "event": {"kind": "state", "event_states": [3, 4]},
        "context_projection": {
            "source_keys": ["forecast_period", "geography_id", "county_fips", "drug_key"],
            "destination_keys": ["forecast_period", "county_fips", "drug_key"],
            "scope": "matching Arkansas county and drug",
        },
    },
    "ndc_monthly_recall_pressure_state": {
        "cadence": "monthly",
        "geography": "national FDA NDC recall evidence",
        "supplier": "recalling firm when available; no pharmacy allocation",
        "direct": False,
        "observation": "forecast",
        "status": "legacy_three_state_proxy",
        "semantics": "next-month no active recall, Class II/III active recall, or Class I active recall state",
        "context_projection": {
            "source_keys": ["forecast_period", "drug_key"],
            "destination_keys": ["forecast_period", "drug_key"],
            "scope": "national NDC recall context; no county allocation",
        },
    },
    "supplier_ndc_monthly_recall_pressure_state": {
        "cadence": "monthly",
        "geography": "national FDA recalling-firm x NDC evidence",
        "supplier": "FDA recalling firm",
        "direct": False,
        "observation": "forecast",
        "status": "legacy_three_state_proxy",
        "semantics": "next-month supplier-by-NDC no active recall, Class II/III active recall, or Class I active recall state",
        "context_projection": {
            "source_keys": ["forecast_period", "supplier", "drug_key"],
            "destination_keys": ["forecast_period", "supplier", "drug_key"],
            "scope": "matching supplier and NDC; no pharmacy allocation",
        },
    },
    "arkansas_weekly_hospital_influenza_admission_pressure_state": {
        "cadence": "weekly",
        "geography": "Arkansas statewide hospital respiratory surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast",
        "status": "legacy_three_state_proxy",
        "semantics": "next-week low, mid, or high Arkansas influenza hospital-admission pressure state; not pharmacy dispensing truth",
        "context_projection": {
            "source_keys": ["forecast_period"],
            "destination_keys": ["forecast_period"],
            "scope": "Arkansas statewide context",
        },
    },
    "ndc_monthly_shortage_supplier_count": {
        "cadence": "monthly",
        "geography": "national FDA evidence joined to Arkansas-exposed NDCs",
        "supplier": "count of FDA-reporting suppliers",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_numeric_proxy",
        "semantics": "next-month count of FDA suppliers reporting active shortage; not local inventory",
        "event": {"kind": "numeric", "event_threshold": 1.0,
                  "relative_tolerance": 0.05},
    },
    "nadac_next_observed_price": {
        "cadence": "weekly",
        "geography": "national NDC acquisition-cost context for Arkansas-exposed drugs",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_numeric_proxy",
        "semantics": "next weekly CMS NADAC acquisition cost per unit; not local inventory or demand",
        "event": {"kind": "not_applicable"},
        "context_projection": {
            "source_keys": ["forecast_period", "drug_key"],
            "destination_keys": ["forecast_period", "drug_key"],
            "scope": "national acquisition-cost context for matching NDC",
        },
    },
    "arkansas_weekly_fluview_wili": {
        "cadence": "weekly",
        "geography": "Arkansas statewide respiratory surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_numeric_proxy",
        "semantics": "next-week Arkansas FluView WILI value; not pharmacy dispensing",
        "event": {"kind": "not_applicable"},
    },
    "arkansas_weekly_hospital_influenza_admissions": {
        "cadence": "weekly",
        "geography": "Arkansas statewide hospital respiratory surveillance",
        "supplier": "none",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_numeric_proxy",
        "semantics": "next-week Arkansas influenza hospital admissions; not pharmacy dispensing",
        "event": {"kind": "not_applicable"},
    },
    "arkansas_region_annual_demand_claims": {
        "cadence": "annual",
        "geography": "five Arkansas DHS/TEFRA regions x drug",
        "supplier": "labeler/drug identity, not fulfillment supplier",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_numeric_proxy",
        "semantics": "next-year CMS Part D demand claims by Arkansas region and drug",
        "event": {"kind": "not_applicable"},
    },
    "ndc_monthly_recall_severity": {
        "cadence": "monthly",
        "geography": "national FDA NDC recall evidence",
        "supplier": "recalling firm when available",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_numeric_proxy",
        "semantics": "next-month numeric FDA recall severity, where 0 is none and 2 is Class I",
        "event": {"kind": "numeric", "event_threshold": 1.0,
                  "relative_tolerance": 0.05},
    },
    "supplier_ndc_monthly_recall_severity": {
        "cadence": "monthly",
        "geography": "national FDA recalling-firm x NDC evidence",
        "supplier": "FDA recalling firm",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "qualified_numeric_proxy",
        "semantics": "next-month numeric supplier-by-NDC FDA recall severity",
        "event": {"kind": "numeric", "event_threshold": 1.0,
                  "relative_tolerance": 0.05},
    },
    "arcos_zip3_drug_next_quarter_distribution_grams": {
        "cadence": "quarterly",
        "geography": "Arkansas ZIP3 x controlled-substance code",
        "supplier": "none in ARCOS Report 01",
        "direct": False,
        "observation": "forecast_proxy",
        "status": "candidate_numeric_proxy",
        "semantics": "next-quarter DEA ARCOS distribution grams; not pharmacy inventory",
    },
    "neighbor_state_supply_event_risk": {
        "cadence": "event_driven",
        "geography": "neighboring state event context",
        "supplier": "not applicable",
        "direct": False,
        "observation": "observed_external_event_context",
        "status": "context_only",
        "semantics": "observed neighboring-state event severity; not Arkansas demand",
    },
}


def _apply_target_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach conservative target semantics to every emitted row."""
    result = frame.copy()
    if "therapeutic_class" not in result:
        result["therapeutic_class"] = ""
    if "forecast_period" not in result:
        result["forecast_period"] = result.get("source_freshness", "")
    metadata = result["target"].map(TARGET_METADATA)
    defaults = {
        "cadence": "unknown", "geography": "unknown",
        "supplier": "unknown", "direct": False, "observation": "unknown",
        "status": "unqualified", "semantics": "unclassified target",
        "event": {"kind": "not_applicable"},
    }
    for key, column in (
        ("cadence", "target_cadence"),
        ("geography", "target_geography_scope"),
        ("supplier", "target_supplier_resolution"),
        ("direct", "target_is_direct_pharmacy_observation"),
        ("observation", "target_observation_type"),
        ("status", "target_promotion_status"),
        ("semantics", "target_semantics"),
    ):
        result[column] = metadata.map(
            lambda value: value.get(key, defaults[key])
            if isinstance(value, dict) else defaults[key]
        )
    result["target_event_definition"] = metadata.map(
        lambda value: json.dumps(value.get("event", defaults["event"]), sort_keys=True)
        if isinstance(value, dict) else json.dumps(defaults["event"], sort_keys=True)
    )
    # Legacy grid rows do not have validated uncertainty estimates.
    result["uncertainty_status"] = result.get(
        "uncertainty_status", pd.Series(index=result.index, dtype="object")
    ).fillna("legacy_unvalidated")
    result["calibration_status"] = result.get(
        "calibration_status", pd.Series(index=result.index, dtype="object")
    ).fillna("not_calibrated")
    return result[UNIVERSAL_OUTPUT_COLUMNS]


def _s(row, key, default=""):
    value = row.get(key, default)
    return default if pd.isna(value) else str(value)


def _county_view(panel: pd.DataFrame, city_to_county: dict[str, dict] | None) -> pd.DataFrame:
    p = panel.copy()
    for col in ("county_fips", "county_name", "arkansas_region"):
        if col not in p:
            p[col] = ""
    if city_to_county:
        mapped = p["city"].astype(str).str.lower().map(city_to_county)
        for col in ("county_fips", "county_name", "arkansas_region"):
            vals = mapped.map(lambda x: x.get(col, "") if isinstance(x, dict) else "")
            existing = p[col].fillna("").astype(str).replace({"nan": "", "None": ""})
            p[col] = existing.where(existing.str.len() > 0, vals)
    p["county_fips"] = p["county_fips"].fillna("").astype(str)
    p["arkansas_region"] = p.apply(
        lambda row: (str(row["arkansas_region"]).strip()
                     if not pd.isna(row["arkansas_region"])
                     and str(row["arkansas_region"]).strip().casefold() not in {"", "nan", "none"}
                     else region_for_county(row["county_fips"])), axis=1)
    return p


def _norm_drug(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _norm_supplier(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _risk_prior(risk_spec: dict) -> float:
    """Return an artifact-supplied prior, never a shortage-count heuristic.

    The neutral default keeps contract tests and exports schema-complete, but
    is explicitly not a forecast until a calibrated risk artifact is supplied.
    """
    value = pd.to_numeric(risk_spec.get("baseline_probability", 0.5),
                          errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        return 0.5
    return float(np.clip(value, 0.0, 1.0))


def _supplier_shortage_context_rows(scores: pd.DataFrame, run_timestamp: str,
                                    model_version: str, max_rows: int,
                                    supplier_filter=None) -> pd.DataFrame:
    """Expose supplier-drug event scores without inventing county allocation."""
    if scores is None or scores.empty or max_rows <= 0:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    wanted_suppliers = {_norm_supplier(value) for value in (supplier_filter or [])}
    rows = []
    for _, score in scores.iterrows():
        probability = float(np.clip(pd.to_numeric(
            score.get("shortage_probability", 0.0), errors="coerce"), 0.0, 1.0))
        supplier = str(score.get("supplier", ""))
        if wanted_suppliers and _norm_supplier(supplier) not in wanted_suppliers:
            continue
        drug = str(score.get("drug", ""))
        month = str(score.get("feature_month", ""))
        rows.append({
            "forecast_timestamp": run_timestamp, "horizon": 30,
            "geography_level": "supplier_drug", "geography_id": supplier,
            "county_fips": "", "county_name": "", "arkansas_region": "",
            "drug_key": drug, "ingredient": drug, "dosage_form": "", "strength": "", "route": "",
            "labeler": supplier, "parent_company": "", "factory": "", "api_source": "FDA shortage reports",
            "target": "supplier_shortage_probability", "prediction": probability,
            "interval_low": max(0.0, probability - 0.15),
            "interval_high": min(1.0, probability + 0.15),
            "risk_score": probability, "confidence": 0.45,
            "source_freshness": month, "event_ids": "[]", "evidence_article_ids": "[]",
            "driver_attribution": json.dumps({
                "source": "S_D/data/by_source/fda_shortages",
                "feature_month": month, "supplier_key": supplier, "drug_key": drug,
                "label_semantics": "observed FDA event; zero is not confirmed no shortage",
            }),
            "evidence_type": "fda_reported_supplier_drug_event_model",
            "model_version": model_version, "feature_window_start": month,
            "feature_window_end": month,
        })
    return pd.DataFrame(rows, columns=UNIVERSAL_OUTPUT_COLUMNS)


def _arkansas_supplier_relevance_rows(scores: pd.DataFrame, panel: pd.DataFrame,
                                      run_timestamp: str, model_version: str,
                                      max_rows: int, supplier_filter=None) -> pd.DataFrame:
    """Project supplier event scores to exact Arkansas drug matches only.

    This is a statewide relevance context row, not evidence that the named
    supplier serves Arkansas pharmacies. No county or regional allocation is
    inferred from the FDA supplier-drug event.
    """
    if scores is None or scores.empty or panel is None or panel.empty or max_rows <= 0:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    wanted_suppliers = {_norm_supplier(value) for value in (supplier_filter or [])}
    mapping = {}
    for _, base in panel.iterrows():
        for field in ("drug_key", "ingredient", "drug"):
            key = _norm_drug(base.get(field, ""))
            if key and key not in mapping:
                mapping[key] = base
    rows = []
    for _, score in scores.iterrows():
        base = mapping.get(_norm_drug(score.get("drug", "")))
        if base is None:
            continue
        probability = float(np.clip(pd.to_numeric(
            score.get("shortage_probability", 0.0), errors="coerce"), 0.0, 1.0))
        supplier = str(score.get("supplier", ""))
        if wanted_suppliers and _norm_supplier(supplier) not in wanted_suppliers:
            continue
        source_drug = str(score.get("drug", ""))
        month = str(score.get("feature_month", ""))
        rows.append({
            "forecast_timestamp": run_timestamp, "horizon": 30,
            "geography_level": "arkansas_supplier_drug", "geography_id": "AR",
            "county_fips": "", "county_name": "", "arkansas_region": "statewide",
            "drug_key": _s(base, "drug_key"), "ingredient": _s(base, "ingredient", source_drug),
            "dosage_form": _s(base, "dosage_form"), "strength": _s(base, "strength"),
            "route": _s(base, "route"), "labeler": supplier,
            "parent_company": "", "factory": "", "api_source": "FDA shortage reports",
            "target": "arkansas_supplier_drug_shortage_pressure", "prediction": probability,
            "interval_low": max(0.0, probability - 0.2),
            "interval_high": min(1.0, probability + 0.2),
            "risk_score": probability, "confidence": 0.25,
            "source_freshness": month, "event_ids": "[]", "evidence_article_ids": "[]",
            "driver_attribution": json.dumps({
                "source": "S_D/data/by_source/fda_shortages",
                "feature_month": month, "supplier_key": supplier,
                "source_drug_key": source_drug,
                "mapping": "exact_normalized_drug_match",
                "geographic_semantics": "statewide_drug_relevance_only",
                "supplier_to_arkansas_allocation_verified": False,
                "label_semantics": "observed FDA event; zero is not confirmed no shortage",
            }),
            "evidence_type": "fda_supplier_event_exact_arkansas_drug_context",
            "model_version": model_version, "feature_window_start": month,
            "feature_window_end": month,
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=UNIVERSAL_OUTPUT_COLUMNS)


def _arcos_context_rows(arcos_panel: pd.DataFrame, panel: pd.DataFrame,
                        run_timestamp: str, max_rows: int) -> pd.DataFrame:
    """Build explicit ZIP3 distribution context rows from defensible mappings."""
    if arcos_panel is None or arcos_panel.empty or max_rows <= 0:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    mapping = {}
    for _, row in panel.iterrows():
        key = str(row.get("drug_key", ""))
        if not key:
            continue
        for field in ("drug", "ingredient"):
            name = _norm_drug(row.get(field, ""))
            if name and name not in mapping:
                mapping[name] = row
    if not mapping:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    latest = (arcos_panel.sort_values("period_index")
              .groupby(["zip3", "drug_name"], as_index=False).tail(1))
    rows = []
    for _, signal in latest.iterrows():
        base = mapping.get(_norm_drug(signal.get("drug_name", "")))
        if base is None:
            continue
        grams = float(signal.get("grams", 0.0))
        rows.append({
            "forecast_timestamp": run_timestamp, "horizon": 0,
            "geography_level": "zip3", "geography_id": str(signal["zip3"]),
            "county_fips": "", "county_name": "", "arkansas_region": "",
            "drug_key": _s(base, "drug_key"), "ingredient": _s(base, "ingredient"),
            "dosage_form": "", "strength": "", "route": "",
            "labeler": "", "parent_company": "", "factory": "",
            "api_source": "DEA ARCOS Report 01",
            "target": "regional_distribution_pressure", "prediction": grams,
            "interval_low": max(0.0, grams * 0.75), "interval_high": grams * 1.25,
            "risk_score": 0.0, "confidence": 0.5,
            "source_freshness": str(signal["period"]), "event_ids": "[]",
            "evidence_article_ids": "[]",
            "driver_attribution": json.dumps({
                "proxy": "reported_controlled_substance_distribution_grams",
                "source": SOURCE_RELATIVE, "source_period": str(signal["period"]),
                "drug_code": str(signal["drug_code"]),
                "inventory_observed": False, "direct_demand_observed": False,
            }), "evidence_type": "observed_distribution_proxy",
            "model_version": "arcos-context-v1",
            "feature_window_start": str(signal["period"]),
            "feature_window_end": str(signal["period"]),
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=UNIVERSAL_OUTPUT_COLUMNS)


def _arcos_forecast_rows(predictions: pd.DataFrame, panel: pd.DataFrame,
                         run_timestamp: str, max_rows: int) -> pd.DataFrame:
    """Expose validated next-quarter ARCOS distribution predictions."""
    if predictions is None or predictions.empty or panel is None or panel.empty or max_rows <= 0:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    mapping = {}
    for _, row in panel.iterrows():
        for field in ("drug", "ingredient"):
            key = _norm_drug(row.get(field, ""))
            if key and key not in mapping:
                mapping[key] = row
    rows = []
    for _, signal in predictions.iterrows():
        base = mapping.get(_norm_drug(signal.get("drug_name", "")))
        if base is None:
            continue
        prediction = max(float(pd.to_numeric(signal.get("prediction", 0.0), errors="coerce")), 0.0)
        period = str(signal.get("forecast_period", ""))
        validation_wape = float(pd.to_numeric(signal.get("validation_wape", 1.0), errors="coerce"))
        rows.append({
            "forecast_timestamp": run_timestamp, "horizon": 91,
            "geography_level": "zip3", "geography_id": str(signal["zip3"]),
            "county_fips": "", "county_name": "", "arkansas_region": "",
            "drug_key": _s(base, "drug_key"), "ingredient": _s(base, "ingredient"),
            "dosage_form": "", "strength": "", "route": "", "labeler": "",
            "parent_company": "", "factory": "", "api_source": "DEA ARCOS Report 01",
            "target": "regional_distribution_pressure_forecast",
            "prediction": prediction,
            "interval_low": max(0.0, prediction * (1.0 - validation_wape)),
            "interval_high": prediction * (1.0 + validation_wape),
            "risk_score": 0.0, "confidence": 0.45, "source_freshness": period,
            "event_ids": "[]", "evidence_article_ids": "[]",
            "driver_attribution": json.dumps({
                "source": SOURCE_RELATIVE, "forecast_period": period,
                "selected_model": str(signal.get("selected_model", "")),
                "validation_wape": validation_wape,
                "proxy": "reported_controlled_substance_distribution_grams",
                "inventory_observed": False, "direct_demand_observed": False,
            }),
            "evidence_type": "validated_distribution_proxy_forecast",
            "model_version": "arcos-next-quarter-v1",
            "feature_window_start": str(signal.get("period", "")),
            "feature_window_end": str(signal.get("period", "")),
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=UNIVERSAL_OUTPUT_COLUMNS)


def build_universal_forecast_grid(panel: pd.DataFrame, cfg=None, *, trained=None,
                                  city_to_county=None, model_version="universal-v1",
                                  forecast_timestamp=None, max_rows=10000, events=None,
                                  supplier_lookup=None, county_fips=None,
                                  regions=None, suppliers=None, state_events=None,
                                  neighbor_states=None, arcos_panel=None,
                                  supplier_shortage_scores=None) -> pd.DataFrame:
    """Create forecast rows at county × drug × supplier × target resolution.

    Rows without a defensible county remain visible with ``county_fips`` empty
    and confidence reduced; they are never relabeled as Arkansas-wide data.
    """
    if panel.empty:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    p = _county_view(panel, city_to_county)
    if county_fips:
        wanted = {str(v).zfill(5) for v in county_fips}
        p = p[p["county_fips"].astype(str).isin(wanted)]
    if regions:
        p = p[p["arkansas_region"].astype(str).isin({str(v) for v in regions})]
    supplier_cols = [c for c in ("labeler", "supplier_key") if c in p]
    supplier = supplier_cols[0] if supplier_cols else None
    keys = [c for c in ("county_fips", "drug_key") if c in p]
    if supplier:
        keys.append(supplier)
    latest = p.sort_values("year").groupby(keys, dropna=False, as_index=False).tail(1).copy()
    # Some legacy panel builds contain repeated feature labels (for example
    # duplicated disease columns).  Keep the first observed column so feature
    # alignment remains deterministic and pandas cannot broadcast ambiguity.
    latest = latest.loc[:, ~latest.columns.duplicated()]
    latest = latest.sort_values("demand_claims", ascending=False).head(max(1, max_rows // (len(HORIZONS_DAYS) * len(TARGETS))))
    if suppliers:
        wanted = {str(v).casefold() for v in suppliers}
        latest = latest[latest[supplier].astype(str).str.casefold().isin(wanted)] if supplier else latest.iloc[0:0]
    if latest.empty:
        return pd.DataFrame(columns=UNIVERSAL_OUTPUT_COLUMNS)
    run_ts = pd.Timestamp(forecast_timestamp or datetime.now(timezone.utc))
    run_ts = run_ts.tz_localize("UTC") if run_ts.tzinfo is None else run_ts.tz_convert("UTC")
    # Context mapping uses the full deduplicated product universe, not the
    # demand-ranked output subset, so valid ARCOS matches cannot be dropped by
    # the forecast row cap before provenance rows are materialized.
    context_panel = p.drop_duplicates(subset=["drug_key"]) if "drug_key" in p else p
    arcos_rows = _arcos_context_rows(arcos_panel, context_panel,
                                     run_ts.isoformat(), max_rows)
    arcos_forecasts = pd.DataFrame()
    if (arcos_panel is not None and not arcos_panel.empty
            and {"log_grams", "period_index", "zip3", "drug_code"} <= set(arcos_panel.columns)):
        arcos_view = build_next_quarter_view(arcos_panel)
        arcos_forecasts = _arcos_forecast_rows(
            forecast_latest_arcos(arcos_view, arcos_panel), context_panel,
            run_ts.isoformat(), max_rows)
    supplier_shortage_rows = _supplier_shortage_context_rows(
        supplier_shortage_scores, run_ts.isoformat(), model_version, max_rows,
        supplier_filter=suppliers)
    arkansas_supplier_rows = _arkansas_supplier_relevance_rows(
        supplier_shortage_scores, context_panel, run_ts.isoformat(), model_version, max_rows,
        supplier_filter=suppliers)
    demand_spec = (trained or {}).get("demand_claims", {})
    demand_model = _restore_demand_model(demand_spec["model"]) if demand_spec.get("model") else None
    risk_spec = (trained or {}).get("shortage_risk", {})
    risk_model = _restore_risk_model(risk_spec.get("model")) if risk_spec.get("model") else None
    risk_prior = _risk_prior(risk_spec)
    risk_source = ("trained_calibrated_risk_model" if risk_model is not None
                   else "unmodeled_neutral_prior")
    if demand_model is not None:
        x = _feature_matrix(latest, demand_spec.get("feature_cols", []),
                            (trained or {}).get("encoder_spec", []))
        ridge_demand = np.expm1(np.clip(demand_model.predict(x), -20.0, 20.0))
        blend = (trained or {}).get("calibrated_blend", {})
        weight = float(blend.get("blend_weight", 1.0))
        latest["_predicted_demand"] = ((1.0 - weight) * pd.to_numeric(
            latest["demand_claims"], errors="coerce").fillna(0).to_numpy() + weight * ridge_demand)
        if risk_model is not None:
            latest["_predicted_risk"] = risk_model.predict_proba(x)
    end_year = int(pd.to_numeric(p["year"], errors="coerce").max())
    rows = []
    total = max(float(pd.to_numeric(latest.get("demand_claims", 0), errors="coerce").sum()), 1.0)
    event_frame = events if events is not None else pd.DataFrame()
    supplier_frame = supplier_lookup if supplier_lookup is not None else pd.DataFrame()
    state_frame = state_events if state_events is not None else pd.DataFrame()
    wanted_states = {str(x).upper() for x in (neighbor_states or
                    ("MO", "TN", "MS", "LA", "TX", "OK"))}
    state_reserve = 0
    if not state_frame.empty and "geography" in state_frame.columns:
        state_codes = state_frame["geography"].fillna("").astype(str).str.split(":").str[0].str.upper()
        state_reserve = int(state_codes.isin(wanted_states).sum() > 0) * len(HORIZONS_DAYS) * int(
            state_codes[state_codes.isin(wanted_states)].nunique())
    row_limit = max(0, max_rows - state_reserve - len(arcos_rows) - len(arcos_forecasts)
                    - len(supplier_shortage_rows) - len(arkansas_supplier_rows))
    region_values = latest["arkansas_region"].fillna("").astype(str).str.strip()
    region_values = sorted(x for x in region_values.unique() if x)
    # Reserve a complete, filterable region summary before materializing the
    # county grid. Small unit-test/export caps intentionally skip summaries
    # rather than allowing them to displace all county rows.
    region_reserve = len(region_values) * len(HORIZONS_DAYS) * len(REGION_TARGETS)
    emit_regions = bool(region_values and max_rows >= region_reserve + state_reserve)
    if emit_regions:
        row_limit = max(0, row_limit - region_reserve)
    for _, base in latest.iterrows():
        demand = float(pd.to_numeric(base.get("_predicted_demand", base.get("demand_claims", 0)), errors="coerce") or 0)
        risk = float(base.get("_predicted_risk", risk_prior))
        county_known = bool(_s(base, "county_fips"))
        evidence = "direct" if county_known else "unresolved"
        supplier_rows = supplier_frame
        if not supplier_frame.empty and "labeler" in supplier_frame:
            supplier_rows = supplier_frame[supplier_frame["labeler"].fillna("").astype(str).eq(_s(base, "labeler"))]
            # A labeler can have many NDCs and facilities. Prefer the exact
            # drug/NDC relationship, then fall back to labeler-only coverage.
            for col, value in (("drug_key", _s(base, "drug_key")), ("ndc", _s(base, "ndc"))):
                if value and col in supplier_rows.columns:
                    exact = supplier_rows[supplier_rows[col].fillna("").astype(str).eq(value)]
                    if not exact.empty:
                        supplier_rows = exact
                        break
        supplier_row = supplier_rows.iloc[0] if not supplier_rows.empty else {}
        matching_events = event_frame
        if not event_frame.empty:
            county = _s(base, "county_fips")
            city = _s(base, "city").lower()
            matching_events = event_frame[
                event_frame.get("county_fips", pd.Series(index=event_frame.index)).fillna("").astype(str).eq(county)
                | event_frame.get("location", pd.Series(index=event_frame.index)).fillna("").astype(str).str.lower().eq(city)
            ]
            event_ids = matching_events.get("event_id", pd.Series(dtype=str)).astype(str).tolist()
            article_ids = matching_events.get("article_id", pd.Series(dtype=str)).astype(str).tolist()
            event_pressure = float(len(matching_events))
        else:
            event_ids, article_ids, event_pressure = [], [], 0.0
        for horizon in HORIZONS_DAYS:
            scale = horizon / 365.0
            values = {"demand_claims": demand * scale,
                      "demand_cost": float(pd.to_numeric(base.get("demand_cost", 0), errors="coerce") or 0) * scale,
                      "demand_shock_index": float(pd.to_numeric(
                          base.get("demand_shock_index", 0.0), errors="coerce") or 0.0),
                      "supply_disruption_risk": risk,
                      "arkansas_shortage_impact": risk * demand / total,
                      }
            for target, prediction in values.items():
                if len(rows) >= row_limit:
                    break
                width = max(abs(float(prediction)) * 0.25, 1e-6)
                rows.append({
                    "forecast_timestamp": run_ts.isoformat(), "horizon": horizon,
                    "geography_level": "county" if county_known else "arkansas_unresolved",
                    "geography_id": _s(base, "county_fips") or _s(base, "city"),
                    "county_fips": _s(base, "county_fips"), "county_name": _s(base, "county_name"),
                    "arkansas_region": _s(base, "arkansas_region"), "drug_key": _s(base, "drug_key"),
                    "ingredient": _s(base, "ingredient"), "dosage_form": _s(base, "dosage_form"),
                    "strength": _s(base, "strength"), "route": _s(base, "route"),
                    "labeler": _s(base, supplier or "labeler"),
                    "parent_company": _s(supplier_row, "parent_company"),
                    "factory": _s(supplier_row, "factory"), "api_source": _s(supplier_row, "api_source"),
                    "target": target, "prediction": float(prediction), "interval_low": max(0.0, float(prediction) - width),
                    "interval_high": float(prediction) + width, "risk_score": risk,
                    "confidence": 0.9 if county_known else 0.35, "source_freshness": str(end_year),
                    "event_ids": json.dumps(event_ids), "evidence_article_ids": json.dumps(article_ids),
                    "driver_attribution": json.dumps({"demand_history": demand, "shortage_exposure": risk,
                                                       "article_event_count": event_pressure,
                                                       "risk_source": risk_source,
                                                       "event_use": "evidence_only"}),
                    "evidence_type": evidence, "model_version": model_version,
                    "feature_window_start": f"{end_year}-01-01", "feature_window_end": f"{end_year}-12-31",
                })
            if len(rows) >= row_limit:
                break
        if len(rows) >= row_limit:
            break

    if emit_regions:
        region_base = latest.copy()
        region_base["_region"] = region_base["arkansas_region"].fillna("").astype(str).str.strip()
        for region, group in region_base.groupby("_region", sort=True):
            if not region:
                continue
            def _region_numeric(column: str) -> pd.Series:
                if column not in group.columns:
                    return pd.Series(0.0, index=group.index)
                return pd.to_numeric(group[column], errors="coerce").fillna(0.0)

            demand = float(pd.to_numeric(group.get("_predicted_demand", group["demand_claims"]),
                                         errors="coerce").fillna(0).sum())
            cost = float(_region_numeric("demand_cost").sum())
            if risk_model is not None and "_predicted_risk" in group:
                risk = float(pd.to_numeric(group["_predicted_risk"], errors="coerce")
                             .dropna().mean())
            else:
                risk = risk_prior
            for horizon in HORIZONS_DAYS:
                scale = horizon / 365.0
                values = {
                    "demand_claims": demand * scale,
                    "demand_cost": cost * scale,
                    "demand_shock_index": 0.0,
                    "supply_disruption_risk": risk,
                    "arkansas_shortage_impact": risk * demand / total,
                }
                for target in REGION_TARGETS:
                    if target not in values:
                        continue
                    prediction = float(values[target])
                    width = max(abs(prediction) * 0.25, 1e-6)
                    rows.append({
                        "forecast_timestamp": run_ts.isoformat(), "horizon": horizon,
                        "geography_level": "region", "geography_id": region,
                        "county_fips": "", "county_name": "", "arkansas_region": region,
                        "drug_key": "", "ingredient": "", "dosage_form": "",
                        "strength": "", "route": "", "labeler": "",
                        "parent_company": "", "factory": "", "api_source": "",
                        "target": target, "prediction": prediction,
                        "interval_low": max(0.0, prediction - width),
                        "interval_high": prediction + width, "risk_score": risk,
                        "confidence": 0.8, "source_freshness": str(end_year),
                        "event_ids": "[]", "evidence_article_ids": "[]",
                        "driver_attribution": json.dumps({
                            "aggregated_county_rows": int(len(group)),
                            "shortage_exposure": risk,
                            "risk_source": risk_source,
                        }), "evidence_type": "aggregated_county_context",
                        "model_version": model_version,
                        "feature_window_start": f"{end_year}-01-01",
                        "feature_window_end": f"{end_year}-12-31",
                    })

    # Neighbor states are emitted only as observed external-event context.
    # No Arkansas demand is copied into another state and no state demand
    # label is fabricated.
    if len(rows) < max_rows and not state_frame.empty:
        required = {"event_id", "event_type", "geography", "start_time", "severity", "confidence"}
        if required <= set(state_frame.columns):
            state = state_frame.copy()
            state["state_code"] = state["geography"].fillna("").astype(str).str.split(":").str[0].str.upper()
            state = state[state["state_code"].isin(wanted_states)].copy()
            if not state.empty:
                state["event_type"] = state["event_type"].astype(str)
                state["start_time"] = pd.to_datetime(state["start_time"], errors="coerce", utc=True)
                state["severity"] = pd.to_numeric(state["severity"], errors="coerce").fillna(0.0)
                state["confidence"] = pd.to_numeric(state["confidence"], errors="coerce").fillna(0.0)
                for state_code, group in state.groupby("state_code", sort=True):
                    observed_severity = float(np.clip(
                        (group["severity"] * group["confidence"]).mean(), 0.0, 1.0))
                    shortage_n = int(group["event_type"].eq("drug_shortage").sum())
                    recall_n = int(group["event_type"].str.contains("recall", case=False, na=False).sum())
                    event_ids = group["event_id"].astype(str).tolist()
                    freshness = group["start_time"].max()
                    freshness_value = "" if pd.isna(freshness) else freshness.strftime("%Y-%m-%d")
                    for horizon in HORIZONS_DAYS:
                        if len(rows) >= max_rows:
                            break
                        rows.append({
                            "forecast_timestamp": run_ts.isoformat(), "horizon": horizon,
                            "geography_level": "neighbor_state", "geography_id": state_code,
                            "county_fips": "", "county_name": "", "arkansas_region": "",
                            "drug_key": "", "ingredient": "", "dosage_form": "", "strength": "", "route": "",
                            "labeler": "", "parent_company": "", "factory": "", "api_source": "",
                            "target": "neighbor_state_supply_event_risk", "prediction": observed_severity,
                            "interval_low": max(0.0, observed_severity - 0.15),
                            "interval_high": min(1.0, observed_severity + 0.15),
                            "risk_score": observed_severity,
                            "confidence": float(group["confidence"].mean()),
                            "source_freshness": freshness_value, "event_ids": json.dumps(event_ids),
                            "evidence_article_ids": "[]",
                            "driver_attribution": json.dumps({"state_event_count": len(group),
                                                               "shortage_event_count": shortage_n,
                                                               "recall_event_count": recall_n,
                                                               "value_semantics": "observed_severity_context"}),
                            "evidence_type": "direct_external_state_event", "model_version": model_version,
                            "feature_window_start": freshness_value, "feature_window_end": freshness_value,
                        })
    if len(arcos_rows):
        rows.extend(arcos_rows.to_dict("records"))
    if len(arcos_forecasts):
        rows.extend(arcos_forecasts.to_dict("records"))
    if len(supplier_shortage_rows):
        rows.extend(supplier_shortage_rows.to_dict("records"))
    if len(arkansas_supplier_rows):
        rows.extend(arkansas_supplier_rows.to_dict("records"))
    return _apply_target_metadata(
        pd.DataFrame(rows[:max_rows], columns=UNIVERSAL_OUTPUT_COLUMNS)
    )


def validate_universal_forecast(frame: pd.DataFrame) -> None:
    missing = set(UNIVERSAL_OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"universal forecast missing columns: {sorted(missing)}")
    if frame.duplicated(["forecast_timestamp", "horizon", "geography_level",
                         "geography_id", "county_fips", "drug_key", "labeler",
                         "target"]).any():
        raise ValueError("duplicate universal forecast keys")
    for col in ("prediction", "interval_low", "interval_high", "risk_score", "confidence"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"forecast {col} must be finite")
    if (pd.to_numeric(frame["interval_low"], errors="coerce")
            > pd.to_numeric(frame["interval_high"], errors="coerce")).any():
        raise ValueError("forecast intervals must be ordered")
    if not pd.to_numeric(frame["confidence"], errors="coerce").between(0, 1).all():
        raise ValueError("forecast confidence must be in [0, 1]")
    if frame["target_promotion_status"].eq("unqualified").any():
        raise ValueError("forecast contains an unqualified target")
    if not frame["target_is_direct_pharmacy_observation"].isin([True, False]).all():
        raise ValueError("target direct-observation metadata must be boolean")
    if not frame["uncertainty_status"].astype(str).isin(
            {"legacy_unvalidated", "not_estimated", "estimated"}).all():
        raise ValueError("unknown forecast uncertainty status")
    if not frame["calibration_status"].astype(str).isin(
            {"not_calibrated", "calibrated", "legacy_unvalidated"}).all():
        raise ValueError("unknown forecast calibration status")
