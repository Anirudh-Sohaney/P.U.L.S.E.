"""Qualification audit for external metrics exposed to pharmacy models.

This module deliberately separates *evaluation evidence* from *promotion*. A
large row count is not enough: a metric must have an observable target, a
leakage-safe chronological test, a permitted output representation, and
enough drug/geography coverage for the claim being made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MIN_HELD_OUT_ROWS = 25
MIN_ROLLING_FOLDS = 3
MIN_METRIC_ACCURACY = 0.65
PART_ONE_ACCURACY = 0.75
PART_ONE_DRUG_COUNT = 100
MIN_STATE_COUNT = 5
MIN_RAW_EVENT_ACCURACY = 0.65
MIN_TRUE_POSITIVE_PRECISION = 0.80

# Event definitions are deliberately explicit. They describe a high-risk
# announcement zone, not a claim that the proxy is pharmacy inventory truth.
_EVENT_DEFINITIONS = {
    "ndc_monthly_shortage_supplier_count": {"kind": "numeric", "threshold": 1},
    "ndc_monthly_shortage_pressure_state": {"kind": "state", "states": [3, 4]},
    "nadac_next_observed_price": {"kind": "not_applicable"},
    "arkansas_region_annual_demand_five_state": {"kind": "state", "states": [3, 4]},
    "arkansas_county_annual_demand_five_state": {"kind": "state", "states": [3, 4]},
    "arkansas_state_annual_partd_demand_five_state": {"kind": "state", "states": [3, 4]},
    "ndc_monthly_recall_severity": {"kind": "numeric", "threshold": 1},
    "supplier_ndc_monthly_recall_severity": {"kind": "numeric", "threshold": 1},
    "arcos_zip3_drug_next_quarter_distribution_state": {"kind": "state", "states": [3, 4]},
    "national_weekly_fluview_respiratory_pressure_state": {"kind": "state", "states": [3, 4]},
    "national_weekly_respnet_rsv_hospitalization_pressure_state": {"kind": "state", "states": [3, 4]},
    "arkansas_weekly_nssp_influenza_ed_pressure_state": {"kind": "state", "states": [3, 4]},
    "arkansas_monthly_apcd_pharmacy_claim_activity_state": {"kind": "state", "states": [3, 4]},
    "arkansas_monthly_atc_therapeutic_demand_state": {"kind": "state", "states": [3, 4]},
}

_SOURCE_METADATA = {
    "arcos_zip3_drug_next_quarter_distribution": {
        "source_url": "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html",
        "source_license_access": "public DEA report; source terms and retrieval date recorded in DATA_SOURCES.md",
        "target_semantics": "next-quarter controlled-substance distribution grams, not pharmacy dispensing",
        "feature_timestamp_boundary": "features through quarter t predict consecutive quarter t+1",
        "geography_scope": "Arkansas ZIP3",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive observed quarters retained; no imputation",
    },
    "arcos_zip3_drug_next_quarter_distribution_state": {
        "source_url": "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html",
        "source_license_access": "public DEA report; source terms and retrieval date recorded in DATA_SOURCES.md",
        "target_semantics": "next-quarter controlled-substance distribution five-quantile state, not pharmacy dispensing",
        "feature_timestamp_boundary": "features through quarter t predict consecutive quarter t+1",
        "geography_scope": "Arkansas ZIP3 x controlled-substance code",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive observed quarters retained; thresholds learned from fit data",
    },
    "arkansas_weekly_fluview_respiratory_pressure": {
        "source_url": "https://www.cdc.gov/fluview/",
        "source_license_access": "public CDC/Delphi surveillance feed; retrieval metadata recorded locally",
        "target_semantics": "next-week Arkansas WILI respiratory pressure, numeric proxy",
        "feature_timestamp_boundary": "current week and prior history predict the next complete week",
        "geography_scope": "Arkansas statewide",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive weekly observations retained",
    },
    "arkansas_weekly_fluview_respiratory_pressure_state": {
        "source_url": "https://www.cdc.gov/fluview/",
        "source_license_access": "public CDC/Delphi surveillance feed; retrieval metadata recorded locally",
        "target_semantics": "next-week Arkansas WILI low/mid/high state proxy",
        "feature_timestamp_boundary": "current week and prior history predict the next complete week",
        "geography_scope": "Arkansas statewide",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive weekly observations retained",
    },
    "national_weekly_fluview_respiratory_pressure_state": {
        "source_url": "https://www.cdc.gov/fluview/",
        "source_license_access": "public CDC/Delphi surveillance feed; retrieval metadata recorded locally",
        "target_semantics": "next-week five-quantile national WILI respiratory-pressure state",
        "feature_timestamp_boundary": "current week and prior history predict the next complete week",
        "geography_scope": "United States national aggregate",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive weekly observations retained; thresholds fit on history",
    },
    "national_weekly_respnet_rsv_hospitalization_pressure_state": {
        "source_url": "https://data.cdc.gov/Public-Health-Surveillance/RESP-NET-Rates-and-Clinical-Data/kvib-3txy",
        "source_license_access": "public CDC RESP-NET Socrata API; retrieval manifest stored locally",
        "target_semantics": "next-week five-quantile national RSV hospitalization pressure, not pharmacy dispensing",
        "feature_timestamp_boundary": "observed week t predicts the consecutive week t+1",
        "geography_scope": "United States RESP-NET surveillance aggregate",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive weekly observations retained; thresholds fit on prior history",
    },
    "nadac_next_observed_price": {
        "source_url": "https://www.medicaid.gov/medicaid/prescription-drugs/pharmacy-pricing",
        "source_license_access": "public CMS NADAC files; historical reconstruction manifest recorded locally",
        "target_semantics": "next observed national NDC acquisition-cost pressure, not inventory",
        "feature_timestamp_boundary": "NADAC observations at t predict the next exactly seven-day observation",
        "geography_scope": "national NDC, Arkansas-exposed subset",
        "supplier_resolution": "none",
        "missingness_censoring": "exact seven-day transitions only; stale gaps excluded",
    },
    "fda_shortage_continuation": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages/",
        "source_license_access": "public FDA/openFDA records; archived observations and retrieval dates recorded locally",
        "target_semantics": "next-month supplier shortage continuation; binary diagnostic rejected",
        "feature_timestamp_boundary": "dated shortage observations through month t predict month t+1",
        "geography_scope": "national NDC9 x supplier, Arkansas-exposed subset",
        "supplier_resolution": "FDA-reported supplier",
        "missingness_censoring": "right-censored continuation rows excluded; no missing record treated as availability",
    },
    "ndc_monthly_shortage_pressure_state": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages/",
        "source_license_access": "public FDA/openFDA records; archived observations and retrieval dates recorded locally",
        "target_semantics": "next-month five-state NDC pressure: none, one, two, three, or four-plus active shortage suppliers",
        "feature_timestamp_boundary": "supplier observations through month t predict consecutive month t+1",
        "geography_scope": "national FDA evidence restricted to Arkansas-exposed NDCs",
        "supplier_resolution": "supplier count, not supplier allocation",
        "missingness_censoring": "only consecutive observed NDC-month transitions retained",
    },
    "ndc_monthly_shortage_supplier_count": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages/",
        "source_license_access": "public FDA/openFDA records; archived observations and retrieval dates recorded locally",
        "target_semantics": "next-month numeric count of FDA suppliers reporting active shortage",
        "feature_timestamp_boundary": "supplier observations through month t predict consecutive month t+1",
        "geography_scope": "national NDC evidence restricted to Arkansas-exposed NDCs",
        "supplier_resolution": "count of FDA-reporting suppliers, not supplier allocation",
        "missingness_censoring": "only consecutive observed NDC-month transitions retained",
    },
    "arkansas_region_annual_demand_state": {
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers",
        "source_license_access": "public CMS Part D provider/drug files; annual source and mapping provenance recorded locally",
        "target_semantics": "next-year regional low/mid/high Medicare Part D demand-claims state",
        "feature_timestamp_boundary": "completed year t features predict consecutive year t+1",
        "geography_scope": "five Arkansas DHS/TEFRA regions x drug",
        "supplier_resolution": "labeler/drug identity, not fulfillment supplier",
        "missingness_censoring": "only consecutive annual region-drug observations retained",
    },
    "arkansas_region_annual_demand_five_state": {
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers",
        "source_license_access": "public CMS Part D provider/drug files; annual source and mapping provenance recorded locally",
        "target_semantics": "next-year five-quantile regional Medicare Part D demand-claims state by drug",
        "feature_timestamp_boundary": "completed year t features predict consecutive year t+1",
        "geography_scope": "five Arkansas DHS/TEFRA regions x drug",
        "supplier_resolution": "labeler/drug identity, not fulfillment supplier",
        "missingness_censoring": "only consecutive annual region-drug observations retained; thresholds fit on training history",
    },
    "arkansas_county_annual_demand_five_state": {
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers",
        "source_license_access": "public CMS Part D provider/drug files; annual source and county provenance recorded locally",
        "target_semantics": "next-year five-quantile Arkansas county Medicare Part D demand-claims state by drug",
        "feature_timestamp_boundary": "completed year t features predict consecutive year t+1",
        "geography_scope": "Arkansas county FIPS x drug",
        "supplier_resolution": "labeler/drug identity, not fulfillment supplier",
        "missingness_censoring": "only consecutive annual county-drug observations retained; thresholds fit on training history",
    },
    "arkansas_region_annual_demand_claims": {
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers",
        "source_license_access": "public CMS Part D provider/drug files; annual source and mapping provenance recorded locally",
        "target_semantics": "next-year numeric regional Medicare Part D demand claims by drug",
        "feature_timestamp_boundary": "completed year t features predict consecutive year t+1",
        "geography_scope": "five Arkansas DHS/TEFRA regions x drug",
        "supplier_resolution": "labeler/drug identity, not fulfillment supplier",
        "missingness_censoring": "only consecutive annual region-drug observations retained",
    },
    "arkansas_weekly_wastewater_influenza_state": {
        "source_url": "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re",
        "source_license_access": "public CDC wastewater activity-level data; local retrieval manifest recorded",
        "target_semantics": "next-week five-quantile Arkansas wastewater Influenza A activity state",
        "feature_timestamp_boundary": "complete wastewater week t predicts consecutive week t+1",
        "geography_scope": "Arkansas statewide site mean",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive statewide weekly aggregates retained",
    },
    "arkansas_weekly_wastewater_rsv_state": {
        "source_url": "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re",
        "source_license_access": "public CDC wastewater activity-level data; local retrieval manifest recorded",
        "target_semantics": "next-week five-quantile Arkansas wastewater RSV activity state",
        "feature_timestamp_boundary": "complete wastewater week t predicts consecutive week t+1",
        "geography_scope": "Arkansas statewide site mean",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive statewide weekly aggregates retained",
    },
    "arkansas_weekly_wastewater_sars_cov_2_state": {
        "source_url": "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re",
        "source_license_access": "public CDC wastewater activity-level data; local retrieval manifest recorded",
        "target_semantics": "next-week five-quantile Arkansas wastewater SARS-CoV-2 activity state",
        "feature_timestamp_boundary": "complete wastewater week t predicts consecutive week t+1",
        "geography_scope": "Arkansas statewide site mean",
        "supplier_resolution": "none",
        "missingness_censoring": "only consecutive statewide aggregates retained",
    },
    "ndc_monthly_recall_pressure_state": {
        "source_url": "https://www.fda.gov/about-fda/open-government-fda-data-sets/recalls-data-sets",
        "source_license_access": "public FDA enforcement/recall records; source archive and normalization documented locally",
        "target_semantics": "next-month NDC recall state: none, Class II/III, or Class I",
        "feature_timestamp_boundary": "recall observations through month t predict consecutive month t+1",
        "geography_scope": "national NDC evidence",
        "supplier_resolution": "recalling firm when available",
        "missingness_censoring": "source-edge and consecutive-month rules documented; no recall is not confirmed stock",
    },
    "supplier_ndc_monthly_recall_pressure_state": {
        "source_url": "https://www.fda.gov/about-fda/open-government-fda-data-sets/recalls-data-sets",
        "source_license_access": "public FDA enforcement/recall records; source archive and normalization documented locally",
        "target_semantics": "next-month supplier-by-NDC recall state: none, Class II/III, or Class I",
        "feature_timestamp_boundary": "recall observations through month t predict consecutive month t+1",
        "geography_scope": "national supplier x NDC evidence, Arkansas-exposed suppliers where mapped",
        "supplier_resolution": "FDA recalling firm",
        "missingness_censoring": "source-edge and consecutive-month rules documented; no recall is not confirmed stock",
    },
    "ndc_monthly_recall_severity": {
        "source_url": "https://www.fda.gov/about-fda/open-government-fda-data-sets/recalls-data-sets",
        "source_license_access": "public FDA enforcement/recall records; source archive and normalization documented locally",
        "target_semantics": "next-month numeric NDC recall severity on the 0-2 FDA class scale",
        "feature_timestamp_boundary": "recall observations through month t predict consecutive month t+1",
        "geography_scope": "national NDC recall evidence",
        "supplier_resolution": "recalling firm when available",
        "missingness_censoring": "source-edge and consecutive-month rules documented; no recall is not confirmed stock",
    },
    "supplier_ndc_monthly_recall_severity": {
        "source_url": "https://www.fda.gov/about-fda/open-government-fda-data-sets/recalls-data-sets",
        "source_license_access": "public FDA enforcement/recall records; source archive and normalization documented locally",
        "target_semantics": "next-month numeric supplier-by-NDC FDA recall severity on the 0-2 class scale",
        "feature_timestamp_boundary": "recall observations through month t predict consecutive month t+1",
        "geography_scope": "national recalling-firm x NDC recall evidence",
        "supplier_resolution": "FDA recalling firm",
        "missingness_censoring": "source-edge and consecutive-month rules documented; no recall is not confirmed stock",
    },
    "global_monthly_supply_chain_pressure_state": {
        "source_url": "https://www.newyorkfed.org/research/policy/gscpi",
        "source_license_access": "public New York Fed GSCPI data; retrieval and vintage recorded in the local source manifest",
        "target_semantics": "next-month global supply-chain pressure state: below 0, 0 through 1, or above 1 standardized GSCPI units",
        "feature_timestamp_boundary": "published GSCPI through month t predicts the next complete month",
        "geography_scope": "global upstream context; no Arkansas allocation",
        "supplier_resolution": "none",
        "missingness_censoring": "only complete consecutive monthly observations retained",
    },
    "arkansas_quarterly_medicaid_prescription_demand": {
        "source_url": "https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data",
        "source_license_access": "public CMS/Medicaid SDUD files; source years and suppression handling recorded locally",
        "target_semantics": "next-quarter Arkansas Medicaid fee-for-service prescription volume, not all-payer pharmacy demand",
        "feature_timestamp_boundary": "completed quarter t predicts consecutive quarter t+1",
        "geography_scope": "Arkansas statewide",
        "supplier_resolution": "none",
        "missingness_censoring": "complete quarterly Arkansas FFS series only; suppressed drug rows are already aggregated by the source pipeline",
    },
    "national_monthly_shortage_breadth_state": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages/",
        "source_license_access": "public FDA shortage observations reconstructed from dated archive records; retrieval and censoring documented locally",
        "target_semantics": "next-month national low/mid/high state of the number of distinct NDCs with an active FDA shortage",
        "feature_timestamp_boundary": "shortage observations through month t predict the consecutive month t+1",
        "geography_scope": "United States national shortage breadth; no Arkansas allocation",
        "supplier_resolution": "none in aggregate; supplier observations deduplicated by NDC",
        "missingness_censoring": "complete monthly archive interval; state thresholds fit within each training fold",
    },
    "arkansas_county_monthly_overdose_pressure_state": {
        "source_url": "https://data.cdc.gov/National-Center-for-Health-Statistics/VSRR-Provisional-County-Level-Drug-Overdose-Death-/gb4e-yj24",
        "source_license_access": "public CDC Socrata API; retrieval and suppression metadata recorded locally",
        "target_semantics": "next-month Arkansas county low/mid/high state of provisional rolling-12-month overdose deaths; not pharmacy dispensing",
        "feature_timestamp_boundary": "published county observation through month t predicts consecutive observed month t+1",
        "geography_scope": "Arkansas county",
        "supplier_resolution": "none",
        "missingness_censoring": "suppressed/null county-months excluded; gaps are not imputed as zero",
    },
    "arkansas_state_annual_partd_demand_five_state": {
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-geography-and-drug",
        "source_license_access": "public CMS data; normalized Arkansas slice and source manifest stored locally",
        "target_semantics": "next-year five-state Medicare Part D total claims by generic drug; external demand proxy, not inventory",
        "feature_timestamp_boundary": "annual claims through year t predict consecutive year t+1",
        "geography_scope": "Arkansas statewide, state FIPS 05",
        "supplier_resolution": "none available in this CMS product",
        "missingness_censoring": "CMS-suppressed cells omitted by source; no imputation",
    },
    "arkansas_monthly_apcd_pharmacy_claim_activity_state": {
        "source_url": "https://achiapcd.atlassian.net/wiki/spaces/ADRS/pages/2778136577/Arkansas%2BAPCD%2BClaim%2BCounts%2Bby%2BMonth",
        "source_license_access": "public Arkansas APCD monthly claim-count workbook; normalized copy and retrieval manifest stored locally",
        "target_semantics": "next-month five-quantile pharmacy-claim activity state by APCD reporting entity; not NDC demand, supplier allocation, or inventory",
        "feature_timestamp_boundary": "prescription-fill claim activity through month t predicts consecutive month t+1",
        "geography_scope": "Arkansas APCD reporting entity, not county or individual pharmacy",
        "supplier_resolution": "none available",
        "missingness_censoring": "only consecutive submitter-month transitions retained; thresholds fit on prior training periods; no gap imputation",
    },
    "arkansas_monthly_atc_therapeutic_demand_state": {
        "source_url": "https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/",
        "source_license_access": "public HHS Medicaid/CHIP NDC claims plus public NLM RxNorm/RxClass crosswalk; local mapping manifest stored locally",
        "target_semantics": "next-month five-state HHS pharmacy claim-line demand fractionally allocated across three-character ATC therapeutic groups for the mapped NDC subset; not all-payer demand or inventory",
        "feature_timestamp_boundary": "observed claims through month t predict consecutive month t+1; class mapping is fixed before evaluation",
        "geography_scope": "Arkansas statewide HHS pharmacy-taxonomy billing providers",
        "supplier_resolution": "none available",
        "missingness_censoring": "suppressed/nonconsecutive HHS NDC cells and NDCs without an RxNorm ATC mapping are excluded; no zero imputation",
    },
}


def _source_metadata(name: str) -> dict[str, Any]:
    """Return required provenance fields for one metric registry record."""
    if name in _SOURCE_METADATA:
        return dict(_SOURCE_METADATA[name])
    if name.startswith("arkansas_weekly_hospital_"):
        return {
            "source_url": "https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy",
            "source_license_access": "public CDC NHSN HRD data; retrieval metadata recorded locally",
            "target_semantics": "next-week Arkansas hospital respiratory admission pressure state",
            "feature_timestamp_boundary": "completed hospital week t predicts the next complete week",
            "geography_scope": "Arkansas statewide by pathogen",
            "supplier_resolution": "none",
            "missingness_censoring": "only consecutive observed weeks retained; pathogen-specific gaps are not imputed",
        }
    if name.startswith("arkansas_weekly_nssp_"):
        return {
            "source_url": "https://data.cdc.gov/Public-Health-Surveillance/NSSP-Emergency-Department-Visit-Trajectories-by-St/rdmq-nq56",
            "source_license_access": "public CDC NSSP trajectory data; retrieval metadata recorded locally",
            "target_semantics": "next-week Arkansas statewide pathogen-specific ED-visit percentage five-state proxy",
            "feature_timestamp_boundary": "complete NSSP week t predicts consecutive complete week t+1",
            "geography_scope": "Arkansas statewide",
            "supplier_resolution": "none",
            "missingness_censoring": "statewide rows only; unavailable pathogen percentages are excluded",
        }
    if name.startswith("arkansas_region_weekly_wastewater_"):
        return {
            "source_url": "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re",
            "source_license_access": "public CDC wastewater WVAL data; county-served mapping and retrieval metadata recorded locally",
            "target_semantics": "next-week Arkansas-region five-state wastewater activity proxy; not pharmacy dispensing",
            "feature_timestamp_boundary": "complete wastewater week t predicts consecutive complete week t+1",
            "geography_scope": "Arkansas DHS region inferred from a single county served by a wastewater site",
            "supplier_resolution": "none",
            "missingness_censoring": "only named single-county service areas and consecutive regional weeks retained; no gaps imputed",
        }
    return {
        "source_url": "",
        "source_license_access": "source documented in DATA_SOURCES.md",
        "target_semantics": "candidate metric; see target-specific evaluation artifact",
        "feature_timestamp_boundary": "features precede the target period",
        "geography_scope": "target-specific; see evaluation artifact",
        "supplier_resolution": "target-specific; see evaluation artifact",
        "missingness_censoring": "target-specific; see evaluation artifact",
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _result(
    *,
    name: str,
    cadence: str,
    target_type: str,
    rows: int,
    folds: int,
    accuracy: float | None,
    balanced_accuracy: float | None = None,
    drug_count: int,
    region_count: int,
    supplier_count: int,
    evidence: str,
    status: str,
    reasons: list[str],
    baseline_skill: dict[str, Any] | None = None,
    drug_count_at_or_above_75_percent: int | None = None,
    region_count_at_or_above_75_percent: int | None = None,
    event_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one stable, reviewable metric registry record."""
    return {
        "metric": name,
        "cadence": cadence,
        "target_type": target_type,
        "held_out_rows": rows,
        "rolling_folds": folds,
        "accuracy_within_5pct_or_exact_state": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "drug_count": drug_count,
        "drug_count_at_or_above_75_percent": drug_count_at_or_above_75_percent,
        "arkansas_region_count": region_count,
        "region_count_at_or_above_75_percent": region_count_at_or_above_75_percent,
        "supplier_count": supplier_count,
        "evidence": evidence,
        "status": status,
        "event_definition": _EVENT_DEFINITIONS.get(
            name, (event_metrics or {}).get("event_definition", {"kind": "not_applicable"})),
        "event_metrics": event_metrics,
        "reasons": reasons,
        "baseline_skill": baseline_skill or {
            "status": "not_recorded",
            "mean_balanced_accuracy_delta_vs_persistence": None,
            "all_folds_beating_persistence": None,
            "incremental_utility_status": "not_tested",
        },
        # Public targets are proxies; this prevents downstream relabeling as
        # direct pharmacy inventory evidence.
        "target_is_direct_pharmacy_observation": False,
        "target_observation_type": "external_proxy_target",
        "target_claim_scope": "proxy_only_not_pharmacy_inventory",
        **_source_metadata(name),
    }


def _qualification_reasons(
    *, rows: int, folds: int, accuracy: float | None,
    balanced_accuracy: float | None = None,
    target_type: str, drug_count: int, region_count: int,
    requires_100_drugs: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if rows < MIN_HELD_OUT_ROWS:
        reasons.append(f"held_out_rows<{MIN_HELD_OUT_ROWS}")
    if folds < MIN_ROLLING_FOLDS:
        reasons.append(f"rolling_folds<{MIN_ROLLING_FOLDS}")
    if accuracy is None or accuracy < MIN_METRIC_ACCURACY:
        reasons.append(f"accuracy<{MIN_METRIC_ACCURACY:.0%}")
    if balanced_accuracy is not None and balanced_accuracy < MIN_METRIC_ACCURACY:
        reasons.append(f"balanced_accuracy<{MIN_METRIC_ACCURACY:.0%}")
    if target_type == "binary_state":
        reasons.append(
            f"binary-only target is prohibited; use at least {MIN_STATE_COUNT} states"
        )
    if requires_100_drugs and drug_count < PART_ONE_DRUG_COUNT:
        reasons.append(f"drug_count<{PART_ONE_DRUG_COUNT}")
    return reasons


def _mean_fold_balanced_delta(payload: dict[str, Any]) -> float | None:
    """Read model-minus-persistence balanced accuracy from fold artifacts."""
    deltas = []
    for fold in payload.get("folds", []):
        model = fold.get("model", {}).get("balanced_accuracy")
        persistence = fold.get("persistence", {}).get("balanced_accuracy")
        if model is not None and persistence is not None:
            deltas.append(float(model) - float(persistence))
    return float(sum(deltas) / len(deltas)) if deltas else None


def audit_metric_library(evaluation_dir: Path) -> dict[str, Any]:
    """Audit current candidates without silently changing their model scores.

    The ARCOS and FluView artifacts expose numeric within-5-percent rates. The
    NADAC artifact uses the historical reconstruction with three chronological
    rolling folds. FDA shortage
    continuation is retained as a rejected diagnostic because it is binary
    and highly imbalanced; its raw accuracy must never be counted as a useful
    state metric.
    """
    arcos = _read(evaluation_dir / "arcos_regional_rolling_metrics.json")
    arcos_fixed = _read(evaluation_dir / "arcos_regional_metrics.json")
    arcos_rows = sum(int(fold.get("test_rows", 0)) for fold in arcos.get("folds", []))
    arcos_rows = max(arcos_rows, int(arcos_fixed.get("n_rows", {}).get("test", 0)))
    # ARCOS rolling output predates the within-5-percent field in some local
    # artifacts. Its WAPE is reported, but that is not substituted for the
    # contractual per-row accuracy measure.
    arcos_accuracy = arcos.get("means", {}).get("selected", {}).get("within_5pct")
    arcos_reasons = _qualification_reasons(
        rows=arcos_rows, folds=int(arcos.get("fold_count", 0)),
        accuracy=arcos_accuracy, target_type="numeric", drug_count=39,
        region_count=85,
    )

    arcos_state = _read(evaluation_dir / "arcos_distribution_state_metrics.json")
    arcos_state_reasons = _qualification_reasons(
        rows=int(arcos_state.get("test_rows", 0)),
        folds=int(arcos_state.get("fold_count", 0)),
        accuracy=arcos_state.get("mean_model_accuracy"),
        balanced_accuracy=arcos_state.get("mean_model_balanced_accuracy"),
        target_type="five_state",
        drug_count=int(arcos_state.get("drug_count", 0)),
        region_count=int(arcos_state.get("zip3_count", 0)),
    )
    if len(arcos_state.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        arcos_state_reasons.append(
            f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows"
        )

    flu = _read(evaluation_dir / "weekly_fluview_ar_proxy_metrics.json")
    flu_rows = sum(int(fold.get("test_rows", 0)) for fold in flu.get("folds", []))
    flu_accuracy = flu.get("mean_model_under_5_percent_error")
    flu_reasons = _qualification_reasons(
        rows=flu_rows, folds=int(flu.get("fold_count", 0)),
        accuracy=flu_accuracy, target_type="numeric", drug_count=0,
        region_count=1,
    )
    flu_state = flu.get("five_state", flu.get("three_state", {}))
    flu_state_reasons = _qualification_reasons(
        rows=int(flu_state.get("test_rows", 0)),
        folds=int(flu_state.get("fold_count", 0)),
        accuracy=flu_state.get("mean_model_accuracy"),
        balanced_accuracy=flu_state.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=0, region_count=1,
    )
    if len(flu_state.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        flu_state_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    national_flu_state = flu.get("national_five_state", {})
    national_flu_reasons = _qualification_reasons(
        rows=int(national_flu_state.get("test_rows", 0)),
        folds=int(national_flu_state.get("fold_count", 0)),
        accuracy=national_flu_state.get("mean_model_accuracy"),
        balanced_accuracy=national_flu_state.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=0, region_count=1,
    )
    if len(national_flu_state.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        national_flu_reasons.append(
            f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    historical_nadac_path = evaluation_dir / (
        "nadac_historical_reconstruction_rolling_metrics.json")
    nadac_path = (historical_nadac_path if historical_nadac_path.exists()
                  else evaluation_dir / "nadac_arkansas_exposed_rolling_metrics.json")
    nadac = _read(nadac_path)
    nadac_rolling = nadac.get("rolling", {})
    nadac_folds = nadac_rolling.get("folds") or []
    nadac_rows = sum(int(fold.get("test_rows", 0)) for fold in nadac_folds)
    if not nadac_rows:
        nadac_rows = int(nadac.get("test_rows", 0))
    fold_accuracy = [fold.get("model", {}).get("under_5_percent_error")
                     for fold in nadac_folds]
    fold_accuracy = [float(value) for value in fold_accuracy if value is not None]
    nadac_accuracy = (float(sum(fold_accuracy) / len(fold_accuracy))
                      if fold_accuracy else None)
    nadac_drug_count = max(
        [int(fold.get("test_ndcs", 0)) for fold in nadac_folds] or
        [int(nadac.get("test_ndcs", 0))])
    nadac_reasons = _qualification_reasons(
        rows=nadac_rows, folds=int(nadac_rolling.get("fold_count", 0)),
        accuracy=nadac_accuracy, target_type="numeric", drug_count=nadac_drug_count,
        region_count=1,
    )
    # Persistence is a valid forecast method when it is selected without test
    # leakage. Record its dominance as a limitation instead of rejecting an
    # otherwise accurate, inventory-relevant acquisition-cost proxy.
    nadac_beats_persistence = nadac.get("model_beats_persistence_wape")
    if nadac_beats_persistence is None and nadac_folds:
        nadac_beats_persistence = all(
            fold.get("model_beats_persistence_wape") is True
            for fold in nadac_folds)
    nadac_model_wape = [fold.get("model", {}).get("wape")
                        for fold in nadac_folds]
    nadac_persistence_wape = [fold.get("persistence", {}).get("wape")
                              for fold in nadac_folds]
    nadac_model_wape = [float(value) for value in nadac_model_wape if value is not None]
    nadac_persistence_wape = [float(value) for value in nadac_persistence_wape
                              if value is not None]

    shortage = _read(evaluation_dir / "arkansas_exposure_shortage_metrics.json")
    shortage_rolling = shortage.get("rolling", {})
    shortage_reasons = _qualification_reasons(
        rows=int(shortage.get("test_rows", 0)),
        folds=int(shortage_rolling.get("fold_count", 0)),
        accuracy=None, target_type="binary_state", drug_count=0,
        region_count=1,
    )

    pressure = _read(evaluation_dir / "shortage_pressure_rolling_metrics.json")
    pressure_reasons = _qualification_reasons(
        rows=int(pressure.get("test_rows", 0)),
        folds=int(pressure.get("fold_count", 0)),
        accuracy=pressure.get("mean_model_accuracy"),
        balanced_accuracy=pressure.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=int(
            pressure.get("per_drug_count_with_minimum_rows", 0)),
        region_count=0,
        requires_100_drugs=True,
    )
    if len(pressure.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        pressure_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    numeric_shortage = _read(
        evaluation_dir / "shortage_supplier_count_numeric_rolling_metrics.json")
    numeric_shortage_reasons = _qualification_reasons(
        rows=int(numeric_shortage.get("test_rows", 0)),
        folds=int(numeric_shortage.get("fold_count", 0)),
        accuracy=numeric_shortage.get("mean_model_within_5_percent_error"),
        target_type="numeric", drug_count=int(numeric_shortage.get("ndc_count", 0)),
        region_count=0,
        requires_100_drugs=True,
    )

    regional = _read(evaluation_dir / "regional_demand_state_metrics.json")
    regional_baseline_delta = _mean_fold_balanced_delta(regional)
    regional_reasons = _qualification_reasons(
        rows=int(regional.get("test_rows", 0)),
        folds=int(regional.get("fold_count", 0)),
        accuracy=regional.get("mean_model_accuracy"),
        balanced_accuracy=regional.get("mean_model_balanced_accuracy"),
        target_type="three_state",
        drug_count=int(regional.get("drug_count", 0)),
        region_count=int(regional.get("region_count", 0)),
        requires_100_drugs=False,
    )
    if len(regional.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        regional_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    regional_numeric = _read(
        evaluation_dir / "regional_demand_numeric_rolling_metrics.json")
    regional_numeric_reasons = _qualification_reasons(
        rows=int(regional_numeric.get("test_rows", 0)),
        folds=int(regional_numeric.get("fold_count", 0)),
        accuracy=regional_numeric.get("mean_model_within_5_percent_error"),
        target_type="numeric", drug_count=int(regional_numeric.get("drug_count", 0)),
        region_count=int(regional_numeric.get("region_count", 0)),
    )

    regional_five = _read(
        evaluation_dir / "regional_demand_five_state_metrics.json")
    regional_five_reasons = _qualification_reasons(
        rows=int(regional_five.get("test_rows", 0)),
        folds=int(regional_five.get("fold_count", 0)),
        accuracy=regional_five.get("mean_model_accuracy"),
        balanced_accuracy=regional_five.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=int(regional_five.get("drug_count", 0)),
        region_count=int(regional_five.get("region_count", 0)),
    )
    if len(regional_five.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        regional_five_reasons.append(
            f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    state_partd = _read(
        evaluation_dir / "arkansas_state_partd_demand_five_state_metrics.json")
    state_partd_reasons = _qualification_reasons(
        rows=int(state_partd.get("test_rows", 0)),
        folds=int(state_partd.get("fold_count", 0)),
        accuracy=state_partd.get("mean_model_accuracy"),
        balanced_accuracy=state_partd.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=int(state_partd.get("drug_count", 0)),
        region_count=int(state_partd.get("region_count", 0)),
    )
    if len(state_partd.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        state_partd_reasons.append(
            f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    state_partd_event_precision = (state_partd.get("event_metrics") or {}).get(
        "true_positive_precision")
    state_partd_event_route = (
        (state_partd.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
        and (state_partd_event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION)
    if state_partd_event_route:
        state_partd_reasons = [reason for reason in state_partd_reasons
                               if not reason.startswith("balanced_accuracy<")]
    elif ((state_partd.get("mean_model_accuracy") or 0.0) < PART_ONE_ACCURACY
          and (state_partd_event_precision or 0.0) < MIN_TRUE_POSITIVE_PRECISION):
        state_partd_reasons.append("raw accuracy<75% and event true-positive precision<80%")

    county_five = _read(
        evaluation_dir / "county_demand_five_state_metrics.json")
    county_five_reasons = _qualification_reasons(
        rows=int(county_five.get("test_rows", 0)),
        folds=int(county_five.get("fold_count", 0)),
        accuracy=county_five.get("mean_model_accuracy"),
        balanced_accuracy=county_five.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=int(county_five.get("drug_count", 0)),
        region_count=int(county_five.get("county_count", 0)),
    )
    if len(county_five.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        county_five_reasons.append(
            f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    county_event_precision = (county_five.get("event_metrics") or {}).get(
        "true_positive_precision")
    county_event_route = (
        (county_five.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
        and (county_event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION)
    if county_event_route:
        county_five_reasons = [reason for reason in county_five_reasons
                               if not reason.startswith("balanced_accuracy<")]
    elif (county_five.get("mean_model_accuracy") or 0.0) < PART_ONE_ACCURACY and (
            county_event_precision is None or
            county_event_precision < MIN_TRUE_POSITIVE_PRECISION):
        county_five_reasons.append(
            "raw accuracy<75% and event true-positive precision<80%")

    recall = _read(evaluation_dir / "recall_pressure_rolling_metrics.json")
    recall_baseline_delta = _mean_fold_balanced_delta(recall)
    recall_reasons = _qualification_reasons(
        rows=int(recall.get("test_rows", 0)), folds=int(recall.get("fold_count", 0)),
        accuracy=recall.get("mean_model_accuracy"),
        balanced_accuracy=recall.get("mean_model_balanced_accuracy"),
        target_type="three_state",
        drug_count=int(recall.get("ndc_count", 0)), region_count=0,
    )
    if len(recall.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        recall_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    supplier_recall = _read(evaluation_dir / "supplier_recall_pressure_rolling_metrics.json")
    supplier_recall_baseline_delta = _mean_fold_balanced_delta(supplier_recall)
    supplier_recall_reasons = _qualification_reasons(
        rows=int(supplier_recall.get("test_rows", 0)),
        folds=int(supplier_recall.get("fold_count", 0)),
        accuracy=supplier_recall.get("mean_model_accuracy"),
        balanced_accuracy=supplier_recall.get("mean_model_balanced_accuracy"),
        target_type="three_state", drug_count=int(supplier_recall.get("ndc_count", 0)),
        region_count=0,
    )
    if len(supplier_recall.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        supplier_recall_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    numeric_recall = _read(evaluation_dir / "recall_numeric_rolling_metrics.json")
    numeric_recall_reasons = _qualification_reasons(
        rows=int(numeric_recall.get("test_rows", 0)),
        folds=int(numeric_recall.get("fold_count", 0)),
        accuracy=numeric_recall.get("mean_model_within_5_percent_error"),
        target_type="numeric", drug_count=int(numeric_recall.get("ndc_count", 0)),
        region_count=0,
    )

    numeric_supplier_recall = _read(
        evaluation_dir / "supplier_recall_numeric_rolling_metrics.json")
    numeric_supplier_recall_reasons = _qualification_reasons(
        rows=int(numeric_supplier_recall.get("test_rows", 0)),
        folds=int(numeric_supplier_recall.get("fold_count", 0)),
        accuracy=numeric_supplier_recall.get("mean_model_within_5_percent_error"),
        target_type="numeric", drug_count=int(numeric_supplier_recall.get("ndc_count", 0)),
        region_count=0,
    )

    gscpi = _read(evaluation_dir / "gscpi_pressure_rolling_metrics.json")
    gscpi_reasons = _qualification_reasons(
        rows=int(gscpi.get("test_rows", 0)), folds=int(gscpi.get("fold_count", 0)),
        accuracy=gscpi.get("mean_model_accuracy"),
        balanced_accuracy=gscpi.get("mean_model_balanced_accuracy"),
        target_type="three_state", drug_count=0, region_count=0,
    )
    if len(gscpi.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        gscpi_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    medicaid_demand = _read(evaluation_dir / "arkansas_medicaid_demand_rolling_metrics.json")
    medicaid_demand_reasons = _qualification_reasons(
        rows=int(medicaid_demand.get("test_rows", 0)),
        folds=int(medicaid_demand.get("fold_count", 0)),
        accuracy=medicaid_demand.get("mean_model_within_5_percent"),
        target_type="numeric", drug_count=0, region_count=1,
    )
    if medicaid_demand.get("mean_learned_ridge_within_5_percent") is not None and \
            medicaid_demand["mean_learned_ridge_within_5_percent"] <= medicaid_demand.get(
                "mean_persistence_within_5_percent", -float("inf")):
        medicaid_demand_reasons.append("learned ridge does not beat the persistence baseline")

    shortage_breadth = _read(evaluation_dir / "shortage_breadth_rolling_metrics.json")
    shortage_breadth_reasons = _qualification_reasons(
        rows=int(shortage_breadth.get("test_rows", 0)),
        folds=int(shortage_breadth.get("fold_count", 0)),
        accuracy=shortage_breadth.get("mean_model_accuracy"),
        balanced_accuracy=shortage_breadth.get("mean_model_balanced_accuracy"),
        target_type="three_state", drug_count=0, region_count=0,
    )
    if len(shortage_breadth.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        shortage_breadth_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")

    hospital = _read(evaluation_dir / "hospital_respiratory_state_metrics.json")
    hospital_results = hospital.get("results", {})
    hospital_candidates = []
    for pathogen in ("covid", "influenza", "rsv"):
        result = hospital_results.get(pathogen, {})
        reasons = _qualification_reasons(
            rows=int(result.get("test_rows", 0)),
            folds=int(result.get("fold_count", 0)),
            accuracy=result.get("mean_model_accuracy"),
            balanced_accuracy=result.get("mean_model_balanced_accuracy"),
            target_type="three_state", drug_count=0, region_count=1,
        )
        if (len(result.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT
                or not all(result.get("state_counts_in_scored_rows", {}).values())):
            reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
        hospital_candidates.append((pathogen, result, reasons))

    nssp = _read(evaluation_dir / "nssp_respiratory_five_state_metrics.json")
    nssp_candidates = []
    for pathogen in ("covid", "influenza", "rsv"):
        result = nssp.get("results", {}).get(pathogen, {})
        event_precision = (result.get("event_metrics") or {}).get(
            "true_positive_precision")
        reasons = _qualification_reasons(
            rows=int(result.get("test_rows", 0)),
            folds=int(result.get("fold_count", 0)),
            accuracy=result.get("mean_model_accuracy"),
            balanced_accuracy=result.get("mean_model_balanced_accuracy"),
            target_type="five_state", drug_count=0, region_count=1,
        )
        # The revised event route permits low balanced accuracy when raw
        # accuracy and high-zone precision both clear their explicit gates.
        if ((result.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
                and (event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION):
            reasons = [reason for reason in reasons
                       if not reason.startswith("balanced_accuracy<")]
        else:
            reasons.append("event true-positive route not met")
        if len(result.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
            reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
        elif any(int(value) == 0 for value in result.get(
                "state_counts_in_scored_rows", {}).values()):
            reasons.append("fewer than 5 target states are observed in scored rows")
        nssp_candidates.append((pathogen, result, reasons))

    respnet = _read(evaluation_dir / "respnet_rsv_state_metrics.json")
    respnet_event_precision = (respnet.get("event_metrics") or {}).get(
        "true_positive_precision")
    respnet_reasons = _qualification_reasons(
        rows=int(respnet.get("test_rows", 0)),
        folds=int(respnet.get("fold_count", 0)),
        accuracy=respnet.get("mean_model_accuracy"),
        balanced_accuracy=respnet.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=0, region_count=1,
    )
    if ((respnet.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
            and (respnet_event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION):
        respnet_reasons = [reason for reason in respnet_reasons
                           if not reason.startswith("balanced_accuracy<")]
    else:
        respnet_reasons.append("event true-positive route not met")
    if len(respnet.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        respnet_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    elif any(int(value) == 0 for value in respnet.get(
            "state_counts_in_scored_rows", {}).values()):
        respnet_reasons.append("fewer than 5 target states are observed in scored rows")

    apcd = _read(evaluation_dir / "apcd_claim_activity_state_metrics.json")
    apcd_event_precision = (apcd.get("event_metrics") or {}).get(
        "true_positive_precision")
    apcd_reasons = _qualification_reasons(
        rows=int(apcd.get("test_rows", 0)),
        folds=int(apcd.get("fold_count", 0)),
        accuracy=apcd.get("mean_model_accuracy"),
        balanced_accuracy=apcd.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=0,
        region_count=int(apcd.get("submitter_count", 0)),
    )
    if ((apcd.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
            and (apcd_event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION):
        apcd_reasons = [reason for reason in apcd_reasons
                        if not reason.startswith("balanced_accuracy<")]
    else:
        apcd_reasons.append("event true-positive route not met")
    if len(apcd.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        apcd_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    elif any(int(value) == 0 for value in apcd.get(
            "state_counts_in_scored_rows", {}).values()):
        apcd_reasons.append("fewer than 5 target states are observed in scored rows")

    atc = _read(evaluation_dir / "atc_therapeutic_demand_state_metrics.json")
    atc_reasons = _qualification_reasons(
        rows=int(atc.get("test_rows", 0)),
        folds=int(atc.get("fold_count", 0)),
        accuracy=atc.get("mean_model_accuracy"),
        balanced_accuracy=atc.get("mean_model_balanced_accuracy"),
        target_type="five_state", drug_count=0, region_count=1,
    )
    if len(atc.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        atc_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    elif any(int(value) == 0 for value in atc.get(
            "state_counts_in_scored_rows", {}).values()):
        atc_reasons.append("fewer than 5 target states are observed in scored rows")

    hospital_numeric = _read(
        evaluation_dir / "hospital_influenza_numeric_rolling_metrics.json")
    hospital_numeric_reasons = _qualification_reasons(
        rows=int(hospital_numeric.get("test_rows", 0)),
        folds=int(hospital_numeric.get("fold_count", 0)),
        accuracy=hospital_numeric.get("mean_model_within_5_percent_error"),
        target_type="numeric", drug_count=0, region_count=1,
    )

    wastewater = _read(evaluation_dir / "wastewater_pressure_five_state_metrics.json")
    wastewater_candidates = []
    for pathogen in ("influenza", "rsv", "sars_cov_2"):
        result = wastewater.get("results", {}).get(pathogen, {})
        reasons = _qualification_reasons(
            rows=int(result.get("test_rows", 0)),
            folds=int(result.get("fold_count", 0)),
            accuracy=result.get("mean_model_accuracy"),
            balanced_accuracy=result.get("mean_model_balanced_accuracy"),
            target_type="five_state", drug_count=0, region_count=1,
        )
        if (len(result.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT
                or not all(result.get("state_counts_in_scored_rows", {}).values())):
            reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    wastewater_candidates.append((pathogen, result, reasons))

    regional_wastewater = _read(
        evaluation_dir / "regional_wastewater_five_state_metrics.json")
    regional_wastewater_candidates = []
    for pathogen, payload in regional_wastewater.get("results", {}).items():
        for representation in ("level", "change"):
            result = payload.get(representation, {})
            event_precision = (result.get("event_metrics") or {}).get(
                "true_positive_precision")
            reasons = _qualification_reasons(
                rows=int(result.get("test_rows", 0)),
                folds=int(result.get("fold_count", 0)),
                accuracy=result.get("mean_model_accuracy"),
                balanced_accuracy=result.get("mean_model_balanced_accuracy"),
                target_type="five_state", drug_count=0,
                region_count=int(result.get("region_count", 0)),
            )
            if ((result.get("mean_model_accuracy") or 0.0) >= MIN_RAW_EVENT_ACCURACY
                    and (event_precision or 0.0) >= MIN_TRUE_POSITIVE_PRECISION):
                reasons = [reason for reason in reasons
                           if not reason.startswith("balanced_accuracy<")]
            else:
                reasons.append("event true-positive route not met")
            if len(result.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
                reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
            elif any(int(value) == 0 for value in result.get(
                    "state_counts_in_scored_rows", {}).values()):
                reasons.append("fewer than 5 target states are observed in scored rows")
            regional_wastewater_candidates.append((
                {"Influenza A virus": "influenza", "SARS-CoV-2": "sars_cov_2",
                 "RSV": "rsv"}.get(pathogen, str(pathogen).lower()),
                representation, result, reasons))

    overdose = _read(evaluation_dir / "overdose_pressure_metrics.json")
    overdose_reasons = _qualification_reasons(
        rows=int(overdose.get("test_rows", 0)),
        folds=int(overdose.get("fold_count", 0)),
        accuracy=overdose.get("mean_model_accuracy"),
        balanced_accuracy=overdose.get("mean_model_balanced_accuracy"),
        target_type="three_state", drug_count=0, region_count=0,
    )
    if len(overdose.get("state_counts_in_scored_rows", {})) < MIN_STATE_COUNT:
        overdose_reasons.append(f"fewer than {MIN_STATE_COUNT} target states are observed in scored rows")
    if int(overdose.get("county_count_with_minimum_rows", 0)) < 1:
        overdose_reasons.append("no county has the minimum held-out row count")
    if overdose.get("vintage_status") != "historical_vintages_available":
        overdose_reasons.append("historical publication vintages are unavailable")

    candidates = [
        _result(name="arcos_zip3_drug_next_quarter_distribution", cadence="quarterly",
                target_type="numeric", rows=arcos_rows,
                folds=int(arcos.get("fold_count", 0)), accuracy=arcos_accuracy,
                drug_count=39, region_count=85, supplier_count=0,
                evidence="DEA ARCOS Report 01", status="qualified_proxy" if not arcos_reasons else "rejected",
                reasons=arcos_reasons,
                baseline_skill={
                    "status": "numeric_baseline_comparison_only",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                "incremental_utility_status": "not_tested",
                }),
        _result(name="arcos_zip3_drug_next_quarter_distribution_state", cadence="quarterly",
                target_type="five_state", rows=int(arcos_state.get("test_rows", 0)),
                folds=int(arcos_state.get("fold_count", 0)),
                accuracy=arcos_state.get("mean_model_accuracy"),
                balanced_accuracy=arcos_state.get("mean_model_balanced_accuracy"),
                drug_count=int(arcos_state.get("drug_count", 0)),
                drug_count_at_or_above_75_percent=int(
                    arcos_state.get("per_drug_count_at_or_above_75_percent", 0)),
                region_count=int(arcos_state.get("zip3_count", 0)),
                region_count_at_or_above_75_percent=sum(
                    float(value) >= PART_ONE_ACCURACY
                    for value in arcos_state.get("per_zip3_accuracy", {}).values()),
                supplier_count=0,
                evidence="DEA ARCOS Report 01, state-encoded distribution pressure",
                status="qualified_proxy" if not arcos_state_reasons else "rejected",
                reasons=arcos_state_reasons,
                event_metrics=arcos_state.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": (
                        arcos_state.get("mean_model_balanced_accuracy", 0.0)
                        - arcos_state.get("mean_persistence_balanced_accuracy", 0.0)),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "distribution state is a regional supply proxy; learned head selected persistence",
                }),
        _result(name="arkansas_weekly_fluview_respiratory_pressure", cadence="weekly",
                target_type="numeric", rows=flu_rows, folds=int(flu.get("fold_count", 0)),
                accuracy=flu_accuracy, drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC FluView", status="qualified_proxy" if not flu_reasons else "rejected",
                reasons=flu_reasons),
        _result(name="arkansas_weekly_fluview_respiratory_pressure_state", cadence="weekly",
                target_type="three_state", rows=int(flu_state.get("test_rows", 0)),
                folds=int(flu_state.get("fold_count", 0)),
                accuracy=flu_state.get("mean_model_accuracy"), drug_count=0,
                balanced_accuracy=flu_state.get("mean_model_balanced_accuracy"),
                region_count=1, supplier_count=0, evidence="CDC FluView",
                status="qualified_proxy" if not flu_state_reasons else "rejected",
                reasons=flu_state_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": flu_state.get(
                        "model_balanced_improvement_vs_persistence"),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "state head did not beat persistence; no pharmacy inventory target",
                }),
        _result(name="national_weekly_fluview_respiratory_pressure_state", cadence="weekly",
                target_type="five_state", rows=int(national_flu_state.get("test_rows", 0)),
                folds=int(national_flu_state.get("fold_count", 0)),
                accuracy=national_flu_state.get("mean_model_accuracy"),
                balanced_accuracy=national_flu_state.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC FluView national",
                status="qualified_proxy" if not national_flu_reasons else "rejected",
                reasons=national_flu_reasons,
                event_metrics=national_flu_state.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": (
                        national_flu_state.get("model_balanced_improvement_vs_persistence")),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "national respiratory context; no pharmacy inventory target",
                }),
        _result(name="national_weekly_respnet_rsv_hospitalization_pressure_state", cadence="weekly",
                target_type="five_state", rows=int(respnet.get("test_rows", 0)),
                folds=int(respnet.get("fold_count", 0)),
                accuracy=respnet.get("mean_model_accuracy"),
                balanced_accuracy=respnet.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC RESP-NET RSV",
                status="qualified_proxy" if not respnet_reasons else "rejected",
                reasons=respnet_reasons, event_metrics=respnet.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": 0.0,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "national RSV hospitalization proxy; no pharmacy inventory target",
                }),
        _result(name="arkansas_monthly_apcd_pharmacy_claim_activity_state", cadence="monthly",
                target_type="five_state", rows=int(apcd.get("test_rows", 0)),
                folds=int(apcd.get("fold_count", 0)),
                accuracy=apcd.get("mean_model_accuracy"),
                balanced_accuracy=apcd.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=int(apcd.get("submitter_count", 0)),
                supplier_count=0, evidence="Arkansas APCD monthly pharmacy claim counts",
                status="qualified_proxy" if not apcd_reasons else "rejected",
                reasons=apcd_reasons, event_metrics=apcd.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "claim-activity proxy; no NDC or inventory label",
                }),
        _result(name="arkansas_monthly_atc_therapeutic_demand_state", cadence="monthly",
                target_type="five_state", rows=int(atc.get("test_rows", 0)),
                folds=int(atc.get("fold_count", 0)),
                accuracy=atc.get("mean_model_accuracy"),
                balanced_accuracy=atc.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="HHS Arkansas pharmacy NDC claims + NLM RxNorm/RxClass ATC mapping",
                status="qualified_proxy" if not atc_reasons else "rejected",
                reasons=atc_reasons, event_metrics=atc.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_or_selected_model",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "therapeutic-class demand proxy; no pharmacy inventory target",
                }),
        _result(name="nadac_next_observed_price", cadence="weekly", target_type="numeric",
                rows=nadac_rows, folds=int(nadac_rolling.get("fold_count", 0)),
                accuracy=nadac_accuracy, drug_count=int(nadac.get("test_ndcs", 0)),
                region_count=1, supplier_count=0, evidence="CMS NADAC",
                status="qualified_proxy" if not nadac_reasons else "rejected",
                reasons=nadac_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_model_wape": (
                        sum(nadac_model_wape) / len(nadac_model_wape)
                        if nadac_model_wape else None),
                    "mean_persistence_wape": (
                        sum(nadac_persistence_wape) / len(nadac_persistence_wape)
                        if nadac_persistence_wape else None),
                    "all_folds_beating_persistence": bool(nadac_beats_persistence),
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "persistence outperformed the model in all historical folds",
                }),
        _result(name="fda_shortage_continuation", cadence="monthly", target_type="binary_state",
                rows=int(shortage.get("test_rows", 0)),
                folds=int(shortage_rolling.get("fold_count", 0)), accuracy=None,
                drug_count=0, region_count=1, supplier_count=0,
                evidence="FDA shortage archive", status="rejected", reasons=shortage_reasons),
        _result(name="ndc_monthly_shortage_pressure_state", cadence="monthly",
                target_type="five_state", rows=int(pressure.get("test_rows", 0)),
                folds=int(pressure.get("fold_count", 0)),
                accuracy=pressure.get("mean_model_accuracy"),
                balanced_accuracy=pressure.get("mean_model_balanced_accuracy"),
                drug_count=int(pressure.get("per_drug_count_with_minimum_rows", 0)),
                drug_count_at_or_above_75_percent=int(
                    pressure.get("per_drug_count_at_or_above_75_percent", 0)),
                region_count=0, supplier_count=0,
                evidence="FDA shortage archive, supplier-count aggregation",
                status="qualified_proxy" if not pressure_reasons else "rejected",
                reasons=pressure_reasons,
                event_metrics=pressure.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": pressure.get(
                        "model_balanced_improvement_vs_persistence"),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "mixed_cross_signal_test",
                    "utility_evidence": (
                        "five-state recall augmentation improved mean balanced accuracy "
                        "but did not improve every chronological fold; utility remains unproven"
                    ),
                }),
        _result(name="ndc_monthly_shortage_supplier_count", cadence="monthly",
                target_type="numeric", rows=int(numeric_shortage.get("test_rows", 0)),
                folds=int(numeric_shortage.get("fold_count", 0)),
                accuracy=numeric_shortage.get("mean_model_within_5_percent_error"),
                drug_count=int(numeric_shortage.get("ndc_count", 0)), region_count=0,
                supplier_count=0,
                drug_count_at_or_above_75_percent=int(
                    numeric_shortage.get("per_drug_count_at_or_above_75_percent", 0)),
                evidence="FDA shortage archive, numeric supplier-count aggregation",
                status="qualified_proxy" if not numeric_shortage_reasons else "rejected",
                reasons=numeric_shortage_reasons,
                event_metrics=numeric_shortage.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "numeric shortage count is an upstream proxy, not inventory truth",
                }),
        _result(name="arkansas_region_annual_demand_state", cadence="annual",
                target_type="three_state", rows=int(regional.get("test_rows", 0)),
                folds=int(regional.get("fold_count", 0)),
                accuracy=regional.get("mean_model_accuracy"),
                balanced_accuracy=regional.get("mean_model_balanced_accuracy"),
                drug_count=int(regional.get("drug_count", 0)),
                region_count=int(regional.get("region_count", 0)), supplier_count=0,
                region_count_at_or_above_75_percent=sum(
                    float(value) >= PART_ONE_ACCURACY
                    for value in regional.get("per_region_accuracy", {}).values()),
                evidence="CMS Medicare Part D county demand aggregated to Arkansas DHS regions",
                status="qualified_proxy" if not regional_reasons else "rejected",
                reasons=regional_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": regional_baseline_delta,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "regional public target is not a private pharmacy inventory target",
                }),
        _result(name="arkansas_region_annual_demand_claims", cadence="annual",
                target_type="numeric", rows=int(regional_numeric.get("test_rows", 0)),
                folds=int(regional_numeric.get("fold_count", 0)),
                accuracy=regional_numeric.get("mean_model_within_5_percent_error"),
                drug_count=int(regional_numeric.get("drug_count", 0)),
                region_count=int(regional_numeric.get("region_count", 0)), supplier_count=0,
                evidence="CMS Medicare Part D county demand, numeric regional claims",
                status="qualified_proxy" if not regional_numeric_reasons else "rejected",
                reasons=regional_numeric_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "numeric regional claims failed the within-5-percent gate",
                }),
        _result(name="arkansas_region_annual_demand_five_state", cadence="annual",
                target_type="five_state", rows=int(regional_five.get("test_rows", 0)),
                folds=int(regional_five.get("fold_count", 0)),
                accuracy=regional_five.get("mean_model_accuracy"),
                balanced_accuracy=regional_five.get("mean_model_balanced_accuracy"),
                drug_count=int(regional_five.get("drug_count", 0)),
                region_count=int(regional_five.get("region_count", 0)), supplier_count=0,
                region_count_at_or_above_75_percent=sum(
                    float(value) >= PART_ONE_ACCURACY
                    for value in regional_five.get("per_region_accuracy", {}).values()),
                evidence="CMS Medicare Part D county demand, five-state regional quantiles",
                status="qualified_proxy" if not regional_five_reasons else "rejected",
                reasons=regional_five_reasons,
                event_metrics=regional_five.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": _mean_fold_balanced_delta(regional_five),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "five-state regional public demand proxy; not private pharmacy inventory truth",
                }),
        _result(name="arkansas_state_annual_partd_demand_five_state", cadence="annual",
                target_type="five_state", rows=int(state_partd.get("test_rows", 0)),
                folds=int(state_partd.get("fold_count", 0)),
                accuracy=state_partd.get("mean_model_accuracy"),
                balanced_accuracy=state_partd.get("mean_model_balanced_accuracy"),
                drug_count=int(state_partd.get("drug_count", 0)),
                drug_count_at_or_above_75_percent=int(
                    sum(float(value) >= PART_ONE_ACCURACY
                        for value in state_partd.get("per_group_accuracy", {}).values())),
                region_count=int(state_partd.get("region_count", 0)),
                region_count_at_or_above_75_percent=sum(
                    float(value) >= PART_ONE_ACCURACY
                    for value in state_partd.get("per_group_accuracy", {}).values()),
                supplier_count=0,
                evidence="CMS Medicare Part D geography and drug, Arkansas state claims",
                status="qualified_proxy" if not state_partd_reasons else "rejected",
                reasons=state_partd_reasons,
                event_metrics=state_partd.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": _mean_fold_balanced_delta(state_partd),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "state-by-drug public demand proxy; not private pharmacy inventory truth",
                }),
        _result(name="arkansas_county_annual_demand_five_state", cadence="annual",
                target_type="five_state", rows=int(county_five.get("test_rows", 0)),
                folds=int(county_five.get("fold_count", 0)),
                accuracy=county_five.get("mean_model_accuracy"),
                balanced_accuracy=county_five.get("mean_model_balanced_accuracy"),
                drug_count=int(county_five.get("drug_count", 0)),
                region_count=int(county_five.get("county_count", 0)), supplier_count=0,
                region_count_at_or_above_75_percent=int(
                    county_five.get("county_count_at_or_above_75_percent", 0)),
                evidence="CMS Medicare Part D county demand, five-state county quantiles",
                status="qualified_proxy" if not county_five_reasons else "rejected",
                reasons=county_five_reasons,
                event_metrics=county_five.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "county public demand proxy; not private pharmacy inventory truth",
                }),
        _result(name="ndc_monthly_recall_pressure_state", cadence="monthly",
                target_type="three_state", rows=int(recall.get("test_rows", 0)),
                folds=int(recall.get("fold_count", 0)),
                accuracy=recall.get("mean_model_accuracy"),
                balanced_accuracy=recall.get("mean_model_balanced_accuracy"),
                drug_count=int(recall.get("ndc_count", 0)), region_count=0,
                supplier_count=0, evidence="FDA drug enforcement recall records",
                status="qualified_proxy" if not recall_reasons else "rejected",
                reasons=recall_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": recall_baseline_delta,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "mixed_cross_signal_test",
                    "utility_evidence": (
                        "five-state recall augmentation improved mean balanced accuracy "
                        "but did not improve every chronological fold; utility remains unproven"
                    ),
                }),
        _result(name="supplier_ndc_monthly_recall_pressure_state", cadence="monthly",
                target_type="three_state", rows=int(supplier_recall.get("test_rows", 0)),
                folds=int(supplier_recall.get("fold_count", 0)),
                accuracy=supplier_recall.get("mean_model_accuracy"),
                balanced_accuracy=supplier_recall.get("mean_model_balanced_accuracy"),
                drug_count=int(supplier_recall.get("ndc_count", 0)), region_count=0,
                supplier_count=int(supplier_recall.get("supplier_count", 0)),
                evidence="FDA drug enforcement recall records, recalling firm preserved",
                status="qualified_proxy" if not supplier_recall_reasons else "rejected",
                reasons=supplier_recall_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": supplier_recall_baseline_delta,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "supplier-by-NDC recall state is upstream recall pressure, not inventory truth",
                }),
        _result(name="ndc_monthly_recall_severity", cadence="monthly",
                target_type="numeric", rows=int(numeric_recall.get("test_rows", 0)),
                folds=int(numeric_recall.get("fold_count", 0)),
                accuracy=numeric_recall.get("mean_model_within_5_percent_error"),
                drug_count=int(numeric_recall.get("ndc_count", 0)), region_count=0,
                drug_count_at_or_above_75_percent=int(
                    numeric_recall.get("per_drug_count_at_or_above_75_percent", 0)),
                supplier_count=0, evidence="FDA drug enforcement records, numeric severity",
                status="qualified_proxy" if not numeric_recall_reasons else "rejected",
                reasons=numeric_recall_reasons,
                event_metrics=numeric_recall.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "numeric severity candidate awaits promoted-surface integration",
                }),
        _result(name="supplier_ndc_monthly_recall_severity", cadence="monthly",
                target_type="numeric", rows=int(numeric_supplier_recall.get("test_rows", 0)),
                folds=int(numeric_supplier_recall.get("fold_count", 0)),
                accuracy=numeric_supplier_recall.get("mean_model_within_5_percent_error"),
                drug_count=int(numeric_supplier_recall.get("ndc_count", 0)), region_count=0,
                drug_count_at_or_above_75_percent=int(
                    numeric_supplier_recall.get("per_drug_count_at_or_above_75_percent", 0)),
                supplier_count=int(numeric_supplier_recall.get("supplier_count", 0)),
                evidence="FDA enforcement records, numeric supplier x NDC severity",
                status="qualified_proxy" if not numeric_supplier_recall_reasons else "rejected",
                reasons=numeric_supplier_recall_reasons,
                event_metrics=numeric_supplier_recall.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "numeric supplier severity awaits promoted-surface integration",
                }),
        _result(name="global_monthly_supply_chain_pressure_state", cadence="monthly",
                target_type="three_state", rows=int(gscpi.get("test_rows", 0)),
                folds=int(gscpi.get("fold_count", 0)),
                accuracy=gscpi.get("mean_model_accuracy"),
                balanced_accuracy=gscpi.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=0, supplier_count=0,
                evidence="New York Fed Global Supply Chain Pressure Index",
                status="qualified_proxy" if not gscpi_reasons else "rejected",
                reasons=gscpi_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": _mean_fold_balanced_delta(gscpi),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "global upstream context failed the balanced-state gate and has no pharmacy inventory label",
                }),
        _result(name="arkansas_quarterly_medicaid_prescription_demand", cadence="quarterly",
                target_type="numeric", rows=int(medicaid_demand.get("test_rows", 0)),
                folds=int(medicaid_demand.get("fold_count", 0)),
                accuracy=medicaid_demand.get("mean_model_within_5_percent"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CMS/Medicaid State Drug Utilization Data",
                status="qualified_proxy" if not medicaid_demand_reasons else "rejected",
                reasons=medicaid_demand_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_learned_ridge_within_5_percent": medicaid_demand.get(
                        "mean_learned_ridge_within_5_percent"),
                    "mean_persistence_within_5_percent": medicaid_demand.get(
                        "mean_persistence_within_5_percent"),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "direct public utilization target failed the numeric accuracy gate",
                }),
        _result(name="national_monthly_shortage_breadth_state", cadence="monthly",
                target_type="three_state", rows=int(shortage_breadth.get("test_rows", 0)),
                folds=int(shortage_breadth.get("fold_count", 0)),
                accuracy=shortage_breadth.get("mean_model_accuracy"),
                balanced_accuracy=shortage_breadth.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=0, supplier_count=0,
                evidence="FDA shortage archive, monthly distinct-NDC breadth",
                status="qualified_proxy" if not shortage_breadth_reasons else "rejected",
                reasons=shortage_breadth_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": _mean_fold_balanced_delta(shortage_breadth),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "raw accuracy is persistence-driven and balanced-state performance fails",
                }),
        *[
            _result(
                name=f"arkansas_weekly_hospital_{pathogen}_admission_pressure_state",
                cadence="weekly", target_type="three_state",
                rows=int(result.get("test_rows", 0)),
                folds=int(result.get("fold_count", 0)),
                accuracy=result.get("mean_model_accuracy"),
                balanced_accuracy=result.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC NHSN Hospital Respiratory Data",
                status="qualified_proxy" if not reasons else "rejected",
                reasons=reasons,
                baseline_skill={
                    "status": "persistence_dominated" if result else "not_available",
                    "mean_balanced_accuracy_delta_vs_persistence": result.get(
                        "model_balanced_improvement_vs_persistence"),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "hospital utilization proxy; no pharmacy inventory target",
                },
            )
            for pathogen, result, reasons in hospital_candidates
        ],
        *[
            _result(
                name=f"arkansas_weekly_nssp_{pathogen}_ed_pressure_state",
                cadence="weekly", target_type="five_state",
                rows=int(result.get("test_rows", 0)),
                folds=int(result.get("fold_count", 0)),
                accuracy=result.get("mean_model_accuracy"),
                balanced_accuracy=result.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC NSSP ED trajectories",
                status="qualified_proxy" if not reasons else "rejected",
                reasons=reasons, event_metrics=result.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": (
                        result.get("mean_model_balanced_accuracy", 0.0)
                        - result.get("mean_persistence_balanced_accuracy", 0.0)),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "NSSP pathogen ED proxy; no pharmacy inventory target",
                },
            )
            for pathogen, result, reasons in nssp_candidates
        ],
        _result(name="arkansas_weekly_hospital_influenza_admissions", cadence="weekly",
                target_type="numeric", rows=int(hospital_numeric.get("test_rows", 0)),
                folds=int(hospital_numeric.get("fold_count", 0)),
                accuracy=hospital_numeric.get("mean_model_within_5_percent_error"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC NHSN Hospital Respiratory Data, numeric influenza admissions",
                status="qualified_proxy" if not hospital_numeric_reasons else "rejected",
                reasons=hospital_numeric_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_tested",
                    "utility_evidence": "numeric hospital admission proxy failed the within-5-percent gate",
                }),
        *[
            _result(
                name=f"arkansas_weekly_wastewater_{pathogen}_state", cadence="weekly",
                target_type="five_state", rows=int(result.get("test_rows", 0)),
                folds=int(result.get("fold_count", 0)),
                accuracy=result.get("mean_model_accuracy"),
                balanced_accuracy=result.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=1, supplier_count=0,
                evidence="CDC Arkansas wastewater viral activity levels",
                status="qualified_proxy" if not reasons else "rejected",
                reasons=reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": (
                        result.get("mean_model_balanced_accuracy", 0.0)
                        - result.get("mean_persistence_accuracy", 0.0)),
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "wastewater respiratory utilization proxy; not pharmacy inventory truth",
                },
            )
            for pathogen, result, reasons in wastewater_candidates
        ],
        *[
            _result(
                name=(f"arkansas_region_weekly_wastewater_{pathogen}_"
                      f"{representation}_pressure_state"),
                cadence="weekly", target_type="five_state",
                rows=int(result.get("test_rows", 0)),
                folds=int(result.get("fold_count", 0)),
                accuracy=result.get("mean_model_accuracy"),
                balanced_accuracy=result.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=int(result.get("region_count", 0)),
                supplier_count=0,
                evidence="CDC wastewater WVAL + Arkansas county-region crosswalk",
                status="qualified_proxy" if not reasons else "rejected",
                reasons=reasons, event_metrics=result.get("event_metrics"),
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": (
                        "regional wastewater level/change proxy; not pharmacy inventory truth"),
                },
            )
            for pathogen, representation, result, reasons
            in regional_wastewater_candidates
        ],
        _result(name="arkansas_county_monthly_overdose_pressure_state", cadence="monthly",
                target_type="three_state", rows=int(overdose.get("test_rows", 0)),
                folds=int(overdose.get("fold_count", 0)),
                accuracy=overdose.get("mean_model_accuracy"),
                balanced_accuracy=overdose.get("mean_model_balanced_accuracy"),
                drug_count=0, region_count=0, supplier_count=0,
                evidence="CDC VSRR provisional county overdose deaths",
                status="qualified_proxy" if not overdose_reasons else "rejected",
                reasons=overdose_reasons,
                baseline_skill={
                    "status": "persistence_dominated",
                    "mean_balanced_accuracy_delta_vs_persistence": None,
                    "all_folds_beating_persistence": False,
                    "incremental_utility_status": "not_demonstrated",
                    "utility_evidence": "county overdose utilization context; no pharmacy inventory target",
                }),
    ]
    qualified = [item for item in candidates if item["status"] == "qualified_proxy"]
    validity_fields = (
        "source_url", "target_semantics", "feature_timestamp_boundary",
        "geography_scope", "supplier_resolution", "missingness_censoring",
        "target_observation_type", "target_claim_scope",
    )
    invalid_validity = [item["metric"] for item in candidates if any(
        not str(item.get(field, "")).strip() for field in validity_fields
    )]
    direct_qualified = [item["metric"] for item in qualified if (
        item.get("target_is_direct_pharmacy_observation") is not False
        or item.get("target_claim_scope") != "proxy_only_not_pharmacy_inventory"
    )]
    qualified_names = {item["metric"] for item in qualified}
    coverage_gates = {
        "geography": {
            "passed": any(item["arkansas_region_count"] > 0 for item in qualified),
            "required_filter": "geography_level/geography_id",
            "passing_metrics": [item["metric"] for item in qualified
                                 if item["arkansas_region_count"] > 0],
        },
        "drug": {
            "passed": any((item["drug_count"] or 0) > 0 for item in qualified),
            "required_filter": "drug_key",
            "passing_metrics": [item["metric"] for item in qualified
                                 if (item["drug_count"] or 0) > 0],
        },
        "supplier": {
            "passed": any((item["supplier_count"] or 0) > 0 for item in qualified),
            "required_filter": "supplier",
            "passing_metrics": [item["metric"] for item in qualified
                                 if (item["supplier_count"] or 0) > 0],
        },
        "disease_symptom": {
            # A single respiratory aggregate is not a disease/symptom split.
            # Keep this false until a qualified target exposes a disease or
            # pathogen filter with the required accuracy route.
            "passed": any(item["metric"].startswith("arkansas_weekly_nssp_")
                           for item in qualified),
            "required_filter": "pathogen",
            "passing_metrics": [item["metric"] for item in qualified
                                 if item["metric"].startswith("arkansas_weekly_nssp_")],
        },
    }
    return {
        "protocol": "metric_library_contract_v1",
        "rules": {
            "minimum_held_out_rows": MIN_HELD_OUT_ROWS,
            "minimum_rolling_folds": MIN_ROLLING_FOLDS,
            "minimum_accuracy": MIN_METRIC_ACCURACY,
            "minimum_balanced_accuracy_for_state_targets": MIN_METRIC_ACCURACY,
            "part_one_accuracy": PART_ONE_ACCURACY,
            "part_one_drug_count": PART_ONE_DRUG_COUNT,
            "numeric_accuracy": "fraction of predictions within 5 percent relative error",
            "state_accuracy": "exact match; binary-only states are rejected",
        },
        "qualified_metric_count": len(qualified),
        "target_validity": {
            "passed": not invalid_validity and not direct_qualified,
            "qualified_targets_are_proxy_only": not direct_qualified,
            "missing_required_provenance": invalid_validity,
            "qualified_direct_inventory_claims": direct_qualified,
            "interpretation": (
                "qualified targets are external proxies; no public direct pharmacy "
                "inventory or fill-disruption label is being claimed"
            ),
        },
        "coverage_gates": coverage_gates,
        "part_one_gates": {
            "weekly_or_monthly_75_percent": any(
                x["cadence"] in {"weekly", "monthly"} and
                x["accuracy_within_5pct_or_exact_state"] is not None and
                x["accuracy_within_5pct_or_exact_state"] >= PART_ONE_ACCURACY
                for x in qualified),
            "per_drug_100_plus_75_percent": any(
                (x.get("drug_count_at_or_above_75_percent") or 0) >= PART_ONE_DRUG_COUNT and
                x["accuracy_within_5pct_or_exact_state"] is not None and
                x["accuracy_within_5pct_or_exact_state"] >= PART_ONE_ACCURACY
                for x in qualified),
            "per_arkansas_region_75_percent": any(
                x["arkansas_region_count"] >= 5 and
                (x.get("region_count_at_or_above_75_percent") or 0) >= 5 and
                x["accuracy_within_5pct_or_exact_state"] is not None and
                x["accuracy_within_5pct_or_exact_state"] >= PART_ONE_ACCURACY
                for x in qualified),
        },
        "candidates": candidates,
    }
