"""Machine-readable provenance for operationally usable input families."""

from __future__ import annotations

from typing import Iterable


# Availability and code integration are intentionally separate. A public URL
# can be usable in principle while the corresponding refresh adapter is still
# required before production operation.
NEAR_REAL_TIME_SOURCE_REGISTRY = {
    **{
        name: {
            "source_url": "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html",
            "source_access": "free_public",
            "cadence": "annual",
            "publication_lag": "annual_report_lag",
            "source_status": "source_documented",
            "adapter_status": "implemented",
            "availability_class": "periodic_training_only",
            "native_geography": "state_or_zip3",
        }
        for name in ("arcos_distribution_grams", "arcos_retail_units",
                     "regional_distribution_pressure")
    },
    **{
        name: {
            "source_url": "https://data.cdc.gov/NNDSS/NNDSS-Weekly-Data/x9gk-5huc",
            "source_access": "free_public",
            "cadence": "weekly",
            "publication_lag": "weekly_reporting_lag",
            "source_status": "source_documented",
            "adapter_status": "implemented",
        }
        for name in ("current_week_cases",)
    },
    **{
        name: {
            "source_url": "https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2",
            "source_access": "free_public",
            "cadence": "event_driven",
            "publication_lag": "posted_after_declaration",
            "source_status": "source_documented",
            "adapter_status": "implemented",
        }
        for name in ("disaster_active", "disaster_severity")
    },
    **{
        name: {
            "source_url": "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
            "source_access": "free_public",
            "cadence": "daily",
            "publication_lag": "approximately_1_to_2_days",
            "source_status": "source_documented",
            "adapter_status": "implemented",
        }
        for name in ("extreme_cold_day", "extreme_heat_day", "precipitation",
                     "snow_depth", "snowfall", "temperature_max", "temperature_mean",
                     "temperature_min", "wind")
    },
    **{
        name: {
            "source_url": "https://cmu-delphi.github.io/delphi-epidata/api/fluview.html",
            "source_access": "free_public",
            "cadence": "weekly",
            "publication_lag": "approximately_1_to_2_weeks",
            "source_status": "source_documented",
            "adapter_status": "implemented",
        }
        for name in ("ili", "num_ili", "wili")
    },
    "respnet_rsv_rate": {
        "source_url": "https://data.cdc.gov/Public-Health-Surveillance/RESP-NET-Rates-and-Clinical-Data/kvib-3txy",
        "source_access": "free_public",
        "cadence": "weekly",
        "publication_lag": "weekly_reporting_and_revision_lag",
        "source_status": "source_documented",
        "adapter_status": "implemented",
    },
    **{
        name: {
            "source_url": "https://www.medicaid.gov/medicaid/prescription-drugs/pharmacy-pricing",
            "source_access": "free_public",
            "cadence": "weekly",
            "publication_lag": "weekly_file_release",
            "source_status": "source_documented",
            "adapter_status": "implemented",
        }
        for name in ("nadac_per_unit_max", "nadac_per_unit_mean", "nadac_per_unit_min")
    },
}

# WHO documents these feeds and fields, but the public FluID object required
# for them was not available in the verified public mart at review time. Keep
# the records for research provenance without allowing them into the
# operational near-real-time registry.
RESEARCH_ONLY_SOURCE_REGISTRY = {
    **{
        name: {
            "source_url": "https://www.who.int/tools/fluid",
            "source_access": "public_landing_page_only",
            "cadence": "weekly",
            "publication_lag": "multi_week_reporting_lag",
            "source_status": "feed_unverified",
            "adapter_status": "blocked_unverified_endpoint",
        }
        for name in ("geospread", "ili_activity", "ili_case", "ili_nb_sites",
                     "ili_outpatients", "intensity")
    },
    "outbreak_event": {
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news",
        "source_access": "free_public",
        "cadence": "event_driven",
        "publication_lag": "posted_after_event_confirmation",
        "source_status": "historical_artifact_only",
        "adapter_status": "blocked_no_verified_refresh",
    },
}

LIVE_SOURCE_REGISTRY = {
    "shortage_active": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages.json",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "implemented",
    },
    "openfda_shortages": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages.json",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "implemented",
    },
    "recall_active": {
        "source_url": "https://open.fda.gov/apis/drug/enforcement/",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "implemented",
    },
    "openfda_enforcement": {
        "source_url": "https://open.fda.gov/apis/drug/enforcement/",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "implemented",
    },
    # These normalized training names are derived from openFDA records but do
    # not yet have a tested end-to-end live feature refresh.
    "fda_shortage_active": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages.json",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "not_integrated",
    },
    "fda_shortage_supplier_count": {
        "source_url": "https://open.fda.gov/apis/drug/drugshortages.json",
        "source_access": "free_public",
        "cadence": "daily_or_weekly",
        "publication_lag": "continuous_api_updates",
        "source_status": "source_documented",
        "adapter_status": "not_integrated",
    },
}


def source_metadata_for_variable(name: str) -> dict:
    """Return source metadata for a normalized variable name, if registered."""
    normalized = str(name).strip().lower()
    if normalized in LIVE_SOURCE_REGISTRY:
        return dict(LIVE_SOURCE_REGISTRY[normalized])
    if normalized.startswith("openfda_shortages"):
        return dict(LIVE_SOURCE_REGISTRY["openfda_shortages"])
    if normalized.startswith("openfda_enforcement"):
        return dict(LIVE_SOURCE_REGISTRY["openfda_enforcement"])
    if normalized in RESEARCH_ONLY_SOURCE_REGISTRY:
        return dict(RESEARCH_ONLY_SOURCE_REGISTRY[normalized])
    return dict(NEAR_REAL_TIME_SOURCE_REGISTRY.get(normalized, {}))


def audit_operational_sources(names: Iterable[str]) -> dict:
    """Audit source documentation without conflating it with adapter readiness."""
    normalized = sorted({str(name).strip().lower() for name in names})
    rows = []
    for name in normalized:
        metadata = source_metadata_for_variable(name)
        rows.append({"variable": name, **metadata})
    uncovered = [row["variable"] for row in rows if not row.get("source_url")]
    not_integrated = [row["variable"] for row in rows
                      if row.get("adapter_status") not in {"implemented"}]
    return {
        "variables": rows,
        "variable_count": len(rows),
        "all_sources_documented": not uncovered,
        "uncovered_variables": uncovered,
        "all_adapters_ready": not not_integrated,
        "adapter_pending_variables": not_integrated,
        "interpretation": (
            "source documentation proves public availability only; adapter readiness "
            "requires a tested point-in-time refresh implementation"
        ),
    }


def audit_near_real_time_sources(names: Iterable[str]) -> dict:
    """Backward-compatible audit for only near-real-time variables."""
    return audit_operational_sources(names)
