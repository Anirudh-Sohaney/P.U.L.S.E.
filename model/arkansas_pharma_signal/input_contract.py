"""Input-variable disposition contract.

Classifies model feature names into the six dispositions documented in
``docs/INPUT_VARIABLE_AUDIT.md``. Fail-closed: names matching no rule are
``PERIODIC_TRAINING_ONLY`` and are never promoted to a live feed by default.

Dependency-free (standard library only).
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .input_sources import audit_operational_sources

LIVE_INPUT = "LIVE_INPUT"
NEAR_REAL_TIME_INPUT = "NEAR_REAL_TIME_INPUT"
LOCAL_PHARMACY_HISTORY_INPUT = "LOCAL_PHARMACY_HISTORY_INPUT"
PERIODIC_TRAINING_ONLY = "PERIODIC_TRAINING_ONLY"
STATIC_IDENTITY_CONTEXT = "STATIC_IDENTITY_CONTEXT"
DERIVED_TRAINING_VARIABLE = "DERIVED_TRAINING_VARIABLE"

DISPOSITIONS = (
    LIVE_INPUT,
    NEAR_REAL_TIME_INPUT,
    LOCAL_PHARMACY_HISTORY_INPUT,
    PERIODIC_TRAINING_ONLY,
    STATIC_IDENTITY_CONTEXT,
    DERIVED_TRAINING_VARIABLE,
)

# Static identity / catalog-derived identity fields (exact names).
_STATIC_EXACT = frozenset({
    "year", "city", "county", "state", "drug", "ndc", "npi", "zip",
    "fips", "geography", "provider_id", "provider_name", "pharmacy",
    "ingredient", "supplier", "establishment",
})

# Catalog-derived identity substrings (crosswalks, dictionaries, rosters...).
_STATIC_SUBSTRINGS = (
    "_ndc", "ndc_", "_npi", "_fips", "_zip", "_crosswalk",
    "_dictionary", "_catalog", "_directory", "_roster",
    "_establishment", "_nppes", "_pharmacy_", "_drug_name",
    "_drug_key", "_ingredient_", "_supplier_",
)

_STATIC_SUFFIXES = ("_id", "_key", "_code", "_name")

# Encoded identity names: catalog/roster keys with "::" namespace prefix.
_STATIC_PREFIXES = (
    "ingredient::", "labeler::", "dosage_form::", "route::",
    "market_cat::", "dea::", "pharm_class::", "therapeutic::",
)

# Deterministic context-by-identity products (encoded interaction keys).
_DERIVED_PREFIXES = ("interaction::",)

# Point-in-time context joins are materialized with this prefix. The suffix
# often contains a raw-source token (for example ``prior_news_*``), so this
# must be classified before the source-family token rules below.
_DERIVED_CONTEXT_PREFIXES = ("prior_",)

# Deterministic transforms of raw series (lag/rolling/MA/delta/anomaly/log).
_DERIVED_RE = re.compile(
    r"(_lag\d*|_rolling_|_change_|_seasonal_|_anomaly|_baseline|"
    r"_delta|_diff|_ma\d|_log|_row_count$|_source_rows$|"
    r"_suppressed_rows$|_supplier_count$)"
)

# Audit-documented historical-only series: valid for training, never live.
_HISTORICAL_ONLY_TOKENS = (
    "covid", "humidity", "usaspending",
    "arkansas_employed_population", "arkansas_labor_force",
    "arkansas_unemployed_population", "consumer_price_index_medical_care",
    "_monthly_2012_2022",
)

# Annual (or slower) demand / population / policy measures.
_PERIODIC_TOKENS = (
    "cms_partd", "partd", "part_d", "medicaid", "arcos", "gscpi",
    "provider", "demand_claims", "y_last", "wuenic",
    "_prevalence", "population", "births", "deaths", "migration",
    "_fills", "_benes", "_clms", "_drug_cst", "_cst_shr",
    "_prscrbrs", "_reimbursed", "_prescriptions", "_suppressed",
    "_obligations", "_tariff", "_trade", "_throughput", "_crude",
    "_pink_sheet", "_life_expectancy", "_mortality", "_immunization",
    "_coverage", "_enrollment", "_applications", "_utilization",
    "_spending", "_beneficiary",
)

# Current weather / disaster / FDA shortage-recall / news / disease surveillance.
_NEAR_REAL_TIME_TOKENS = (
    "temperature", "precipitation", "wind", "snow", "weather", "extreme",
    "nadac", "acquisition_cost", "ppi", "pharma_producer_price_index",
    "bls", "fred",
    "fema", "disaster", "declaration",
    "shortage", "recall", "enforcement",
    "news", "article", "mention", "sentiment", "headline", "corpus",
    "ili", "wili", "flu", "influenza", "nndss", "wastewater",
    "outbreak", "surveillance", "cases", "transmission", "specimen",
    "intensity", "spread", "syndromic", "respiratory", "_test",
    "unemployment", "consumer_price_index_all_items",
)

# NWS active alerts are event-driven and available at operational cadence.
# Keep this separate from weekly/monthly feeds so the disposition records the
# stronger freshness contract.
_LIVE_INPUT_TOKENS = ("nws", "weather_api", "noaa_alert", "openfda_")

# These canonical names are openFDA-backed in the authoritative variable
# inventory even though their source prefix is normalized away.
_LIVE_EXACT = frozenset({"shortage_active", "recall_active"})

# These names are present in historical WHO-derived artifacts, but the
# documented FluID endpoint was not verified as a usable public refresh feed.
_WHO_RESEARCH_ONLY_EXACT = frozenset({
    "geospread", "ili_activity", "ili_case", "ili_nb_sites",
    "ili_outpatients", "intensity", "outbreak_event",
})

# These are observations a future pharmacy supplies from its own history at
# prediction time. They are not public external feeds, but they are valid
# operational inputs for the eventual pharmacy-level regression and should
# not be mistaken for unavailable annual context.
_LOCAL_HISTORY_EXACT = frozenset({
    "value", "value_last", "ma2", "prev_q", "lag2", "q1", "q2", "q3", "q4", "quarter",
    "log_demand_change", "relative_demand_change",
})


def _is_static(name: str) -> bool:
    if name in _STATIC_EXACT:
        return True
    if name.startswith(_STATIC_PREFIXES):
        return True
    if name.endswith(_STATIC_SUFFIXES):
        return True
    return any(tok in name for tok in _STATIC_SUBSTRINGS)


def _classify(name: str) -> tuple[str, str]:
    """Return (disposition, rationale) for a single feature name."""
    n = name.strip().lower()
    if not n:
        return PERIODIC_TRAINING_ONLY, "empty name; fail closed"

    if n in _LOCAL_HISTORY_EXACT:
        return LOCAL_PHARMACY_HISTORY_INPUT, (
            "supplied by the individual pharmacy from its own current history; "
            "not a public external feed"
        )

    if n in _LIVE_EXACT:
        return LIVE_INPUT, "canonical openFDA shortage/enforcement feed variable"

    if n in _WHO_RESEARCH_ONLY_EXACT:
        return PERIODIC_TRAINING_ONLY, (
            "WHO historical artifact retained for research; public refresh endpoint "
            "was not verified"
        )

    if n.startswith(_DERIVED_CONTEXT_PREFIXES):
        return DERIVED_TRAINING_VARIABLE, (
            "point-in-time prior/context feature derived from an earlier raw observation"
        )

    if n.startswith(_DERIVED_PREFIXES):
        return DERIVED_TRAINING_VARIABLE, (
            "deterministic context-by-identity product (interaction key)"
        )

    if _DERIVED_RE.search(n):
        return DERIVED_TRAINING_VARIABLE, (
            "deterministic transform (lag/rolling/MA/delta/anomaly/log) of a raw series"
        )

    if _is_static(n):
        return STATIC_IDENTITY_CONTEXT, "static identity or catalog-derived identity field"

    # Source-prefixed feeds win over periodic/historical tokens: news_/article_
    # are near-real-time news feeds, ww_ is wastewater surveillance.
    if n.startswith(("news_", "article_", "ww_")):
        return NEAR_REAL_TIME_INPUT, (
            "source-prefixed near-real-time feed (news/article/wastewater)"
        )

    if any(tok in n for tok in _HISTORICAL_ONLY_TOKENS):
        return PERIODIC_TRAINING_ONLY, "historical-only series per audit; not live"

    if any(tok in n for tok in _PERIODIC_TOKENS):
        return PERIODIC_TRAINING_ONLY, "annual (or slower) demand/population/policy measure"

    if any(tok in n for tok in _LIVE_INPUT_TOKENS):
        return LIVE_INPUT, "event-driven National Weather Service alert feed"

    if any(tok in n for tok in _NEAR_REAL_TIME_TOKENS):
        return NEAR_REAL_TIME_INPUT, (
            "current weather/disaster/FDA/news/disease-surveillance source"
        )

    return PERIODIC_TRAINING_ONLY, "unknown name; fail closed to training-only"


def disposition_for_variable(name: str) -> str:
    """Classify a feature name into one of the five dispositions.

    Fail-closed: names matching no rule are ``PERIODIC_TRAINING_ONLY``.
    """
    return _classify(name)[0]


def audit_feature_names(names: Iterable[str]) -> List[dict]:
    """Return one row per name: ``variable``, ``disposition``, ``rationale``."""
    return [
        {"variable": name, "disposition": disp, "rationale": reason}
        for name in names
        for disp, reason in [_classify(name)]
    ]


def operational_dispositions() -> frozenset:
    """Dispositions available at forecast time, including local history."""
    return frozenset({LIVE_INPUT, NEAR_REAL_TIME_INPUT,
                      LOCAL_PHARMACY_HISTORY_INPUT})


def is_operational_feature(name: str) -> bool:
    """Return whether a feature can be rebuilt from operational sources.

    Derived variables inherit availability from the raw feature family. This
    prevents an annual or historical transform from becoming operational merely
    because its transformed name does not contain the raw source token.
    """
    disposition = disposition_for_variable(name)
    if str(name).strip().lower() in _LOCAL_HISTORY_EXACT:
        return True
    if disposition in operational_dispositions() or disposition == STATIC_IDENTITY_CONTEXT:
        return True
    if disposition != DERIVED_TRAINING_VARIABLE:
        return False
    normalized = str(name).strip().lower()
    # A prior/context feature inherits availability from the raw field it was
    # joined from. This avoids treating ``prior_news_*`` as historical merely
    # because the source token also contains a historical-series word.
    if normalized.startswith(_DERIVED_CONTEXT_PREFIXES):
        return is_operational_feature(normalized[len("prior_"):])
    if normalized.startswith(_DERIVED_PREFIXES):
        context = normalized[len("interaction::"):].split("::", 1)[0]
        return is_operational_feature(context)
    # Local demand history is an operational input for the eventual pharmacy
    # regression, even though the token ``y_last`` overlaps a periodic-name
    # heuristic used for external annual features.
    if normalized.startswith("y_last"):
        return True
    transform_re = re.compile(
        r"(?:_lag(?:_?\d+)?|_rolling_(?:mean|std)(?:_\d+)?|"
        r"_change(?:_?\d+)?|_seasonal_(?:anomaly|baseline)|"
        r"_anomaly|_baseline|_delta|_diff|_ma(?:_?\d+)?|_log|"
        r"_row_count|_source_rows|_suppressed_rows|_supplier_count)$"
    )
    raw_name = normalized
    while True:
        stripped = transform_re.sub("", raw_name)
        if stripped == raw_name:
            break
        raw_name = stripped
    if raw_name == normalized:
        return False
    raw_disposition = disposition_for_variable(raw_name)
    return raw_disposition in {
        LIVE_INPUT, NEAR_REAL_TIME_INPUT, LOCAL_PHARMACY_HISTORY_INPUT,
        STATIC_IDENTITY_CONTEXT,
    }


def is_local_history_feature(name: str) -> bool:
    """Return whether a field is supplied by the pharmacy's own history."""
    return str(name).strip().lower() in _LOCAL_HISTORY_EXACT


def summarize_dispositions(names: Iterable[str]) -> dict:
    """Aggregate a feature-name audit into a machine-readable contract.

    Returns row-level audit plus counts by every disposition.  ``operational_ready``
    is fail-closed: true only when no feature is ``PERIODIC_TRAINING_ONLY``.
    """
    rows = audit_feature_names(names)
    counts = {disp: sum(1 for r in rows if r["disposition"] == disp)
              for disp in DISPOSITIONS}
    operational = operational_dispositions()
    source_names = [row["variable"] for row in rows
                    if row["disposition"] in {LIVE_INPUT, NEAR_REAL_TIME_INPUT}]
    source_audit = audit_operational_sources(source_names)
    return {
        "rows": rows,
        "counts": counts,
        "operational_feature_count": sum(counts[d] for d in operational),
        "periodic_training_only_count": counts[PERIODIC_TRAINING_ONLY],
        "derived_training_variable_count": counts[DERIVED_TRAINING_VARIABLE],
        "derived_non_operational_count": sum(
            1 for row in rows
            if row["disposition"] == DERIVED_TRAINING_VARIABLE
            and not is_operational_feature(row["variable"])),
        "local_history_count": sum(
            1 for row in rows if is_local_history_feature(row["variable"])),
        "non_operational_feature_count": sum(
            1 for row in rows if not is_operational_feature(row["variable"])),
        "static_identity_context_count": counts[STATIC_IDENTITY_CONTEXT],
        "disposition_operational_ready": all(is_operational_feature(row["variable"])
                                              for row in rows),
        "operational_ready": (
            all(is_operational_feature(row["variable"])
                for row in rows) and source_audit["all_adapters_ready"]
        ),
        "operational_source_audit": source_audit,
        "near_real_time_source_audit": source_audit,
    }
