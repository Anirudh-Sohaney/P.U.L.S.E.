"""Auditable event output schema for the language-understanding layer."""

from __future__ import annotations

import pandas as pd

EVENT_COLUMNS = [
    "event_id", "article_id", "event_type", "event_subtype", "event_start",
    "event_end", "location", "county_fips", "disease", "variant", "drug",
    "therapeutic_class", "supplier", "parent_company", "factory", "api_source",
    "supply_impact", "demand_impact", "severity", "probability", "lead_time_days",
    "evidence_span", "negation_status", "uncertainty_status", "causal_status",
    "source", "source_timestamp", "effective_date", "confidence", "evidence_type",
]


def empty_events() -> pd.DataFrame:
    """Return an empty frame with the canonical event columns."""
    return pd.DataFrame(columns=EVENT_COLUMNS)


def validate_events(events: pd.DataFrame) -> None:
    """Validate event columns, unique IDs, and confidence bounds."""
    missing = set(EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"event schema missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    confidence = pd.to_numeric(events["confidence"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("event confidence must be in [0, 1]")


def available_events(events: pd.DataFrame, as_of: object) -> pd.DataFrame:
    """Enforce article/event time availability before feature construction."""
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    published = pd.to_datetime(events["source_timestamp"], errors="coerce", utc=True)
    return events[published.notna() & (published <= cutoff)].copy()
