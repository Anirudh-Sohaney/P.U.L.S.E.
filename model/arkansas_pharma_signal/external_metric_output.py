"""Schema-preserving output helpers for qualified external metrics."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import numpy as np

from .universal_forecast import (
    TARGET_METADATA,
    UNIVERSAL_OUTPUT_COLUMNS,
    _apply_target_metadata,
)


PROMOTED_STATUSES = frozenset({
    "qualified_five_state_proxy",
    "qualified_numeric_proxy",
})

# Historical three-state evaluators remain readable from their artifacts, but
# cannot enter the operational metric surface or feature store.
LEGACY_STATUSES = frozenset({"legacy_three_state_proxy"})


def validate_metric_rows(frame: pd.DataFrame) -> None:
    """Fail closed on rows that violate the promoted metric contract.

    The universal schema alone cannot distinguish a qualified five-state
    signal from a binary diagnostic or an unknown target. Validate after
    metadata enrichment, before serialization or feature-store attachment.
    """
    required = {
        "target", "forecast_period", "prediction", "target_cadence",
        "target_semantics", "target_promotion_status", "source_freshness",
        "driver_attribution",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metric rows missing contract columns: {sorted(missing)}")
    targets = frame["target"].astype(str)
    unknown = sorted(set(targets) - set(TARGET_METADATA))
    if unknown:
        raise ValueError(f"metric rows contain unknown targets: {unknown}")
    status = frame["target_promotion_status"].astype(str)
    if (~status.isin(PROMOTED_STATUSES)).any():
        bad = sorted(status[~status.isin(PROMOTED_STATUSES)].unique())
        raise ValueError(f"only qualified metrics may be serialized: {bad}")
    metadata = targets.map(TARGET_METADATA)
    expected_status = metadata.map(lambda value: value.get("status", "")
                                   if isinstance(value, dict) else "")
    if not status.eq(expected_status).all():
        raise ValueError("metric promotion status does not match target metadata")
    predictions = pd.to_numeric(frame["prediction"], errors="coerce")
    if predictions.isna().any() or (~np.isfinite(predictions.to_numpy(float))).any():
        raise ValueError("metric predictions must be finite numeric values")
    state_rows = status.eq("qualified_five_state_proxy")
    if state_rows.any():
        state_values = predictions[state_rows]
        if (~state_values.isin([0, 1, 2, 3, 4])).any():
            raise ValueError("five-state metric predictions must be 0, 1, 2, 3, or 4")
        state_events = metadata[state_rows].map(
            lambda value: value.get("event", {}).get("kind")
            if isinstance(value, dict) else None)
        if not state_events.eq("state").all():
            raise ValueError("qualified five-state metrics must declare state events")
    numeric_rows = status.eq("qualified_numeric_proxy")
    if numeric_rows.any():
        numeric_events = metadata[numeric_rows].map(
            lambda value: value.get("event", {})
            if isinstance(value, dict) else {})
        invalid_numeric_events = numeric_events.map(
            lambda event: event.get("kind") not in {"numeric", "not_applicable"})
        if invalid_numeric_events.any():
            raise ValueError("qualified numeric metrics have invalid event definitions")
        missing_numeric_definition = numeric_events.map(
            lambda event: event.get("kind") == "numeric"
            and ("event_threshold" not in event
                 or "relative_tolerance" not in event))
        if missing_numeric_definition.any():
            raise ValueError("numeric event metrics require threshold and tolerance")
    for column in ("forecast_period", "target_cadence", "target_semantics", "source_freshness"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"metric {column} cannot be empty")
    if frame["driver_attribution"].isna().any():
        raise ValueError("metric driver attribution cannot be null")
    if "horizon" in frame:
        horizons = pd.to_numeric(frame["horizon"], errors="coerce")
        if horizons.isna().any() or (horizons <= 0).any():
            raise ValueError("qualified metric horizons must be positive")
    if not frame["uncertainty_status"].astype(str).eq("not_estimated").all():
        raise ValueError("qualified metrics must declare uncertainty as not_estimated")
    if not frame["calibration_status"].astype(str).eq("not_calibrated").all():
        raise ValueError("qualified metrics must declare calibration as not_calibrated")
    for column in ("confidence", "interval_low", "interval_high", "risk_score"):
        if column in frame and frame[column].notna().any():
            raise ValueError(f"qualified metrics cannot expose unvalidated {column}; use null")


def build_metric_rows(
    metrics: pd.DataFrame,
    *,
    model_version: str,
    forecast_timestamp: str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Normalize metric predictions into filterable output rows.

    ``metrics`` must carry a stable metric target, period, prediction, and
    geography/drug keys. Missing supplier or county allocation is represented
    by an empty value rather than inferred from the metric name. Candidate
    outputs may skip promotion validation, but remain metadata-bearing and are
    rejected by the feature-store adapter until separately promoted.
    """
    required = {"target", "forecast_period", "prediction"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metric rows missing columns: {sorted(missing)}")
    frame = metrics.copy()
    now = forecast_timestamp or datetime.now(timezone.utc).isoformat()
    defaults = {
        "horizon": None, "geography_level": "external_metric",
        "geography_id": "", "county_fips": "", "county_name": "",
        "arkansas_region": "", "drug_key": "", "ingredient": "", "therapeutic_class": "", "pathogen": "",
        "dosage_form": "", "strength": "", "route": "", "labeler": "", "supplier": "",
        "parent_company": "", "factory": "", "api_source": "",
        "interval_low": None, "interval_high": None, "risk_score": None,
        "confidence": None, "event_ids": "[]", "evidence_article_ids": "[]",
        "driver_attribution": "{}", "evidence_type": "public_observation",
        "feature_window_start": "", "feature_window_end": "",
        "uncertainty_status": "not_estimated", "calibration_status": "not_calibrated",
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    frame["forecast_timestamp"] = now
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    if "source_freshness" not in frame:
        frame["source_freshness"] = ""
    freshness = frame["source_freshness"].astype(str)
    frame["source_freshness"] = frame["source_freshness"].where(
        ~freshness.isin(("", "nan", "None")), frame["forecast_period"].astype(str))
    frame["model_version"] = model_version
    frame["target_observation_type"] = "forecast"
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["target"] = frame["target"].astype(str)
    keys = ["forecast_period", "target", "geography_level", "geography_id",
            "county_fips", "drug_key", "therapeutic_class", "labeler", "supplier"]
    if frame.duplicated(keys).any():
        raise ValueError("metric output contains duplicate filterable keys")
    frame = _apply_target_metadata(frame)
    unknown = sorted(set(frame["target"].astype(str)) - set(TARGET_METADATA))
    if unknown:
        raise ValueError(f"metric rows contain unknown targets: {unknown}")
    cadence_horizon = {"weekly": 7, "monthly": 30, "quarterly": 91, "annual": 365}
    inferred_horizon = frame["target_cadence"].map(cadence_horizon)
    frame["horizon"] = frame["horizon"].where(
        frame["horizon"].notna() & frame["horizon"].gt(0), inferred_horizon)
    frame["horizon"] = frame["horizon"].astype(int)
    if validate:
        validate_metric_rows(frame)
    return frame[UNIVERSAL_OUTPUT_COLUMNS]


def write_metric_rows(metrics: pd.DataFrame, path, *, model_version: str,
                      forecast_timestamp: str | None = None) -> pd.DataFrame:
    """Write a compressed CSV after applying the public metric contract."""
    result = build_metric_rows(metrics, model_version=model_version,
                               forecast_timestamp=forecast_timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, compression="gzip")
    return result
