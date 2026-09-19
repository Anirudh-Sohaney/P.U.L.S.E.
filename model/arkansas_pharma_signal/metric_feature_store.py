"""Grain-preserving adapter from qualified forecast rows to model features."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .external_metric_output import PROMOTED_STATUSES, validate_metric_rows
from .universal_forecast import TARGET_METADATA


DEFAULT_KEYS = ["forecast_period", "geography_level", "geography_id",
                "county_fips", "drug_key", "labeler", "supplier"]


def _feature_name(target: str) -> str:
    return "metric__" + re.sub(r"[^a-z0-9]+", "_", str(target).lower()).strip("_")


def metric_context_vector(row: pd.Series | dict, metadata: dict) -> np.ndarray:
    """Encode an attached feature row for the modular model.

    The first half contains feature values in the metadata order and the
    second half contains one explicit missingness indicator per feature. No
    missing value is converted to a substantive zero without its indicator.
    """
    features = list(metadata.get("feature_columns", metadata.get("features", {})))
    values, missing = [], []
    for feature in features:
        value = row.get(feature, pd.NA)
        is_missing = pd.isna(value)
        values.append(0.0 if is_missing else float(value))
        marker = row.get(f"{feature}__missing", int(is_missing))
        missing.append(float(marker) if not pd.isna(marker) else 1.0)
    return np.asarray([*values, *missing], dtype=np.float32)


def build_metric_feature_store(rows: pd.DataFrame,
                               keys: Iterable[str] = DEFAULT_KEYS) -> tuple[pd.DataFrame, dict]:
    """Pivot qualified metric rows without changing their declared grain."""
    keys = list(keys)
    required = set(keys) | {"target", "prediction", "target_promotion_status",
                            "target_semantics", "source_freshness", "driver_attribution"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"metric rows missing feature-store columns: {sorted(missing)}")
    frame = rows.copy()
    # CSV readers commonly reinterpret intentionally empty optional allocation
    # fields as NaN. Normalize only those fields; period/geography/target keys
    # remain strict and continue to fail closed when absent.
    for column in ("county_fips", "drug_key", "labeler", "supplier"):
        if column in keys:
            frame[column] = frame[column].fillna("").astype(str)
    if frame[keys + ["target"]].isna().any().any():
        raise ValueError("feature-store keys and target cannot be null")
    if frame.duplicated(keys + ["target"]).any():
        raise ValueError("duplicate metric grain before feature-store pivot")
    status = frame["target_promotion_status"].astype(str)
    if (~status.isin(PROMOTED_STATUSES)).any():
        bad = sorted(status[~status.isin(PROMOTED_STATUSES)].unique())
        raise ValueError(f"unqualified metrics cannot enter the feature store: {bad}")
    # Direct callers may bypass the serializer, so reapply the remaining
    # promotion contract before any metric becomes a regression feature.
    validate_metric_rows(frame)
    frame["feature_name"] = frame["target"].map(_feature_name)
    collisions = (frame[["target", "feature_name"]].drop_duplicates()
                  .groupby("feature_name")["target"].nunique())
    if collisions.gt(1).any():
        raise ValueError("metric target names collide after feature normalization")
    values = (frame.pivot(index=keys, columns="feature_name", values="prediction")
              .reset_index())
    values.columns.name = None
    metadata = {}
    for name, group in frame.groupby("feature_name", sort=False):
        first = group.iloc[0]
        metadata[name] = {
            "target": str(first["target"]),
            "cadence": str(first.get("target_cadence", "")),
            "source_freshness": sorted(group["source_freshness"].astype(str).unique()),
            "semantics": str(first["target_semantics"]),
            "promotion_status": str(first["target_promotion_status"]),
            "uncertainty_status": str(first.get("uncertainty_status", "")),
            "calibration_status": str(first.get("calibration_status", "")),
            "provenance_examples": group["driver_attribution"].astype(str).head(3).tolist(),
            "context_projection": TARGET_METADATA.get(str(first["target"]), {}).get(
                "context_projection"),
        }
    return values, {"keys": keys, "features": metadata,
                    "feature_count": len(metadata),
                    "missing_values_are_not_imputed": True}


def join_metric_features(base: pd.DataFrame, feature_store: pd.DataFrame,
                         keys: Iterable[str] = DEFAULT_KEYS,
                         *, how: str = "left") -> pd.DataFrame:
    """Join only matching metric grains; no geography or drug broadcasting."""
    keys = list(keys)
    missing = set(keys).difference(base.columns) | set(keys).difference(feature_store.columns)
    if missing:
        raise ValueError(f"join keys missing from feature frames: {sorted(missing)}")
    if feature_store.duplicated(keys).any():
        raise ValueError("feature store has duplicate join keys")
    result = base.merge(feature_store, on=keys, how=how, validate="many_to_one")
    return result


def project_metric_features(base: pd.DataFrame, feature_store: pd.DataFrame,
                            metadata: dict) -> tuple[pd.DataFrame, dict]:
    """Apply only target-declared context projections to a local base frame.

    Projection is intentionally separate from exact-grain joining. Missing
    destination geography keys skip only the incompatible feature and expose
    all-one missingness, rather than inferring geography or broadcasting it.
    """
    result = base.copy()
    projected, skipped = [], {}
    for feature_name, feature_meta in metadata.get("features", {}).items():
        policy = feature_meta.get("context_projection")
        if not policy:
            skipped[feature_name] = "target has no declared context projection"
            continue
        source_keys = list(policy["source_keys"])
        destination_keys = list(policy["destination_keys"])
        absent_source = sorted(set(source_keys) - set(feature_store.columns))
        absent_destination = sorted(set(destination_keys) - set(result.columns))
        if absent_source:
            raise ValueError(f"projection source keys missing: {absent_source}")
        if absent_destination:
            result[feature_name] = pd.NA
            result[f"{feature_name}__missing"] = 1
            skipped[feature_name] = f"destination keys missing: {absent_destination}"
            continue
        source = feature_store[source_keys + [feature_name]].copy()
        # The wide store has one row for every target grain. Keep only rows
        # where this feature exists before checking projection uniqueness.
        source = source[source[feature_name].notna()].copy()
        if source.duplicated(source_keys).any():
            raise ValueError(f"projection source is not unique at declared grain: {feature_name}")
        rename = dict(zip(source_keys, destination_keys))
        source = source.rename(columns=rename)
        result = result.merge(source, on=destination_keys, how="left",
                              validate="many_to_one")
        result[f"{feature_name}__missing"] = result[feature_name].isna().astype("int8")
        projected.append({"feature": feature_name, "scope": policy["scope"],
                          "source_keys": source_keys,
                          "destination_keys": destination_keys})
    return result, {"projection_mode": "context", "projected_features": projected,
                    "skipped_features": skipped,
                    "values_are_not_imputed": True}


def attach_qualified_metric_features(
    base: pd.DataFrame,
    rows: pd.DataFrame,
    keys: Iterable[str] = DEFAULT_KEYS,
    *,
    how: str = "left",
    join_mode: str = "exact",
) -> tuple[pd.DataFrame, dict]:
    """Attach qualified metrics to a training frame at exactly its declared grain.

    The function is the integration seam for downstream pharmacy regressions.
    It does not fill missing forecasts or relax geography/drug/supplier keys;
    instead it adds explicit missingness indicators for each metric feature so
    the consuming model can learn coverage rather than treating absence as a
    zero signal.
    """
    if join_mode not in {"exact", "context"}:
        raise ValueError("join_mode must be 'exact' or 'context'")
    feature_store, metadata = build_metric_feature_store(rows, keys=keys)
    feature_names = list(metadata["features"])
    if join_mode == "exact":
        joined = join_metric_features(base, feature_store, keys=keys, how=how)
        for feature_name in feature_names:
            joined[f"{feature_name}__missing"] = joined[feature_name].isna().astype("int8")
        projection_metadata = {"projection_mode": "exact", "projected_features": [],
                              "skipped_features": {}}
    else:
        joined, projection_metadata = project_metric_features(base, feature_store, metadata)
    metadata = dict(metadata)
    metadata["attached_rows"] = int(len(joined))
    metadata["feature_columns"] = feature_names
    metadata["missingness_columns"] = [f"{name}__missing" for name in feature_names]
    metadata["model_context_order"] = [*feature_names, *metadata["missingness_columns"]]
    metadata["model_context_width"] = 2 * len(feature_names)
    metadata["values_are_not_imputed"] = True
    metadata.update(projection_metadata)
    return joined, metadata
