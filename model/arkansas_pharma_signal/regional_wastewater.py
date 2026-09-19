"""Weekly Arkansas-region wastewater pressure proxy.

CDC wastewater observations are attached to named counties served by a site,
not to pharmacy locations. This adapter uses only unambiguous single-county
service labels, maps those counties through the existing Arkansas DHS region
crosswalk, and keeps the resulting signal explicitly proxy-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .event_accuracy import score_event_predictions
from .wastewater_pressure import (
    FEATURES, _metrics, evaluate_wastewater_state_rolling,
)


# County names in the CDC service-area field are normalized against the Census
# county naming convention. The corresponding region assignment is the same
# crosswalk used by geography.REGION_COUNTIES.
COUNTY_REGION = {
    "benton": "northwest", "boone": "northwest", "garland": "central",
    "greene": "northeast", "hempstead": "southwest", "jefferson": "southeast",
    "lonoke": "central", "mississippi": "northeast", "ouachita": "southwest",
    "pope": "northwest", "pulaski": "central", "saline": "central",
    "saint francis": "southeast", "st francis": "southeast",
    "washington": "northwest", "white": "northeast",
}


def _county_key(value: object) -> str:
    value = str(value or "").strip().casefold()
    for suffix in (" county", ", ar", ", arkansas"):
        if value.endswith(suffix):
            value = value[:-len(suffix)].strip()
    return " ".join(value.split())


def build_regional_wastewater_weekly_view(
    source: pd.DataFrame, *, pathogen: str,
) -> pd.DataFrame:
    """Build consecutive region-week rows with population-weighted WVAL."""
    required = {"week_end", "pathogen_target", "site_wval", "counties_served"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"wastewater data missing columns: {sorted(missing)}")
    frame = source.copy()
    frame = frame[frame["pathogen_target"].astype(str).eq(pathogen)].copy()
    frame["region"] = frame["counties_served"].map(
        lambda value: COUNTY_REGION.get(_county_key(value), ""))
    frame = frame[frame["region"].ne("")].copy()
    frame["week"] = pd.to_datetime(frame["week_end"], errors="coerce").dt.to_period("W-SAT")
    frame["value"] = pd.to_numeric(frame["site_wval"], errors="coerce")
    frame["population"] = pd.to_numeric(frame.get("population_served", 1.0), errors="coerce")
    frame["population"] = frame["population"].fillna(1.0).clip(lower=1.0)
    frame = frame.dropna(subset=["week", "value"])
    if frame.empty:
        raise ValueError(f"no usable regional wastewater rows for pathogen {pathogen!r}")
    frame["weighted_value"] = frame["value"] * frame["population"]
    grouped = (frame.groupby(["region", "week"], as_index=False)
               .agg(weighted_value=("weighted_value", "sum"),
                    population=("population", "sum"),
                    site_count=("site", "nunique")))
    grouped["value"] = grouped["weighted_value"] / grouped["population"]
    grouped = grouped.sort_values(["region", "week"])
    grouped["next_week"] = grouped.groupby("region")["week"].shift(-1)
    grouped["target"] = grouped.groupby("region")["value"].shift(-1)
    grouped["target"] = grouped["target"].where(
        grouped["next_week"].eq(grouped["week"] + 1))
    grouped = grouped.dropna(subset=["target"]).copy()
    grouped["current_value"] = grouped["value"]
    for lag in (1, 2, 4):
        grouped[f"lag{lag}_value"] = grouped.groupby("region")["value"].shift(lag)
    grouped["lag1_value"] = grouped["lag1_value"].fillna(grouped["current_value"])
    grouped["lag2_value"] = grouped["lag2_value"].fillna(grouped["current_value"])
    grouped["lag4_value"] = grouped["lag4_value"].fillna(grouped["current_value"])
    grouped["rolling4_value"] = (
        grouped.groupby("region")["value"].shift(1).groupby(grouped["region"])
        .rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        .fillna(grouped["current_value"])
    )
    week_number = grouped["week"].dt.week.astype(float)
    grouped["week_sin"] = np.sin(2 * np.pi * week_number / 52.0)
    grouped["week_cos"] = np.cos(2 * np.pi * week_number / 52.0)
    grouped["year"] = grouped["week"].dt.year.astype(int)
    return grouped.reset_index(drop=True)


def evaluate_regional_wastewater_state_rolling(
    view: pd.DataFrame, *, min_train_rows: int = 52,
    validation_rows: int = 13, test_rows: int = 13, step_rows: int = 13,
) -> dict[str, Any]:
    """Evaluate next-week five-state pressure separately by region."""
    required = {"region", "week", "target", "current_value"}
    missing = required.difference(view.columns)
    if missing:
        raise ValueError(f"regional wastewater view missing columns: {sorted(missing)}")
    regional = {}
    for region, rows in view.groupby("region", sort=True):
        result = evaluate_wastewater_state_rolling(
            rows.sort_values("week").reset_index(drop=True),
            min_train_rows=min_train_rows, validation_rows=validation_rows,
            test_rows=test_rows, step_rows=step_rows)
        if result.get("fold_count", 0):
            regional[str(region)] = result
    if not regional:
        return {"fold_count": 0, "publishable_candidate": False, "regions": {}}
    test_rows_total = sum(int(item["test_rows"]) for item in regional.values())
    weighted = lambda key: float(sum(
        item.get(key, 0.0) * item["test_rows"] for item in regional.values()
    ) / max(test_rows_total, 1))
    state_counts = {str(state): sum(
        int(item.get("state_counts_in_scored_rows", {}).get(str(state), 0))
        for item in regional.values()) for state in range(5)}
    predicted_events = sum(item["event_metrics"].get("predicted_event_count", 0)
                           for item in regional.values())
    true_positives = sum(item["event_metrics"].get("true_positive_count", 0)
                         for item in regional.values())
    actual_events = sum(item["event_metrics"].get("actual_event_count", 0)
                        for item in regional.values())
    event_metrics = {
        "event_applicable": True,
        "event_definition": {"kind": "state", "event_states": [3, 4]},
        "predicted_event_count": int(predicted_events),
        "true_positive_count": int(true_positives),
        "true_positive_precision": (
            float(true_positives / predicted_events) if predicted_events else None),
        "actual_event_count": int(actual_events),
    }
    return {
        "protocol": "rolling_origin_next_week_arkansas_region_wastewater_five_state",
        "fold_count": int(sum(item["fold_count"] for item in regional.values())),
        "test_rows": int(test_rows_total),
        "region_count": len(regional), "regions": sorted(regional),
        "mean_model_accuracy": weighted("mean_model_accuracy"),
        "mean_model_balanced_accuracy": weighted("mean_model_balanced_accuracy"),
        "mean_persistence_accuracy": weighted("mean_persistence_accuracy"),
        "state_counts_in_scored_rows": state_counts,
        "event_metrics": event_metrics,
        "state_count": 5,
        "publishable_candidate": bool(
            len(regional) >= 3 and test_rows_total >= 25
            and weighted("mean_model_accuracy") >= 0.65
            and weighted("mean_model_balanced_accuracy") >= 0.65
            and all(state_counts.values())),
        "scope": "Arkansas region wastewater viral-activity proxy; not pharmacy dispensing",
        "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
        "per_region": regional,
    }


def build_regional_wastewater_change_view(level_view: pd.DataFrame) -> pd.DataFrame:
    """Convert a level view to a next-week five-state change target."""
    required = {"region", "week", "target", "current_value"}
    missing = required.difference(level_view.columns)
    if missing:
        raise ValueError(f"regional wastewater view missing columns: {sorted(missing)}")
    view = level_view.copy()
    view["target_change"] = view["target"].astype(float) - view["current_value"].astype(float)
    return view


def _predict_change(train: pd.DataFrame, target: pd.DataFrame,
                    thresholds: tuple[float, ...], alpha: float) -> np.ndarray:
    labels = np.digitize(train["target_change"].to_numpy(float), thresholds).astype(int)
    probabilities = []
    for state in range(5):
        from .regression import LogisticRidge
        model = LogisticRidge(alpha=alpha).fit(
            train[list(FEATURES)].to_numpy(float), (labels == state).astype(float),
            list(FEATURES))
        probabilities.append(model.predict_proba(
            target[list(FEATURES)].to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_regional_wastewater_change_state_rolling(
    view: pd.DataFrame, *, min_train_rows: int = 52,
    validation_rows: int = 13, test_rows: int = 13, step_rows: int = 13,
) -> dict[str, Any]:
    """Evaluate five ordered states of next-week wastewater change."""
    required = {"region", "week", "target_change"}
    missing = required.difference(view.columns)
    if missing:
        raise ValueError(f"regional wastewater change view missing columns: {sorted(missing)}")
    regional = {}
    for region, rows in view.groupby("region", sort=True):
        periods = sorted(rows["week"].unique())
        folds = []
        event_actual, event_predicted = [], []
        for index in range(min_train_rows, len(periods) - validation_rows - test_rows + 1,
                           max(1, step_rows)):
            train = rows[rows["week"].isin(periods[:index])]
            validation = rows[rows["week"].isin(periods[index:index + validation_rows])]
            test = rows[rows["week"].isin(
                periods[index + validation_rows:index + validation_rows + test_rows])]
            if train.empty or validation.empty or test.empty:
                continue
            thresholds = tuple(np.quantile(train["target_change"], np.arange(1, 5) / 5))
            validation_actual = np.digitize(validation["target_change"], thresholds)
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    validation_actual, _predict_change(train, validation, thresholds, alpha)
                )["balanced_accuracy"]
            alpha = max(scores, key=scores.get)
            actual = np.digitize(test["target_change"], thresholds)
            learned = _predict_change(pd.concat([train, validation]), test, thresholds, alpha)
            persistence_state = int(np.digitize(0.0, thresholds))
            persistence = np.full(len(test), persistence_state, dtype=int)
            validation_persistence = np.full(len(validation), persistence_state, dtype=int)
            selected = (persistence if _metrics(
                validation_actual, validation_persistence)["balanced_accuracy"] >= scores[alpha]
                else learned)
            event_actual.extend(actual.tolist())
            event_predicted.extend(selected.tolist())
            folds.append({
                "validation_start": str(validation["week"].min()),
                "validation_end": str(validation["week"].max()),
                "test_start": str(test["week"].min()), "test_end": str(test["week"].max()),
                "train_rows": int(len(train)), "validation_rows": int(len(validation)),
                "test_rows": int(len(test)), "selected_alpha": float(alpha),
                "selected_model": "persistence_zero_change" if selected is persistence
                else "logistic_one_vs_rest",
                "thresholds": [float(value) for value in thresholds],
                "model": _metrics(actual, selected),
                "persistence": _metrics(actual, persistence),
            })
        if folds:
            regional[str(region)] = {
                "fold_count": len(folds), "test_rows": sum(x["test_rows"] for x in folds),
                "mean_model_accuracy": float(np.mean([x["model"]["accuracy"] for x in folds])),
                "mean_model_balanced_accuracy": float(np.mean([
                    x["model"]["balanced_accuracy"] for x in folds])),
                "mean_persistence_accuracy": float(np.mean([
                    x["persistence"]["accuracy"] for x in folds])),
                "state_counts_in_scored_rows": {
                    str(state): sum(x["model"]["state_counts"][str(state)] for x in folds)
                    for state in range(5)},
                "event_metrics": score_event_predictions(
                    np.asarray(event_actual), np.asarray(event_predicted), kind="state",
                    event_states={3, 4}),
                "folds": folds,
            }
    if not regional:
        return {"fold_count": 0, "publishable_candidate": False, "regions": {}}
    total = sum(x["test_rows"] for x in regional.values())
    weighted = lambda key: float(sum(
        x[key] * x["test_rows"] for x in regional.values()) / max(total, 1))
    counts = {str(state): sum(x["state_counts_in_scored_rows"][str(state)]
                              for x in regional.values()) for state in range(5)}
    predicted = sum(x["event_metrics"]["predicted_event_count"] for x in regional.values())
    true_positive = sum(x["event_metrics"]["true_positive_count"] for x in regional.values())
    actual_event = sum(x["event_metrics"]["actual_event_count"] for x in regional.values())
    return {
        "protocol": "rolling_origin_next_week_arkansas_region_wastewater_change_five_state",
        "fold_count": int(sum(x["fold_count"] for x in regional.values())),
        "test_rows": int(total), "region_count": len(regional),
        "regions": sorted(regional), "mean_model_accuracy": weighted("mean_model_accuracy"),
        "mean_model_balanced_accuracy": weighted("mean_model_balanced_accuracy"),
        "mean_persistence_accuracy": weighted("mean_persistence_accuracy"),
        "state_counts_in_scored_rows": counts,
        "event_metrics": {
            "event_applicable": True,
            "event_definition": {"kind": "state", "event_states": [3, 4]},
            "predicted_event_count": int(predicted),
            "true_positive_count": int(true_positive),
            "true_positive_precision": float(true_positive / predicted) if predicted else None,
            "actual_event_count": int(actual_event),
        },
        "state_count": 5,
        "publishable_candidate": bool(
            len(regional) >= 3 and total >= 25 and weighted("mean_model_accuracy") >= 0.65
            and weighted("mean_model_balanced_accuracy") >= 0.65 and all(counts.values())),
        "scope": "Arkansas region wastewater week-over-week change proxy; not pharmacy dispensing",
        "state_definition": {"0": "strong_decrease", "1": "decrease", "2": "stable",
                             "3": "increase", "4": "strong_increase"},
        "per_region": regional,
    }
