"""Arkansas county overdose-pressure candidate from CDC VSRR data.

The CDC source is a provisional rolling-12-month death count, not a pharmacy
dispensing or medication-specific label. Suppressed county-month rows are
absent from the public response and must remain missing. This module therefore
only evaluates consecutive observed transitions and reports the candidate as a
medical context signal unless it clears the normal promotion gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge


FEATURES = (
    "provisional_drug_overdose", "lag1_overdose", "lag2_overdose",
    "rolling3_overdose",
)


def build_overdose_observations(source: pd.DataFrame, *, enforce_vintage: bool = False) -> pd.DataFrame:
    """Normalize observed rows, optionally requiring historical source vintages."""
    required = {"st_abbrev", "fips", "monthendingdate", "provisional_drug_overdose"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"overdose source missing columns: {sorted(missing)}")
    frame = source.copy()
    frame = frame[frame["st_abbrev"].astype(str).str.upper().eq("AR")].copy()
    frame["county_fips"] = frame["fips"].astype(str).str.replace(r"\\D", "", regex=True).str.zfill(5)
    frame["period"] = pd.to_datetime(frame["monthendingdate"], errors="coerce").dt.to_period("M")
    frame["value"] = pd.to_numeric(frame["provisional_drug_overdose"], errors="coerce")
    frame = frame.dropna(subset=["county_fips", "period", "value"])
    if "data_as_of" in frame:
        frame["data_as_of"] = pd.to_datetime(frame["data_as_of"], errors="coerce")
        if enforce_vintage:
            # A current snapshot cannot be reused as if it were known at every
            # historical origin. Keep only observations whose vintage was
            # available by the end of that observation month.
            frame = frame[frame["data_as_of"].le(
                frame["period"].dt.to_timestamp(how="end"))]
    aggregations = {"value": "max"}
    if "data_as_of" in frame:
        aggregations["data_as_of"] = "max"
    frame = frame.groupby(["county_fips", "period"], as_index=False).agg(aggregations)
    frame = frame.rename(columns={"value": "provisional_drug_overdose"})
    return frame.sort_values(["county_fips", "period"]).reset_index(drop=True)


def build_overdose_view(source: pd.DataFrame, *, enforce_vintage: bool = True) -> pd.DataFrame:
    """Normalize Arkansas VSRR rows into leakage-safe county-month transitions."""
    frame = build_overdose_observations(source, enforce_vintage=enforce_vintage)
    grouped = frame.groupby("county_fips", sort=False)
    frame["next_period"] = grouped["period"].shift(-1)
    frame["target_overdose"] = grouped["provisional_drug_overdose"].shift(-1)
    frame["lag1_overdose"] = grouped["provisional_drug_overdose"].shift(1)
    frame["lag2_overdose"] = grouped["provisional_drug_overdose"].shift(2)
    frame["rolling3_overdose"] = grouped["provisional_drug_overdose"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    # A missing public row can represent suppression, not zero. Do not bridge it.
    frame = frame[frame["next_period"].eq(frame["period"] + 1)].copy()
    return frame.reset_index(drop=True)


def _state(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), [lower, upper]).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for value in range(3):
        mask = actual == value
        recalls.append(float(np.mean(predicted[mask] == value)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(value): int(np.sum(actual == value)) for value in range(3)},
    }


def _predict(train: pd.DataFrame, target: pd.DataFrame,
             lower: float, upper: float, alpha: float) -> np.ndarray:
    labels = _state(train["target_overdose"], lower, upper)
    probabilities = []
    for value in range(3):
        model = LogisticRidge(alpha=alpha).fit(
            train[list(FEATURES)].fillna(0).to_numpy(float),
            (labels == value).astype(float), list(FEATURES))
        probabilities.append(model.predict_proba(
            target[list(FEATURES)].fillna(0).to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_overdose_pressure(view: pd.DataFrame, *, min_train_months: int = 24,
                               min_rows_per_county: int = 25) -> dict[str, Any]:
    """Run rolling-origin state evaluation and retain county coverage evidence."""
    if view.empty:
        return {
            "protocol": "rolling_origin_next_month_arkansas_county_overdose_pressure_state",
            "fold_count": 0, "test_rows": 0, "county_count": 0,
            "county_count_with_minimum_rows": 0,
            "minimum_rows_per_county": min_rows_per_county,
            "mean_model_accuracy": None, "mean_model_balanced_accuracy": None,
            "mean_persistence_accuracy": None, "state_counts_in_scored_rows": {},
            "per_county_accuracy": {}, "publishable_candidate": False,
            "scope": "Arkansas county provisional rolling-12-month overdose pressure; not pharmacy dispensing",
            "state_definition": {"0": "low", "1": "mid", "2": "high"},
            "vintage_status": "historical_vintages_unavailable",
            "rejection_reasons": [
                "available source snapshot has no historical publication vintages",
                "chronological next-month testing cannot be reconstructed without vintage data",
            ],
            "folds": [],
        }
    periods = sorted(view["period"].unique())
    folds = []
    county_correct: dict[str, int] = {}
    county_rows: dict[str, int] = {}
    for test_period in periods:
        validation_period = test_period - 1
        train = view[view["period"] < validation_period]
        validation = view[view["period"].eq(validation_period)]
        test = view[view["period"].eq(test_period)]
        if (len(train["period"].unique()) < min_train_months or
                validation.empty or test.empty):
            continue
        lower, upper = np.quantile(train["target_overdose"], [1 / 3, 2 / 3])
        validation_actual = _state(validation["target_overdose"], lower, upper)
        candidates = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidates[alpha] = _metrics(
                validation_actual, _predict(train, validation, lower, upper, alpha)
            )["balanced_accuracy"]
        alpha = max(candidates, key=candidates.get)
        actual = _state(test["target_overdose"], lower, upper)
        learned = _predict(pd.concat([train, validation]), test, lower, upper, alpha)
        persistence = _state(test["provisional_drug_overdose"], lower, upper)
        persistence_validation = _state(validation["provisional_drug_overdose"], lower, upper)
        selected = (persistence if _metrics(validation_actual, persistence_validation)[
            "balanced_accuracy"] >= candidates[alpha] else learned)
        method = "persistence" if selected is persistence else "logistic_one_vs_rest"
        model_metrics = _metrics(actual, selected)
        persistence_metrics = _metrics(actual, persistence)
        for county, truth, prediction in zip(test["county_fips"], actual, selected):
            county = str(county)
            county_rows[county] = county_rows.get(county, 0) + 1
            county_correct[county] = county_correct.get(county, 0) + int(truth == prediction)
        folds.append({
            "validation_period": str(validation_period), "test_period": str(test_period),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_model": method,
            "thresholds": {"low_mid": float(lower), "mid_high": float(upper)},
            "model": model_metrics, "persistence": persistence_metrics,
        })
    if not folds:
        raise ValueError("no eligible county overdose folds")
    county_accuracy = {county: county_correct[county] / county_rows[county]
                       for county in sorted(county_rows)}
    eligible_counties = sum(count >= min_rows_per_county for count in county_rows.values())
    accuracy = float(np.mean([fold["model"]["accuracy"] for fold in folds]))
    balanced = float(np.mean([fold["model"]["balanced_accuracy"] for fold in folds]))
    counts = {str(state): sum(fold["model"]["state_counts"][str(state)] for fold in folds)
              for state in range(3)}
    return {
        "protocol": "rolling_origin_next_month_arkansas_county_overdose_pressure_state",
        "fold_count": len(folds), "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "county_count": len(county_accuracy), "county_count_with_minimum_rows": eligible_counties,
        "minimum_rows_per_county": min_rows_per_county,
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "mean_persistence_accuracy": float(np.mean(
            [fold["persistence"]["accuracy"] for fold in folds])),
        "state_counts_in_scored_rows": counts, "per_county_accuracy": county_accuracy,
        "publishable_candidate": bool(accuracy >= 0.65 and balanced >= 0.65 and
                                       eligible_counties > 0 and all(counts.values())),
        "scope": "Arkansas county provisional rolling-12-month overdose pressure; not pharmacy dispensing",
        "vintage_status": "historical_vintages_available",
        "state_definition": {"0": "low", "1": "mid", "2": "high"},
        "source_limitations": [
            "suppressed county-month rows are excluded, not imputed as zero",
            "the public value is a rolling-12-month provisional count",
            "the target has no medication, supplier, or pharmacy-fill identity",
        ],
        "folds": folds,
    }
