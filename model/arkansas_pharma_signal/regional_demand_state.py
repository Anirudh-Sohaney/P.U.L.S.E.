"""Annual Arkansas-region demand-state metric from public CMS demand data.

This is a lower-frequency context metric, not a substitute for weekly
inventory labels. County observations are aggregated to the published five
Arkansas DHS/TEFRA regions and drug, then the next year's demand is classified
as low, mid, or high using cut points learned from the training partition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge, RidgeLinear
from .event_accuracy import score_event_predictions


REGIONS = ("central", "northeast", "northwest", "southeast", "southwest")
FEATURES = ("demand_claims", "lag1_demand", "lag2_demand", "rolling3_demand")


def build_regional_demand_view(
        county_demand: pd.DataFrame,
        *, allowed_regions: tuple[str, ...] = REGIONS) -> pd.DataFrame:
    required = {"year", "arkansas_region", "drug_key", "demand_claims"}
    missing = required.difference(county_demand.columns)
    if missing:
        raise ValueError(f"county demand missing columns: {sorted(missing)}")
    frame = county_demand.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["arkansas_region"] = frame["arkansas_region"].astype(str).str.lower().str.strip()
    frame["drug_key"] = frame["drug_key"].astype(str).str.strip()
    frame["demand_claims"] = pd.to_numeric(frame["demand_claims"], errors="raise")
    frame = (frame.groupby(["year", "arkansas_region", "drug_key"], as_index=False)
             ["demand_claims"].sum())
    frame = frame[frame["arkansas_region"].isin(allowed_regions)].copy()
    frame = frame.sort_values(["arkansas_region", "drug_key", "year"])
    grouped = frame.groupby(["arkansas_region", "drug_key"], sort=False)
    frame["next_year"] = grouped["year"].shift(-1)
    frame["target_demand_claims"] = grouped["demand_claims"].shift(-1)
    frame["lag1_demand"] = grouped["demand_claims"].shift(1)
    frame["lag2_demand"] = grouped["demand_claims"].shift(2)
    frame["rolling3_demand"] = grouped["demand_claims"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    frame = frame[frame["next_year"].eq(frame["year"] + 1)].copy()
    return frame.reset_index(drop=True)


def build_state_demand_view(state_demand: pd.DataFrame) -> pd.DataFrame:
    """Normalize Arkansas state-by-drug Part D rows to the shared demand view."""
    required = {"year", "state", "drug_key", "demand_claims"}
    missing = required.difference(state_demand.columns)
    if missing:
        raise ValueError(f"state demand missing columns: {sorted(missing)}")
    frame = state_demand.rename(columns={"state": "arkansas_region"}).copy()
    frame["arkansas_region"] = frame["arkansas_region"].astype(str).str.lower().str.strip()
    return build_regional_demand_view(frame, allowed_regions=("arkansas",))


def _state(values: np.ndarray, *thresholds: float) -> np.ndarray:
    """Map values to states using thresholds learned from the fit partition."""
    return np.digitize(np.asarray(values, dtype=float), thresholds).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray, state_count: int = 3) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recall = []
    for value in range(state_count):
        mask = actual == value
        recall.append(float(np.mean(predicted[mask] == value)) if mask.any() else 0.0)
    return {"accuracy": float(np.mean(actual == predicted)),
            "balanced_accuracy": float(np.mean(recall)),
            "state_counts": {str(x): int(np.sum(actual == x)) for x in range(state_count)}}


def _predict(train: pd.DataFrame, target: pd.DataFrame,
             thresholds: tuple[float, ...], alpha: float,
             state_count: int = 3) -> np.ndarray:
    train_state = _state(train["target_demand_claims"].to_numpy(), *thresholds)
    probabilities = []
    for value in range(state_count):
        model = LogisticRidge(alpha=alpha).fit(
            train[list(FEATURES)].fillna(0).to_numpy(float),
            (train_state == value).astype(float), list(FEATURES))
        probabilities.append(model.predict_proba(
            target[list(FEATURES)].fillna(0).to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_regional_demand_state(
        view: pd.DataFrame, min_train_years: int = 4, state_count: int = 3,
        *, group_column: str = "arkansas_region",
        required_groups: tuple[str, ...] = REGIONS,
        protocol_label: str = "arkansas_region") -> dict[str, Any]:
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    if group_column not in view:
        raise ValueError(f"demand view missing grouping column: {group_column}")
    years = sorted(view["year"].unique())
    folds = []
    event_actual: list[int] = []
    event_predicted: list[int] = []
    per_region_correct: dict[str, int] = {}
    per_region_rows: dict[str, int] = {}
    per_drug_correct: dict[str, int] = {}
    per_drug_rows: dict[str, int] = {}
    for test_year in years:
        validation_year = test_year - 1
        train = view[view["year"] < validation_year]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if len(train["year"].unique()) < min_train_years or validation.empty or test.empty:
            continue
        thresholds = tuple(np.quantile(
            train["target_demand_claims"],
            np.arange(1, state_count) / state_count,
        ))
        validation_actual = _state(validation["target_demand_claims"], *thresholds)
        candidates = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidates[alpha] = _metrics(
                validation_actual, _predict(train, validation, thresholds, alpha, state_count),
                state_count,
            )["balanced_accuracy"]
        alpha = max(candidates, key=candidates.get)
        actual = _state(test["target_demand_claims"], *thresholds)
        learned = _predict(pd.concat([train, validation]), test, thresholds, alpha, state_count)
        persistence = _state(test["demand_claims"], *thresholds)
        persistence_validation = _state(validation["demand_claims"], *thresholds)
        selected = (persistence if _metrics(
            validation_actual, persistence_validation, state_count
        )[
            "balanced_accuracy"] >= candidates[alpha] else learned)
        method = "persistence" if selected is persistence else "logistic_one_vs_rest"
        model_metrics = _metrics(actual, selected, state_count)
        persistence_metrics = _metrics(actual, persistence, state_count)
        event_actual.extend(actual.tolist())
        event_predicted.extend(selected.tolist())
        for group, truth, prediction in zip(test[group_column], actual, selected):
            per_region_rows[group] = per_region_rows.get(group, 0) + 1
            per_region_correct[group] = per_region_correct.get(group, 0) + int(truth == prediction)
        for drug, truth, prediction in zip(test["drug_key"], actual, selected):
            drug = str(drug)
            per_drug_rows[drug] = per_drug_rows.get(drug, 0) + 1
            per_drug_correct[drug] = per_drug_correct.get(drug, 0) + int(truth == prediction)
        folds.append({"validation_year": int(validation_year), "test_year": int(test_year),
                      "train_rows": int(len(train)), "validation_rows": int(len(validation)),
                      "test_rows": int(len(test)), "selected_model": method,
            "thresholds": {str(i): float(value) for i, value in enumerate(thresholds)},
                      "model": model_metrics, "persistence": persistence_metrics})
    if not folds:
        raise ValueError("no eligible regional demand folds")
    region_accuracy = {region: per_region_correct[region] / per_region_rows[region]
                       for region in sorted(per_region_rows)}
    per_drug = {
        drug: {"test_rows": per_drug_rows[drug],
               "accuracy": per_drug_correct[drug] / per_drug_rows[drug]}
        for drug in sorted(per_drug_rows)
    }
    accuracy = float(np.mean([x["model"]["accuracy"] for x in folds]))
    balanced = float(np.mean([x["model"]["balanced_accuracy"] for x in folds]))
    counts = {str(state): sum(x["model"]["state_counts"][str(state)] for x in folds)
              for state in range(state_count)}
    return {
        "protocol": f"rolling_origin_next_year_{protocol_label}_drug_demand_{state_count}_state",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "region_count": len(region_accuracy), "group_count": len(region_accuracy),
        "drug_count": int(view["drug_key"].nunique()),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "mean_persistence_accuracy": float(np.mean([x["persistence"]["accuracy"] for x in folds])),
        "state_counts_in_scored_rows": counts, "per_region_accuracy": region_accuracy,
        "per_group_accuracy": region_accuracy,
        "per_drug": per_drug,
        "event_metrics": score_event_predictions(
            np.asarray(event_actual), np.asarray(event_predicted), kind="state",
            event_states=set(range(max(0, state_count - 2), state_count))),
        "state_count": state_count,
        "publishable_candidate": bool(accuracy >= 0.65 and balanced >= 0.65
                                       and set(region_accuracy) == set(required_groups)
                                       and all(counts.values())),
        "scope": f"annual CMS Part D demand state by {protocol_label} and drug; not weekly inventory",
        "state_definition": {str(i): f"quantile_{i + 1}_of_{state_count}"
                             for i in range(state_count)}, "folds": folds,
    }


def _numeric_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score annual claims numerically with an explicit relative-error floor."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    absolute = np.abs(actual - predicted)
    relative = absolute / np.maximum(np.abs(actual), 1.0)
    return {
        "wape": float(absolute.sum() / max(np.abs(actual).sum(), 1e-12)),
        "mae": float(absolute.mean()) if actual.size else 0.0,
        "within_5_percent_error": float(np.mean(relative < 0.05)) if actual.size else 0.0,
        "within_20_percent_error": float(np.mean(relative < 0.20)) if actual.size else 0.0,
    }


def evaluate_regional_demand_numeric(
        view: pd.DataFrame, min_train_years: int = 4) -> dict[str, Any]:
    """Evaluate next-year regional demand claims as a numeric proxy."""
    years = sorted(view["year"].unique())
    folds = []
    for test_year in years:
        validation_year = test_year - 1
        train = view[view["year"] < validation_year]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if len(train["year"].unique()) < min_train_years or validation.empty or test.empty:
            continue
        candidates = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                train[list(FEATURES)].fillna(0).to_numpy(float),
                np.log1p(np.maximum(train["target_demand_claims"].to_numpy(float), 0.0)),
                list(FEATURES))
            pred = np.expm1(model.predict(validation[list(FEATURES)].fillna(0).to_numpy(float)))
            candidates[alpha] = _numeric_metrics(
                validation["target_demand_claims"], pred)["wape"]
        selected_alpha = min(candidates, key=candidates.get)
        combined = pd.concat([train, validation])
        model = RidgeLinear(alpha=selected_alpha).fit(
            combined[list(FEATURES)].fillna(0).to_numpy(float),
            np.log1p(np.maximum(combined["target_demand_claims"].to_numpy(float), 0.0)),
            list(FEATURES))
        actual = test["target_demand_claims"].to_numpy(float)
        learned = np.expm1(model.predict(test[list(FEATURES)].fillna(0).to_numpy(float)))
        persistence = test["demand_claims"].to_numpy(float)
        validation_persistence = validation["demand_claims"].to_numpy(float)
        persistence_score = _numeric_metrics(
            validation["target_demand_claims"], validation_persistence)["wape"]
        selected = persistence if persistence_score <= candidates[selected_alpha] else learned
        folds.append({
            "validation_year": int(validation_year), "test_year": int(test_year),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_alpha": float(selected_alpha),
            "selected_model": "persistence" if selected is persistence else "log_ridge",
            "model": _numeric_metrics(actual, selected),
            "persistence": _numeric_metrics(actual, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    within5 = float(np.mean([x["model"]["within_5_percent_error"] for x in folds]))
    return {
        "protocol": "rolling_origin_next_year_arkansas_region_drug_demand_numeric",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "region_count": int(view["arkansas_region"].nunique()),
        "drug_count": int(view["drug_key"].nunique()),
        "mean_model_wape": float(np.mean([x["model"]["wape"] for x in folds])),
        "mean_persistence_wape": float(np.mean([x["persistence"]["wape"] for x in folds])),
        "mean_model_within_5_percent_error": within5,
        "publishable_candidate": bool(len(folds) >= 3 and within5 >= 0.65),
        "scope": "annual CMS Part D claims numeric proxy by Arkansas region and drug; not inventory",
        "folds": folds,
    }
