"""Five-state annual Arkansas county-by-drug demand proxy evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .county_evaluation import _feature_matrix, build_next_year_view
from .regression import LogisticRidge
from .event_accuracy import score_event_predictions


def _metrics(actual: np.ndarray, predicted: np.ndarray, state_count: int) -> dict:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = [
        float(np.mean(predicted[actual == state] == state))
        if np.any(actual == state) else 0.0
        for state in range(state_count)
    ]
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state))
                         for state in range(state_count)},
    }


def _predict(train: pd.DataFrame, target: pd.DataFrame,
             train_states: np.ndarray, alpha: float, state_count: int) -> np.ndarray:
    features = [
        "log1p_demand_claims", "log1p_demand_fills", "log1p_demand_cost",
        "log1p_source_row_count", "log1p_mapped_city_count", "mapping_confidence",
    ]
    x_train = _feature_matrix(train)
    x_target = _feature_matrix(target)
    probabilities = []
    for state in range(state_count):
        model = LogisticRidge(alpha=alpha).fit(
            x_train, (train_states == state).astype(float), features)
        probabilities.append(model.predict_proba(x_target))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_county_demand_state_rolling(
    outcomes: pd.DataFrame,
    *,
    min_train_years: int = 4,
    state_count: int = 5,
    folds: Optional[Iterable[tuple[int, int]]] = None,
) -> Dict:
    """Evaluate next-year county/drug demand states by chronological folds."""
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    view = build_next_year_view(outcomes)
    years = sorted(int(value) for value in view["year"].unique())
    if folds is None:
        folds = tuple((year - 1, year) for year in years)
    results = []
    county_correct: dict[str, int] = {}
    county_rows: dict[str, int] = {}
    state_counts = {str(state): 0 for state in range(state_count)}
    event_actual = []
    event_predicted = []
    for validation_year, test_year in folds:
        train = view[view["year"] < validation_year]
        validation = view[view["year"].eq(validation_year)]
        test = view[view["year"].eq(test_year)]
        if len(train["year"].unique()) < min_train_years or validation.empty or test.empty:
            continue
        thresholds = np.quantile(
            train["y_target"].to_numpy(float), np.arange(1, state_count) / state_count)
        if len(np.unique(thresholds)) < state_count - 1:
            continue
        train_states = np.digitize(train["y_target"], thresholds)
        validation_states = np.digitize(validation["y_target"], thresholds)
        test_states = np.digitize(test["y_target"], thresholds)
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            learned = _predict(train, validation, train_states, alpha, state_count)
            scores[alpha] = _metrics(validation_states, learned, state_count)[
                "balanced_accuracy"]
        alpha = max(scores, key=scores.get)
        learned = _predict(
            pd.concat([train, validation]), test,
            np.digitize(pd.concat([train, validation])["y_target"], thresholds),
            alpha, state_count)
        persistence = np.digitize(test["demand_claims"], thresholds)
        validation_persistence = np.digitize(validation["demand_claims"], thresholds)
        if _metrics(validation_states, validation_persistence, state_count)[
                "balanced_accuracy"] >= scores[alpha]:
            selected, selected_name = persistence, "persistence"
        else:
            selected, selected_name = learned, "logistic_one_vs_rest"
        model_metrics = _metrics(test_states, selected, state_count)
        persistence_metrics = _metrics(test_states, persistence, state_count)
        event_actual.extend(test_states.tolist())
        event_predicted.extend(selected.tolist())
        for county, actual, prediction in zip(
                test["county_fips"].astype(str), test_states, selected):
            county_rows[county] = county_rows.get(county, 0) + 1
            county_correct[county] = county_correct.get(county, 0) + int(actual == prediction)
        for state, count in model_metrics["state_counts"].items():
            state_counts[state] += count
        results.append({
            "validation_year": int(validation_year), "test_year": int(test_year),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_model": selected_name,
            "thresholds": [float(value) for value in thresholds],
            "model": model_metrics, "persistence": persistence_metrics,
        })
    county_accuracy = {key: county_correct[key] / county_rows[key]
                       for key in sorted(county_rows)}
    accuracy = float(np.mean([row["model"]["accuracy"] for row in results])) if results else None
    balanced = float(np.mean([row["model"]["balanced_accuracy"] for row in results])) if results else None
    event_metrics = score_event_predictions(
        np.asarray(event_actual), np.asarray(event_predicted), kind="state",
        event_states={3, 4}) if event_actual else None
    return {
        "protocol": "rolling_origin_next_year_arkansas_county_drug_demand_five_state",
        "target": "arkansas_county_annual_demand_five_state",
        "state_count": state_count, "fold_count": len(results),
        "test_rows": int(sum(row["test_rows"] for row in results)),
        "drug_count": int(view["drug_key"].nunique()),
        "county_count": int(view["county_fips"].nunique()),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "event_metrics": event_metrics,
        "state_counts_in_scored_rows": state_counts,
        "county_rows_minimum": 25, "county_accuracy": county_accuracy,
        "county_count_at_or_above_75_percent": int(sum(
            accuracy >= 0.75 for key, accuracy in county_accuracy.items()
            if county_rows[key] >= 25)),
        "publishable_candidate": bool(
            len(results) >= 3 and accuracy is not None and accuracy >= 0.65
            and balanced is not None and balanced >= 0.65
            and all(state_counts.values())),
        "scope": "CMS Part D county/drug demand proxy; not pharmacy inventory or supplier allocation",
        "state_definition": {str(i): f"quantile_{i + 1}_of_{state_count}"
                             for i in range(state_count)},
        "folds": results,
    }
