"""Leakage-safe monthly Arkansas pharmacy demand proxies from HHS NDC data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .event_accuracy import score_event_predictions
from .regression import LogisticRidge, RidgeLinear


FEATURES = (
    "demand_claim_lines", "lag1_demand", "lag2_demand", "rolling3_demand",
    "pharmacy_provider_count", "lag1_provider_count", "market_lag1_demand",
    "market_rolling3_demand", "labeler_lag1_demand", "market_share",
    "calendar_month",
)


def build_monthly_pharmacy_demand_view(
        source: pd.DataFrame, *, group_columns: tuple[str, ...] = ("drug_key",)
) -> pd.DataFrame:
    """Build consecutive observed month-to-next-month NDC transitions.

    HHS cell suppression makes an absent row ambiguous. Therefore only pairs
    with explicitly observed consecutive months are eligible targets; missing
    cells are never converted to zero demand.
    """
    required = {"month", "demand_claim_lines", "pharmacy_provider_count", *group_columns}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"monthly pharmacy demand missing columns: {sorted(missing)}")
    frame = source.copy()
    frame["month"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    if "drug_key" in frame:
        frame["drug_key"] = frame["drug_key"].astype(str).str.strip()
        frame["labeler_key"] = frame["drug_key"].str[:5]
    for column in group_columns:
        frame[column] = frame[column].astype(str).str.strip()
    for column in ("demand_claim_lines", "demand_paid", "pharmacy_provider_count"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["month", *group_columns, "demand_claim_lines",
                                 "pharmacy_provider_count"])
    frame = (frame.groupby(["month", *group_columns], as_index=False)
             .agg(demand_claim_lines=("demand_claim_lines", "sum"),
                  demand_paid=("demand_paid", "sum") if "demand_paid" in frame else
                  ("demand_claim_lines", "sum"),
                   pharmacy_provider_count=("pharmacy_provider_count", "sum")))
    frame = frame.sort_values([*group_columns, "month"])
    if "labeler_key" not in frame:
        frame["labeler_key"] = ""
    market = (frame.groupby("month", as_index=False)["demand_claim_lines"].sum()
              .rename(columns={"demand_claim_lines": "market_demand"})
              .sort_values("month"))
    market["market_lag1_demand"] = market["market_demand"].shift(1)
    market["market_rolling3_demand"] = market["market_demand"].shift(1).rolling(
        3, min_periods=1).mean()
    frame = frame.merge(market[["month", "market_lag1_demand",
                                "market_rolling3_demand"]], on="month", how="left")
    labeler = (frame.groupby(["labeler_key", "month"], as_index=False)
               ["demand_claim_lines"].sum().sort_values(["labeler_key", "month"]))
    labeler["labeler_lag1_demand"] = labeler.groupby("labeler_key")[
        "demand_claim_lines"].shift(1)
    frame = frame.merge(labeler[["labeler_key", "month", "labeler_lag1_demand"]],
                        on=["labeler_key", "month"], how="left")
    frame["market_share"] = frame["demand_claim_lines"] / frame.groupby(
        "month")["demand_claim_lines"].transform("sum").clip(lower=1.0)
    grouped = frame.groupby(list(group_columns), sort=False)
    frame["next_month"] = grouped["month"].shift(-1)
    frame["target_demand_claims"] = grouped["demand_claim_lines"].shift(-1)
    frame["lag1_demand"] = grouped["demand_claim_lines"].shift(1)
    frame["lag2_demand"] = grouped["demand_claim_lines"].shift(2)
    frame["rolling3_demand"] = grouped["demand_claim_lines"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    frame["lag1_provider_count"] = grouped["pharmacy_provider_count"].shift(1)
    frame["calendar_month"] = frame["month"].dt.month
    frame = frame[frame["next_month"].eq(frame["month"] + 1)].copy()
    return frame.reset_index(drop=True)


def _state(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), thresholds).astype(int)


def _state_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(5):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state)) for state in range(5)},
    }


def _numeric_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.reindex(columns=FEATURES, fill_value=0.0).fillna(0.0).to_numpy(float)


def _predict_state(train: pd.DataFrame, target: pd.DataFrame,
                   thresholds: np.ndarray, alpha: float) -> np.ndarray:
    labels = _state(train["target_demand_claims"], thresholds)
    probabilities = []
    for state in range(5):
        model = LogisticRidge(alpha=alpha).fit(
            _matrix(train), (labels == state).astype(float), list(FEATURES))
        probabilities.append(model.predict_proba(_matrix(target)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_monthly_pharmacy_demand(
        view: pd.DataFrame, *, min_train_months: int = 24,
        validation_months: int = 1,
        group_columns: tuple[str, ...] = ("drug_key",)) -> dict[str, Any]:
    """Evaluate next-month numeric and five-state NDC demand transitions."""
    months = sorted(view["month"].unique())
    folds = []
    all_actual: list[int] = []
    all_predicted: list[int] = []
    numeric_actual: list[float] = []
    numeric_predicted: list[float] = []
    per_group_correct: dict[str, int] = {}
    per_group_rows: dict[str, int] = {}
    for index in range(min_train_months, len(months) - 1):
        train_end = index - validation_months
        if train_end < min_train_months:
            continue
        train = view[view["month"].isin(months[:train_end])]
        validation = view[view["month"].isin(months[train_end:index])]
        test = view[view["month"].eq(months[index])]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = np.quantile(train["target_demand_claims"], np.arange(1, 5) / 5)
        if len(np.unique(thresholds)) < 4:
            continue
        validation_actual = _state(validation["target_demand_claims"], thresholds)
        candidates = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidates[alpha] = _state_metrics(
                validation_actual, _predict_state(train, validation, thresholds, alpha)
            )["balanced_accuracy"]
        alpha = max(candidates, key=candidates.get)
        combined = pd.concat([train, validation])
        learned_states = _predict_state(combined, test, thresholds, alpha)
        persistence_states = _state(test["demand_claim_lines"], thresholds)
        persistence_validation = _state(validation["demand_claim_lines"], thresholds)
        selected_states = (persistence_states if _state_metrics(
            validation_actual, persistence_validation)["balanced_accuracy"] >= candidates[alpha]
            else learned_states)
        # Numeric ridge is evaluated independently; persistence remains an allowed baseline.
        numeric_model = RidgeLinear(alpha=alpha).fit(
            _matrix(combined), np.log1p(np.maximum(
                combined["target_demand_claims"].to_numpy(float), 0.0)), list(FEATURES))
        learned_numeric = np.expm1(numeric_model.predict(_matrix(test)))
        persistence_numeric = test["demand_claim_lines"].to_numpy(float)
        validation_numeric_model = RidgeLinear(alpha=alpha).fit(
            _matrix(train), np.log1p(np.maximum(
                train["target_demand_claims"].to_numpy(float), 0.0)), list(FEATURES))
        validation_learned = np.expm1(validation_numeric_model.predict(_matrix(validation)))
        selected_numeric = (persistence_numeric if _numeric_metrics(
            validation["target_demand_claims"], validation["demand_claim_lines"]
        )["wape"] <= _numeric_metrics(
            validation["target_demand_claims"], validation_learned)["wape"]
        else learned_numeric)
        actual = _state(test["target_demand_claims"], thresholds)
        state_metrics = _state_metrics(actual, selected_states)
        numeric_metrics = _numeric_metrics(test["target_demand_claims"], selected_numeric)
        all_actual.extend(actual.tolist())
        all_predicted.extend(selected_states.tolist())
        numeric_actual.extend(test["target_demand_claims"].tolist())
        numeric_predicted.extend(selected_numeric.tolist())
        group_key = test[list(group_columns)].astype(str).agg("|".join, axis=1)
        for key in group_key:
            per_group_rows[key] = per_group_rows.get(key, 0) + 1
        for key, truth, prediction in zip(group_key,
                                           test["target_demand_claims"], selected_numeric):
            per_group_correct[str(key)] = per_group_correct.get(str(key), 0) + int(
                abs(float(truth) - float(prediction)) / max(abs(float(truth)), 1.0) < 0.05)
        folds.append({
            "validation_month": str(months[index - 1]), "test_month": str(months[index]),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_state_model":
                "persistence" if selected_states is persistence_states else "logistic_one_vs_rest",
            "selected_numeric_model": "persistence" if selected_numeric is persistence_numeric
                else "log_ridge",
            "state": state_metrics, "numeric": numeric_metrics,
            "persistence_state": _state_metrics(actual, persistence_states),
            "persistence_numeric": _numeric_metrics(
                test["target_demand_claims"], persistence_numeric),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    per_group_accuracy = {key: per_group_correct.get(key, 0) / rows
                          for key, rows in per_group_rows.items()}
    eligible = {key: score for key, score in per_group_accuracy.items()
                if per_group_rows[key] >= 25}
    state_actual = np.asarray(all_actual)
    state_predicted = np.asarray(all_predicted)
    mean_state_accuracy = float(np.mean([x["state"]["accuracy"] for x in folds]))
    mean_state_balanced_accuracy = float(np.mean(
        [x["state"]["balanced_accuracy"] for x in folds]))
    state_event_metrics = score_event_predictions(
        state_actual, state_predicted, kind="state", event_states={3, 4})
    publishable = bool(
        len(folds) >= 3
        and all(np.sum(state_actual == state) > 0 for state in range(5))
        and mean_state_accuracy >= 0.65
        and (mean_state_balanced_accuracy >= 0.65
             or state_event_metrics["true_positive_precision"] >= 0.80)
    )
    return {
        "protocol": "rolling_origin_next_month_arkansas_pharmacy_ndc_demand",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "drug_count": int(view["drug_key"].nunique()) if "drug_key" in view else 0,
        "group_columns": list(group_columns),
        "group_count": int(view[list(group_columns)].drop_duplicates().shape[0]),
        "county_count": int(view["county_fips"].nunique()) if "county_fips" in view else 0,
        "pharmacy_demand_state_count": 5,
        "mean_model_accuracy": mean_state_accuracy,
        "mean_model_balanced_accuracy": mean_state_balanced_accuracy,
        "mean_model_within_5_percent_error": float(np.mean(
            [x["numeric"]["within_5_percent_error"] for x in folds])),
        "mean_model_wape": float(np.mean([x["numeric"]["wape"] for x in folds])),
        "state_counts_in_scored_rows": {
            str(state): int(np.sum(state_actual == state)) for state in range(5)},
        "per_drug_rows_minimum": 25,
        "per_drug_count_with_minimum_rows": len(eligible),
        "per_drug_count_at_or_above_75_percent": int(sum(
            score >= 0.75 for score in eligible.values())),
        "per_group_accuracy": per_group_accuracy,
        "event_metrics": state_event_metrics,
        "numeric_event_metrics": score_event_predictions(
            np.asarray(numeric_actual), np.asarray(numeric_predicted), kind="numeric",
            event_threshold=float(np.quantile(numeric_actual, 0.6))),
        "publishable_candidate": publishable,
        "scope": "HHS Medicaid/CHIP Arkansas pharmacy-taxonomy billing provider NDC demand proxy; not all-payer inventory",
        "state_definition": {str(i): f"rolling_fit_quantile_{i + 1}_of_5" for i in range(5)},
        "event_definition": {"kind": "state", "event_states": [3, 4]},
        "folds": folds,
    }
