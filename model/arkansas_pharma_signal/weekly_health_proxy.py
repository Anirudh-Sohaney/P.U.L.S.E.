"""Leak-safe weekly Arkansas respiratory-demand proxy evaluation.

FluView ILI/WILI is an observed Arkansas healthcare-utilization signal, not a
pharmacy dispense or inventory label. This module evaluates it as an upstream
near-term demand proxy that may inform the pharmacy model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .regression import LogisticRidge, RidgeLinear
from .event_accuracy import score_event_predictions


FEATURE_COLUMNS = [
    "current_value", "lag1_value", "lag2_value", "lag4_value",
    "rolling4_value", "week_sin", "week_cos",
]


def _week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    return pd.to_datetime(
        year.astype(str) + "-W" + week.astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce")


def load_fluview_weekly_proxy(path: Path | pd.DataFrame, target: str = "wili") -> pd.DataFrame:
    """Build strict consecutive-week rows from the latest FluView release."""
    required = {"region", "epiweek", "issue", target}
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FluView proxy missing columns: {sorted(missing)}")
    frame["region"] = frame["region"].astype(str).str.lower().str.strip()
    frame["epiweek"] = pd.to_numeric(frame["epiweek"], errors="coerce")
    frame["issue"] = pd.to_numeric(frame["issue"], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=["region", "epiweek", "issue", target]).copy()
    frame["year"] = (frame["epiweek"] // 100).astype(int)
    frame["week"] = (frame["epiweek"] % 100).astype(int)
    frame["week_start"] = _week_start(frame["year"], frame["week"])
    frame["issue_date"] = _week_start(
        (frame["issue"] // 100).astype(int), (frame["issue"] % 100).astype(int))
    frame = frame.dropna(subset=["week_start", "issue_date"])
    # The source contains revisions; retain the latest published issue for
    # each epidemiological week, while forecasting only from prior weeks.
    frame = (frame.sort_values(["region", "week_start", "issue_date"])
             .drop_duplicates(["region", "week_start"], keep="last"))
    frame = frame.sort_values(["region", "week_start"]).reset_index(drop=True)
    group = frame.groupby("region", sort=False)
    frame["previous_week"] = group["week_start"].shift(1)
    frame["next_week"] = group["week_start"].shift(-1)
    frame["value_last"] = group[target].shift(1)
    frame["value_last2"] = group[target].shift(2)
    frame["value_last4"] = group[target].shift(4)
    frame["rolling4_value"] = group[target].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    frame["target"] = group[target].shift(-1)
    frame["valid_next_week"] = frame["next_week"].sub(frame["week_start"]).eq(pd.Timedelta(days=7))
    frame = frame[frame["valid_next_week"]].copy()
    frame["lag1_value"] = frame["value_last"].fillna(frame[target])
    frame["lag2_value"] = frame["value_last2"].fillna(frame["lag1_value"])
    frame["lag4_value"] = frame["value_last4"].fillna(frame["lag2_value"])
    frame["current_value"] = frame[target]
    frame["rolling4_value"] = frame["rolling4_value"].fillna(frame["current_value"])
    frame["week_sin"] = np.sin(2 * np.pi * frame["week"] / 52.0)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week"] / 52.0)
    return frame[["region", "year", "week", "week_start", "target",
                  *FEATURE_COLUMNS]].reset_index(drop=True)


def _wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.abs(actual - prediction).sum() /
                 max(np.abs(actual).sum(), 1e-12))


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    ape = np.abs(actual - prediction) / np.maximum(np.abs(actual), 1e-12)
    return {
        "wape": _wape(actual, prediction),
        "under_5_percent_error": float(np.mean(ape < 0.05)),
        "median_absolute_percentage_error": float(np.median(ape)),
    }


def evaluate_weekly_health_proxy(
    view: pd.DataFrame,
    *,
    region: str = "ar",
    folds: Optional[Iterable[tuple[int, int, int]]] = None,
) -> dict:
    """Evaluate next-week proxy forecasts over chronological year folds."""
    data = view[view["region"].eq(region)].sort_values("week_start")
    if folds is None:
        folds = ((2018, 2019, 2020), (2019, 2020, 2021),
                 (2020, 2021, 2022), (2021, 2022, 2023),
                 (2022, 2023, 2024), (2023, 2024, 2025),
                 (2024, 2025, 2026))
    results = []
    for train_year, validation_year, test_year in folds:
        train = data[data["year"] <= train_year]
        validation = data[data["year"] == validation_year]
        test = data[data["year"] == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        candidates = {}
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                train[FEATURE_COLUMNS].to_numpy(),
                np.log1p(np.maximum(train["target"].to_numpy(), 0.0)),
                feature_names=FEATURE_COLUMNS)
            pred = np.expm1(model.predict(validation[FEATURE_COLUMNS].to_numpy()))
            candidates[alpha] = _wape(validation["target"], pred)
        selected_alpha = min(candidates, key=candidates.get)
        model = RidgeLinear(alpha=selected_alpha).fit(
            train[FEATURE_COLUMNS].to_numpy(),
            np.log1p(np.maximum(train["target"].to_numpy(), 0.0)),
            feature_names=FEATURE_COLUMNS)
        actual = test["target"].to_numpy(dtype=float)
        prediction = np.expm1(model.predict(test[FEATURE_COLUMNS].to_numpy()))
        persistence = test["current_value"].to_numpy(dtype=float)
        model_metrics, persistence_metrics = _metrics(actual, prediction), _metrics(actual, persistence)
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "train_rows": len(train),
            "validation_rows": len(validation), "test_rows": len(test),
            "selected_alpha": selected_alpha, "model": model_metrics,
            "persistence": persistence_metrics,
            "model_beats_persistence_wape": model_metrics["wape"] < persistence_metrics["wape"],
        })
    if not results:
        raise ValueError(f"no complete weekly proxy folds for region {region!r}")
    return {
        "target": "next_week_fluview_wili_proxy",
        "region": region,
        "fold_count": len(results), "minimum_required_folds": 3,
        "folds": results,
        "mean_model_wape": float(np.mean([r["model"]["wape"] for r in results])),
        "mean_persistence_wape": float(np.mean([r["persistence"]["wape"] for r in results])),
        "mean_model_under_5_percent_error": float(np.mean([
            r["model"]["under_5_percent_error"] for r in results])),
        "mean_persistence_under_5_percent_error": float(np.mean([
            r["persistence"]["under_5_percent_error"] for r in results])),
        "publishable_rolling_candidate": bool(
            len(results) >= 3 and all(r["model_beats_persistence_wape"] for r in results)),
        "scope": "Arkansas weekly respiratory healthcare-demand proxy; not pharmacy dispensing truth",
    }


def _pressure_state(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map a continuous respiratory signal to ordered low/mid/high states."""
    return np.digitize(np.asarray(values, dtype=float), [lower, upper], right=False).astype(int)


def _state_metrics(actual: np.ndarray, predicted: np.ndarray,
                   state_count: int = 3) -> dict:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(state_count):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state))
                         for state in range(state_count)},
    }


def _fit_weekly_state(train: pd.DataFrame, target: pd.DataFrame,
                      alpha: float, thresholds: np.ndarray,
                      state_count: int) -> np.ndarray:
    """Fit one-v-rest state models using only thresholds learned from train."""
    train_state = np.digitize(train["target"].to_numpy(), thresholds)
    x_train = train[FEATURE_COLUMNS].to_numpy(float)
    x_target = target[FEATURE_COLUMNS].to_numpy(float)
    probabilities = []
    for state in range(state_count):
        model = LogisticRidge(alpha=alpha).fit(
            x_train, (train_state == state).astype(float), FEATURE_COLUMNS)
        probabilities.append(model.predict_proba(x_target))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_weekly_health_state(
    view: pd.DataFrame,
    *,
    region: str = "ar",
    folds: Optional[Iterable[tuple[int, int, int]]] = None,
    state_count: int = 3,
) -> dict:
    """Evaluate an Arkansas weekly respiratory-pressure state signal.

    Low/mid/high cut points are recalculated from each training partition's
    tertiles. This prevents test-period information from defining the state
    labels while keeping the output ordered and non-binary.
    """
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    data = view[view["region"].eq(region)].sort_values("week_start")
    if folds is None:
        folds = ((2018, 2019, 2020), (2019, 2020, 2021),
                 (2020, 2021, 2022), (2021, 2022, 2023),
                 (2022, 2023, 2024), (2023, 2024, 2025),
                 (2024, 2025, 2026))
    results = []
    event_actual: list[int] = []
    event_predicted: list[int] = []
    for train_year, validation_year, test_year in folds:
        train = data[data["year"] <= train_year]
        validation = data[data["year"] == validation_year]
        test = data[data["year"] == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        thresholds = np.quantile(
            train["target"].to_numpy(float), np.arange(1, state_count) / state_count)
        if len(np.unique(thresholds)) < state_count - 1:
            continue
        candidates = {}
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            prediction = _fit_weekly_state(
                train, validation, alpha, thresholds, state_count)
            actual = np.digitize(validation["target"].to_numpy(float), thresholds)
            candidates[alpha] = _state_metrics(
                actual, prediction, state_count)["balanced_accuracy"]
        validation_actual = np.digitize(validation["target"].to_numpy(float), thresholds)
        persistence_validation = np.digitize(
            validation["current_value"].to_numpy(float), thresholds)
        persistence_score = _state_metrics(
            validation_actual, persistence_validation, state_count)["balanced_accuracy"]
        selected_alpha = max(candidates, key=candidates.get)
        selected_model = "logistic_one_vs_rest"
        if persistence_score > candidates[selected_alpha]:
            selected_model = "persistence"
        actual = np.digitize(test["target"].to_numpy(float), thresholds)
        prediction = (test["current_value"].to_numpy(float)
                      if selected_model == "persistence" else
                      _fit_weekly_state(train, test, selected_alpha, thresholds, state_count))
        if selected_model == "persistence":
            prediction = np.digitize(prediction, thresholds)
        persistence = np.digitize(test["current_value"].to_numpy(float), thresholds)
        event_actual.extend(actual.tolist())
        event_predicted.extend(prediction.tolist())
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "selected_alpha": selected_alpha,
            "selected_model": selected_model,
            "thresholds": [float(value) for value in thresholds],
            "model": _state_metrics(actual, prediction, state_count),
            "persistence": _state_metrics(actual, persistence, state_count),
        })
    if not results:
        raise ValueError(f"no complete weekly state folds for region {region!r}")
    model_accuracy = float(np.mean([x["model"]["accuracy"] for x in results]))
    model_balanced = float(np.mean([x["model"]["balanced_accuracy"] for x in results]))
    persistence_balanced = float(np.mean([
        x["persistence"]["balanced_accuracy"] for x in results]))
    state_counts = {str(state): sum(x["model"]["state_counts"][str(state)] for x in results)
                    for state in range(state_count)}
    return {
        "protocol": f"rolling_origin_next_week_arkansas_fluview_{state_count}_state",
        "target": "next_week_fluview_wili_state",
        "region": region, "state_count": state_count,
        "state_definition": {str(i): f"quantile_{i + 1}_of_{state_count}"
                             for i in range(state_count)},
        "fold_count": len(results), "test_rows": int(sum(x["test_rows"] for x in results)),
        "mean_model_accuracy": model_accuracy,
        "mean_model_balanced_accuracy": model_balanced,
        "mean_persistence_accuracy": float(np.mean([
            x["persistence"]["accuracy"] for x in results])),
        "mean_persistence_balanced_accuracy": persistence_balanced,
        "model_balanced_improvement_vs_persistence": model_balanced - persistence_balanced,
        "state_counts_in_scored_rows": state_counts,
        "event_metrics": score_event_predictions(
            np.asarray(event_actual), np.asarray(event_predicted), kind="state",
            event_states=set(range(max(0, state_count - 2), state_count))),
        "publishable_candidate": bool(
            len(results) >= 3 and model_accuracy >= 0.65 and model_balanced >= 0.65
            and all(state_counts.values())),
        "scope": "Arkansas weekly respiratory healthcare-utilization pressure proxy; not pharmacy dispensing truth",
        "folds": results,
    }


def evaluate_fluview_divergence_state(
    view: pd.DataFrame,
    *,
    folds: Optional[Iterable[tuple[int, int, int]]] = None,
    state_count: int = 5,
) -> dict:
    """Screen five-state national-minus-Arkansas WILI divergence.

    This is a derived upstream-context candidate, not a promoted target. The
    target is next week's national WILI minus Arkansas WILI; thresholds are
    fit-only and persistence is the intentionally transparent baseline.
    """
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    paired = (view[view["region"].isin(["ar", "nat"])]
              .pivot(index="week_start", columns="region", values="current_value")
              .dropna(subset=["ar", "nat"]).sort_index())
    paired["divergence"] = paired["nat"] - paired["ar"]
    paired["target"] = paired["divergence"].shift(-1)
    paired["year"] = pd.to_datetime(paired.index).isocalendar().year.astype(int).to_numpy()
    paired = paired.dropna(subset=["target"])
    if folds is None:
        folds = ((2018, 2019, 2020), (2019, 2020, 2021),
                 (2020, 2021, 2022), (2021, 2022, 2023),
                 (2022, 2023, 2024), (2023, 2024, 2025),
                 (2024, 2025, 2026))
    results = []
    for train_year, validation_year, test_year in folds:
        train = paired[paired["year"] <= train_year]
        test = paired[paired["year"] == test_year]
        if train.empty or test.empty:
            continue
        thresholds = np.quantile(
            train["target"].to_numpy(float), np.arange(1, state_count) / state_count)
        actual = np.digitize(test["target"].to_numpy(float), thresholds)
        prediction = np.digitize(test["divergence"].to_numpy(float), thresholds)
        metrics = _state_metrics(actual, prediction, state_count)
        results.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "thresholds": [float(value) for value in thresholds],
            "model": metrics,
        })
    state_counts = {str(state): sum(
        row["model"]["state_counts"][str(state)] for row in results)
                    for state in range(state_count)}
    accuracy = float(np.mean([row["model"]["accuracy"] for row in results])) if results else None
    balanced = float(np.mean([row["model"]["balanced_accuracy"] for row in results])) if results else None
    return {
        "protocol": "rolling_origin_next_week_fluview_national_minus_arkansas_divergence_five_state",
        "target": "next_week_fluview_national_minus_arkansas_wili_state",
        "state_count": state_count, "fold_count": len(results),
        "test_rows": int(sum(row["test_rows"] for row in results)),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "state_counts_in_scored_rows": state_counts,
        "publishable_candidate": bool(
            len(results) >= 3 and accuracy is not None and accuracy >= 0.65
            and balanced is not None and balanced >= 0.65
            and all(state_counts.values())),
        "scope": "national-minus-Arkansas respiratory context; not pharmacy dispensing truth",
        "folds": results,
    }
