"""Five-state monthly FDA shortage-pressure metric.

The public FDA archive exposes supplier-by-NDC shortage observations, but a
binary continuation label is both imbalanced and not a useful three-state
output. This module aggregates the observed supplier rows to an NDC-month:

* ``0``: no supplier reported active shortage;
* ``1``: exactly one supplier reported active shortage;
* ``2``: exactly two suppliers reported active shortage;
* ``3``: exactly three suppliers reported active shortage;
* ``4``: four or more suppliers reported active shortage.

The resulting state is a market-pressure proxy. It does not claim that every
pharmacy is short, and it is not an Arkansas-only label. Features are built
from months at or before the forecast origin; the target is the next observed
month. One-vs-rest logistic ridge models are used so the evaluator remains
compatible with the project's auditable numpy-only regression layer.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .regression import LogisticRidge, RidgeLinear
from .event_accuracy import score_event_predictions


STATE_NAMES = {
    0: "none", 1: "single_supplier", 2: "two_suppliers",
    3: "three_suppliers", 4: "four_or_more_suppliers",
}
FEATURE_COLUMNS = [
    "active_supplier_count",
    "observed_supplier_count",
    "lag1_pressure",
    "lag2_pressure",
    "rolling3_pressure",
    "calendar_month",
]


def build_shortage_breadth_panel(shortage_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the FDA archive to monthly count of active shortage NDCs."""
    required = {"ndc9", "month", "shortage_active"}
    missing = required.difference(shortage_panel.columns)
    if missing:
        raise ValueError(f"shortage panel missing columns: {sorted(missing)}")
    frame = shortage_panel.copy()
    frame["ndc9"] = frame["ndc9"].astype(str).str.strip()
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M")
    frame["shortage_active"] = pd.to_numeric(frame["shortage_active"], errors="raise").astype(int)
    if not frame["shortage_active"].isin([0, 1]).all():
        raise ValueError("shortage_active must be binary in the source archive")
    monthly = (frame.groupby(["month", "ndc9"], as_index=False)["shortage_active"].max()
               .groupby("month", as_index=False)["shortage_active"].sum()
               .rename(columns={"shortage_active": "active_ndc_count"}))
    expected = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M")
    monthly = (monthly.set_index("month").reindex(expected, fill_value=0)
               .rename_axis("month").reset_index())
    return monthly


def _breadth_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(3):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "state_counts": {str(state): int(np.sum(actual == state)) for state in range(3)},
    }


def evaluate_shortage_breadth_rolling(
    source: pd.DataFrame, *, min_train_months: int = 24,
    validation_months: int = 3, test_months: int = 3, step_months: int = 1,
) -> dict[str, Any]:
    """Evaluate next-month low/mid/high national shortage breadth."""
    frame = build_shortage_breadth_panel(source)
    periods = list(frame["month"])
    folds = []
    for index in range(min_train_months, len(periods) - validation_months - test_months + 1,
                       max(1, step_months)):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:index + validation_months + test_months]
        thresholds = frame[frame.month.isin(train_periods)]["active_ndc_count"].quantile(
            [1 / 3, 2 / 3]).to_numpy()
        if len(set(thresholds)) < 2:
            continue
        fold = frame.copy()
        fold["current_state"] = pd.cut(
            fold["active_ndc_count"], bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
            labels=False, include_lowest=True,
        ).astype(int)
        fold["target_state"] = fold["current_state"].shift(-1)
        fold["lag1_state"] = fold["current_state"].shift(1)
        fold["lag2_state"] = fold["current_state"].shift(2)
        fold["rolling3_state"] = fold["current_state"].shift(1).rolling(3, min_periods=1).mean()
        fold["calendar_month"] = fold["month"].dt.month
        train = fold[fold.month.isin(train_periods)].dropna(subset=["target_state"])
        validation = fold[fold.month.isin(validation_periods)].dropna(subset=["target_state"])
        test = fold[fold.month.isin(test_periods)].dropna(subset=["target_state"])
        features = ["current_state", "lag1_state", "lag2_state", "rolling3_state", "calendar_month"]
        if train.empty or validation.empty or test.empty:
            continue
        def predict(train_frame: pd.DataFrame, target_frame: pd.DataFrame, alpha: float) -> np.ndarray:
            probabilities = []
            for state in range(3):
                model = LogisticRidge(alpha=alpha).fit(
                    train_frame[features].fillna(0).to_numpy(float),
                    train_frame["target_state"].eq(state).astype(float), features)
                probabilities.append(model.predict_proba(target_frame[features].fillna(0).to_numpy(float)))
            return np.column_stack(probabilities).argmax(axis=1)
        scores = {alpha: _breadth_metrics(
            validation.target_state, predict(train, validation, alpha))["balanced_accuracy"]
                  for alpha in (0.1, 1.0, 10.0, 100.0)}
        alpha = max(scores, key=scores.get)
        learned = predict(pd.concat([train, validation]), test, alpha)
        persistence = test.current_state.to_numpy(int)
        val_persistence = validation.current_state.to_numpy(int)
        selected = persistence if _breadth_metrics(
            validation.target_state, val_persistence)["balanced_accuracy"] >= scores[alpha] else learned
        folds.append({
            "validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
            "test_period": f"{test_periods[0]}/{test_periods[-1]}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "thresholds_fit_on_train": thresholds.tolist(),
            "selected_model": "persistence" if selected is persistence else "logistic_one_vs_rest",
            "model": _breadth_metrics(test.target_state, selected),
            "persistence": _breadth_metrics(test.target_state, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    accuracy = float(np.mean([fold["model"]["accuracy"] for fold in folds]))
    balanced = float(np.mean([fold["model"]["balanced_accuracy"] for fold in folds]))
    counts = {str(state): sum(fold["model"]["state_counts"][str(state)] for fold in folds)
              for state in range(3)}
    return {
        "protocol": "rolling_origin_next_month_fda_national_shortage_breadth_state",
        "fold_count": len(folds), "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "mean_model_accuracy": accuracy, "mean_model_balanced_accuracy": balanced,
        "state_counts_in_scored_rows": counts,
        "publishable_candidate": bool(len(folds) >= 3 and accuracy >= 0.65
                                       and balanced >= 0.65 and all(counts.values())),
        "scope": "national count of active FDA shortage NDCs; no Arkansas allocation",
        "state_definition": {"0": "low_breadth", "1": "mid_breadth", "2": "high_breadth"},
        "folds": folds,
    }


def _numeric_supplier_count_metrics(actual: np.ndarray,
                                    predicted: np.ndarray) -> dict[str, float]:
    """Score supplier-count forecasts with explicit zero handling."""
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


def evaluate_numeric_supplier_count_rolling(
        view: pd.DataFrame, min_train_months: int = 24,
        validation_months: int = 3, test_months: int = 3) -> dict[str, Any]:
    """Evaluate next-month active FDA supplier count as a numeric target."""
    folds = []
    event_actual: list[float] = []
    event_predicted: list[float] = []
    per_drug_correct: dict[str, int] = {}
    per_drug_rows: dict[str, int] = {}
    for train, validation, test, val_start, val_end, test_start, test_end in _monthly_splits(
            view, min_train_months, validation_months, test_months):
        candidates = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = RidgeLinear(alpha=alpha).fit(
                _matrix(train, FEATURE_COLUMNS),
                np.log1p(train["target_supplier_count"].to_numpy(float)),
                FEATURE_COLUMNS)
            prediction = np.expm1(model.predict(_matrix(validation, FEATURE_COLUMNS)))
            candidates[alpha] = _numeric_supplier_count_metrics(
                validation["target_supplier_count"], prediction)["wape"]
        alpha = min(candidates, key=candidates.get)
        model = RidgeLinear(alpha=alpha).fit(
            _matrix(pd.concat([train, validation]), FEATURE_COLUMNS),
            np.log1p(pd.concat([train, validation])["target_supplier_count"].to_numpy(float)),
            FEATURE_COLUMNS)
        actual = test["target_supplier_count"].to_numpy(float)
        learned = np.expm1(model.predict(_matrix(test, FEATURE_COLUMNS)))
        persistence = test["active_supplier_count"].to_numpy(float)
        validation_persistence = validation["active_supplier_count"].to_numpy(float)
        validation_persistence_score = _numeric_supplier_count_metrics(
            validation["target_supplier_count"], validation_persistence)["wape"]
        selected = persistence if validation_persistence_score <= candidates[alpha] else learned
        event_actual.extend(actual.tolist())
        event_predicted.extend(selected.tolist())
        for drug, truth, prediction in zip(
                test["ndc9"].astype(str), actual, selected):
            relative_error = abs(float(prediction) - float(truth)) / max(abs(float(truth)), 1.0)
            drug = str(drug)
            per_drug_rows[drug] = per_drug_rows.get(drug, 0) + 1
            per_drug_correct[drug] = per_drug_correct.get(drug, 0) + int(relative_error < 0.05)
        folds.append({
            "validation_period": f"{val_start}-{val_end}",
            "test_period": f"{test_start}-{test_end}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "selected_alpha": float(alpha),
            "selected_model": "persistence" if selected is persistence else "log_ridge",
            "model": _numeric_supplier_count_metrics(actual, selected),
            "persistence": _numeric_supplier_count_metrics(actual, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    within5 = float(np.mean([x["model"]["within_5_percent_error"] for x in folds]))
    per_drug_accuracy = {
        drug: per_drug_correct[drug] / per_drug_rows[drug]
        for drug in sorted(per_drug_rows)
    }
    eligible_drugs = {
        drug: accuracy for drug, accuracy in per_drug_accuracy.items()
        if per_drug_rows[drug] >= 25
    }
    drugs_at_75 = sum(accuracy >= 0.75 for accuracy in eligible_drugs.values())
    return {
        "protocol": "rolling_origin_next_month_ndc_shortage_supplier_count_numeric",
        "fold_count": len(folds),
        "test_rows": int(sum(x["test_rows"] for x in folds)),
        "ndc_count": int(view["ndc9"].nunique()),
        "mean_model_wape": float(np.mean([x["model"]["wape"] for x in folds])),
        "mean_persistence_wape": float(np.mean([x["persistence"]["wape"] for x in folds])),
        "mean_model_within_5_percent_error": within5,
        "per_drug_rows_minimum": 25,
        "per_drug_count_with_minimum_rows": len(eligible_drugs),
        "per_drug_count_at_or_above_75_percent": int(drugs_at_75),
        "per_drug_accuracy": per_drug_accuracy,
        "event_metrics": score_event_predictions(
            np.asarray(event_actual), np.asarray(event_predicted), kind="numeric",
            event_threshold=1.0),
        "publishable_candidate": bool(len(folds) >= 3 and within5 >= 0.65),
        "scope": "national FDA active supplier count for Arkansas-exposed NDCs; not inventory truth",
        "folds": folds,
    }


def build_pressure_panel(shortage_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate supplier observations into a complete drug-month pressure panel."""
    required = {"ndc9", "supplier", "month", "shortage_active"}
    missing = required.difference(shortage_panel.columns)
    if missing:
        raise ValueError(f"shortage panel missing columns: {sorted(missing)}")
    frame = shortage_panel.copy()
    frame["ndc9"] = frame["ndc9"].astype(str).str.strip()
    frame["supplier"] = frame["supplier"].astype(str).str.strip()
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M")
    frame["shortage_active"] = pd.to_numeric(
        frame["shortage_active"], errors="raise").astype(int)
    if not frame["shortage_active"].isin([0, 1]).all():
        raise ValueError("shortage_active must be binary in the source archive")

    grouped = (frame.groupby(["ndc9", "month"], as_index=False)
               .agg(active_supplier_count=("shortage_active", "sum"),
                    observed_supplier_count=("supplier", "nunique")))
    grouped["pressure_state"] = np.select(
        [grouped["active_supplier_count"].eq(0),
         grouped["active_supplier_count"].eq(1),
         grouped["active_supplier_count"].eq(2),
         grouped["active_supplier_count"].eq(3)],
        [0, 1, 2, 3], default=4).astype(int)
    grouped = grouped.sort_values(["ndc9", "month"]).reset_index(drop=True)
    by_drug = grouped.groupby("ndc9", sort=False)
    grouped["lag1_pressure"] = by_drug["pressure_state"].shift(1)
    grouped["lag2_pressure"] = by_drug["pressure_state"].shift(2)
    grouped["rolling3_pressure"] = by_drug["pressure_state"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    grouped["calendar_month"] = grouped["month"].dt.month
    return grouped


def build_next_month_pressure_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact next observed monthly pressure state as the target."""
    frame = panel.sort_values(["ndc9", "month"]).copy()
    by_drug = frame.groupby("ndc9", sort=False)
    next_month = by_drug["month"].shift(-1)
    expected = frame["month"] + 1
    consecutive = next_month == expected
    frame["target_month"] = next_month.where(consecutive)
    frame["target_pressure_state"] = by_drug["pressure_state"].shift(-1).where(consecutive)
    frame = frame.dropna(subset=["target_pressure_state"]).copy()
    frame["target_pressure_state"] = frame["target_pressure_state"].astype(int)
    return frame.reset_index(drop=True)


def build_next_month_numeric_supplier_count_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact next-month active supplier count as a numeric target."""
    frame = panel.sort_values(["ndc9", "month"]).copy()
    by_drug = frame.groupby("ndc9", sort=False)
    next_month = by_drug["month"].shift(-1)
    consecutive = next_month.eq(frame["month"] + 1)
    frame["target_month"] = next_month.where(consecutive)
    frame["target_supplier_count"] = by_drug["active_supplier_count"].shift(-1).where(consecutive)
    frame = frame.dropna(subset=["target_supplier_count"]).copy()
    frame["target_supplier_count"] = frame["target_supplier_count"].astype(float)
    return frame.reset_index(drop=True)


def _matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = frame.reindex(columns=list(columns), fill_value=0.0).copy()
    return values.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)


def _fit_multiclass(train: pd.DataFrame, validation: pd.DataFrame,
                    test: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit one-v-rest logistic models and return validation/test states."""
    models = []
    x_train = _matrix(train, FEATURE_COLUMNS)
    x_validation = _matrix(validation, FEATURE_COLUMNS)
    x_test = _matrix(test, FEATURE_COLUMNS)
    for state in STATE_NAMES:
        model = LogisticRidge(alpha=alpha).fit(
            x_train, train["target_pressure_state"].eq(state).astype(float).to_numpy(),
            FEATURE_COLUMNS)
        models.append(model)
    validation_probabilities = np.column_stack([
        model.predict_proba(x_validation) for model in models])
    test_probabilities = np.column_stack([
        model.predict_proba(x_test) for model in models])
    return validation_probabilities.argmax(axis=1), test_probabilities.argmax(axis=1)


def _state_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    classes = sorted(set(actual.tolist()) | set(predicted.tolist()))
    recalls = []
    for state in classes:
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_recall": dict(zip((str(x) for x in classes), recalls)),
        "state_counts": {str(x): int(np.sum(actual == x)) for x in classes},
    }


def _persistence(frame: pd.DataFrame) -> np.ndarray:
    return frame["pressure_state"].astype(int).to_numpy()


def _monthly_splits(frame: pd.DataFrame, min_train_months: int = 24,
                    validation_months: int = 3, test_months: int = 3):
    periods = sorted(frame["month"].unique())
    for index in range(min_train_months,
                       len(periods) - validation_months - test_months + 1):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:
                                index + validation_months + test_months]
        expected = pd.period_range(train_periods[-1] + 1,
                                   test_periods[-1], freq="M")
        if len(expected) != validation_months + test_months:
            continue
        yield (
            frame[frame["month"].isin(train_periods)],
            frame[frame["month"].isin(validation_periods)],
            frame[frame["month"].isin(test_periods)],
            str(validation_periods[0]), str(validation_periods[-1]),
            str(test_periods[0]), str(test_periods[-1]),
        )


def evaluate_pressure_rolling(view: pd.DataFrame, min_train_months: int = 24,
                              validation_months: int = 3,
                              test_months: int = 3) -> dict[str, Any]:
    """Evaluate monthly pressure states with strict rolling-origin folds."""
    folds = []
    per_drug_correct: dict[str, int] = {}
    per_drug_rows: dict[str, int] = {}
    observed_state_counts: dict[str, int] = {str(state): 0 for state in STATE_NAMES}
    event_actual: list[int] = []
    event_predicted: list[int] = []
    for train, validation, test, val_start, val_end, test_start, test_end in _monthly_splits(
            view, min_train_months, validation_months, test_months):
        selected_alpha = None
        best_score = -np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0):
            val_pred, _ = _fit_multiclass(train, validation, validation, alpha)
            score = _state_metrics(validation["target_pressure_state"], val_pred)[
                "balanced_accuracy"]
            if score > best_score:
                best_score, selected_alpha = score, alpha
        val_pred, model_pred = _fit_multiclass(train, validation, test, selected_alpha)
        model_metrics = _state_metrics(test["target_pressure_state"], model_pred)
        persistence_metrics = _state_metrics(test["target_pressure_state"], _persistence(test))
        event_actual.extend(test["target_pressure_state"].to_numpy(int).tolist())
        event_predicted.extend(np.asarray(model_pred, dtype=int).tolist())
        for drug, actual, predicted in zip(
                test["ndc9"].astype(str),
                test["target_pressure_state"].to_numpy(int),
                model_pred):
            per_drug_rows[drug] = per_drug_rows.get(drug, 0) + 1
            per_drug_correct[drug] = per_drug_correct.get(drug, 0) + int(actual == predicted)
        for state, count in model_metrics["state_counts"].items():
            observed_state_counts[state] = observed_state_counts.get(state, 0) + count
        folds.append({
            "validation_period": f"{val_start}-{val_end}",
            "test_period": f"{test_start}-{test_end}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "selected_alpha": selected_alpha,
            "model": model_metrics, "persistence": persistence_metrics,
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    mean_accuracy = float(np.mean([x["model"]["accuracy"] for x in folds]))
    mean_balanced = float(np.mean([x["model"]["balanced_accuracy"] for x in folds]))
    persistence_balanced = float(np.mean([
        x["persistence"]["balanced_accuracy"] for x in folds]))
    per_drug_accuracy = {
        drug: per_drug_correct[drug] / per_drug_rows[drug]
        for drug in sorted(per_drug_rows)
    }
    eligible_drugs = {
        drug: accuracy for drug, accuracy in per_drug_accuracy.items()
        if per_drug_rows[drug] >= 25
    }
    drugs_at_75 = sum(accuracy >= 0.75 for accuracy in eligible_drugs.values())
    event_metrics = score_event_predictions(
        np.asarray(event_actual), np.asarray(event_predicted),
        kind="state", event_states={3, 4})
    raw_route = bool(mean_accuracy >= 0.75 and mean_balanced >= 0.75)
    event_route = bool(mean_accuracy >= 0.65
                       and event_metrics["true_positive_precision"] is not None
                       and event_metrics["true_positive_precision"] >= 0.80)
    return {
        "protocol": "rolling_origin_next_month_ndc_shortage_pressure_five_state",
        "fold_count": len(folds),
        "test_rows": int(sum(x["test_rows"] for x in folds)),
        "mean_model_accuracy": mean_accuracy,
        "mean_model_balanced_accuracy": mean_balanced,
        "mean_persistence_accuracy": float(np.mean([
            x["persistence"]["accuracy"] for x in folds])),
        "mean_persistence_balanced_accuracy": persistence_balanced,
        "model_balanced_improvement_vs_persistence": mean_balanced - persistence_balanced,
        "state_counts_in_scored_rows": observed_state_counts,
        "per_drug_rows_minimum": 25,
        "per_drug_count_with_minimum_rows": len(eligible_drugs),
        "per_drug_count_at_or_above_75_percent": int(drugs_at_75),
        "per_drug_accuracy": per_drug_accuracy,
        "event_metrics": event_metrics,
        "raw_accuracy_route_passed": raw_route,
        "event_true_positive_route_passed": event_route,
        "publishable_candidate": bool(
            len(folds) >= 3 and (raw_route or event_route)),
        "scope": "national FDA supplier-count pressure proxy for Arkansas-exposed NDCs; not inventory truth",
        "state_definition": STATE_NAMES,
        "folds": folds,
    }
