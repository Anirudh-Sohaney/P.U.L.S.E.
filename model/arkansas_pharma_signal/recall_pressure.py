"""Termination-aware three-state FDA drug-recall pressure evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge
from .event_accuracy import score_event_predictions


FEATURES = ["recall_state", "lag1_state", "lag2_state", "rolling3_state", "calendar_month"]


def _extract_ndcs(row: pd.Series) -> list[str]:
    text = " ".join(str(row.get(column, "")) for column in
                    ("product_description", "code_info"))
    matches = re.findall(r"(?i)\bNDC\s*:?\s*([0-9]{4,5}-[0-9]{3,4}-[0-9]{1,2})\b", text)
    normalized = set()
    for value in matches:
        parts = value.split("-")
        # FDA NDCs may use 4-4-2, 5-3-2, or 5-4-1 segments. Padding to the
        # canonical 5-4-2 form makes the first nine digits joinable to the
        # shortage archive's NDC9 product key.
        normalized.add("".join(part.zfill(width) for part, width in
                                zip(parts, (5, 4, 2)))[:9])
    return sorted(normalized)


def load_recall_events(paths: list[Path]) -> pd.DataFrame:
    """Load deduplicated FDA enforcement events and extract product NDCs."""
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "event_id" in frame:
        frame = frame.drop_duplicates("event_id").reset_index(drop=True)
    starts = pd.to_datetime(frame["recall_initiation_date"].astype(str), errors="coerce")
    ends = pd.to_datetime(frame.get("termination_date", pd.Series(index=frame.index)),
                          errors="coerce")
    reported = pd.to_datetime(frame.get("report_date", pd.Series(index=frame.index)),
                              errors="coerce")
    records = []
    for index, row in frame.iterrows():
        start = starts.iloc[index]
        if pd.isna(start):
            continue
        severity = {"Class I": 2, "Class II": 1, "Class III": 1}.get(
            str(row.get("classification", "")), 1)
        for ndc in _extract_ndcs(row):
            records.append({
                "ndc": ndc, "supplier": str(row.get("recalling_firm", "")).strip(),
                "start": start.to_period("M"),
                "end": ends.iloc[index].to_period("M") if pd.notna(ends.iloc[index]) else pd.NaT,
                "reported": (reported.iloc[index].to_period("M")
                             if pd.notna(reported.iloc[index]) else start.to_period("M")),
                "severity": severity, "event_id": str(row.get("event_id", "")),
            })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("no FDA recall events with extractable NDC identifiers")
    return result.drop_duplicates(["ndc", "start", "end", "reported", "severity", "event_id"])


def build_supplier_recall_panel(events: pd.DataFrame, *, include_source_edge: bool = False) -> pd.DataFrame:
    """Build three-state monthly recall pressure for supplier x NDC pairs."""
    required = {"ndc", "supplier", "start", "end", "severity"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"supplier recall events missing columns: {sorted(missing)}")
    events = events[events["supplier"].fillna("").astype(str).str.strip().ne("")].copy()
    if events.empty:
        raise ValueError("no recall events contain a supplier identifier")
    source_last = max(events["start"].max(), events["end"].dropna().max()
                      if events["end"].notna().any() else events["start"].max())
    bounds = events.groupby(["supplier", "ndc"], as_index=False)["start"].min()
    base = pd.concat([
        pd.DataFrame({"supplier": row.supplier, "ndc": row.ndc,
                      "month": pd.period_range(row.start, source_last, freq="M")})
        for row in bounds.itertuples(index=False)
    ], ignore_index=True)
    active_frames = []
    for row in events.itertuples(index=False):
        end = row.end if pd.notna(row.end) else source_last
        active_frames.append(pd.DataFrame({
            "supplier": row.supplier, "ndc": row.ndc,
            "month": pd.period_range(row.start, end, freq="M"),
            "severity": int(row.severity),
        }))
    active = (pd.concat(active_frames, ignore_index=True)
              .groupby(["supplier", "ndc", "month"], as_index=False)["severity"].max())
    panel = base.merge(active, on=["supplier", "ndc", "month"], how="left")
    panel["recall_state"] = panel["severity"].fillna(0).astype(int)
    panel = panel.drop(columns="severity").sort_values(["supplier", "ndc", "month"])
    grouped = panel.groupby(["supplier", "ndc"], sort=False)
    panel["next_month"] = grouped["month"].shift(-1)
    panel["target_state"] = grouped["recall_state"].shift(-1)
    panel["lag1_state"] = grouped["recall_state"].shift(1)
    panel["lag2_state"] = grouped["recall_state"].shift(2)
    panel["rolling3_state"] = grouped["recall_state"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    panel["calendar_month"] = panel["month"].dt.month
    if not include_source_edge:
        panel = panel[panel["next_month"].eq(panel["month"] + 1)].copy()
        panel["target_state"] = panel["target_state"].astype(int)
    return panel.reset_index(drop=True)


def build_recall_panel(events: pd.DataFrame, *, include_source_edge: bool = False) -> pd.DataFrame:
    """Build active recall states, carrying ongoing events to the source edge."""
    required = {"ndc", "start", "end", "severity"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"recall events missing columns: {sorted(missing)}")
    source_last = max(events["start"].max(), events["end"].dropna().max()
                      if events["end"].notna().any() else events["start"].max())
    # Build the complete observed interval once, then expand recall intervals
    # and aggregate their maximum severity. This avoids filtering every event
    # group once for every month (which is quadratic on the archive).
    bounds = events.groupby("ndc", as_index=False)["start"].min()
    base = pd.concat([
        pd.DataFrame({"ndc": row.ndc,
                      "month": pd.period_range(row.start, source_last, freq="M")})
        for row in bounds.itertuples(index=False)
    ], ignore_index=True)
    active_frames = []
    for row in events.itertuples(index=False):
        end = row.end if pd.notna(row.end) else source_last
        active_frames.append(pd.DataFrame({
            "ndc": row.ndc,
            "month": pd.period_range(row.start, end, freq="M"),
            "severity": int(row.severity),
        }))
    active = (pd.concat(active_frames, ignore_index=True)
              .groupby(["ndc", "month"], as_index=False)["severity"].max())
    panel = base.merge(active, on=["ndc", "month"], how="left")
    panel["recall_state"] = panel["severity"].fillna(0).astype(int)
    panel = panel.drop(columns="severity").sort_values(["ndc", "month"]).reset_index(drop=True)
    grouped = panel.groupby("ndc", sort=False)
    panel["next_month"] = grouped["month"].shift(-1)
    panel["target_state"] = grouped["recall_state"].shift(-1)
    panel["lag1_state"] = grouped["recall_state"].shift(1)
    panel["lag2_state"] = grouped["recall_state"].shift(2)
    panel["rolling3_state"] = grouped["recall_state"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    panel["calendar_month"] = panel["month"].dt.month
    if not include_source_edge:
        panel = panel[panel["next_month"].eq(panel["month"] + 1)].copy()
        panel["target_state"] = panel["target_state"].astype(int)
    return panel.reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int); predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(3):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {"accuracy": float(np.mean(actual == predicted)),
            "balanced_accuracy": float(np.mean(recalls)),
            "state_counts": {str(state): int(np.sum(actual == state)) for state in range(3)}}


def _predict(train: pd.DataFrame, target: pd.DataFrame, alpha: float) -> np.ndarray:
    probabilities = []
    x_train = train[FEATURES].fillna(0).to_numpy(float)
    x_target = target[FEATURES].fillna(0).to_numpy(float)
    for state in range(3):
        model = LogisticRidge(alpha=alpha).fit(
            x_train, train["target_state"].eq(state).astype(float), FEATURES)
        probabilities.append(model.predict_proba(x_target))
    return np.column_stack(probabilities).argmax(axis=1)


def _source_edge_recall_predictions(
        panel: pd.DataFrame, *, group_columns: tuple[str, ...],
        validation_months: int = 3, min_train_months: int = 24,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a chronological logistic model or persistence for the source edge."""
    latest_month = panel["month"].max()
    latest = panel[panel["month"].eq(latest_month)].copy()
    history = panel[panel["month"].lt(latest_month)].copy()
    history = history[history["target_state"].notna()].copy()
    periods = sorted(history["month"].unique())
    if len(periods) < min_train_months + validation_months or latest.empty:
        latest["forecast_state"] = latest["recall_state"].astype(int)
        return latest, {"method": "persistence", "reason": "insufficient_history"}
    validation_periods = periods[-validation_months:]
    train_periods = periods[:-validation_months]
    train = history[history["month"].isin(train_periods)]
    validation = history[history["month"].isin(validation_periods)]
    if train.empty or validation.empty:
        latest["forecast_state"] = latest["recall_state"].astype(int)
        return latest, {"method": "persistence", "reason": "empty_split"}
    scores = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        scores[alpha] = _metrics(
            validation["target_state"].to_numpy(int),
            _predict(train, validation, alpha),
        )["balanced_accuracy"]
    alpha = max(scores, key=scores.get)
    learned_validation = _predict(train, validation, alpha)
    persistence_validation = validation["recall_state"].to_numpy(int)
    learned_score = _metrics(validation["target_state"], learned_validation)
    persistence_score = _metrics(validation["target_state"], persistence_validation)
    if persistence_score["balanced_accuracy"] >= learned_score["balanced_accuracy"]:
        latest["forecast_state"] = latest["recall_state"].astype(int)
        method = "persistence"
    else:
        combined = pd.concat([train, validation], ignore_index=True)
        latest["forecast_state"] = _predict(combined, latest, alpha)
        method = "logistic_one_vs_rest"
    return latest, {
        "method": method, "alpha": float(alpha),
        "validation_learned_balanced_accuracy": learned_score["balanced_accuracy"],
        "validation_persistence_balanced_accuracy": persistence_score["balanced_accuracy"],
        "validation_months": validation_months,
        "group_columns": list(group_columns),
    }


def evaluate_recall_rolling(view: pd.DataFrame, min_train_months: int = 24,
                            validation_months: int = 3, test_months: int = 3,
                            step_months: int = 1) -> dict[str, Any]:
    periods = sorted(view["month"].unique())
    folds = []
    per_drug_correct: dict[str, int] = {}
    per_drug_rows: dict[str, int] = {}
    for index in range(min_train_months, len(periods) - validation_months - test_months + 1,
                       max(1, step_months)):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:index + validation_months + test_months]
        expected = pd.period_range(train_periods[-1] + 1, test_periods[-1], freq="M")
        if len(expected) != validation_months + test_months:
            continue
        train = view[view.month.isin(train_periods)]
        validation = view[view.month.isin(validation_periods)]
        test = view[view.month.isin(test_periods)]
        if train.empty or validation.empty or test.empty:
            continue
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            scores[alpha] = _metrics(
                validation.target_state, _predict(train, validation, alpha)
            )["balanced_accuracy"]
        alpha = max(scores, key=scores.get)
        actual = test.target_state.to_numpy(int)
        learned = _predict(pd.concat([train, validation]), test, alpha)
        persistence = test.recall_state.to_numpy(int)
        validation_persistence = validation.recall_state.to_numpy(int)
        selected = (persistence if _metrics(validation.target_state, validation_persistence)[
            "balanced_accuracy"] >= scores[alpha] else learned)
        folds.append({"validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
                      "test_period": f"{test_periods[0]}/{test_periods[-1]}",
                      "train_rows": int(len(train)), "validation_rows": int(len(validation)),
                      "test_rows": int(len(test)),
                      "selected_model": "persistence" if selected is persistence else "logistic_one_vs_rest",
                      "model": _metrics(actual, selected),
                      "persistence": _metrics(actual, persistence)})
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    accuracy = float(np.mean([x["model"]["accuracy"] for x in folds]))
    balanced = float(np.mean([x["model"]["balanced_accuracy"] for x in folds]))
    counts = {str(state): sum(x["model"]["state_counts"][str(state)] for x in folds)
              for state in range(3)}
    return {"protocol": "rolling_origin_next_month_fda_recall_state",
            "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
            "ndc_count": int(view["ndc"].nunique()), "mean_model_accuracy": accuracy,
            "mean_model_balanced_accuracy": balanced,
            "state_counts_in_scored_rows": counts,
            "publishable_candidate": bool(len(folds) >= 3 and accuracy >= 0.65
                                           and balanced >= 0.65 and all(counts.values())),
            "scope": "FDA NDC recall-severity pressure proxy; not pharmacy availability",
            "state_definition": {"0": "no_active_recall", "1": "active_class_II_or_III",
                                 "2": "active_class_I"}, "folds": folds}


def evaluate_supplier_recall_rolling(view: pd.DataFrame, min_train_months: int = 24,
                                     validation_months: int = 3, test_months: int = 3,
                                     step_months: int = 1) -> dict[str, Any]:
    """Evaluate supplier x NDC recall states with the shared rolling protocol."""
    result = evaluate_recall_rolling(view, min_train_months, validation_months,
                                     test_months, step_months)
    result["protocol"] = "rolling_origin_next_month_fda_supplier_ndc_recall_state"
    result["supplier_count"] = int(view["supplier"].nunique())
    result["ndc_count"] = int(view["ndc"].nunique())
    result["supplier_ndc_pair_count"] = int(view[["supplier", "ndc"]].drop_duplicates().shape[0])
    result["scope"] = "FDA supplier x NDC recall-severity pressure proxy; not pharmacy availability"
    return result


def _numeric_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score numeric recall severity without treating zero as missing."""
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


def evaluate_recall_numeric_rolling(view: pd.DataFrame, min_train_months: int = 24,
                                    validation_months: int = 3, test_months: int = 3,
                                    step_months: int = 1) -> dict[str, Any]:
    """Evaluate next-month recall severity as a numeric proxy."""
    periods = sorted(view["month"].unique())
    folds = []
    event_actual: list[float] = []
    event_predicted: list[float] = []
    per_drug_correct: dict[str, int] = {}
    per_drug_rows: dict[str, int] = {}
    for index in range(min_train_months, len(periods) - validation_months - test_months + 1,
                       max(1, step_months)):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:index + validation_months + test_months]
        expected = pd.period_range(train_periods[-1] + 1, test_periods[-1], freq="M")
        if len(expected) != validation_months + test_months:
            continue
        train = view[view.month.isin(train_periods)]
        validation = view[view.month.isin(validation_periods)]
        test = view[view.month.isin(test_periods)]
        if train.empty or validation.empty or test.empty:
            continue
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            scores[alpha] = _numeric_metrics(
                validation.target_state.to_numpy(float),
                _predict(train, validation, alpha),
            )["wape"]
        alpha = min(scores, key=scores.get)
        actual = test.target_state.to_numpy(float)
        learned = _predict(pd.concat([train, validation]), test, alpha)
        persistence = test.recall_state.to_numpy(float)
        validation_persistence = validation.recall_state.to_numpy(float)
        selected = (persistence if _numeric_metrics(
            validation.target_state.to_numpy(float), validation_persistence)["wape"]
            <= scores[alpha] else learned)
        event_actual.extend(actual.tolist())
        event_predicted.extend(selected.tolist())
        for drug, truth, prediction in zip(
                test["ndc"].astype(str), actual, selected):
            relative_error = abs(float(prediction) - float(truth)) / max(abs(float(truth)), 1.0)
            drug = str(drug)
            per_drug_rows[drug] = per_drug_rows.get(drug, 0) + 1
            per_drug_correct[drug] = per_drug_correct.get(drug, 0) + int(relative_error < 0.05)
        folds.append({
            "validation_period": f"{validation_periods[0]}/{validation_periods[-1]}",
            "test_period": f"{test_periods[0]}/{test_periods[-1]}",
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "selected_model": "persistence" if selected is persistence else "logistic_one_vs_rest",
            "model": _numeric_metrics(actual, selected),
            "persistence": _numeric_metrics(actual, persistence),
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    mean_within5 = float(np.mean([
        fold["model"]["within_5_percent_error"] for fold in folds]))
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
        "protocol": "rolling_origin_next_month_fda_recall_numeric_severity",
        "fold_count": len(folds),
        "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "ndc_count": int(view["ndc"].nunique()),
        "mean_model_wape": float(np.mean([fold["model"]["wape"] for fold in folds])),
        "mean_persistence_wape": float(np.mean([fold["persistence"]["wape"] for fold in folds])),
        "mean_model_within_5_percent_error": mean_within5,
        "per_drug_rows_minimum": 25,
        "per_drug_count_with_minimum_rows": len(eligible_drugs),
        "per_drug_count_at_or_above_75_percent": int(drugs_at_75),
        "per_drug_accuracy": per_drug_accuracy,
        "event_metrics": score_event_predictions(
            np.asarray(event_actual), np.asarray(event_predicted), kind="numeric",
            event_threshold=1.0),
        "publishable_candidate": bool(len(folds) >= 3 and mean_within5 >= 0.65),
        "scope": "FDA recall severity numeric proxy; not pharmacy availability",
        "folds": folds,
    }


def evaluate_supplier_recall_numeric_rolling(
        view: pd.DataFrame, min_train_months: int = 24,
        validation_months: int = 3, test_months: int = 3,
        step_months: int = 1) -> dict[str, Any]:
    """Evaluate supplier-by-NDC numeric recall severity."""
    result = evaluate_recall_numeric_rolling(
        view, min_train_months, validation_months, test_months, step_months)
    result["protocol"] = "rolling_origin_next_month_fda_supplier_ndc_recall_numeric_severity"
    result["supplier_count"] = int(view["supplier"].nunique())
    result["ndc_count"] = int(view["ndc"].nunique())
    result["supplier_ndc_pair_count"] = int(
        view[["supplier", "ndc"]].drop_duplicates().shape[0])
    return result
