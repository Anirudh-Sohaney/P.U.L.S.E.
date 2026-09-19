"""Strict next-quarter evaluation of Arkansas DEA ARCOS distribution proxies.

ARCOS Report 01 measures grams reported by manufacturers/distributors, grouped
by Arkansas ZIP3 and controlled-substance code. It is a regional distribution
proxy, not pharmacy inventory or a direct demand observation. Missing source
quarters remain missing; they are never converted to zero.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .regression import LogisticRidge, RidgeLinear
from .event_accuracy import score_event_predictions

SOURCE_RELATIVE = (
    "targeted_additions/dea_arcos_arkansas/data/"
    "arcos_arkansas_retail_summary.csv.gz"
)
QUARTER_COLUMNS = ["quarter1_grams", "quarter2_grams", "quarter3_grams", "quarter4_grams"]
BASE_FEATURES = ["current_log", "lag1_log", "lag4_log", "rolling4_log", "calendar_quarter"]
ARCOS_STATE_NAMES = {0: "low", 1: "mid", 2: "high"}


def _grams(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    return float(str(value).replace(",", "").strip())


def load_arcos(path) -> pd.DataFrame:
    """Load and normalize the public Arkansas ARCOS summary artifact."""
    raw = pd.read_csv(path, low_memory=False)
    required = {"year", "drug_code", "zip3", *QUARTER_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"ARCOS artifact missing columns: {sorted(missing)}")
    rows = []
    for _, row in raw.iterrows():
        for quarter, column in enumerate(QUARTER_COLUMNS, start=1):
            grams = _grams(row[column])
            if not np.isfinite(grams):
                continue
            rows.append({
                "year": int(row["year"]), "quarter": quarter,
                "period": f"{int(row['year']):04d}-Q{quarter}",
                "period_index": int(row["year"]) * 4 + quarter - 1,
                "drug_code": str(row["drug_code"]),
                "drug_name": str(row.get("drug_name", "")),
                "zip3": str(row["zip3"]).zfill(3), "grams": grams,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("ARCOS artifact has no numeric quarterly observations")
    out = (out.groupby(["year", "quarter", "period", "period_index",
                        "drug_code", "drug_name", "zip3"], as_index=False)["grams"]
           .sum().sort_values(["zip3", "drug_code", "period_index"])
           .reset_index(drop=True))
    out["log_grams"] = np.log1p(out["grams"].clip(lower=0))
    return out


def build_next_quarter_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Create rows whose target is the exact next calendar quarter."""
    df = panel.sort_values(["zip3", "drug_code", "period_index"]).reset_index(drop=True)
    group = df.groupby(["zip3", "drug_code"], sort=False)
    prior_index = group["period_index"].shift(1)
    prior4_index = group["period_index"].shift(4)
    df["lag1_log"] = group["log_grams"].shift(1).where(
        df["period_index"] - prior_index == 1)
    df["lag4_log"] = group["log_grams"].shift(4).where(
        df["period_index"] - prior4_index == 4)
    # Rolling history is valid only when all four preceding calendar quarters
    # exist; this prevents a source gap from becoming an artificial zero.
    vals = []
    for _, sub in df.groupby(["zip3", "drug_code"], sort=False):
        arr = sub["log_grams"].to_numpy(float)
        idx = sub["period_index"].to_numpy(int)
        out = np.full(len(sub), np.nan)
        for i in range(len(sub)):
            if i >= 4 and idx[i] - idx[i - 4] == 4 and np.all(np.diff(idx[i - 4:i + 1]) == 1):
                out[i] = float(np.mean(arr[i - 4:i]))
        vals.extend(out)
    df["rolling4_log"] = vals
    next_index = group["period_index"].shift(-1)
    consecutive = next_index - df["period_index"] == 1
    df["target"] = group["grams"].shift(-1).where(consecutive)
    df["target_period_index"] = next_index.where(consecutive)
    df["target_year"] = (df["target_period_index"] // 4).astype("Int64")
    df["target_quarter"] = (df["target_period_index"] % 4 + 1).astype("Int64")
    df["target_log"] = np.log1p(df["target"].clip(lower=0))
    df["current_log"] = df["log_grams"]
    df["calendar_quarter"] = df["quarter"]
    # The current observation is known at forecast time; lagged fields are
    # retained as explicit historical context. Missing history is imputed from
    # training statistics later, never from future observations.
    return df.dropna(subset=["target"]).reset_index(drop=True)


def build_latest_scoring_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Build current rows for a next-quarter forecast without a target.

    The final observed quarter for a ZIP3/drug pair has no future label and is
    therefore absent from ``build_next_quarter_view``.  This function keeps
    that row and computes only history available at its forecast timestamp.
    """
    df = panel.sort_values(["zip3", "drug_code", "period_index"]).copy()
    group = df.groupby(["zip3", "drug_code"], sort=False)
    prior_index = group["period_index"].shift(1)
    prior4_index = group["period_index"].shift(4)
    df["lag1_log"] = group["log_grams"].shift(1).where(
        df["period_index"] - prior_index == 1)
    df["lag4_log"] = group["log_grams"].shift(4).where(
        df["period_index"] - prior4_index == 4)
    rolling = np.full(len(df), np.nan)
    for _, sub in df.groupby(["zip3", "drug_code"], sort=False):
        positions = sub.index.to_numpy()
        values = sub["log_grams"].to_numpy(float)
        indices = sub["period_index"].to_numpy(int)
        for i in range(4, len(sub)):
            if (indices[i] - indices[i - 4] == 4
                    and np.all(np.diff(indices[i - 4:i + 1]) == 1)):
                rolling[positions[i]] = float(np.mean(values[i - 4:i]))
    df["rolling4_log"] = rolling
    df["current_log"] = df["log_grams"]
    df["calendar_quarter"] = df["quarter"]
    return df.groupby(["zip3", "drug_code"], as_index=False).tail(1).reset_index(drop=True)


def forecast_latest_arcos(view: pd.DataFrame, panel: pd.DataFrame,
                          validation_quarters: int = 4) -> pd.DataFrame:
    """Forecast each latest ZIP3/drug distribution row for the next quarter.

    Candidate selection uses only the final observed validation quarters and
    chooses between the learned log-ridge model and temporal baselines. The
    returned values are distribution proxies, not direct demand labels.
    """
    if view.empty or panel.empty:
        return pd.DataFrame()
    periods = sorted(view["period_index"].unique())
    if len(periods) <= validation_quarters:
        return pd.DataFrame()
    fit_periods = periods[:-validation_quarters]
    val_periods = periods[-validation_quarters:]
    fit = view[view["period_index"].isin(fit_periods)]
    validation = view[view["period_index"].isin(val_periods)]
    if fit.empty or validation.empty:
        return pd.DataFrame()
    model_val, _ = _fit_predict(fit, validation)
    candidates = {
        "ridge_log1p": _metric(validation["target"].to_numpy(float), model_val),
        **{name: _metric(validation["target"].to_numpy(float), pred)
           for name, pred in _baselines(validation).items()},
    }
    selected = min(candidates, key=lambda name: candidates[name]["wape"])
    latest = build_latest_scoring_view(panel)
    # A pair that disappeared from later ARCOS releases is not a current
    # forecast candidate.  Keep it in historical evaluation, but fail closed
    # rather than emitting a stale forecast for its last observed quarter.
    latest_period = int(panel["period_index"].max())
    latest = latest[latest["period_index"].eq(latest_period)].reset_index(drop=True)
    if latest.empty:
        return pd.DataFrame()
    if selected == "ridge_log1p":
        prediction, _ = _fit_predict(view, latest.assign(target_log=np.nan))
    else:
        prediction = _baselines(latest)[selected]
    next_index = latest["period_index"].to_numpy(int) + 1
    out = latest[["zip3", "drug_code", "drug_name", "period", "grams"]].copy()
    out["prediction"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    out["forecast_period_index"] = next_index
    out["forecast_period"] = [
        f"{int(index // 4):04d}-Q{int(index % 4) + 1}" for index in next_index
    ]
    out["selected_model"] = selected
    out["validation_wape"] = float(candidates[selected]["wape"])
    return out.reset_index(drop=True)


def _feature_spec(train: pd.DataFrame) -> List[str]:
    zips = sorted(train["zip3"].unique())
    drugs = sorted(train["drug_code"].unique())
    return BASE_FEATURES + [f"zip3_{x}" for x in zips] + [f"drug_{x}" for x in drugs]


def _matrix(frame: pd.DataFrame, spec: List[str], means: Dict[str, float] | None = None) -> np.ndarray:
    base = frame.reindex(columns=BASE_FEATURES, fill_value=np.nan).copy()
    for col in BASE_FEATURES:
        fill = 0.0 if means is None else means.get(col, 0.0)
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(fill)
    identity = {}
    for prefix, key in (("zip3_", "zip3"), ("drug_", "drug_code")):
        for col in (x for x in spec if x.startswith(prefix)):
            identity[col] = (frame[key].astype(str) == col[len(prefix):]).astype(float)
    if identity:
        base = pd.concat([base, pd.DataFrame(identity, index=frame.index)], axis=1)
    return base.reindex(columns=spec, fill_value=0.0).to_numpy(float)


def _metric(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float); pred = np.maximum(np.asarray(pred, float), 0.0)
    relative_error = np.abs(pred - y) / np.maximum(np.abs(y), 1e-9)
    return {"mae": float(np.mean(np.abs(y - pred))),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "wape": float(np.sum(np.abs(y - pred)) / max(np.sum(np.abs(y)), 1e-9)),
            "within_5pct": float(np.mean(relative_error <= 0.05))}


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    spec = _feature_spec(train)
    means = {c: float(train[c].mean()) for c in BASE_FEATURES}
    model = RidgeLinear(alpha=10.0).fit(
        _matrix(train, spec, means), train["target_log"].to_numpy(float), spec)
    return np.expm1(model.predict(_matrix(test, spec, means))), spec


def _baselines(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    prev = np.expm1(frame["current_log"].to_numpy(float))
    seasonal = np.expm1(frame["lag4_log"].fillna(frame["current_log"]).to_numpy(float))
    return {"previous_quarter": prev, "seasonal_quarter": seasonal}


def evaluate_arcos_view(view: pd.DataFrame, train_end_year: int = 2018,
                        validation_years: Tuple[int, int] = (2019, 2020)) -> Dict:
    train = view[view["year"] <= train_end_year].copy()
    val = view[view["year"].between(*validation_years)].copy()
    test = view[view["year"] > validation_years[1]].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("ARCOS chronological split is empty")
    val_pred, _ = _fit_predict(train, val)
    test_pred, _ = _fit_predict(pd.concat([train, val]), test)
    yv, yt = val["target"].to_numpy(float), test["target"].to_numpy(float)
    val_rows = [{"model": "ridge_log1p", **_metric(yv, val_pred)}]
    for name, pred in _baselines(val).items():
        val_rows.append({"model": name, **_metric(yv, pred)})
    selected = min(val_rows, key=lambda x: x["wape"])["model"]
    test_preds = {"ridge_log1p": test_pred, **_baselines(test)}
    leaderboard = []
    for row in val_rows:
        out = {**row, "selected": row["model"] == selected}
        out.update({"test_" + k: v for k, v in
                    _metric(yt, test_preds[row["model"]]).items()})
        leaderboard.append(out)
    naive_wape = min(r["test_wape"] for r in leaderboard if r["model"] != "ridge_log1p")
    selected_test = next(r for r in leaderboard if r["model"] == selected)
    improvement = (naive_wape - selected_test["test_wape"]) / max(naive_wape, 1e-9)
    return {"split": "strict_next_quarter_arcos_zip3_drug", "n_rows": {
        "train": len(train), "validation": len(val), "test": len(test)},
        "years": {"train": [int(train.year.min()), int(train.year.max())],
                  "validation": [int(val.year.min()), int(val.year.max())],
                  "test": [int(test.year.min()), int(test.year.max())]},
        "n_zip3": int(view.zip3.nunique()), "n_drugs": int(view.drug_code.nunique()),
        "selected_model": selected, "best_naive_wape": naive_wape,
        "selected_test_wape": selected_test["test_wape"],
        "improvement_vs_best_naive": improvement,
        "publishable_candidate": bool(improvement >= 0.10),
        "leaderboard": pd.DataFrame(leaderboard),
        "scope": "ARCOS reported controlled-substance distribution proxy; not inventory or direct demand.",
    }


def evaluate_arcos_rolling_view(view: pd.DataFrame, min_train_quarters: int = 12,
                                test_window_quarters: int = 4,
                                validation_quarters: int = 4) -> Dict:
    """Rolling-origin evaluation with validation-only candidate selection.

    Each fold reserves the final ``validation_quarters`` from the available
    training history. The selected candidate is then refit on all pre-test
    observations and evaluated once on the future test window.
    """
    periods = sorted(view["period_index"].unique())
    folds = []
    start = min_train_quarters
    while start + test_window_quarters <= len(periods):
        train_periods = periods[:start]
        fit_periods = train_periods[:-validation_quarters]
        val_periods = train_periods[-validation_quarters:]
        fit = view[view.period_index.isin(fit_periods)]
        val = view[view.period_index.isin(val_periods)]
        train = view[view.period_index.isin(train_periods)]
        test = view[view.period_index.isin(periods[start:start + test_window_quarters])]
        if len(fit) and len(val) and len(train) and len(test):
            val_pred, _ = _fit_predict(fit, val)
            val_candidates = {"ridge_log1p": _metric(
                val.target.to_numpy(float), val_pred), **{
                    name: _metric(val.target.to_numpy(float), pred)
                    for name, pred in _baselines(val).items()}}
            selected = min(val_candidates,
                           key=lambda name: val_candidates[name]["wape"])
            model_pred, _ = _fit_predict(train, test)
            test_candidates = {"ridge_log1p": model_pred, **_baselines(test)}
            y = test.target.to_numpy(float)
            rows = {"fold": len(folds) + 1, "train_end": int(periods[start - 1]),
                    "test_start": int(periods[start]), "test_end": int(periods[start + test_window_quarters - 1]),
                    "validation_quarters": validation_quarters,
                    "selected_model": selected,
                    "selected": _metric(y, test_candidates[selected])}
            rows.update({name: _metric(y, pred)
                         for name, pred in test_candidates.items()})
            folds.append(rows)
        start += test_window_quarters
    fdf = pd.DataFrame(folds)
    if fdf.empty:
        return {"fold_count": 0, "publishable_rolling_candidate": False, "folds": fdf}
    means = {name: {k: float(fdf[name].apply(lambda x: x[k]).mean())
                    for k in ("mae", "rmse", "wape", "within_5pct")}
             for name in ("ridge_log1p", "previous_quarter", "seasonal_quarter", "selected")}
    best_naive = min(means["previous_quarter"]["wape"], means["seasonal_quarter"]["wape"])
    imp = (best_naive - means["selected"]["wape"]) / max(best_naive, 1e-9)
    return {"split": "rolling_origin_next_quarter_arcos_zip3_drug", "fold_count": len(fdf),
            "min_train_quarters": min_train_quarters,
            "test_window_quarters": test_window_quarters,
            "validation_quarters": validation_quarters,
            "means": means, "improvement_vs_best_naive": imp,
            "publishable_rolling_candidate": bool(len(fdf) >= 3 and imp >= 0.10),
            "folds": fdf, "scope": "ARCOS reported controlled-substance distribution proxy; not inventory or direct demand."}


def _arcos_state(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Encode log-distribution into training-defined low/mid/high states."""
    return np.digitize(np.asarray(values, dtype=float), thresholds).astype(int)


def _arcos_state_metrics(actual: np.ndarray, predicted: np.ndarray,
                         state_count: int) -> Dict:
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


def _arcos_state_predict(train: pd.DataFrame, target: pd.DataFrame,
                         train_states: np.ndarray, alpha: float,
                         state_count: int) -> np.ndarray:
    """Predict ARCOS states with the auditable one-v-rest logistic layer."""
    spec = _feature_spec(train)
    means = {column: float(train[column].mean()) for column in BASE_FEATURES}
    x_train = _matrix(train, spec, means)
    x_target = _matrix(target, spec, means)
    probabilities = []
    for state in range(state_count):
        model = LogisticRidge(alpha=alpha).fit(
            x_train, (train_states == state).astype(float), spec)
        probabilities.append(model.predict_proba(x_target))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_arcos_state_rolling_view(view: pd.DataFrame,
                                      min_train_quarters: int = 12,
                                      test_window_quarters: int = 4,
                                      validation_quarters: int = 4,
                                      state_count: int = 5) -> Dict:
    """Evaluate a leakage-safe quarterly ARCOS ordered pressure state.

    Thresholds are learned from the fit portion of each fold and held fixed
    for validation and test. Persistence is eligible for selection because
    the metric is intended as a stable external context signal, not a claim
    that a learned model improves the observed distribution series.
    """
    if state_count < 3:
        raise ValueError("state_count must be at least 3")
    periods = sorted(view["period_index"].unique())
    folds = []
    event_actual: list[int] = []
    event_predicted: list[int] = []
    per_drug_correct: Dict[str, int] = {}
    per_drug_rows: Dict[str, int] = {}
    per_zip_correct: Dict[str, int] = {}
    per_zip_rows: Dict[str, int] = {}
    state_counts = {str(state): 0 for state in range(state_count)}
    start = min_train_quarters
    while start + test_window_quarters <= len(periods):
        train_periods = periods[:start]
        fit_periods = train_periods[:-validation_quarters]
        val_periods = train_periods[-validation_quarters:]
        fit = view[view.period_index.isin(fit_periods)]
        validation = view[view.period_index.isin(val_periods)]
        train = view[view.period_index.isin(train_periods)]
        test = view[view.period_index.isin(
            periods[start:start + test_window_quarters])]
        if len(fit) and len(validation) and len(train) and len(test):
            thresholds = np.quantile(
                fit["target_log"].to_numpy(float),
                np.arange(1, state_count) / state_count,
            )
            if len(np.unique(thresholds)) < 2:
                start += test_window_quarters
                continue
            fit_states = _arcos_state(fit["target_log"], thresholds)
            validation_states = _arcos_state(validation["target_log"], thresholds)
            test_states = _arcos_state(test["target_log"], thresholds)
            validation_persistence = _arcos_state(validation["current_log"], thresholds)
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                learned = _arcos_state_predict(
                    fit, validation, fit_states, alpha, state_count)
                scores[alpha] = _arcos_state_metrics(
                    validation_states, learned, state_count)[
                    "balanced_accuracy"]
            alpha = max(scores, key=scores.get)
            learned = _arcos_state_predict(
                train, test, _arcos_state(train["target_log"], thresholds),
                alpha, state_count)
            persistence = _arcos_state(test["current_log"], thresholds)
            persistence_validation_score = _arcos_state_metrics(
                validation_states, validation_persistence, state_count)["balanced_accuracy"]
            learned_validation_score = scores[alpha]
            if learned_validation_score > persistence_validation_score:
                selected, selected_name = learned, "logistic_one_vs_rest"
            else:
                selected, selected_name = persistence, "persistence"
            event_actual.extend(test_states.tolist())
            event_predicted.extend(selected.tolist())
            metrics = _arcos_state_metrics(test_states, selected, state_count)
            persistence_metrics = _arcos_state_metrics(test_states, persistence, state_count)
            for key, actual, predicted in zip(
                    test["drug_code"].astype(str), test_states, selected):
                per_drug_rows[key] = per_drug_rows.get(key, 0) + 1
                per_drug_correct[key] = per_drug_correct.get(key, 0) + int(actual == predicted)
            for key, actual, predicted in zip(test["zip3"].astype(str), test_states, selected):
                per_zip_rows[key] = per_zip_rows.get(key, 0) + 1
                per_zip_correct[key] = per_zip_correct.get(key, 0) + int(actual == predicted)
            for state, count in metrics["state_counts"].items():
                state_counts[state] += count
            folds.append({
                "fold": len(folds) + 1,
                "train_end": int(periods[start - 1]),
                "test_start": int(periods[start]),
                "test_end": int(periods[start + test_window_quarters - 1]),
                "selected_model": selected_name,
                "thresholds_log1p": [float(value) for value in thresholds],
                "test_rows": int(len(test)), "model": metrics,
                "persistence": persistence_metrics,
            })
        start += test_window_quarters
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    per_drug_accuracy = {key: per_drug_correct[key] / per_drug_rows[key]
                         for key in sorted(per_drug_rows)}
    per_zip_accuracy = {key: per_zip_correct[key] / per_zip_rows[key]
                        for key in sorted(per_zip_rows)}
    drugs_75 = sum(value >= 0.75 for key, value in per_drug_accuracy.items()
                   if per_drug_rows[key] >= 25)
    return {
        "protocol": "rolling_origin_next_quarter_arcos_zip3_drug_distribution_state",
        "state_count": state_count,
        "fold_count": len(folds),
        "test_rows": int(sum(row["test_rows"] for row in folds)),
        "drug_count": int(view["drug_code"].nunique()),
        "zip3_count": int(view["zip3"].nunique()),
        "mean_model_accuracy": float(np.mean([
            row["model"]["accuracy"] for row in folds])),
        "mean_model_balanced_accuracy": float(np.mean([
            row["model"]["balanced_accuracy"] for row in folds])),
        "mean_persistence_accuracy": float(np.mean([
            row["persistence"]["accuracy"] for row in folds])),
        "mean_persistence_balanced_accuracy": float(np.mean([
            row["persistence"]["balanced_accuracy"] for row in folds])),
        "state_counts_in_scored_rows": state_counts,
        "per_drug_rows_minimum": 25,
        "per_drug_count_at_or_above_75_percent": int(drugs_75),
        "per_drug_accuracy": per_drug_accuracy,
        "per_zip3_accuracy": per_zip_accuracy,
        "event_metrics": score_event_predictions(
            np.asarray(event_actual), np.asarray(event_predicted), kind="state",
            event_states={3, 4}),
        "publishable_candidate": bool(
            len(folds) >= 3
            and all(state_counts.values())
            and np.mean([row["model"]["accuracy"] for row in folds]) >= 0.65
            and np.mean([row["model"]["balanced_accuracy"] for row in folds]) >= 0.65),
        "scope": "DEA ARCOS controlled-substance distribution state proxy; not inventory or direct demand",
        "state_definition": {str(state): f"quantile_{state + 1}_of_{state_count}"
                             for state in range(state_count)},
        "folds": folds,
    }
