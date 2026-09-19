"""FDA-reported supplier-drug shortage event forecasting.

Scope: this is FDA-reported supplier-drug event forecasting. It is not
pharmacy inventory, not county allocation, and not proof of true shortage
absence. A zero label means no observed FDA shortage event for that
supplier-drug pair in that month, not a confirmed absence of shortage.

The evaluation builds a complete monthly supplier-drug panel from the real
normalized FDA shortage records under ``data/S_D/data/by_source/fda_shortages``
and evaluates a strict next-month view: features at month t predict the
shortage_event observed at t+1 for the same supplier-drug pair.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .evaluate import (
    _auroc,
    _auprc,
    _binary_metrics,
    _brier,
    _select_binary_threshold,
)
from .regression import LogisticRidge

#: Base model features at month t. All event features are strictly lagged
#: (months < t); calendar month carries no t+1 information. Supplier/drug
#: one-hot columns are appended per split/fold and fitted on the training
#: slice only, so identities unseen at train time map to all-zero rows.
#: ``current_event`` (month t) is kept in the view for the previous-month
#: persistence baseline only and is deliberately excluded.
FEATURE_COLS = [
    "lag1_event",
    "lag2_event",
    "trailing_events_3",
    "trailing_events_6",
    "drug_global_lag1_event",
    "drug_global_trailing_events_3",
    "drug_global_trailing_events_6",
    "calendar_month",
]

SOURCE_DIR = "S_D/data/by_source/fda_shortages"


def load_archived_fda_shortage_panel(path: Path) -> pd.DataFrame:
    """Load the dated national FDA archive panel as an evaluation view.

    This target is intentionally separate from ``load_fda_shortage_records``:
    archived status rows are national supply evidence and include explicit
    right-censoring, while the legacy event records are the local pipeline's
    event source.  No missing post-archive months are synthesized.
    """
    required = {"ndc9", "supplier", "month", "shortage_active",
                "right_censored"}
    frame = pd.read_csv(path, dtype={"ndc9": str, "supplier": str,
                                     "month": str})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"archived shortage panel missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"ndc9": "drug",
                                  "shortage_active": "shortage_event"}).copy()
    frame["drug"] = frame["drug"].astype(str).str.strip()
    frame["supplier"] = frame["supplier"].astype(str).str.strip()
    frame["month"] = frame["month"].astype(str).str.strip()
    frame["shortage_event"] = pd.to_numeric(
        frame["shortage_event"], errors="raise").astype(int)
    frame["right_censored"] = pd.to_numeric(
        frame["right_censored"], errors="raise").astype(int)
    if frame[["drug", "supplier", "month"]].eq("").any().any():
        raise ValueError("archived shortage panel has empty identifiers")
    if not frame["shortage_event"].isin([0, 1]).all():
        raise ValueError("archived shortage target must be binary")
    return frame.sort_values(["supplier", "drug", "month"]).reset_index(drop=True)


def normalize_supplier(name: object) -> str:
    """Deterministic supplier key: lowercase, alphanumeric runs, single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def normalize_drug(name: object) -> str:
    """Deterministic drug key: lowercase, alphanumeric runs, single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _filter_records(raw_rows: List[Dict]) -> pd.DataFrame:
    """Keep shortage_active records with a drug key, a manufacturer, and a
    valid event_date; normalize supplier and drug keys deterministically."""
    rows: List[Dict] = []
    for rec in raw_rows:
        obs = rec.get("observation", {})
        if obs.get("metric") != "shortage_active":
            continue
        drug = rec.get("drug", {})
        ingredient = drug.get("ingredient") or ""
        ndc = drug.get("ndc") or ""
        manufacturer = drug.get("manufacturer") or ""
        if not (ingredient or ndc):
            continue
        if not manufacturer:
            continue
        event_date = (rec.get("source", {}) or {}).get("event_date")
        if not event_date:
            continue
        try:
            dt = pd.Timestamp(event_date)
        except (ValueError, TypeError):
            continue
        rows.append({
            "supplier": normalize_supplier(manufacturer),
            "drug": normalize_drug(ingredient or ndc),
            "event_date": dt,
            "month": dt.strftime("%Y-%m"),
        })
    return pd.DataFrame(rows)


def load_fda_shortage_records(cfg: Config) -> pd.DataFrame:
    """Load and filter the real FDA shortage records through the cfg-aware
    loader. Fails clearly when the source records are absent."""
    src = cfg.data_dir / SOURCE_DIR
    files = sorted(src.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No FDA shortage source records under {src}. "
            "Run the S_D data pipeline first or check --root."
        )
    raw: List[Dict] = []
    for f in files:
        raw.extend(io.read_json(f))
    records = _filter_records(raw)
    if records.empty:
        raise FileNotFoundError(
            f"No valid shortage_active records with supplier and drug keys "
            f"under {src}."
        )
    return records


def build_monthly_panel(records: pd.DataFrame) -> pd.DataFrame:
    """Complete monthly supplier-drug panel from each pair's first observed
    FDA event month through the global last observed month.

    Every observed (supplier, drug) pair gets a row for every month from its
    own first observed FDA event month to the global last observed month.
    Months before a pair was first observed are not filled. ``shortage_event``
    is 1 when any FDA shortage event was reported for that pair in that month.
    Zero means no observed FDA event, not confirmed no shortage.
    """
    last_month = records["month"].max()
    first = (
        records.groupby(["supplier", "drug"])["month"].min()
        .rename("first_month").reset_index()
    )
    frames = []
    for _, p in first.iterrows():
        months = pd.period_range(
            p["first_month"], last_month, freq="M"
        ).strftime("%Y-%m")
        frames.append(pd.DataFrame({
            "supplier": p["supplier"], "drug": p["drug"], "month": months,
        }))
    panel = pd.concat(frames, ignore_index=True)
    events = (
        records.groupby(["supplier", "drug", "month"])
        .size()
        .rename("n")
        .reset_index()
    )
    panel = panel.merge(events, on=["supplier", "drug", "month"], how="left")
    panel["shortage_event"] = panel["n"].fillna(0).gt(0).astype(int)
    panel = panel.drop(columns="n")
    # Legacy event records have no archive censoring field; their observed
    # event labels are treated as uncensored by this legacy panel builder.
    panel["right_censored"] = 0
    return panel.sort_values(["supplier", "drug", "month"]).reset_index(drop=True)


def _add_drug_global_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lagged market-wide shortage context for each normalized drug.

    The target remains supplier-drug specific.  The added context is the
    maximum observed FDA event across suppliers for the same drug, shifted so
    the supplier model cannot see the current or target month's event.
    """
    drug_month = (panel.groupby(["drug", "month"], as_index=False)
                  ["shortage_event"].max()
                  .sort_values(["drug", "month"]))
    grouped = drug_month.groupby("drug", sort=False)
    drug_month["drug_global_lag1_event"] = grouped["shortage_event"].shift(1)
    drug_month["drug_global_trailing_events_3"] = grouped["shortage_event"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    drug_month["drug_global_trailing_events_6"] = grouped["shortage_event"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum())
    cols = ["drug", "month", "drug_global_lag1_event",
            "drug_global_trailing_events_3", "drug_global_trailing_events_6"]
    return panel.merge(drug_month[cols], on=["drug", "month"], how="left")


def build_next_month_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Strict next-month view: features at month t predict target at t+1.

    Features use only months strictly before t (lag1/lag2/trailing counts),
    calendar month, and supplier/drug one-hot columns fitted per split on the
    training slice. The target is the shortage_event observed at t+1 for the
    same supplier-drug pair. ``current_event`` (month t) is retained for the
    persistence baseline only and is not a model feature.
    """
    df = _add_drug_global_history(panel).sort_values(
        ["supplier", "drug", "month"]).copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    df["year"] = df["month_dt"].dt.year
    df["calendar_month"] = df["month_dt"].dt.month
    g = df.groupby(["supplier", "drug"], sort=False)
    df["lag1_event"] = g["shortage_event"].shift(1)
    df["lag2_event"] = g["shortage_event"].shift(2)
    df["trailing_events_3"] = g["shortage_event"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    )
    df["trailing_events_6"] = g["shortage_event"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()
    )
    df["current_event"] = df["shortage_event"]
    df["target"] = g["shortage_event"].shift(-1)
    df["target_right_censored"] = g["right_censored"].shift(-1)
    # Months before a pair's first observed event have no observed FDA event;
    # zero is the documented "no observed event" value, not confirmed absence.
    for col in ("lag1_event", "lag2_event", "trailing_events_3",
                "trailing_events_6", "drug_global_lag1_event",
                "drug_global_trailing_events_3",
                "drug_global_trailing_events_6"):
        df[col] = df[col].fillna(0.0)
    view = df.dropna(subset=["target"]).copy()
    cols = ["supplier", "drug", "month", "year", "calendar_month",
            "current_event", "lag1_event",
            "lag2_event", "trailing_events_3", "trailing_events_6",
            "drug_global_lag1_event", "drug_global_trailing_events_3",
            "drug_global_trailing_events_6", "target",
            "target_right_censored"]
    return view[cols].reset_index(drop=True)


def build_supplier_shortage_scores(records: pd.DataFrame) -> pd.DataFrame:
    """Fit the monthly event model and score the latest feature month per pair.

    This is an optional context artifact for the universal output.  Scores are
    FDA-reported supplier-drug event probabilities, not pharmacy inventory
    probabilities and not a geographic allocation model.
    """
    panel = build_monthly_panel(records)
    view = build_next_month_view(panel)
    if view.empty:
        return pd.DataFrame(columns=["supplier", "drug", "feature_month",
                                     "shortage_probability", "current_event"])
    spec = _feature_spec(view)
    model = LogisticRidge(alpha=10.0, max_iter=30).fit(
        _feature_matrix(view, spec), view["target"].to_numpy(dtype=float), spec)
    # Build the latest feature row directly because the strict next-month view
    # necessarily drops each pair's final month when constructing its target.
    latest_view = _add_drug_global_history(panel).sort_values(
        ["supplier", "drug", "month"]).copy()
    groups = latest_view.groupby(["supplier", "drug"], sort=False)
    latest_view["calendar_month"] = pd.to_datetime(latest_view["month"]).dt.month
    latest_view["lag1_event"] = groups["shortage_event"].shift(1).fillna(0.0)
    latest_view["lag2_event"] = groups["shortage_event"].shift(2).fillna(0.0)
    latest_view["trailing_events_3"] = groups["shortage_event"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0.0)
    latest_view["trailing_events_6"] = groups["shortage_event"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0.0)
    latest_view = latest_view.rename(columns={"shortage_event": "current_event"})
    for col in ("drug_global_lag1_event", "drug_global_trailing_events_3",
                "drug_global_trailing_events_6"):
        latest_view[col] = latest_view[col].fillna(0.0)
    latest_view = latest_view.groupby(["supplier", "drug"], as_index=False).tail(1)
    probabilities = model.predict_proba(_feature_matrix(latest_view, spec))
    return pd.DataFrame({
        "supplier": latest_view["supplier"].astype(str).to_numpy(),
        "drug": latest_view["drug"].astype(str).to_numpy(),
        "feature_month": latest_view["month"].astype(str).to_numpy(),
        "shortage_probability": np.clip(probabilities, 0.0, 1.0),
        "current_event": latest_view["current_event"].astype(float).to_numpy(),
        "evidence_type": "fda_reported_supplier_drug_event_model",
    })


def _one_hot_spec(train: pd.DataFrame) -> List[str]:
    """Deterministic one-hot supplier/drug columns fitted on the training
    slice only; identities absent from training map to all-zero rows."""
    suppliers = sorted(train["supplier"].unique())
    drugs = sorted(train["drug"].unique())
    return [f"supplier_{s}" for s in suppliers] + [f"drug_{d}" for d in drugs]


def _feature_spec(train: pd.DataFrame, include_identity: bool = True) -> List[str]:
    """Feature list fitted on ``train``.

    Identity columns are useful for the smaller legacy panel, but can make a
    national archived NDC panel needlessly dense.  The archive evaluator can
    disable them and retain the temporal/global shortage features.
    """
    return FEATURE_COLS + (_one_hot_spec(train) if include_identity else [])


def _feature_matrix(view: pd.DataFrame, spec: List[str]) -> np.ndarray:
    """Build numeric features, zero-filling identities unseen in training."""
    base = view.reindex(columns=FEATURE_COLS, fill_value=0.0).copy()
    identity = {}
    for prefix, key in (("supplier_", "supplier"), ("drug_", "drug")):
        for column in (c for c in spec if c.startswith(prefix)):
            if column in FEATURE_COLS:
                continue
            identity[column] = (view[key] == column[len(prefix):]).astype(float)
    if identity:
        base = pd.concat([base, pd.DataFrame(identity, index=view.index)], axis=1)
    return base.reindex(columns=spec, fill_value=0.0).to_numpy(dtype=float)


def _metric_row(y_true: np.ndarray, proba: np.ndarray,
                threshold: float) -> Dict[str, float]:
    """Binary metrics at a fixed threshold plus ranking/calibration scores."""
    binm = _binary_metrics(y_true, proba, threshold)
    return {
        "accuracy": binm["binary_accuracy"],
        "balanced_accuracy": binm["balanced_accuracy"],
        "precision": binm["binary_precision"],
        "recall": binm["binary_recall"],
        "f1": binm["binary_f1"],
        "auroc": _auroc(y_true, proba),
        "auprc": _auprc(y_true, proba),
        "brier": _brier(y_true, np.clip(proba, 0.0, 1.0)),
        "threshold": float(threshold),
    }


def _lb_row(name: str, family: str, val: Dict[str, float],
            test: Dict[str, float], selected: bool = False) -> Dict:
    row: Dict = {"model": name, "family": family, "selected": selected}
    for k, v in val.items():
        row[f"validation_{k}"] = v
    for k, v in test.items():
        row[f"test_{k}"] = v
    return row


def evaluate_supplier_shortage_view(
    view: pd.DataFrame,
    panel: pd.DataFrame,
    train_cutoff: int = 2020,
    val_year: int = 2021,
    test_year: int = 2022,
    include_identity: bool = True,
    exclude_right_censored: bool = True,
) -> Dict:
    """Chronological train/validation/test evaluation on a next-month view.

    Selection (the logistic threshold) uses validation only; test metrics
    are reported for the selected model and the previous-month persistence
    baseline. ``test_year`` is evaluated where available.
    """
    observed = view[view["target_right_censored"].eq(0)].copy()
    train = observed[observed["year"] <= train_cutoff]
    val = observed[observed["year"] == val_year]
    test = observed[observed["year"] == test_year]
    if not exclude_right_censored:
        train = view[view["year"] <= train_cutoff]
        val = view[view["year"] == val_year]
        test = view[view["year"] == test_year]
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Chronological split empty: train<= {train_cutoff} rows={len(train)}, "
            f"val {val_year} rows={len(val)}, test {test_year} rows={len(test)}. "
            "Source records do not cover the required periods."
        )
    spec = _feature_spec(train, include_identity=include_identity)
    X_tr, y_tr = _feature_matrix(train, spec), train["target"].to_numpy(dtype=float)
    X_va, y_va = _feature_matrix(val, spec), val["target"].to_numpy(dtype=float)
    X_te, y_te = _feature_matrix(test, spec), test["target"].to_numpy(dtype=float)

    model = LogisticRidge(alpha=10.0, max_iter=30).fit(X_tr, y_tr, spec)
    val_proba = model.predict_proba(X_va)
    threshold = _select_binary_threshold(y_va, val_proba)
    test_proba = model.predict_proba(X_te)

    persist_va = val["current_event"].to_numpy(dtype=float)
    persist_te = test["current_event"].to_numpy(dtype=float)

    ldf = pd.DataFrame([
        _lb_row("previous_month_persistence", "persistence",
                _metric_row(y_va, persist_va, 0.5),
                _metric_row(y_te, persist_te, 0.5)),
        _lb_row("logistic_ridge", "logistic_ridge",
                _metric_row(y_va, val_proba, threshold),
                _metric_row(y_te, test_proba, threshold),
                selected=True),
    ])

    return {
        "split": "strict_next_month_supplier_drug_time",
        "train_months": [train["month"].min(), train["month"].max()],
        "validation_months": [val["month"].min(), val["month"].max()],
        "test_months": [test["month"].min(), test["month"].max()],
        "n_rows": {"train": int(len(train)), "validation": int(len(val)),
                   "test": int(len(test))},
        "n_pairs": int(panel[["supplier", "drug"]].drop_duplicates().shape[0]),
        "n_suppliers": int(panel["supplier"].nunique()),
        "n_drugs": int(panel["drug"].nunique()),
        "event_counts": {"train": int(y_tr.sum()), "validation": int(y_va.sum()),
                         "test": int(y_te.sum())},
        "censoring": {
            "excluded_target_rows": int(view["target_right_censored"].eq(1).sum())
            if exclude_right_censored else 0,
            "scored_target_rows": int(len(train) + len(val) + len(test)),
            "policy": ("exclude_right_censored_targets" if exclude_right_censored
                       else "include_right_censored_targets"),
        },
        "label_rate": {"validation": float(y_va.mean()), "test": float(y_te.mean())},
        "selected_model": "logistic_ridge",
        "selected_threshold": float(threshold),
        "candidates": ["previous_month_persistence", "logistic_ridge"],
        "leaderboard": ldf,
        "scope": ("FDA-reported supplier-drug event forecasting; not pharmacy "
                  "inventory, not county allocation, not proof of true "
                  "shortage absence"),
    }


def evaluate_supplier_shortage(cfg: Config, train_cutoff: int = 2020,
                               val_year: int = 2021,
                               test_year: int = 2022) -> Dict:
    records = load_fda_shortage_records(cfg)
    panel = build_monthly_panel(records)
    view = build_next_month_view(panel)
    return evaluate_supplier_shortage_view(view, panel, train_cutoff,
                                           val_year, test_year)


def supplier_shortage_rolling_cutoffs(
    months: List[str],
    min_train_months: int = 24,
    test_window_months: int = 4,
    min_folds: int = 3,
) -> List[Tuple[int, int]]:
    """Rolling-origin monthly cutoffs as (train_end, test_end) month indices.

    Shrinks the test window until at least ``min_folds`` folds fit, so short
    histories still produce the required fold count when data supports it.
    """
    n = len(months)
    best: List[Tuple[int, int]] = []
    for tw in range(test_window_months, 0, -1):
        cutoffs: List[Tuple[int, int]] = []
        train_end = min_train_months
        while train_end + tw <= n:
            cutoffs.append((train_end, train_end + tw))
            train_end += tw
        if len(cutoffs) >= min_folds:
            return cutoffs
        best = cutoffs
    return best


def evaluate_supplier_shortage_rolling_view(
    view: pd.DataFrame,
    min_train_months: int = 24,
    test_window_months: int = 4,
    include_identity: bool = True,
    exclude_right_censored: bool = True,
) -> Dict:
    """Rolling-origin monthly evaluation with at least 3 folds when the data
    supports it. Each fold trains on all months before the test window and
    reports the logistic model with a threshold selected on the final
    pre-test validation month against the persistence baseline."""
    if exclude_right_censored:
        view = view[view["target_right_censored"].eq(0)].copy()
    months = sorted(view["month"].unique())
    cutoffs = supplier_shortage_rolling_cutoffs(
        months, min_train_months, test_window_months)
    folds: List[Dict] = []
    for train_end, test_end in cutoffs:
        # Reserve the final month before the test window for threshold
        # selection. It must not also be part of the fitting slice.
        fit_end = train_end - 1
        if fit_end < 1 or test_end <= train_end:
            continue
        train = view[view["month"].isin(months[:fit_end])]
        validation = view[view["month"].isin(months[fit_end:train_end])]
        test = view[view["month"].isin(months[train_end:test_end])]
        spec = _feature_spec(train, include_identity=include_identity)
        X_tr, y_tr = _feature_matrix(train, spec), \
            train["target"].to_numpy(dtype=float)
        X_va, y_va = _feature_matrix(validation, spec), \
            validation["target"].to_numpy(dtype=float)
        X_te, y_te = _feature_matrix(test, spec), \
            test["target"].to_numpy(dtype=float)
        model = LogisticRidge(alpha=10.0, max_iter=30).fit(X_tr, y_tr, spec)
        threshold = _select_binary_threshold(y_va, model.predict_proba(X_va))
        proba = model.predict_proba(X_te)
        persist = test["current_event"].to_numpy(dtype=float)
        folds.append({
            "fold": len(folds) + 1,
            # ``train_months`` is the full pre-test history for compatibility;
            # ``fit_train_months`` is the exact model-fitting slice.
            "train_months": [months[0], months[train_end - 1]],
            "fit_train_months": [months[0], months[fit_end - 1]],
            "validation_months": [months[fit_end], months[train_end - 1]],
            "test_months": [months[train_end], months[test_end - 1]],
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "test_positives": int(y_te.sum()),
            "logistic_ridge": _metric_row(y_te, proba, threshold),
            "previous_month_persistence": _metric_row(y_te, persist, 0.5),
        })
    folds_df = pd.DataFrame(folds)

    def _mean_metrics(candidate: str) -> Dict[str, float]:
        if not folds:
            return {}
        keys = list(folds[0][candidate].keys())
        out: Dict[str, float] = {}
        for k in keys:
            vals = [f[candidate][k] for f in folds
                    if np.isfinite(f[candidate][k])]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    return {
        "split": "rolling_origin_next_month_supplier_drug",
        "fold_count": len(folds),
        "folds_with_positives": int(sum(
            f["test_positives"] > 0 for f in folds)),
        "min_train_months": min_train_months,
        "test_window_months": test_window_months,
        "censoring_policy": ("exclude_right_censored_targets"
                              if exclude_right_censored
                              else "include_right_censored_targets"),
        "logistic_ridge_mean": _mean_metrics("logistic_ridge"),
        "previous_month_persistence_mean": _mean_metrics(
            "previous_month_persistence"),
        "folds": folds_df,
        "scope": ("FDA-reported supplier-drug event forecasting; not pharmacy "
                  "inventory, not county allocation, not proof of true "
                  "shortage absence"),
    }


def evaluate_supplier_shortage_rolling(
    cfg: Config, min_train_months: int = 24,
    test_window_months: int = 4,
) -> Dict:
    records = load_fda_shortage_records(cfg)
    panel = build_monthly_panel(records)
    view = build_next_month_view(panel)
    return evaluate_supplier_shortage_rolling_view(
        view, min_train_months, test_window_months)


def evaluate_archived_supplier_shortage(
    archive_path: Path,
    *,
    train_cutoff: int = 2020,
    validation_year: int = 2021,
    test_year: int = 2022,
    min_train_months: int = 24,
    test_window_months: int = 4,
    include_identity: bool = False,
) -> Dict:
    """Evaluate the dated public archive with censor-safe monthly scoring."""
    panel = load_archived_fda_shortage_panel(archive_path)
    view = build_next_month_view(panel)
    single = evaluate_supplier_shortage_view(
        view, panel, train_cutoff=train_cutoff, val_year=validation_year,
        test_year=test_year, include_identity=include_identity,
        exclude_right_censored=True)
    single["leaderboard"] = single["leaderboard"].to_dict("records")
    rolling = evaluate_supplier_shortage_rolling_view(
        view, min_train_months=min_train_months,
        test_window_months=test_window_months,
        include_identity=include_identity, exclude_right_censored=True)
    rolling["folds"] = rolling["folds"].to_dict("records")
    return {
        "protocol": "public_fda_archive_supplier_drug_next_month_v1",
        "archive_path": str(archive_path),
        "panel_rows": int(len(panel)),
        "view_rows": int(len(view)),
        "supplier_count": int(panel["supplier"].nunique()),
        "drug_count": int(panel["drug"].nunique()),
        "single_split": single,
        "rolling": rolling,
        "scope": ("national FDA-reported supplier-drug shortage continuation; "
                  "used as upstream supplier evidence, not Arkansas inventory "
                  "or proof of shortage absence"),
    }


def load_arkansas_exposed_supplier_panel(
    archive_path: Path, exposure_path: Path,
) -> pd.DataFrame:
    """Filter national supplier observations to NDC9s observed in Arkansas."""
    panel = load_archived_fda_shortage_panel(archive_path)
    exposure = pd.read_csv(exposure_path, usecols=["ndc9"], dtype={"ndc9": str})
    ndcs = (exposure["ndc9"].astype(str).str.replace(r"\D", "", regex=True)
            .str.zfill(9))
    panel["drug"] = panel["drug"].astype(str).str.replace(
        r"\D", "", regex=True).str.zfill(9)
    panel = panel[panel["drug"].isin(set(ndcs))].copy()
    if panel.empty:
        raise ValueError("no archived supplier rows match Arkansas exposure NDC9s")
    return panel.reset_index(drop=True)


def evaluate_arkansas_exposed_supplier_shortage(
    archive_path: Path, exposure_path: Path, **kwargs,
) -> Dict:
    """Evaluate supplier shortage continuation for Arkansas-exposed NDCs."""
    panel = load_arkansas_exposed_supplier_panel(archive_path, exposure_path)
    view = build_next_month_view(panel)
    allowed = {"train_cutoff", "val_year", "test_year", "include_identity"}
    single = evaluate_supplier_shortage_view(
        view, panel, exclude_right_censored=True,
        **{key: value for key, value in kwargs.items() if key in allowed})
    single["leaderboard"] = single["leaderboard"].to_dict("records")
    rolling = evaluate_supplier_shortage_rolling_view(
        view, min_train_months=kwargs.get("min_train_months", 24),
        test_window_months=kwargs.get("test_window_months", 4),
        include_identity=kwargs.get("include_identity", False),
        exclude_right_censored=True)
    rolling["folds"] = rolling["folds"].to_dict("records")
    return {
        "protocol": "arkansas_exposed_ndc_supplier_next_month_v1",
        "archive_path": str(archive_path),
        "exposure_path": str(exposure_path),
        "panel_rows": int(len(panel)), "view_rows": int(len(view)),
        "supplier_count": int(panel["supplier"].nunique()),
        "drug_count": int(panel["drug"].nunique()),
        "single_split": single, "rolling": rolling,
        "scope": ("Arkansas Medicaid-exposed NDCs joined to national FDA "
                  "supplier-drug shortage observations; not county allocation "
                  "or pharmacy inventory truth"),
    }
