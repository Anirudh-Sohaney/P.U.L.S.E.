"""Strict next-period evaluation with baselines, ablations, and ensemble.

Time-based split only: train (feature years <= train_cutoff-1), validation
(feature year == train_cutoff) for ensemble weights, test (feature years >
train_cutoff). Targets are year t+1 from features at year t, so no same-year
target leakage. Writes model/artifacts/evaluation/leaderboard.csv and
metrics.json.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .datasets import ablation_features, build_next_period, feature_columns_for_mode
from .input_contract import PERIODIC_TRAINING_ONLY, disposition_for_variable
from .neural import NeuralSignalBlender
from .regression import LogisticRidge, RidgeLinear, _fill_nan

DEMAND_TARGET = "demand_claims_t1"
RISK_LABEL = "shortage_events_t1"
DEMAND_ONLY_COLS = ["y_last_log", "demand_claims_lag1_log", "demand_claims_ma2_log"]
DEMAND_PUBLISHABLE_THRESHOLD = 0.10


def impute_fit_apply(fit: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """Fill NaN in `apply` with per-column means computed from `fit`."""
    fit = np.asarray(fit, dtype=float)
    apply = np.array(np.asarray(apply, dtype=float))
    means = np.nanmean(fit, axis=0)
    means[~np.isfinite(means)] = 0.0
    for j in range(apply.shape[1]):
        col = apply[:, j]
        nan = ~np.isfinite(col)
        if nan.any():
            col[nan] = means[j]
    return apply


def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            y_last: Optional[np.ndarray] = None) -> Dict[str, float]:
    """WAPE/MAE/RMSE/sMAPE/R2/directional-accuracy on the raw target scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {k: float("nan") for k in
                ("mae", "rmse", "wape", "smape", "r2", "dir_acc")}
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    wape = float(np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1e-9))
    smape = float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-9)))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-9))
    dir_acc = float("nan")
    if y_last is not None:
        yl = np.asarray(y_last, dtype=float)
        yl = yl[mask]
        valid = np.isfinite(yl) & (np.abs(yp - yl) > 0)
        if valid.sum() > 0:
            actual = np.sign(yt[valid] - yl[valid])
            predicted = np.sign(yp[valid] - yl[valid])
            dir_acc = float(np.mean(actual == predicted))
    return {"mae": mae, "rmse": rmse, "wape": wape, "smape": smape,
            "r2": r2, "dir_acc": dir_acc}


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    pos = y_true == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    s_pos = score[pos]
    s_neg = score[neg]
    gt = np.sum(s_pos[:, None] > s_neg[None, :])
    eq = np.sum(s_pos[:, None] == s_neg[None, :])
    return float((gt + 0.5 * eq) / (len(s_pos) * len(s_neg)))


def _auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y)
    n_pos = tp[-1]
    precision = tp / np.arange(1, len(sorted_y) + 1)
    recall = tp / n_pos
    ap = 0.0
    prev = 0.0
    for p, r in zip(precision, recall):
        if r >= prev:
            ap += (r - prev) * p
            prev = r
    return float(ap)


def _brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true) ** 2))


def _topk_recall(y_true: np.ndarray, score: np.ndarray,
                 k: Optional[int] = None) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    k = int(k if k is not None else pos.sum())
    order = np.argsort(-score)[:k]
    return float(y_true[order].sum() / pos.sum())


def _topk_stats(y_true: np.ndarray, score: np.ndarray,
                k: Optional[int] = None) -> Dict[str, float]:
    """Inventory-triage metrics at top-k, where k defaults to positives."""
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return {
            "k": 0, "positives": 0, "positives_captured": 0,
            "topk_recall": float("nan"), "precision_at_k": float("nan"),
            "lift_at_k": float("nan"),
        }
    k = int(k if k is not None else n_pos)
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-score)[:k]
    captured = int(y_true[order].sum())
    precision = float(captured / k)
    base_rate = float(n_pos / max(len(y_true), 1))
    return {
        "k": k,
        "positives": n_pos,
        "positives_captured": captured,
        "topk_recall": float(captured / n_pos),
        "precision_at_k": precision,
        "lift_at_k": float(precision / max(base_rate, 1e-12)),
    }


def _unit_score(fit_score: np.ndarray, apply_score: np.ndarray) -> np.ndarray:
    """Scale an arbitrary risk score to [0, 1] using fit-period bounds."""
    fit_score = np.asarray(fit_score, dtype=float)
    apply_score = np.asarray(apply_score, dtype=float)
    finite = np.isfinite(fit_score)
    if not finite.any():
        return np.zeros(len(apply_score), dtype=float)
    lo = float(np.nanpercentile(fit_score[finite], 1))
    hi = float(np.nanpercentile(fit_score[finite], 99))
    if hi <= lo:
        hi = float(np.nanmax(fit_score[finite]))
        lo = float(np.nanmin(fit_score[finite]))
    if hi <= lo:
        return np.zeros(len(apply_score), dtype=float)
    return np.clip((np.nan_to_num(apply_score, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def _zscore(fit_score: np.ndarray, apply_score: np.ndarray) -> np.ndarray:
    """Standardize a risk score with fit-period moments."""
    fit_score = np.asarray(fit_score, dtype=float)
    apply_score = np.asarray(apply_score, dtype=float)
    finite = np.isfinite(fit_score)
    if not finite.any():
        return np.zeros(len(apply_score), dtype=float)
    mean = float(np.nanmean(fit_score[finite]))
    std = float(np.nanstd(fit_score[finite]))
    if std <= 1e-12:
        return np.zeros(len(apply_score), dtype=float)
    return (np.nan_to_num(apply_score, nan=mean) - mean) / std


def _fit_ridge_log(X_train: np.ndarray, y_train: np.ndarray,
                   feature_names: List[str], alpha: float = 10.0,
                   sample_weight: Optional[np.ndarray] = None) -> RidgeLinear:
    y_log = np.log1p(np.clip(y_train, 0, None))
    return RidgeLinear(alpha=alpha).fit(X_train, y_log, feature_names,
                                        sample_weight=sample_weight)


def _ridge_log_predict(model: RidgeLinear, X: np.ndarray) -> np.ndarray:
    # Guard diagnostic high-dimensional fits against overflow on the raw
    # count scale. The bound is far above observed Arkansas claim counts and
    # keeps one unstable candidate from corrupting benchmark metrics.
    return np.expm1(np.clip(model.predict(X), -20.0, 20.0))


def _select_raw_blend_weight(y_true: np.ndarray, base: np.ndarray,
                             model_pred: np.ndarray) -> tuple:
    """Validation-selected convex weight minimizing raw-scale WAPE."""
    best_w = 0.0
    best_wape = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        pred = (1.0 - w) * base + w * model_pred
        score = metrics(y_true, pred, base)["wape"]
        if score < best_wape:
            best_wape = score
            best_w = float(w)
    return best_w, best_wape


def _residual_target(y_t1: np.ndarray, y_last: np.ndarray) -> np.ndarray:
    """log1p(y_{t+1}) - log1p(y_t): log-gap the residual models learn."""
    return np.log1p(np.clip(y_t1, 0, None)) - np.log1p(np.clip(y_last, 0, None))


def residual_to_raw(y_last: np.ndarray, resid_pred: np.ndarray,
                    lo: float, hi: float) -> np.ndarray:
    """expm1(log1p(y_last) + clipped_residual_pred), clip on train percentiles."""
    resid_pred = np.clip(np.asarray(resid_pred, dtype=float), lo, hi)
    return np.expm1(np.log1p(np.clip(np.asarray(y_last, dtype=float), 0, None))
                    + resid_pred)


def _model_params(m) -> int:
    return int(m.n_params) if hasattr(m, "n_params") else int(m.coef.size + 1)


def _layer_uplift(ldf: pd.DataFrame, city_drug_last_wape: float,
                  demand_only_ridge_wape: float) -> pd.DataFrame:
    """WAPE delta per layer/model vs the two fixed reference baselines."""
    out = ldf.copy()
    out["wape_delta_vs_city_drug_last"] = out["wape"] - city_drug_last_wape
    out["wape_delta_vs_demand_only_ridge"] = out["wape"] - demand_only_ridge_wape
    return out[["model", "family", "subset", "wape",
                "wape_delta_vs_city_drug_last", "wape_delta_vs_demand_only_ridge"]]


def _simplex_weights(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Non-negative weights minimizing ||A w - b||; normalized to sum 1."""
    k = A.shape[1]
    try:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.ones(k) / k
    # ponytail: single projection pass; iterate if weights drift negative
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        w = np.ones(k) / k
    else:
        w = w / s
    return w


def _neural_subsample(index: np.ndarray, weights: Optional[np.ndarray] = None,
                      seed: int = 0, cap: int = 120_000) -> np.ndarray:
    if len(index) <= cap:
        return index
    rng = np.random.default_rng(seed)
    if weights is not None:
        p = np.asarray(weights, dtype=float).copy()
        pos = p > 0
        p = np.where(pos, p, p.mean() if p.mean() > 0 else 1.0)
        if p.sum() > 0:
            p = p / p.sum()
            return rng.choice(index, size=cap, replace=False, p=p)
    return rng.choice(index, size=cap, replace=False)


def _year_split(view: pd.DataFrame, train_cutoff: int,
                test_end_year: Optional[int] = None) -> tuple:
    train_mask = view["year"] <= train_cutoff - 1
    val_mask = view["year"] == train_cutoff
    test_mask = view["year"] > train_cutoff
    if test_end_year is not None:
        test_mask = test_mask & (view["year"] <= test_end_year)
    return train_mask, val_mask, test_mask


def evaluate_next_period(
    panel: pd.DataFrame,
    train_cutoff: int = 2021,
    target: str = DEMAND_TARGET,
    risk_label: str = RISK_LABEL,
    neural: bool = True,
    test_end_year: Optional[int] = None,
    feature_mode: str = "operational",
) -> Dict:
    """Run the full strict next-period leaderboard and metrics."""
    if feature_mode not in ("operational", "full"):
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    # Fit encoders on training feature years only. The validation year is used
    # for candidate selection, never for fitting identity categories.
    fit_mask = panel["year"] <= train_cutoff - 1
    view, _ = build_next_period(panel, fit_mask=fit_mask)
    view = view.dropna(subset=[target])
    if risk_label not in view.columns or view[risk_label].isna().all():
        view[risk_label] = 0.0
    view[risk_label] = view[risk_label].fillna(0.0)

    train_mask, val_mask, test_mask = _year_split(
        view, train_cutoff, test_end_year=test_end_year)
    train = view[train_mask]
    val = view[val_mask]
    test = view[test_mask]

    result: Dict = {
        "split": "strict_next_period_time",
        "feature_mode": feature_mode,
        "operational_ready": feature_mode == "operational",
        "feature_years": {"train": f"<= {train_cutoff - 1}",
                          "validation": str(train_cutoff),
                          "test": (f"> {train_cutoff}" if test_end_year is None
                                   else f"{train_cutoff + 1}-{test_end_year}")},
        "n_rows": {"train": int(len(train)), "validation": int(len(val)),
                   "test": int(len(test))},
    }
    if len(train) < 30 or len(test) < 5:
        result["error"] = f"insufficient rows train={len(train)} test={len(test)}"
        return result

    feat_groups = ablation_features(view)
    if feature_mode == "operational":
        operational_cols = set(feature_columns_for_mode(view, "operational"))
        feat_groups = {key: [c for c in cols if c in operational_cols]
                       for key, cols in feat_groups.items()}
    full_cols = feat_groups["full"]
    y_train = train[target].to_numpy(dtype=float)
    y_val = val[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    y_last_test = test["y_last"].to_numpy(dtype=float)
    w_train = train["y_last"].to_numpy(dtype=float)

    X_train = _fill_nan(train[full_cols].to_numpy(dtype=float))
    X_val = impute_fit_apply(X_train, val[full_cols].to_numpy(dtype=float))
    X_test = impute_fit_apply(X_train, test[full_cols].to_numpy(dtype=float))

    leaderboard: List[Dict] = []

    # ---- baselines (raw-scale predictions) ----------------------------------
    baselines: Dict[str, np.ndarray] = _fit_baselines(train, test)
    for name, pred in baselines.items():
        leaderboard.append(_lb_row(name, "baseline", name,
                                   y_test, pred, y_last_test, 0))
    best_naive = min(baselines, key=lambda n: metrics(y_test, baselines[n],
                                                      y_last_test)["wape"])
    naive_wape = metrics(y_test, baselines[best_naive], y_last_test)["wape"]
    ridge0_wape = metrics(y_test, baselines["demand_only_ridge"], y_last_test)["wape"]

    # ---- ablation ridges ----------------------------------------------------
    models: Dict[str, Dict] = {}
    for name, cols in feat_groups.items():
        if not cols:
            continue
        Xt = _fill_nan(train[cols].to_numpy(dtype=float))
        model = _fit_ridge_log(Xt, y_train, cols, sample_weight=w_train)
        pred_test = _ridge_log_predict(
            model, impute_fit_apply(Xt, test[cols].to_numpy(dtype=float)))
        models[f"ridge_{name}"] = {"model": model, "pred_test": pred_test}
        leaderboard.append(_lb_row(f"ridge_{name}", "ridge_linear", name,
                                   y_test, pred_test, y_last_test,
                                   int(model.coef.size + 1)))

    # Conservative production candidate: validation first chooses the Medicaid
    # feature scope, then chooses the raw-scale blend into city_drug_last.
    # This prevents broader identifier bridges from entering the production
    # blend unless they help on validation.
    medicaid_candidates = [
        ("all", feat_groups.get("full", [])),
        ("none", feat_groups.get("full_no_medicaid", [])),
        ("exact", feat_groups.get("full_medicaid_exact", [])),
        ("bridge", feat_groups.get("full_medicaid_bridge", [])),
    ]
    calib_rows: List[Dict] = []
    for variant, cols in medicaid_candidates:
        if not cols:
            continue
        Xt = _fill_nan(train[cols].to_numpy(dtype=float))
        model = _fit_ridge_log(Xt, y_train, cols, alpha=0.01, sample_weight=w_train)
        Xv = impute_fit_apply(Xt, val[cols].to_numpy(dtype=float))
        Xe = impute_fit_apply(Xt, test[cols].to_numpy(dtype=float))
        pred_val = _ridge_log_predict(model, Xv)
        pred_test = _ridge_log_predict(model, Xe)
        blend_w, blend_val_wape = _select_raw_blend_weight(
            y_val, val["y_last"].to_numpy(dtype=float), pred_val)
        calib_rows.append({
            "variant": variant,
            "cols": cols,
            "model": model,
            "pred_test": pred_test,
            "blend_weight": blend_w,
            "validation_wape": blend_val_wape,
        })
    if calib_rows:
        selected = min(calib_rows, key=lambda r: r["validation_wape"])
        for row in calib_rows:
            pred = ((1.0 - row["blend_weight"]) * y_last_test
                    + row["blend_weight"] * row["pred_test"])
            name = f"calibrated_ridge_blend_{row['variant']}"
            leaderboard.append(_lb_row(
                name, "validated_convex_blend_diagnostic", row["variant"],
                y_test, pred, y_last_test, int(row["model"].coef.size + 2)))
        calib_blend = ((1.0 - selected["blend_weight"]) * y_last_test
                       + selected["blend_weight"] * selected["pred_test"])
        leaderboard.append(_lb_row(
            "calibrated_ridge_blend", "validated_convex_blend",
            selected["variant"], y_test, calib_blend, y_last_test,
            int(selected["model"].coef.size + 2)))
        result["calibrated_blend"] = {
            "ridge_alpha": 0.01,
            "blend_weight": selected["blend_weight"],
            "validation_wape": selected["validation_wape"],
            "selected_medicaid_variant": selected["variant"],
            "candidate_validation": [
                {"variant": r["variant"],
                 "blend_weight": r["blend_weight"],
                 "validation_wape": r["validation_wape"],
                 "n_features": len(r["cols"])}
                for r in calib_rows
            ],
            "members": ["city_drug_last", f"ridge_full_{selected['variant']}_alpha_0.01"],
        }

    # ---- neural blend -------------------------------------------------------
    neural_model = None
    neural_test = None
    if neural:
        Xu = _fill_nan(train[full_cols].to_numpy(dtype=float))
        Xn = _neural_subsample(np.arange(len(train)), weights=w_train, cap=120_000)
        y_log = np.log1p(np.clip(y_train[Xn], 0, None)).reshape(-1, 1)
        try:
            nm = NeuralSignalBlender(epochs=200).fit(Xu[Xn], y_log, full_cols)
            neural_test = np.expm1(nm.predict(X_test).reshape(-1))
            neural_model = nm
            leaderboard.append(_lb_row("neural_blend", "neural_signal_blender_mlp",
                                       "full", y_test, neural_test, y_last_test,
                                       nm.n_params))
        except (ValueError, np.linalg.LinAlgError):
            neural_model = None

    # ---- residual (shock) models: learn log1p(y_t1) - log1p(y_last) --------
    last_train = np.log1p(np.clip(train["y_last"].to_numpy(dtype=float), 0, None))
    last_val = np.log1p(np.clip(val["y_last"].to_numpy(dtype=float), 0, None))
    last_test = np.log1p(np.clip(y_last_test, 0, None))
    resid_train = _residual_target(y_train, train["y_last"].to_numpy(dtype=float))
    resid_lo = float(np.nanpercentile(resid_train, 1))
    resid_hi = float(np.nanpercentile(resid_train, 99))

    resid_feat_groups: Dict[str, List[str]] = {
        "history": [c for c in feat_groups["history_only"] if c != "y_last_log"],
        "external": feat_groups["all_external"],
        "full": [c for c in feat_groups["full"] if c != "y_last_log"],
    }
    resid_col_idx = [i for i, c in enumerate(full_cols) if c != "y_last_log"]
    Xr_tr = X_train[:, resid_col_idx]
    Xr_val = X_val[:, resid_col_idx]
    Xr_test = X_test[:, resid_col_idx]

    residual_models: Dict[str, Dict] = {}
    for name, cols in resid_feat_groups.items():
        if not cols:
            continue
        Xt = _fill_nan(train[cols].to_numpy(dtype=float))
        model = RidgeLinear(alpha=10.0).fit(Xt, resid_train, cols)
        pred_val = residual_to_raw(
            val["y_last"].to_numpy(dtype=float),
            model.predict(impute_fit_apply(Xt, val[cols].to_numpy(dtype=float))),
            resid_lo, resid_hi)
        pred_test = residual_to_raw(
            y_last_test,
            model.predict(impute_fit_apply(Xt, test[cols].to_numpy(dtype=float))),
            resid_lo, resid_hi)
        residual_models[f"residual_ridge_{name}"] = {
            "model": model, "pred_val": pred_val, "pred_test": pred_test}
        leaderboard.append(_lb_row(f"residual_ridge_{name}", "residual_ridge", name,
                                   y_test, pred_test, y_last_test,
                                   int(model.coef.size + 1)))

    if neural:
        try:
            nm_idx = _neural_subsample(np.arange(len(train)))
            nm_res = NeuralSignalBlender(epochs=200).fit(
                Xr_tr[nm_idx], resid_train[nm_idx].reshape(-1, 1),
                resid_feat_groups["full"])
            pred_val = residual_to_raw(
                val["y_last"].to_numpy(dtype=float),
                nm_res.predict(Xr_val).reshape(-1), resid_lo, resid_hi)
            pred_test = residual_to_raw(
                y_last_test, nm_res.predict(Xr_test).reshape(-1),
                resid_lo, resid_hi)
            residual_models["residual_neural"] = {
                "model": nm_res, "pred_val": pred_val, "pred_test": pred_test}
            leaderboard.append(_lb_row("residual_neural", "neural_signal_blender_mlp",
                                       "full", y_test, pred_test, y_last_test,
                                       nm_res.n_params))
        except (ValueError, np.linalg.LinAlgError):
            pass

    # ---- residual ensemble: city_drug_last anchor, residual members only when
    #      they beat the anchor on validation WAPE -----------------------------
    city_last_wape = metrics(y_val, val["y_last"].to_numpy(dtype=float),
                             val["y_last"].to_numpy(dtype=float))["wape"]
    improving = [n for n, m in residual_models.items()
                 if metrics(y_val, m["pred_val"],
                            val["y_last"].to_numpy(dtype=float))["wape"] < city_last_wape]
    if improving:
        comp_val = np.column_stack(
            [last_val] + [np.log1p(np.clip(residual_models[n]["pred_val"], 0, None))
                          for n in improving])
        comp_test = np.column_stack(
            [last_test] + [np.log1p(np.clip(residual_models[n]["pred_test"], 0, None))
                           for n in improving])
        w = _simplex_weights(comp_val, np.log1p(np.clip(y_val, 0, None)))
        ens_test = np.expm1(comp_test @ w)
        weights_list = w.tolist()
    else:
        ens_test = y_last_test.copy()
        weights_list = [1.0]
    residual_models["residual_ensemble"] = {
        "members": ["city_drug_last"] + improving,
        "weights": weights_list,
        "pred_test": ens_test}
    ens_params = (0 if not improving
                  else sum(_model_params(residual_models[n]["model"]) for n in improving) + 1)
    leaderboard.append(_lb_row("residual_ensemble", "nnls_ensemble", "residual",
                               y_test, ens_test, y_last_test, ens_params))

    # ---- ensemble: best naive + full ridge + neural, validation weights -----
    if "ridge_full" in models and neural_model is not None:
        ridge_val = _ridge_log_predict(models["ridge_full"]["model"], X_val)
        ridge_test = models["ridge_full"]["pred_test"]
        neural_val = np.expm1(neural_model.predict(X_val).reshape(-1))
        comp_val = np.column_stack([
            np.log1p(np.clip(val["y_last"].to_numpy(dtype=float), 0, None)),
            np.log1p(np.clip(ridge_val, 0, None)),
            np.log1p(np.clip(neural_val, 0, None)),
        ])
        comp_test = np.column_stack([
            np.log1p(np.clip(test["y_last"].to_numpy(dtype=float), 0, None)),
            np.log1p(np.clip(ridge_test, 0, None)),
            np.log1p(np.clip(neural_test, 0, None)),
        ])
        w = _simplex_weights(comp_val, np.log1p(np.clip(y_val, 0, None)))
        ens_test = np.expm1(comp_test @ w)
        ens_params = int(3) + models["ridge_full"]["model"].coef.size + 1 \
            + neural_model.n_params
        leaderboard.append(_lb_row("ensemble", "nnls_ensemble", "blend",
                                   y_test, ens_test, y_last_test, ens_params))
        models["ensemble"] = {"weights": w.tolist(), "pred_test": ens_test}

    # ---- leaderboard with improvements -------------------------------------
    ldf = pd.DataFrame(leaderboard)
    ldf = _with_improvements(ldf, naive_wape, ridge0_wape)
    result["leaderboard"] = ldf
    best = ldf.loc[ldf["wape"].idxmin()]
    city_drug_last_wape = metrics(y_test, baselines["city_drug_last"],
                                  y_last_test)["wape"]
    best_is_learned = best["model"] not in set(baselines.keys())
    gap = (city_drug_last_wape - float(best["wape"])) / max(city_drug_last_wape, 1e-9)
    publishable = best_is_learned and gap >= 0.10
    result["best_model"] = {"model": best["model"], "wape": float(best["wape"])}
    result["best_naive"] = {"model": best_naive, "wape": float(naive_wape)}
    result["publishable_candidate"] = bool(publishable)
    if publishable:
        result["publishability_reason"] = (
            f"{best['model']} beats city_drug_last naive by {gap:.1%} WAPE "
            f"on strict test (>=10% threshold)")
    else:
        result["publishability_reason"] = (
            f"best strict-test model {best['model']} wape {float(best['wape']):.4f} "
            f"vs city_drug_last {city_drug_last_wape:.4f}; relative gap "
            f"{gap:.1%} below the 10% publishable threshold; "
            f"learned/calibrated model uplift is not large enough for a "
            f"marketable inventory-impact claim")
    result["layer_uplift"] = _layer_uplift(ldf, city_drug_last_wape, ridge0_wape)
    result["residual"] = {
        "clip_bounds": [resid_lo, resid_hi],
        "improving_members": improving,
        "ensemble_members": residual_models["residual_ensemble"]["members"],
        "ensemble_weights": residual_models["residual_ensemble"]["weights"],
    }

    # ---- shortage risk ------------------------------------------------------
    result["shortage_risk"] = _evaluate_risk(train, val, test, full_cols, risk_label,
                                             feature_mode=feature_mode)
    return result


def annual_fold_cutoffs(panel: pd.DataFrame,
                        min_train_years: int = 4) -> List[int]:
    """Return annual validation cutoffs with at least one future test year."""
    years = sorted(int(y) for y in panel["year"].dropna().unique())
    out: List[int] = []
    max_year = max(years) if years else None
    for cutoff in years:
        train_years = [y for y in years if y <= cutoff - 1]
        # A feature year t+1 needs a real target year t+2, so the final
        # observed panel year cannot be used as a test feature year.
        test_years = [y for y in years if y > cutoff and y < max_year]
        if len(train_years) >= min_train_years and test_years:
            out.append(cutoff)
    return out


def evaluate_demand_rolling(
    panel: pd.DataFrame,
    min_train_years: int = 4,
    test_window_years: Optional[int] = None,
    neural: bool = False,
    feature_mode: str = "operational",
) -> Dict:
    """Run rolling-origin annual demand evaluations.

    Each fold fits on feature years <= cutoff-1, selects the Medicaid scope and
    blend weight on feature year ``cutoff``, and evaluates the selected model
    on later feature years. ``best_model_by_test`` is retained as a diagnostic
    only; publishability uses the validation-selected calibrated blend and a
    validation-selected naive baseline. The optional bounded test window is
    useful for near-term planning without changing the fold's selection rules.
    """
    if feature_mode not in ("operational", "full"):
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    cutoffs = annual_fold_cutoffs(panel, min_train_years=min_train_years)
    rows: List[Dict] = []
    for cutoff in cutoffs:
        test_end_year = None
        if test_window_years is not None:
            test_end_year = cutoff + int(test_window_years)

        result = evaluate_next_period(
            panel,
            train_cutoff=cutoff,
            neural=neural,
            test_end_year=test_end_year,
            feature_mode=feature_mode,
        )
        fold_panel = panel
        if "leaderboard" not in result or "calibrated_blend" not in result:
            continue

        # Select the naive comparator on validation, not after seeing test
        # labels. The selected demand model is already selected on validation
        # inside evaluate_next_period.
        view, _ = build_next_period(
            fold_panel,
            fit_mask=fold_panel["year"] <= cutoff - 1,
        )
        train = view[view["year"] <= cutoff - 1]
        val = view[view["year"] == cutoff]
        test = view[view["year"] > cutoff]
        if test_end_year is not None:
            test = test[test["year"] <= test_end_year]
        if len(train) < 30 or len(val) < 5 or len(test) < 5:
            continue

        val_baselines = _fit_baselines(train, val)
        test_baselines = _fit_baselines(train, test)
        y_val = val[DEMAND_TARGET].to_numpy(dtype=float)
        y_test = test[DEMAND_TARGET].to_numpy(dtype=float)
        best_naive = min(
            val_baselines,
            key=lambda name: metrics(y_val, val_baselines[name],
                                     val["y_last"].to_numpy(dtype=float))["wape"],
        )
        naive_wape = metrics(
            y_test, test_baselines[best_naive], test["y_last"].to_numpy(dtype=float)
        )["wape"]

        leaderboard = result["leaderboard"]
        selected_rows = leaderboard.loc[
            leaderboard["model"] == "calibrated_ridge_blend"
        ]
        if selected_rows.empty:
            continue
        selected_wape = float(selected_rows.iloc[0]["wape"])
        selected_improvement = (
            (naive_wape - selected_wape) / max(naive_wape, 1e-9)
        )
        diagnostic = result.get("best_model", {})
        diagnostic_wape = float(diagnostic.get("wape", float("nan")))
        diagnostic_improvement = (
            (naive_wape - diagnostic_wape) / max(naive_wape, 1e-9)
            if np.isfinite(diagnostic_wape) else float("nan")
        )
        blend = result["calibrated_blend"]
        rows.append({
            "cutoff_year": int(cutoff),
            "train_years": f"<= {cutoff - 1}",
            "validation_year": int(cutoff),
            "test_years": (f"> {cutoff}" if test_end_year is None
                           else f"{cutoff + 1}-{test_end_year}"),
            "test_window_years": (None if test_window_years is None
                                  else int(test_window_years)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "best_naive": best_naive,
            "best_naive_wape": float(naive_wape),
            "selected_model": "calibrated_ridge_blend",
            "selected_medicaid_variant": blend.get("selected_medicaid_variant"),
            "selected_model_validation_wape": float(blend["validation_wape"]),
            "selected_model_wape": selected_wape,
            "selected_blend_weight": float(blend["blend_weight"]),
            "selected_improvement_vs_best_naive": float(selected_improvement),
            "diagnostic_best_model_by_test": diagnostic.get("model"),
            "diagnostic_best_model_wape": diagnostic_wape,
            "diagnostic_improvement_vs_best_naive": float(diagnostic_improvement),
            "beats_naive": bool(selected_improvement > 0),
            "clears_10pct_gate": bool(
                selected_improvement >= DEMAND_PUBLISHABLE_THRESHOLD),
        })

    folds = pd.DataFrame(rows)
    summary: Dict = {
        "split": "rolling_strict_next_period_annual_demand",
        "feature_mode": feature_mode,
        "operational_ready": feature_mode == "operational",
        "min_train_years": int(min_train_years),
        "test_window_years": (None if test_window_years is None
                              else int(test_window_years)),
        "publishability_threshold": DEMAND_PUBLISHABLE_THRESHOLD,
        "fold_count": int(len(folds)),
        "folds": folds,
    }
    if folds.empty:
        summary.update({
            "publishable_rolling_candidate": False,
            "publishability_reason": "no eligible rolling demand folds",
        })
        return summary

    imp = folds["selected_improvement_vs_best_naive"].to_numpy(dtype=float)
    finite = imp[np.isfinite(imp)]
    mean_imp = float(np.mean(finite)) if finite.size else float("nan")
    median_imp = float(np.median(finite)) if finite.size else float("nan")
    min_imp = float(np.min(finite)) if finite.size else float("nan")
    n_clear = int(np.sum(finite >= DEMAND_PUBLISHABLE_THRESHOLD))
    n_beat = int(np.sum(finite > 0))
    all_beat = bool(finite.size == len(folds) and n_beat == len(folds))
    publishable = bool(
        len(folds) >= 3 and all_beat and np.isfinite(mean_imp)
        and mean_imp >= DEMAND_PUBLISHABLE_THRESHOLD
    )
    summary.update({
        "mean_improvement_vs_best_naive": mean_imp,
        "median_improvement_vs_best_naive": median_imp,
        "min_improvement_vs_best_naive": min_imp,
        "folds_clearing_10pct_gate": n_clear,
        "fraction_folds_clearing_10pct_gate": float(n_clear / len(folds)),
        "folds_beating_naive": n_beat,
        "fraction_folds_beating_naive": float(n_beat / len(folds)),
        "publishable_rolling_candidate": publishable,
        "publishability_reason": (
            f"all {len(folds)} rolling folds beat their validation-selected "
            f"naive baseline and mean improvement {mean_imp:.1%} is >=10%"
            if publishable else
            f"rolling evidence below gate: {n_beat}/{len(folds)} folds beat "
            f"naive, {n_clear}/{len(folds)} clear 10%, mean improvement "
            f"{mean_imp:.1%}"
        ),
    })
    return summary


def _feature_groups_for_mode(feat_groups: Dict[str, List[str]],
                             feature_mode: str) -> Dict[str, List[str]]:
    """Restrict ablation feature groups to the requested training mode.

    ``full`` passes groups through; ``operational`` drops columns whose input
    disposition is ``PERIODIC_TRAINING_ONLY`` (annual CMS Part D / Medicaid /
    provider measures that are never live feeds). Equivalent to filtering each
    group by ``feature_columns_for_mode(view, "operational")``.
    """
    if feature_mode == "full":
        return feat_groups
    if feature_mode == "operational":
        operational_cols = {
            c for c in feat_groups["full"]
            if disposition_for_variable(c) != PERIODIC_TRAINING_ONLY
        }
        return {key: [c for c in cols if c in operational_cols]
                for key, cols in feat_groups.items()}
    raise ValueError(f"unknown feature_mode: {feature_mode!r}")


def evaluate_risk_rolling(
    panel: pd.DataFrame,
    min_train_years: int = 4,
    risk_label: str = RISK_LABEL,
    feature_mode: str = "operational",
) -> Dict:
    """Rolling-origin annual shortage-risk evaluation.

    Each fold uses feature years <= cutoff-1 for training, cutoff for model
    selection, and cutoff+1 for testing. Targets are still t+1 labels built by
    `build_next_period`, so the test fold asks whether feature-year cutoff+1
    ranks shortages in year cutoff+2.
    """
    if feature_mode not in ("operational", "full"):
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    years = sorted(int(y) for y in panel["year"].dropna().unique())
    out: Dict = {
        "split": "rolling_origin_annual_shortage_risk",
        "feature_mode": feature_mode,
        "operational_ready": feature_mode == "operational",
        "risk_label": risk_label,
        "min_train_years": int(min_train_years),
    }
    if len(years) < min_train_years + 3:
        out["error"] = "insufficient years for rolling risk evaluation"
        out["folds"] = pd.DataFrame()
        return out

    rows: List[Dict] = []
    min_year = min(years)
    max_feature_year = max(years) - 1
    for cutoff in range(min_year + min_train_years, max_feature_year):
        fit_mask = panel["year"] <= cutoff - 1
        view, _ = build_next_period(panel, fit_mask=fit_mask)
        if risk_label not in view.columns:
            view[risk_label] = 0.0
        view[risk_label] = view[risk_label].fillna(0.0)
        feat_groups = _feature_groups_for_mode(ablation_features(view), feature_mode)
        full_cols = feat_groups["full"]
        train = view[view["year"] <= cutoff - 1]
        val = view[view["year"] == cutoff]
        test = view[view["year"] == cutoff + 1]
        if len(train) < 30 or len(val) < 5 or len(test) < 5:
            continue
        risk = _evaluate_risk(train, val, test, full_cols, risk_label,
                              feature_mode=feature_mode)
        if risk.get("note"):
            continue
        test_best = next(
            row for row in risk["leaderboard"]
            if row["model"] == risk["best_model_by_test"])
        rows.append({
            "cutoff_year": int(cutoff),
            "train_years": f"{int(train['year'].min())}-{int(train['year'].max())}",
            "validation_year": int(cutoff),
            "test_feature_year": int(cutoff + 1),
            "test_target_year": int(cutoff + 2),
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "selected_model": risk["selected_model"],
            "selected_family": risk["selected_family"],
            "best_model_by_test": risk["best_model_by_test"],
            "test_best_auroc": test_best["test_auroc"],
            "test_best_auprc": test_best["test_auprc"],
            "test_best_brier": test_best["test_brier"],
            "test_best_topk_recall": test_best["test_topk_recall"],
            "test_best_precision_at_k": test_best["test_precision_at_k"],
            "test_best_lift_at_k": test_best["test_lift_at_k"],
            "test_best_positives_captured": test_best["test_positives_captured"],
            "test_auroc": risk["auroc"],
            "test_auprc": risk["auprc"],
            "test_brier": risk["brier"],
            "test_topk_recall": risk["topk_recall"],
            "test_precision_at_k": risk["precision_at_k"],
            "test_lift_at_k": risk["lift_at_k"],
            "test_positives_captured": risk["positives_captured"],
            "test_k": risk["k"],
            "label_rate_validation": risk["label_rate_validation"],
            "label_rate_test": risk["label_rate_test"],
            "publishable_candidate": risk["publishable_candidate"],
            "best_model_by_test_is_diagnostic": True,
        })

    folds = pd.DataFrame(rows)
    out["folds"] = folds
    out["n_folds"] = int(len(folds))
    if folds.empty:
        out["publishable_rolling_candidate"] = False
        out["publishability_reason"] = "no eligible rolling shortage-risk folds"
        return out

    topk = folds["test_topk_recall"].to_numpy(dtype=float)
    precision = folds["test_precision_at_k"].to_numpy(dtype=float)
    lift = folds["test_lift_at_k"].to_numpy(dtype=float)
    captured = folds["test_positives_captured"].to_numpy(dtype=float)
    k = folds["test_k"].to_numpy(dtype=float)
    beats_base = precision > folds["label_rate_test"].to_numpy(dtype=float)
    publishable = bool(
        len(folds) >= 3
        and np.nanmean(topk) >= 0.25
        and np.nanmedian(lift) >= 5.0
        and bool(np.all(beats_base))
    )
    out.update({
        "mean_topk_recall": float(np.nanmean(topk)),
        "median_topk_recall": float(np.nanmedian(topk)),
        "min_topk_recall": float(np.nanmin(topk)),
        "mean_precision_at_k": float(np.nanmean(precision)),
        "median_precision_at_k": float(np.nanmedian(precision)),
        "mean_lift_at_k": float(np.nanmean(lift)),
        "median_lift_at_k": float(np.nanmedian(lift)),
        "total_positives_captured": int(np.nansum(captured)),
        "total_k": int(np.nansum(k)),
        "folds_beating_label_rate": int(np.sum(beats_base)),
        "publishable_rolling_candidate": publishable,
        "publishability_reason": (
            "rolling selected risk model meets shortage-triage thresholds"
            if publishable else
            "rolling selected risk model does not meet the configured gate: "
            ">=3 folds, mean top-k recall >=25%, median lift >=5x, and every "
            "fold precision above the fold label rate"
        ),
    })
    return out


def _fit_baselines(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Naive baselines (raw scale) keyed by name."""
    tgt = DEMAND_TARGET
    y_test = test[tgt].to_numpy(dtype=float)
    y_last_test = test["y_last"].to_numpy(dtype=float)
    global_mean = float(np.nanmean(train[tgt].to_numpy(dtype=float)))

    def _map_means(group_col: str) -> np.ndarray:
        means = train.groupby(group_col)[tgt].mean()
        g = test[group_col].map(means)
        return g.fillna(global_mean).to_numpy(dtype=float)

    lag1_test = test["demand_claims_lag1"].to_numpy(dtype=float)
    ma = 0.5 * (y_last_test + np.where(np.isfinite(lag1_test), lag1_test, y_last_test))

    return {
        "global_mean": np.full(len(y_test), global_mean),
        "drug_mean": _map_means("drug_key"),
        "city_mean": _map_means("city"),
        "city_drug_last": y_last_test,
        "city_drug_ma": ma,
        "demand_only_ridge": _demand_only_ridge_pred(train, test),
    }


def _demand_only_ridge_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cols = [c for c in DEMAND_ONLY_COLS if c in train.columns]
    X = _fill_nan(train[cols].to_numpy(dtype=float))
    y = train[DEMAND_TARGET].to_numpy(dtype=float)
    m = _fit_ridge_log(X, y, cols)
    return _ridge_log_predict(m, impute_fit_apply(X, test[cols].to_numpy(dtype=float)))


def _risk_metric_row(y_true: np.ndarray, score: np.ndarray,
                     proba: Optional[np.ndarray] = None,
                     prefix: str = "") -> Dict:
    proba = score if proba is None else proba
    m = {
        f"{prefix}auroc": _auroc(y_true, score),
        f"{prefix}auprc": _auprc(y_true, score),
        f"{prefix}brier": _brier(y_true, np.clip(proba, 0.0, 1.0)),
        f"{prefix}label_rate": float(np.mean(y_true)),
    }
    m.update({f"{prefix}{k}": v for k, v in _topk_stats(y_true, score).items()})
    return m


def _select_binary_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select a shortage operating threshold from validation labels only."""
    best_f1, best_threshold = 0.0, 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = np.asarray(probability) >= threshold
        actual = np.asarray(y_true) > 0
        tp = float(np.sum(predicted & actual))
        fp = float(np.sum(predicted & ~actual))
        fn = float(np.sum(~predicted & actual))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def _select_balanced_binary_threshold(
    y_true: np.ndarray, probability: np.ndarray,
) -> float:
    """Select a threshold maximizing validation balanced accuracy."""
    actual = np.asarray(y_true, dtype=float) > 0
    best_score, best_f1, best_threshold = -1.0, -1.0, 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = np.asarray(probability, dtype=float) >= threshold
        tp = float(np.sum(predicted & actual))
        tn = float(np.sum(~predicted & ~actual))
        fp = float(np.sum(predicted & ~actual))
        fn = float(np.sum(~predicted & actual))
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        balanced = (recall + specificity) / 2.0
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        if balanced > best_score or (balanced == best_score and f1 > best_f1):
            best_score, best_f1, best_threshold = balanced, f1, float(threshold)
    return best_threshold


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray,
                    threshold: float) -> Dict[str, float]:
    """Report raw and balanced binary metrics at a validation-selected point."""
    actual = np.asarray(y_true, dtype=float) > 0
    predicted = np.asarray(probability, dtype=float) >= float(threshold)
    tp = float(np.sum(predicted & actual))
    tn = float(np.sum(~predicted & ~actual))
    fp = float(np.sum(predicted & ~actual))
    fn = float(np.sum(~predicted & actual))
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    return {
        "binary_threshold": float(threshold),
        "binary_accuracy": float((tp + tn) / max(len(actual), 1)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "binary_precision": float(tp / max(tp + fp, 1.0)),
        "binary_recall": float(recall),
        "binary_f1": float(2.0 * tp / max(2.0 * tp + fp + fn, 1.0)),
    }


def _risk_selection_key(row: Dict) -> tuple:
    """Validation objective for deployable shortage triage."""
    return (
        float(row.get("validation_topk_recall", float("-inf"))),
        float(row.get("validation_auprc", float("-inf"))),
        -float(row.get("validation_brier", float("inf"))),
    )


def _frame_score(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return frame[column].fillna(0).to_numpy(dtype=float)


def _add_risk_candidate(rows: List[Dict], name: str, family: str, subset: str,
                        train_score: np.ndarray, val_score: np.ndarray,
                        test_score: np.ndarray, y_val: np.ndarray,
                        y_test: np.ndarray, n_params: int,
                        val_proba: Optional[np.ndarray] = None,
                        test_proba: Optional[np.ndarray] = None) -> None:
    """Append one deployable risk score with validation and test evidence."""
    val_prob = (_unit_score(train_score, val_score)
                if val_proba is None else np.asarray(val_proba, dtype=float))
    test_prob = (_unit_score(train_score, test_score)
                 if test_proba is None else np.asarray(test_proba, dtype=float))
    row: Dict = {
        "model": name,
        "family": family,
        "subset": subset,
        "n_params": int(n_params),
    }
    row.update(_risk_metric_row(y_val, val_score, val_prob, "validation_"))
    row.update(_risk_metric_row(y_test, test_score, test_prob, "test_"))
    threshold = _select_binary_threshold(y_val, val_prob)
    row.update({f"validation_{k}": v for k, v in
                _binary_metrics(y_val, val_prob, threshold).items()})
    row.update({f"test_{k}": v for k, v in
                _binary_metrics(y_test, test_prob, threshold).items()})
    rows.append(row)


def _evaluate_risk(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   full_cols: List[str], risk_label: str,
                   feature_mode: str = "operational") -> Dict:
    if feature_mode not in ("operational", "full"):
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    yt = (train[risk_label].fillna(0) > 0).astype(float).to_numpy()
    yv = (val[risk_label].fillna(0) > 0).astype(float).to_numpy()
    ye = (test[risk_label].fillna(0) > 0).astype(float).to_numpy()
    out: Dict = {"label": risk_label, "family": "validation_selected_risk_leaderboard"}
    if yt.sum() < 5 or yv.sum() < 2 or ye.sum() < 2:
        out["note"] = ("too few positive labels for stable risk metrics; "
                       f"train_pos={int(yt.sum())} val_pos={int(yv.sum())} "
                       f"test_pos={int(ye.sum())}")
        out["label_rate_validation"] = float(np.mean(yv)) if len(yv) else float("nan")
        out["label_rate_test"] = float(np.mean(ye))
        out.update({"auroc": float("nan"), "auprc": float("nan"),
                    "brier": float("nan"), "topk_recall": float("nan"),
                    "precision_at_k": float("nan"), "lift_at_k": float("nan")})
        return out

    rows: List[Dict] = []
    X = _fill_nan(train[full_cols].to_numpy(dtype=float))
    m = LogisticRidge(alpha=10.0, max_iter=30).fit(X, yt, full_cols)
    train_logistic = m.predict_proba(X)
    val_logistic = m.predict_proba(impute_fit_apply(X, val[full_cols].to_numpy(dtype=float)))
    test_logistic = m.predict_proba(impute_fit_apply(X, test[full_cols].to_numpy(dtype=float)))
    _add_risk_candidate(
        rows, "logistic_full", "logistic_ridge", "full",
        train_logistic, val_logistic, test_logistic, yv, ye,
        int(m.coef.size + 1), val_logistic, test_logistic)

    exposure_cols = [
        "shortage_events", "shortage_active", "na_shortage_active_mean",
        "na_recall_active_mean", "labeler_shortage_n", "recall_count_y",
        "labeler_recall_n",
    ]
    available = [c for c in exposure_cols if c in train.columns]
    for col in available:
        _add_risk_candidate(
            rows, col, "feature_year_exposure", col,
            _frame_score(train, col), _frame_score(val, col), _frame_score(test, col),
            yv, ye, 0)

    if available:
        train_z = np.column_stack([_zscore(_frame_score(train, c), _frame_score(train, c))
                                   for c in available])
        val_z = np.column_stack([_zscore(_frame_score(train, c), _frame_score(val, c))
                                 for c in available])
        test_z = np.column_stack([_zscore(_frame_score(train, c), _frame_score(test, c))
                                  for c in available])
        _add_risk_candidate(
            rows, "mean_supply_exposure", "exposure_blend", "supply",
            train_z.mean(axis=1), val_z.mean(axis=1), test_z.mean(axis=1),
            yv, ye, len(available))
        _add_risk_candidate(
            rows, "max_supply_exposure", "exposure_blend", "supply",
            train_z.max(axis=1), val_z.max(axis=1), test_z.max(axis=1),
            yv, ye, len(available))

        train_combo = 0.5 * _zscore(train_logistic, train_logistic) \
            + 0.5 * train_z.max(axis=1)
        val_combo = 0.5 * _zscore(train_logistic, val_logistic) \
            + 0.5 * val_z.max(axis=1)
        test_combo = 0.5 * _zscore(train_logistic, test_logistic) \
            + 0.5 * test_z.max(axis=1)
        _add_risk_candidate(
            rows, "logistic_max_exposure_blend", "score_blend",
            "full_plus_supply", train_combo, val_combo, test_combo,
            yv, ye, int(m.coef.size + 1 + len(available)))

    leaderboard = sorted(rows, key=_risk_selection_key, reverse=True)
    selected = leaderboard[0]
    test_best = sorted(
        rows,
        key=lambda r: (
            float(r.get("test_topk_recall", float("-inf"))),
            float(r.get("test_auprc", float("-inf"))),
            -float(r.get("test_brier", float("inf"))),
        ),
        reverse=True,
    )[0]
    selected_lift = float(selected.get("test_lift_at_k", float("nan")))
    selected_recall = float(selected.get("test_topk_recall", float("nan")))
    publishable = bool(
        selected_recall >= 0.25
        and selected_lift >= 5.0
        and float(selected.get("validation_topk_recall", 0.0)) >= 0.05
    )
    out.update({
        "selected_model": selected["model"],
        "selected_family": selected["family"],
        "selection_metric": "validation_topk_recall_then_auprc_then_brier",
        "best_model_by_test": test_best["model"],
        "best_model_by_test_is_diagnostic": True,
        "leaderboard": leaderboard,
        "publishable_candidate": publishable,
        "publishability_reason": (
            "selected risk model meets inventory-triage thresholds"
            if publishable else
            "validation-selected risk model does not meet the configured "
            "triage thresholds: test top-k recall >=25%, lift >=5x, and "
            "validation top-k recall >=5% are required"
        ),
        "auroc": selected["test_auroc"],
        "auprc": selected["test_auprc"],
        "brier": selected["test_brier"],
        "topk_recall": selected["test_topk_recall"],
        "precision_at_k": selected["test_precision_at_k"],
        "lift_at_k": selected["test_lift_at_k"],
        "positives_captured": selected["test_positives_captured"],
        "k": selected["test_k"],
        "label_rate_validation": float(np.mean(yv)),
        "label_rate_test": float(np.mean(ye)),
        "n_params": int(selected["n_params"]),
    })
    return out


def _lb_row(model: str, family: str, subset: str,
            y_true: np.ndarray, y_pred: np.ndarray, y_last: np.ndarray,
            params: int) -> Dict:
    m = metrics(y_true, y_pred, y_last)
    return {"model": model, "family": family, "subset": subset, "params": params,
            **m}


def _with_improvements(ldf: pd.DataFrame, naive_wape: float,
                       ridge_wape: float) -> pd.DataFrame:
    ldf = ldf.copy()
    ldf["improvement_vs_best_naive"] = (
        (naive_wape - ldf["wape"]) / max(naive_wape, 1e-9)
    )
    ldf["improvement_vs_demand_only_ridge"] = (
        (ridge_wape - ldf["wape"]) / max(ridge_wape, 1e-9)
    )
    return ldf.sort_values("wape").reset_index(drop=True)
