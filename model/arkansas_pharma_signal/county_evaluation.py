"""Annual county x drug next-year demand evaluation on real CMS-derived outcomes.

Contract: feature year ``t`` predicts real ``demand_claims`` at ``t+1`` for the
same ``(county_fips, drug_key)``. Only rows with an observed ``t+1`` label are
kept; no synthetic shortage labels are introduced. All features come from year
``t`` only, so there is no same-year target leakage.

Split: train feature years <= 2020, validation 2021, test > 2021. Candidates
are the previous-year naive (demand at ``t``) and a leakage-safe ridge/log1p
model; the candidate with lower validation WAPE is selected, and the
publishable flag requires a documented 10% test WAPE improvement over the
naive. Rolling-origin folds reuse the same no-leakage contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import io
from .evaluate import _fit_ridge_log, _ridge_log_predict, impute_fit_apply, metrics
from .regression import _fill_nan

ARTIFACT_PATH = "outcomes/county_demand.csv.gz"
TARGET = "demand_claims"
# All feature-year-t source columns; nothing from t+1 enters the feature
# matrix. Kept for source-column documentation only.
FEATURE_COLS = ["demand_claims", "demand_fills", "demand_cost",
                "mapping_confidence", "source_row_count", "mapped_city_count"]
# Transformed feature names in exact column order of _feature_matrix output;
# these are the coefficient names on the fitted ridge model.
FEATURE_MATRIX_COLS = [
    "log1p_demand_claims", "log1p_demand_fills", "log1p_demand_cost",
    "log1p_source_row_count", "log1p_mapped_city_count", "mapping_confidence",
]
PUBLISHABLE_THRESHOLD = 0.10
TRAIN_CUTOFF = 2021  # train <= 2020, validation == 2021, test > 2021


def load_county_demand(cfg) -> pd.DataFrame:
    """Load the real county outcome artifact, failing clearly when absent."""
    path = cfg.artifact_path(ARTIFACT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"county demand artifact missing: {path}. "
            "Run `build-county-outcomes` first.")
    return io.load_csv(path)


def build_next_year_view(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Feature year t -> observed demand_claims at t+1 for the same pair.

    ``y_target`` is demand at ``t+1``; ``y_last`` is demand at ``t`` (the
    previous-year naive prediction). Rows without an observed ``t+1`` label
    are dropped.
    """
    df = outcomes.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["county_fips"] = df["county_fips"].astype(str)
    df["drug_key"] = df["drug_key"].astype(str)
    df = df.sort_values(["county_fips", "drug_key", "year"])
    g = df.groupby(["county_fips", "drug_key"], sort=False)
    df["y_target"] = g["demand_claims"].shift(-1)
    df["y_last"] = df["demand_claims"]
    view = df.dropna(subset=["y_target"]).copy()
    view["y_target"] = pd.to_numeric(view["y_target"], errors="coerce")
    view = view.dropna(subset=["y_target"])
    return view.reset_index(drop=True)


def _feature_matrix(view: pd.DataFrame) -> np.ndarray:
    """Leakage-safe feature matrix: log1p demand/counts, raw confidence."""
    X = pd.DataFrame(index=view.index)
    for c in ("demand_claims", "demand_fills", "demand_cost",
              "source_row_count", "mapped_city_count"):
        X[f"log1p_{c}"] = np.log1p(
            pd.to_numeric(view[c], errors="coerce").clip(lower=0))
    X["mapping_confidence"] = pd.to_numeric(
        view["mapping_confidence"], errors="coerce")
    assert list(X.columns) == FEATURE_MATRIX_COLS
    return _fill_nan(X.to_numpy(dtype=float))


def _split_masks(view: pd.DataFrame, train_cutoff: int = TRAIN_CUTOFF) -> tuple:
    train = view["year"] <= train_cutoff - 1
    val = view["year"] == train_cutoff
    test = view["year"] > train_cutoff
    return train, val, test


def _fit_candidates(train: pd.DataFrame, val: pd.DataFrame,
                    test: pd.DataFrame) -> Dict:
    """Fit ridge/log1p on train; return raw-scale predictions per split."""
    X_tr = _feature_matrix(train)
    y_tr = train[TARGET].to_numpy(dtype=float)
    model = _fit_ridge_log(X_tr, y_tr, FEATURE_MATRIX_COLS, alpha=10.0)
    return {
        "model": model,
        "train": _ridge_log_predict(model, X_tr),
        "val": _ridge_log_predict(model, impute_fit_apply(X_tr, _feature_matrix(val))),
        "test": _ridge_log_predict(model, impute_fit_apply(X_tr, _feature_matrix(test))),
    }


def _mapping_confidence_summary(view: pd.DataFrame) -> Dict:
    conf = pd.to_numeric(view["mapping_confidence"], errors="coerce")
    return {
        "mean": float(conf.mean()) if conf.notna().any() else float("nan"),
        "min": float(conf.min()) if conf.notna().any() else float("nan"),
        "max": float(conf.max()) if conf.notna().any() else float("nan"),
        "rows_with_confidence": int(conf.notna().sum()),
    }


def evaluate_county_demand(
    outcomes: pd.DataFrame,
    train_cutoff: int = TRAIN_CUTOFF,
) -> Dict:
    """Strict next-year county x drug evaluation with validation selection."""
    view = build_next_year_view(outcomes)
    train_mask, val_mask, test_mask = _split_masks(view, train_cutoff)
    train, val, test = view[train_mask], view[val_mask], view[test_mask]

    result: Dict = {
        "split": "strict_next_year_county_drug_time",
        "target": TARGET,
        "feature_years": {"train": f"<= {train_cutoff - 1}",
                          "validation": str(train_cutoff),
                          "test": f"> {train_cutoff}"},
        "n_rows": {"train": int(len(train)), "validation": int(len(val)),
                   "test": int(len(test))},
        "n_counties": int(view["county_fips"].nunique()),
        "n_drugs": int(view["drug_key"].nunique()),
        "mapping_confidence": _mapping_confidence_summary(view),
    }
    if len(train) < 30 or len(val) < 5 or len(test) < 5:
        result["error"] = (f"insufficient rows train={len(train)} "
                           f"val={len(val)} test={len(test)}")
        return result

    y_val = val["y_target"].to_numpy(dtype=float)
    y_test = test["y_target"].to_numpy(dtype=float)
    naive_val = val["y_last"].to_numpy(dtype=float)
    naive_test = test["y_last"].to_numpy(dtype=float)

    ridge = _fit_candidates(train, val, test)
    ridge_val = ridge["val"]
    ridge_test = ridge["test"]

    naive_val_wape = metrics(y_val, naive_val, naive_val)["wape"]
    ridge_val_wape = metrics(y_val, ridge_val, naive_val)["wape"]
    selected = ("ridge_log1p" if ridge_val_wape < naive_val_wape
                else "previous_year_naive")
    selected_test = ridge_test if selected == "ridge_log1p" else naive_test

    leaderboard = pd.DataFrame([
        {"model": "previous_year_naive", "family": "naive",
         "validation_wape": naive_val_wape,
         **metrics(y_test, naive_test, naive_test)},
        {"model": "ridge_log1p", "family": "ridge_linear_log1p",
         "validation_wape": ridge_val_wape,
         **metrics(y_test, ridge_test, naive_test)},
    ])
    selected_row = leaderboard.loc[leaderboard["model"] == selected].iloc[0]
    naive_row = leaderboard.loc[leaderboard["model"] == "previous_year_naive"].iloc[0]
    improvement = (naive_row["wape"] - selected_row["wape"]) / max(naive_row["wape"], 1e-9)
    publishable = bool(selected == "ridge_log1p"
                       and improvement >= PUBLISHABLE_THRESHOLD)

    result.update({
        "candidates": ["previous_year_naive", "ridge_log1p"],
        "selection_metric": "validation_wape",
        "selected_model": selected,
        "selected_validation_wape": float(selected_row["validation_wape"]),
        "naive_validation_wape": float(naive_val_wape),
        "ridge_validation_wape": float(ridge_val_wape),
        "test_improvement_vs_naive": float(improvement),
        "publishability_threshold": PUBLISHABLE_THRESHOLD,
        "publishable_candidate": publishable,
        "publishability_reason": (
            f"selected {selected} beats previous-year naive by "
            f"{improvement:.1%} WAPE on strict test (>=10% threshold)"
            if publishable else
            f"selected {selected} test WAPE {float(selected_row['wape']):.4f} vs "
            f"naive {float(naive_row['wape']):.4f}; relative gap "
            f"{improvement:.1%} below the 10% publishable threshold"
        ),
        "leaderboard": leaderboard,
    })
    return result


def county_rolling_cutoffs(view: pd.DataFrame,
                           min_train_years: int = 4) -> List[int]:
    """Validation cutoffs with >= min_train_years of train and a future test."""
    years = sorted(int(y) for y in view["year"].dropna().unique())
    max_year = max(years) if years else None
    out: List[int] = []
    for cutoff in years:
        train_years = [y for y in years if y <= cutoff - 1]
        test_years = [y for y in years if y > cutoff and y < max_year]
        if len(train_years) >= min_train_years and test_years:
            out.append(cutoff)
    return out


def evaluate_county_demand_rolling(
    outcomes: pd.DataFrame,
    min_train_years: int = 4,
) -> Dict:
    """Rolling-origin next-year county x drug folds with the same no-leakage
    contract: each fold trains on years <= cutoff-1, selects on cutoff, and
    tests on later years."""
    view = build_next_year_view(outcomes)
    cutoffs = county_rolling_cutoffs(view, min_train_years=min_train_years)
    rows: List[Dict] = []
    for cutoff in cutoffs:
        train_mask, val_mask, test_mask = _split_masks(view, cutoff)
        train, val, test = view[train_mask], view[val_mask], view[test_mask]
        if len(train) < 30 or len(val) < 5 or len(test) < 5:
            continue
        y_val = val["y_target"].to_numpy(dtype=float)
        y_test = test["y_target"].to_numpy(dtype=float)
        naive_val = val["y_last"].to_numpy(dtype=float)
        naive_test = test["y_last"].to_numpy(dtype=float)
        ridge = _fit_candidates(train, val, test)
        selected = ("ridge_log1p"
                    if metrics(y_val, ridge["val"], naive_val)["wape"]
                    < metrics(y_val, naive_val, naive_val)["wape"]
                    else "previous_year_naive")
        pred = ridge["test"] if selected == "ridge_log1p" else naive_test
        m = metrics(y_test, pred, naive_test)
        naive_wape = metrics(y_test, naive_test, naive_test)["wape"]
        improvement = (naive_wape - m["wape"]) / max(naive_wape, 1e-9)
        rows.append({
            "cutoff_year": int(cutoff),
            "train_years": f"<= {cutoff - 1}",
            "validation_year": int(cutoff),
            "test_years": f"> {cutoff}",
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "selected_model": selected,
            "selected_model_wape": m["wape"],
            "naive_wape": naive_wape,
            "improvement_vs_naive": float(improvement),
            "beats_naive": bool(improvement > 0),
            "clears_10pct_gate": bool(improvement >= PUBLISHABLE_THRESHOLD),
        })

    folds = pd.DataFrame(rows)
    summary: Dict = {
        "split": "rolling_origin_next_year_county_drug",
        "target": TARGET,
        "min_train_years": int(min_train_years),
        "publishability_threshold": PUBLISHABLE_THRESHOLD,
        "fold_count": int(len(folds)),
        "folds": folds,
    }
    if folds.empty:
        summary.update({
            "publishable_rolling_candidate": False,
            "publishability_reason": "no eligible rolling county-demand folds",
        })
        return summary

    imp = folds["improvement_vs_naive"].to_numpy(dtype=float)
    finite = imp[np.isfinite(imp)]
    mean_imp = float(np.mean(finite)) if finite.size else float("nan")
    n_beat = int(np.sum(finite > 0))
    all_beat = bool(finite.size == len(folds) and n_beat == len(folds))
    publishable = bool(len(folds) >= 3 and all_beat
                       and np.isfinite(mean_imp)
                       and mean_imp >= PUBLISHABLE_THRESHOLD)
    summary.update({
        "mean_improvement_vs_naive": mean_imp,
        "folds_beating_naive": n_beat,
        "fraction_folds_beating_naive": float(n_beat / len(folds)),
        "publishable_rolling_candidate": publishable,
        "publishability_reason": (
            f"all {len(folds)} rolling folds beat the previous-year naive and "
            f"mean improvement {mean_imp:.1%} is >=10%"
            if publishable else
            f"rolling evidence below gate: {n_beat}/{len(folds)} folds beat "
            f"naive, mean improvement {mean_imp:.1%}"
        ),
    })
    return summary


def evaluate_county_demand_by_region(
    outcomes: pd.DataFrame,
    min_train_years: int = 4,
) -> Dict:
    """Evaluate the same rolling contract independently within each region.

    Regions are reporting slices derived from county FIPS; this function does
    not pool or impute across regions and does not create a supplier target.
    """
    required = "arkansas_region"
    if required not in outcomes.columns:
        raise ValueError("county outcomes missing arkansas_region")
    frame = outcomes.copy()
    regions = sorted(
        value for value in frame[required].fillna("").astype(str).str.strip().unique()
        if value
    )
    rows: List[Dict] = []
    for region in regions:
        regional = frame[frame[required].fillna("").astype(str).str.strip().eq(region)]
        view = build_next_year_view(regional)
        rolling = evaluate_county_demand_rolling(
            regional, min_train_years=min_train_years)
        rows.append({
            "arkansas_region": region,
            "n_rows": int(len(view)),
            "n_counties": int(view["county_fips"].nunique()),
            "n_drugs": int(view["drug_key"].nunique()),
            "fold_count": int(rolling["fold_count"]),
            "mean_improvement_vs_naive": rolling.get("mean_improvement_vs_naive"),
            "folds_beating_naive": rolling.get("folds_beating_naive", 0),
            "fraction_folds_beating_naive": rolling.get("fraction_folds_beating_naive"),
            "publishable_rolling_candidate": rolling.get(
                "publishable_rolling_candidate", False),
            "publishability_reason": rolling.get("publishability_reason", ""),
        })
    return {
        "split": "rolling_origin_next_year_county_drug_by_region",
        "target": TARGET,
        "min_train_years": int(min_train_years),
        "region_count": len(rows),
        "regions": rows,
    }
