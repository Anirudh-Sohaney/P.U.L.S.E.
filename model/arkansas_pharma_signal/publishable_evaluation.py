"""Strict evaluation protocol for the public test-suite artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .evaluate import (_auprc, _auroc, _brier, _binary_metrics,
                       _select_binary_threshold, impute_fit_apply, metrics)
from .arcos_evaluation import evaluate_arcos_rolling_view, evaluate_arcos_view
from .regression import LogisticRidge, RidgeLinear


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_suite(root: Path) -> dict:
    """Verify manifest hashes and target grains before any model is fitted."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    geography = manifest.get("geography_definition", {})
    expected_regions = ["northwest", "northeast", "central", "southwest", "southeast"]
    if (geography.get("name") != "Arkansas DHS/TEFRA five-region county partition"
            or geography.get("source_url") !=
            "https://humanservices.arkansas.gov/wp-content/uploads/TEFRAAttachI.pdf"
            or geography.get("regions") != expected_regions
            or geography.get("county_count") != 75
            or geography.get("partition_policy") !=
            "each Arkansas county FIPS is assigned exactly once"):
        raise ValueError("publishable suite geography contract is missing or invalid")
    required_tables = {
        "arkansas_ndc_next_quarter_test.csv.gz": {
            "cadence": "quarterly", "geography": "Arkansas statewide",
            "direct_observation": True,
            "targets": {"target_next_medicaid_prescriptions",
                        "target_next_shortage_active"},
        },
        "arkansas_county_drug_next_year_test.csv.gz": {
            "cadence": "annual", "geography": "Arkansas county",
            "direct_observation": True,
            "targets": {"target_next_demand_claims"},
        },
        "supplier_ndc_next_month_test.csv.gz": {
            "cadence": "monthly", "geography": "national evidence restricted to Arkansas-exposed NDCs",
            "direct_observation": True,
            "targets": {"target_next_shortage_active"},
        },
        "arcos_zip3_drug_next_quarter_test.csv.gz": {
            "cadence": "quarterly", "geography": "Arkansas ZIP3",
            "direct_observation": True,
            "targets": {"target_next_distribution_grams"},
        },
    }
    tables = manifest.get("tables", {})
    if set(tables) != set(required_tables):
        raise ValueError("publishable suite table contract does not match evaluator")
    for name, required in required_tables.items():
        qualification = tables[name].get("target_qualification", {})
        for field, expected in required.items():
            if field == "targets":
                actual_targets = set(tables[name].get("targets", []))
                if not expected.issubset(actual_targets):
                    raise ValueError(
                        f"publishable suite target mismatch for {name}")
                continue
            if qualification.get(field) != expected:
                raise ValueError(
                    f"publishable suite qualification mismatch for {name}: {field}")
    checks = []
    for name, expected in manifest["artifacts"].items():
        path = root / "data" / name
        exists = path.is_file()
        actual = _hash(path) if exists else ""
        checks.append({"artifact": name, "exists": exists,
                       "hash_matches": actual == expected["sha256"],
                       "rows": int(expected["rows"])})
    if not all(c["exists"] and c["hash_matches"] for c in checks):
        raise ValueError("publishable suite artifact hash verification failed")
    return {"manifest_version": manifest["dataset_version"], "checks": checks}


def validate_table_grains(state: pd.DataFrame, county: pd.DataFrame,
                          supplier: pd.DataFrame) -> dict:
    """Fail closed on duplicate keys or non-consecutive target periods."""
    state_qid = state["year"].astype(int) * 4 + state["quarter"].astype(int)
    target_qid = (state["target_year"].astype(int) * 4
                  + state["target_quarter"].astype(int))
    if state.duplicated(["ndc9", "year", "quarter"]).any():
        raise ValueError("state suite contains duplicate NDC9-quarter keys")
    if not target_qid.sub(state_qid).eq(1).all():
        raise ValueError("state suite contains a non-consecutive target quarter")
    if county.duplicated(["county_fips", "drug_key", "year"]).any():
        raise ValueError("county suite contains duplicate county-drug-year keys")
    if not county["target_year"].sub(county["year"]).eq(1).all():
        raise ValueError("county suite contains a non-consecutive target year")
    month = pd.to_datetime(supplier["month"])
    target_month = pd.to_datetime(supplier["target_month"])
    month_id = month.dt.year * 12 + month.dt.month
    target_id = target_month.dt.year * 12 + target_month.dt.month
    if supplier.duplicated(["ndc9", "supplier", "month"]).any():
        raise ValueError("supplier suite contains duplicate NDC9-supplier-month keys")
    if not target_id.sub(month_id).eq(1).all():
        raise ValueError("supplier suite contains a non-consecutive target month")
    return {"state_rows": int(len(state)), "county_rows": int(len(county)),
            "supplier_rows": int(len(supplier)),
            "all_targets_consecutive": True, "duplicate_keys": False}


def assess_target_adequacy(state: pd.DataFrame, county: pd.DataFrame,
                           supplier: pd.DataFrame,
                           arcos: pd.DataFrame | None = None) -> dict:
    """Report label coverage without upgrading proxies into local inventory."""
    uncensored = supplier[supplier["target_next_right_censored"].fillna(0).eq(0)]
    onset_pool = uncensored[uncensored["shortage_active"].eq(0)]
    result = {
        "state_demand": {
            "rows": int(len(state)),
            "cadence": "quarterly",
            "geography": "Arkansas statewide",
            "supplier_resolution": "none",
            "direct_observation": True,
        },
        "county_demand": {
            "rows": int(len(county)),
            "cadence": "annual",
            "geography": "Arkansas county",
            "supplier_resolution": "labeler_only",
            "direct_observation": True,
        },
        "supplier_shortage_evidence": {
            "rows_total": int(len(supplier)),
            "rows_uncensored": int(len(uncensored)),
            "right_censored_rows": int(len(supplier) - len(uncensored)),
            "cadence": "monthly",
            "geography": "national evidence restricted to Arkansas-exposed NDCs",
            "supplier_resolution": "FDA supplier",
            "direct_observation": True,
            "onset_candidate_rows": int(len(onset_pool)),
            "onset_candidate_positive_rows": int(
                onset_pool["target_next_shortage_active"].eq(1).sum()),
            "onset_candidate_is_thin": bool(len(onset_pool) < 500),
        },
        "requested_local_inventory_target": {
            "grain": "Arkansas county x supplier x drug x week/month",
            "available": False,
            "synthetic_labels_created": False,
            "promotion_eligible": False,
        },
    }
    if arcos is not None:
        result["arcos_distribution_proxy"] = {
            "rows": int(len(arcos)),
            "cadence": "quarterly",
            "geography": "Arkansas ZIP3",
            "drug_resolution": "DEA controlled-substance code",
            "supplier_resolution": "none",
            "direct_observation": True,
            "inventory_target": False,
        }
    return result


def validate_arcos_grain(arcos: pd.DataFrame) -> dict:
    """Fail closed on ARCOS duplicate keys and non-consecutive targets."""
    required = {"zip3", "drug_code", "period_index", "target_period_index",
                "target", "target_year", "target_quarter"}
    missing = required.difference(arcos.columns)
    if missing:
        raise ValueError(f"ARCOS suite missing columns: {sorted(missing)}")
    if arcos.duplicated(["zip3", "drug_code", "period_index"]).any():
        raise ValueError("ARCOS suite contains duplicate ZIP3-drug-quarter keys")
    if not arcos["target_period_index"].sub(arcos["period_index"]).eq(1).all():
        raise ValueError("ARCOS suite contains a non-consecutive target quarter")
    return {"arcos_rows": int(len(arcos)), "duplicate_keys": False,
            "all_targets_consecutive": True}


def _arcos_json(result: dict) -> dict:
    """Convert ARCOS dataframe diagnostics into manifest-friendly records."""
    out = dict(result)
    if isinstance(out.get("leaderboard"), pd.DataFrame):
        out["leaderboard"] = out["leaderboard"].to_dict(orient="records")
    if isinstance(out.get("folds"), pd.DataFrame):
        out["folds"] = out["folds"].to_dict(orient="records")
    return out


def _numeric_features(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = frame.reindex(columns=list(columns), fill_value=0.0)
    values = values.apply(pd.to_numeric, errors="coerce")
    for column in values.columns:
        name = str(column).lower()
        if any(token in name for token in
               ("demand", "claims", "fills", "cost", "supplier_count",
                "article", "news_", "flu_week_count")):
            values[column] = np.log1p(values[column].clip(lower=0))
    return values.to_numpy(dtype=float)


def _within_relative_error(y_true: np.ndarray, y_pred: np.ndarray,
                           tolerance: float = 0.05) -> float:
    """Return the fraction of finite targets within a relative-error band."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return float("nan")
    relative_error = np.abs(predicted[mask] - actual[mask]) / np.maximum(
        np.abs(actual[mask]), 1e-9)
    return float(np.mean(relative_error <= tolerance))


def _fit_ridge(train: pd.DataFrame, validation: pd.DataFrame,
               test: pd.DataFrame, feature_columns: list[str], target: str,
               baseline_column: str) -> dict:
    x_train = _numeric_features(train, feature_columns)
    x_validation = _numeric_features(validation, feature_columns)
    x_test = _numeric_features(test, feature_columns)
    x_train = impute_fit_apply(x_train, x_train)
    x_validation = impute_fit_apply(x_train, x_validation)
    x_test = impute_fit_apply(x_train, x_test)
    y_train = np.log1p(pd.to_numeric(train[target], errors="coerce").clip(lower=0))
    train_log_max = float(np.nanmax(y_train))
    y_validation = pd.to_numeric(validation[target], errors="coerce").to_numpy(float)
    y_test = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
    baseline_validation = pd.to_numeric(validation[baseline_column], errors="coerce").to_numpy(float)
    baseline_test = pd.to_numeric(test[baseline_column], errors="coerce").to_numpy(float)
    candidates = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = RidgeLinear(alpha=alpha).fit(x_train, y_train, feature_columns)
        candidates[alpha] = (
            np.maximum(np.expm1(np.clip(model.predict(x_validation), 0.0,
                                        train_log_max)), 0.0),
            np.maximum(np.expm1(np.clip(model.predict(x_test), 0.0,
                                        train_log_max)), 0.0),
        )
    selected_alpha = min(
        candidates,
        key=lambda alpha: metrics(y_validation, candidates[alpha][0])['wape'],
    )
    prediction_validation, prediction_test = candidates[selected_alpha]
    persistence_validation_metrics = metrics(y_validation, baseline_validation)
    persistence_test_metrics = metrics(y_test, baseline_test)
    model_validation_metrics = metrics(y_validation, prediction_validation)
    model_test_metrics = metrics(y_test, prediction_test)
    persistence_validation_metrics["within_5pct"] = _within_relative_error(
        y_validation, baseline_validation)
    persistence_test_metrics["within_5pct"] = _within_relative_error(
        y_test, baseline_test)
    model_validation_metrics["within_5pct"] = _within_relative_error(
        y_validation, prediction_validation)
    model_test_metrics["within_5pct"] = _within_relative_error(
        y_test, prediction_test)
    return {
        "target": target,
        "feature_columns": feature_columns,
        "selected_alpha": selected_alpha,
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "persistence": {
            "validation": persistence_validation_metrics,
            "test": persistence_test_metrics,
        },
        "ridge_log1p": {
            "validation": model_validation_metrics,
            "test": model_test_metrics,
        },
    }


def _binary_result(y_true: np.ndarray, probability: np.ndarray,
                   threshold: float) -> dict:
    return {
        **_binary_metrics(y_true, probability, threshold),
        "auroc": _auroc(y_true, probability),
        "auprc": _auprc(y_true, probability),
        "brier": _brier(y_true, np.clip(probability, 0.0, 1.0)),
        "positive_rows": int(np.sum(y_true > 0)),
        "rows": int(len(y_true)),
    }


def _fit_logistic(train: pd.DataFrame, validation: pd.DataFrame,
                  test: pd.DataFrame, feature_columns: list[str],
                  target: str, persistence_column: str) -> dict:
    x_train = _numeric_features(train, feature_columns)
    x_train = impute_fit_apply(x_train, x_train)
    x_validation = impute_fit_apply(x_train, _numeric_features(validation, feature_columns))
    x_test = impute_fit_apply(x_train, _numeric_features(test, feature_columns))
    y_train = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    y_validation = pd.to_numeric(validation[target], errors="coerce").to_numpy(float)
    y_test = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
    model = LogisticRidge(alpha=10.0, max_iter=100).fit(x_train, y_train, feature_columns)
    validation_probability = model.predict_proba(x_validation)
    threshold = _select_binary_threshold(y_validation, validation_probability)
    test_probability = model.predict_proba(x_test)
    return {
        "target": target,
        "feature_columns": feature_columns,
        "threshold_selected_on": "validation_only",
        "threshold": threshold,
        "logistic_ridge": {
            "validation": _binary_result(y_validation, validation_probability, threshold),
            "test": _binary_result(y_test, test_probability, threshold),
        },
        "persistence": {
            "validation": _binary_result(y_validation,
                pd.to_numeric(validation[persistence_column], errors="coerce").to_numpy(float), 0.5),
            "test": _binary_result(y_test,
                pd.to_numeric(test[persistence_column], errors="coerce").to_numpy(float), 0.5),
        },
    }


def _validate_consecutive(frame: pd.DataFrame, left: str, right: str,
                           periods: str) -> dict:
    if frame.duplicated([left, periods]).any():
        raise ValueError(f"duplicate feature keys in {left}")
    return {"rows": int(len(frame)), "duplicate_keys": False}


def evaluate_suite(suite_root: Path) -> dict:
    """Verify and evaluate all compatible benchmark tasks separately."""
    verification = verify_suite(suite_root)
    data = suite_root / "data"
    state = pd.read_csv(data / "arkansas_ndc_next_quarter_test.csv.gz", dtype={"ndc9": str})
    county = pd.read_csv(data / "arkansas_county_drug_next_year_test.csv.gz",
                         dtype={"county_fips": str, "drug_key": str})
    supplier = pd.read_csv(data / "supplier_ndc_next_month_test.csv.gz",
                           dtype={"ndc9": str, "supplier": str})
    arcos = pd.read_csv(data / "arcos_zip3_drug_next_quarter_test.csv.gz",
                        dtype={"zip3": str, "drug_code": str})
    arcos = arcos.rename(columns={"target_next_distribution_grams": "target"})
    table_validation = validate_table_grains(state, county, supplier)
    arcos_validation = validate_arcos_grain(arcos)
    target_adequacy = assess_target_adequacy(state, county, supplier, arcos)
    state_train = state[state.year <= 2019]
    state_validation = state[state.year == 2020]
    state_test = state[state.year >= 2021]
    state_features = ["medicaid_prescriptions",
                      "feature_lag1_medicaid_prescriptions",
                      "feature_lag2_medicaid_prescriptions",
                      "feature_ma2_medicaid_prescriptions",
                      "fda_shortage_active", "fda_shortage_supplier_count",
                      "prior_context_missing"] + \
        [c for c in state.columns if c.startswith("prior_")]
    state_features = list(dict.fromkeys(state_features))
    county_train = county[county.year <= 2020]
    county_validation = county[county.year == 2021]
    county_test = county[county.year >= 2022]
    county_features = ["demand_claims", "demand_fills", "demand_cost",
                       "mapping_confidence"]
    supplier_censored = int(supplier["target_next_right_censored"].fillna(0).sum())
    supplier = supplier[supplier["target_next_right_censored"].fillna(0).eq(0)].copy()
    supplier["feature_year"] = pd.to_datetime(supplier["month"]).dt.year
    supplier["month_number"] = pd.to_datetime(supplier["month"]).dt.month
    supplier_train = supplier[supplier.feature_year <= 2020]
    supplier_validation = supplier[supplier.feature_year == 2021]
    supplier_test = supplier[supplier.feature_year == 2022]
    supplier_features = ["shortage_active", "month_number"]
    return {
        "protocol": "grain_preserving_public_suite_v1",
        "verification": verification,
        "table_validation": table_validation,
        "arcos_validation": arcos_validation,
        "target_adequacy": target_adequacy,
        "supplier_target_rows_excluded_as_right_censored": supplier_censored,
        "tasks": {
            "arkansas_ndc_next_quarter_demand": _fit_ridge(
                state_train, state_validation, state_test, state_features,
                "target_next_medicaid_prescriptions", "medicaid_prescriptions"),
            "arkansas_ndc_next_quarter_shortage": _fit_logistic(
                state_train, state_validation, state_test, state_features,
                "target_next_shortage_active", "fda_shortage_active"),
            "arkansas_county_drug_next_year_demand": _fit_ridge(
                county_train, county_validation, county_test, county_features,
                "target_next_demand_claims", "demand_claims"),
            "supplier_ndc_next_month_continuation": _fit_logistic(
                supplier_train, supplier_validation, supplier_test, supplier_features,
                "target_next_shortage_active", "shortage_active"),
            "arcos_zip3_drug_next_quarter_distribution": _arcos_json(
                evaluate_arcos_view(arcos)),
        },
        "promotion_policy": {
            "numeric": "selected model must beat persistence by >=10% WAPE on held-out test",
            "binary": "report raw, balanced accuracy, AUROC, AUPRC, Brier; raw accuracy alone is insufficient",
            "locality": "county and supplier tasks remain separate; no composite score is computed",
        },
    }


def _rolling_year_splits(frame: pd.DataFrame, min_train_years: int = 4):
    """Yield train, validation, and immediately following test years."""
    years = sorted(int(year) for year in frame["year"].dropna().unique())
    for validation_year in years:
        train_years = [year for year in years if year < validation_year]
        test_year = validation_year + 1
        if (len(train_years) < min_train_years
                or test_year not in years):
            continue
        yield (
            f"<= {validation_year - 1}", str(validation_year), str(test_year),
            frame[frame["year"] < validation_year],
            frame[frame["year"] == validation_year],
            frame[frame["year"] == test_year],
        )


def _rolling_month_splits(frame: pd.DataFrame, min_train_months: int = 12,
                          validation_months: int = 3,
                          test_months: int = 3):
    """Yield chronological month-block folds without skipping calendar gaps."""
    periods = sorted(pd.to_datetime(frame["month"]).dt.to_period("M").unique())
    for index in range(min_train_months,
                       len(periods) - validation_months - test_months + 1):
        train_periods = periods[:index]
        validation_periods = periods[index:index + validation_months]
        test_periods = periods[index + validation_months:
                                index + validation_months + test_months]
        expected = pd.period_range(train_periods[-1] + 1,
                                   test_periods[-1], freq="M")
        if len(expected) != len(test_periods) + validation_months:
            continue
        period = pd.to_datetime(frame["month"]).dt.to_period("M")
        yield (
            f"<= {train_periods[-1]}",
            f"{validation_periods[0]}-{validation_periods[-1]}",
            f"{test_periods[0]}-{test_periods[-1]}",
            frame[period.isin(train_periods)],
            frame[period.isin(validation_periods)],
            frame[period.isin(test_periods)],
        )


def _rolling_task_result(name: str, splits, feature_columns: list[str],
                         target: str, baseline: str, binary: bool) -> dict:
    folds = []
    for train_label, validation_label, test_label, train, validation, test in splits:
        if binary:
            result = _fit_logistic(train, validation, test, feature_columns,
                                   target, baseline)
            model_metrics = result["logistic_ridge"]
            persistence_metrics = result["persistence"]
            model_score = float(model_metrics["test"]["balanced_accuracy"])
            persistence_score = float(
                persistence_metrics["test"]["balanced_accuracy"])
            row = {
                "task": name,
                "train_period": train_label,
                "validation_period": validation_label,
                "test_period": test_label,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "model_accuracy": float(model_metrics["test"]["binary_accuracy"]),
                "persistence_accuracy": float(
                    persistence_metrics["test"]["binary_accuracy"]),
                "model_balanced_accuracy": model_score,
                "persistence_balanced_accuracy": persistence_score,
                "model_auroc": float(model_metrics["test"]["auroc"]),
                "persistence_auroc": float(persistence_metrics["test"]["auroc"]),
                "balanced_improvement_vs_persistence": model_score - persistence_score,
                "accuracy_goal_met": bool(
                    float(model_metrics["test"]["binary_accuracy"]) >= 0.75
                    and model_score >= 0.75),
            }
        else:
            result = _fit_ridge(train, validation, test, feature_columns,
                                target, baseline)
            model_metrics = result["ridge_log1p"]
            persistence_metrics = result["persistence"]
            model_wape = float(model_metrics["test"]["wape"])
            persistence_wape = float(persistence_metrics["test"]["wape"])
            improvement = ((persistence_wape - model_wape)
                           / max(persistence_wape, 1e-12))
            row = {
                "task": name,
                "train_period": train_label,
                "validation_period": validation_label,
                "test_period": test_label,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "model_wape": model_wape,
                "persistence_wape": persistence_wape,
                "improvement_vs_persistence": improvement,
                "accuracy_goal_met": bool(
                    float(model_metrics["test"]["within_5pct"]) >= 0.75),
            }
        folds.append(row)
    result = {"task": name, "folds": folds, "fold_count": len(folds)}
    if not folds:
        result["promotion_candidate"] = False
        result["reason"] = "no eligible chronological folds"
        return result
    table = pd.DataFrame(folds)
    if binary:
        mean_accuracy = float(table["model_accuracy"].mean())
        mean_balanced = float(table["model_balanced_accuracy"].mean())
        mean_skill = float(table["balanced_improvement_vs_persistence"].mean())
        result.update({
            "mean_model_accuracy": mean_accuracy,
            "mean_model_balanced_accuracy": mean_balanced,
            "mean_balanced_improvement_vs_persistence": mean_skill,
            "requested_accuracy_goal_met": bool(
                mean_accuracy >= 0.75 and mean_balanced >= 0.75),
            "promotion_candidate": bool(
                mean_accuracy >= 0.75 and mean_balanced >= 0.75
                and mean_skill >= 0.01),
        })
    else:
        improvements = table["improvement_vs_persistence"]
        mean_improvement = float(improvements.mean())
        result.update({
            "mean_improvement_vs_persistence": mean_improvement,
            "folds_beating_persistence": int((improvements > 0).sum()),
            "requested_accuracy_goal_met": bool(table["accuracy_goal_met"].mean() >= 1.0),
            "promotion_candidate": bool(
                len(folds) > 0 and (improvements > 0).all()
                and mean_improvement >= 0.10),
        })
    return result


def evaluate_suite_rolling(suite_root: Path, min_train_years: int = 4,
                           min_train_months: int = 12) -> dict:
    """Evaluate every public task using strict chronological rolling folds."""
    verification = verify_suite(suite_root)
    data = suite_root / "data"
    state = pd.read_csv(data / "arkansas_ndc_next_quarter_test.csv.gz",
                        dtype={"ndc9": str})
    county = pd.read_csv(data / "arkansas_county_drug_next_year_test.csv.gz",
                         dtype={"county_fips": str, "drug_key": str})
    supplier = pd.read_csv(data / "supplier_ndc_next_month_test.csv.gz",
                           dtype={"ndc9": str, "supplier": str})
    arcos = pd.read_csv(data / "arcos_zip3_drug_next_quarter_test.csv.gz",
                        dtype={"zip3": str, "drug_code": str})
    arcos = arcos.rename(columns={"target_next_distribution_grams": "target"})
    arcos_validation = validate_arcos_grain(arcos)
    target_adequacy = assess_target_adequacy(state, county, supplier, arcos)
    supplier = supplier[supplier["target_next_right_censored"].fillna(0).eq(0)].copy()
    supplier["month"] = pd.to_datetime(supplier["month"]).dt.strftime("%Y-%m")
    state_features = ["medicaid_prescriptions",
                      "feature_lag1_medicaid_prescriptions",
                      "feature_lag2_medicaid_prescriptions",
                      "feature_ma2_medicaid_prescriptions",
                      "fda_shortage_active", "fda_shortage_supplier_count",
                      "prior_context_missing"] + \
        [column for column in state.columns if column.startswith("prior_")]
    state_features = list(dict.fromkeys(state_features))
    county_features = ["demand_claims", "demand_fills", "demand_cost",
                       "mapping_confidence"]
    supplier_features = ["shortage_active", "month_number"]
    supplier["month_number"] = pd.to_datetime(supplier["month"]).dt.month
    return {
        "protocol": "grain_preserving_public_suite_rolling_v1",
        "verification": verification,
        "arcos_validation": arcos_validation,
        "target_adequacy": target_adequacy,
        "tasks": {
            "arkansas_ndc_next_quarter_demand": _rolling_task_result(
                "arkansas_ndc_next_quarter_demand",
                _rolling_year_splits(state, min_train_years), state_features,
                "target_next_medicaid_prescriptions", "medicaid_prescriptions", False),
            "arkansas_ndc_next_quarter_shortage": _rolling_task_result(
                "arkansas_ndc_next_quarter_shortage",
                _rolling_year_splits(state, min_train_years), state_features,
                "target_next_shortage_active", "fda_shortage_active", True),
            "arkansas_county_drug_next_year_demand": _rolling_task_result(
                "arkansas_county_drug_next_year_demand",
                _rolling_year_splits(county, min_train_years), county_features,
                "target_next_demand_claims", "demand_claims", False),
            "supplier_ndc_next_month_continuation": _rolling_task_result(
                "supplier_ndc_next_month_continuation",
                _rolling_month_splits(supplier, min_train_months), supplier_features,
                "target_next_shortage_active", "shortage_active", True),
            "arcos_zip3_drug_next_quarter_distribution": _arcos_json(
                evaluate_arcos_rolling_view(arcos)),
        },
        "promotion_policy": {
            "numeric": "all rolling folds beat persistence and mean WAPE improvement is >=10%",
            "binary": "mean raw and balanced accuracy are >=75% and balanced skill over persistence is >=1 percentage point",
            "locality": "county, statewide, and supplier tasks remain separate; no composite score is computed",
        },
    }
