"""Strict next-period split, no-leakage, leaderboard, and no-synthetic tests.

Logic tests use tiny deterministic frames; integration tests read prebuilt
artifacts when present so the suite stays fast in CI.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import arkansas_pharma_signal.datasets as datasets_module
from arkansas_pharma_signal.datasets import (
    ablation_features,
    apply_encoder,
    build_next_period,
    feature_columns_for_mode,
    fit_encoder_spec,
)
from arkansas_pharma_signal.config import Config
from arkansas_pharma_signal.features import build_pharmacy_access_features
from arkansas_pharma_signal.evaluate import (
    annual_fold_cutoffs,
    evaluate_demand_rolling,
    evaluate_next_period,
)
from arkansas_pharma_signal.input_contract import (
    PERIODIC_TRAINING_ONLY,
    disposition_for_variable,
)

TARGET_COLS = {"demand_claims_t1", "demand_fills_t1", "demand_cost_t1",
               "shortage_events_t1", "shortage_active_t1"}


def _toy_panel():
    rows = []
    for year in range(2013, 2017):
        for city, ing in (("X", "metformin"), ("Y", "insulin")):
            rows.append({
                "year": year, "city": city, "drug": "d",
                "drug_key": ing, "demand_claims": 100 + year,
                "demand_fills": 90 + year, "demand_cost": 500 + year,
                "shortage_events": 1.0 if year == 2015 else 0.0,
                "shortage_active": 0.0,
                "ingredient": ing, "labeler": "L",
            })
    return pd.DataFrame(rows)


def _toy_annual_panel():
    rows = []
    for year in range(2013, 2019):
        for idx in range(16):
            rows.append({
                "year": year,
                "city": f"city_{idx % 4}",
                "drug": f"drug_{idx}",
                "drug_key": f"drug_{idx}",
                "demand_claims": 100.0 + idx * 3 + (year - 2013) * (idx + 2),
                "demand_fills": 90.0 + idx * 2,
                "demand_cost": 500.0 + idx * 5,
                "shortage_events": 0.0,
                "shortage_active": 0.0,
            })
    return pd.DataFrame(rows)


def test_annual_fold_cutoffs_are_chronological():
    cutoffs = annual_fold_cutoffs(_toy_annual_panel(), min_train_years=2)
    assert cutoffs == [2015, 2016]
    assert all(a < b for a, b in zip(cutoffs, cutoffs[1:]))


def test_annual_rolling_uses_validation_selected_rows():
    result = evaluate_demand_rolling(
        _toy_annual_panel(), min_train_years=2,
        test_window_years=1, neural=False,
    )
    folds = result["folds"]
    assert result["fold_count"] == 2
    assert (folds["validation_year"] < folds["test_years"].str[:4].astype(int)).all()
    assert (folds["selected_model"] == "calibrated_ridge_blend").all()
    assert folds["diagnostic_best_model_by_test"].notna().all()
    assert result["publishable_rolling_candidate"] is False


def test_next_period_target_is_year_plus_one():
    panel = _toy_panel()
    view, _ = build_next_period(panel)
    # No synthetic targets: every t+1 demand is an exact real demand value.
    assert set(view["demand_claims_t1"]) <= set(panel["demand_claims"])
    # Max-year rows (no t+1) are dropped.
    assert (view["year"] < panel["year"].max()).all()
    # For the same city-drug, target equals the real demand at year t+1.
    t1 = panel.rename(columns={"year": "y1", "demand_claims": "d1"})[
        ["drug_key", "city", "y1", "d1"]]
    view["y1"] = view["year"] + 1
    check = view.merge(t1, on=["drug_key", "city", "y1"])
    assert (check["demand_claims_t1"] == check["d1"]).all()


def test_no_same_year_target_leakage():
    panel = _toy_panel()
    view, _ = build_next_period(panel)
    core = {"year", "city", "drug", "drug_key", "y_last", "y1"}
    for col in view.columns:
        if col in core or col in TARGET_COLS:
            continue
        assert "t1" not in col, f"feature {col} leaks the t+1 target"


def test_encoder_fit_mask_limits_categories():
    panel = _toy_panel()
    view, spec = build_next_period(panel, fit_mask=panel["year"] <= 2014)
    cats = {c for grp in spec for c in grp["categories"]}
    assert "metformin" in cats and "insulin" in cats  # both in fit rows
    full_spec = fit_encoder_spec(panel, None)
    assert {c for grp in full_spec for c in grp["categories"]} >= cats


def test_encoder_unknown_category_is_zero():
    panel = _toy_panel()
    spec = fit_encoder_spec(panel, panel["year"] <= 2014)
    unknown = panel[panel["year"] == 2016].copy()
    unknown["ingredient"] = "never_seen_ingredient"
    encoded = apply_encoder(unknown, spec)
    assert "ingredient::metformin" in encoded.columns
    ingredient_cols = [c for c in encoded.columns if c.startswith("ingredient::")]
    assert (encoded[ingredient_cols] == 0).all().all()


def _mode_view():
    """Synthetic view covering every feature disposition."""
    cols = [
        "y_last_log", "demand_claims_lag1_log",          # history (derived)
        "ingredient::metformin", "labeler::a_s_medication_solut",  # static
        "shortage_active", "shortage_events", "news_mentions",     # near-real-time
        "ar_ili_mean", "ww_covid_ar",                              # near-real-time
        "medicaid_rx_annual", "n_provider_types",                  # periodic
        "demand_benes", "cost_per_fill",                           # periodic
        "interaction::ar_ili_mean::ingredient_metformin",          # derived
    ]
    return pd.DataFrame({c: [0.0] for c in cols})


def test_evaluate_next_period_records_feature_mode():
    panel = _toy_panel()
    for mode in ("operational", "full"):
        result = evaluate_next_period(panel, neural=False, feature_mode=mode)
        assert result["feature_mode"] == mode
    with pytest.raises(ValueError):
        evaluate_next_period(panel, neural=False, feature_mode="bogus")


def test_feature_columns_for_mode_operational_excludes_periodic():
    view = _mode_view()
    op = feature_columns_for_mode(view, "operational")
    for excluded in ("medicaid_rx_annual", "n_provider_types",
                     "demand_benes", "cost_per_fill"):
        assert excluded not in op, excluded
    for kept in ("shortage_active", "shortage_events", "news_mentions",
                 "ar_ili_mean", "ww_covid_ar", "y_last_log",
                 "ingredient::metformin", "labeler::a_s_medication_solut",
                 "interaction::ar_ili_mean::ingredient_metformin"):
        assert kept in op, kept
    assert all(disposition_for_variable(c) != PERIODIC_TRAINING_ONLY for c in op)


def test_feature_columns_for_mode_monkeypatched_ablation(monkeypatch):
    cols = ["shortage_active", "cms_fills", "medicaid_rx_annual",
            "n_provider_types", "y_last_log", "ingredient::drug",
            "interaction::shortage_active::ingredient_drug"]
    view = pd.DataFrame({c: [0.0] for c in cols})
    monkeypatch.setattr(datasets_module, "ablation_features",
                        lambda v: {"full": cols})
    op = feature_columns_for_mode(view, "operational")
    assert set(op) == {"shortage_active", "y_last_log",
                       "ingredient::drug",
                       "interaction::shortage_active::ingredient_drug"}
    assert feature_columns_for_mode(view, "full") == cols
    with pytest.raises(ValueError):
        feature_columns_for_mode(view, "bogus")


def test_feature_columns_for_mode_full_equals_ablation_full():
    view = _mode_view()
    assert feature_columns_for_mode(view, "full") == ablation_features(view)["full"]
    assert set(feature_columns_for_mode(view, "operational")) <= set(
        feature_columns_for_mode(view, "full"))


def test_feature_columns_for_mode_invalid_mode_errors():
    view = _mode_view()
    try:
        feature_columns_for_mode(view, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")


def test_operational_mode_contract_on_built_view():
    panel_path = Path(__file__).resolve().parents[1] / "artifacts" / "panel" / "panel.csv"
    if not panel_path.exists():
        return
    view, _ = build_next_period(pd.read_csv(panel_path))
    op = feature_columns_for_mode(view, "operational")
    full = feature_columns_for_mode(view, "full")
    assert set(op) <= set(full)
    assert op, "operational mode must keep at least the history layer"
    assert all(disposition_for_variable(c) != PERIODIC_TRAINING_ONLY for c in op)


def test_pharmacy_access_features_are_real_city_year_aggregates():
    root = Path(__file__).resolve().parents[2]
    cfg = Config(root=str(root))
    access = build_pharmacy_access_features(cfg)
    assert {"year", "city_key", "pharmacy_facility_count"} <= set(access.columns)
    assert len(access) > 0
    assert (access["pharmacy_facility_count"] > 0).all()


def test_leaderboard_artifact_schema():
    lb = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "leaderboard.csv"
    if not lb.exists():
        return
    df = pd.read_csv(lb)
    required = {"model", "family", "subset", "wape", "mae", "rmse", "smape",
                "r2", "dir_acc", "params",
                "improvement_vs_best_naive", "improvement_vs_demand_only_ridge"}
    assert required <= set(df.columns)
    assert len(df) >= 8  # 6 baselines + at least 2 model families


def test_leaderboard_has_residual_models():
    lb = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "leaderboard.csv"
    if not lb.exists():
        return
    df = pd.read_csv(lb)
    required = {"residual_ridge_history", "residual_ridge_external",
                "residual_ridge_full", "residual_neural", "residual_ensemble"}
    assert required <= set(df["model"])


def test_leaderboard_has_medicaid_ablation():
    lb = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "leaderboard.csv"
    if not lb.exists():
        return
    df = pd.read_csv(lb)
    assert "ridge_medicaid" in set(df["model"])
    assert "ridge_medicaid_exact" in set(df["model"])
    assert "ridge_medicaid_bridge" in set(df["model"])


def test_calibrated_blend_records_medicaid_variant():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "metrics.json"
    if not path.exists():
        return
    blend = json.loads(path.read_text()).get("calibrated_blend", {})
    if not blend:
        return
    assert blend.get("selected_medicaid_variant") in {"all", "none", "exact", "bridge"}
    assert isinstance(blend.get("candidate_validation"), list)
    assert blend["candidate_validation"]


def test_layer_uplift_artifact_schema():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "layer_uplift.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    required = {"model", "wape", "wape_delta_vs_city_drug_last",
                "wape_delta_vs_demand_only_ridge"}
    assert required <= set(df.columns)
    row = df[df["model"] == "city_drug_last"]
    assert not row.empty
    assert abs(row.iloc[0]["wape_delta_vs_city_drug_last"]) < 1e-9


def test_shortage_risk_artifact_schema():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "metrics.json"
    if not path.exists():
        return
    risk = json.loads(path.read_text()).get("shortage_risk", {})
    required = {
        "label", "family", "selected_model", "selected_family",
        "best_model_by_test", "best_model_by_test_is_diagnostic",
        "leaderboard", "publishable_candidate", "auroc", "auprc",
        "brier", "topk_recall", "precision_at_k", "lift_at_k",
        "positives_captured", "k", "label_rate_validation",
        "label_rate_test", "n_params",
    }
    assert required <= set(risk)
    assert isinstance(risk["leaderboard"], list)
    assert risk["leaderboard"]
    row_required = {
        "model", "family", "validation_topk_recall",
        "validation_auprc", "validation_brier", "test_topk_recall",
        "test_precision_at_k", "test_lift_at_k",
        "test_positives_captured", "test_k",
    }
    assert row_required <= set(risk["leaderboard"][0])
    assert risk["best_model_by_test_is_diagnostic"] is True


def test_rolling_shortage_risk_artifact_schema():
    metrics = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "risk_rolling_metrics.json"
    folds = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "risk_rolling.csv"
    if not metrics.exists() or not folds.exists():
        return
    data = json.loads(metrics.read_text())
    required = {
        "split", "risk_label", "n_folds", "mean_topk_recall",
        "median_topk_recall", "min_topk_recall", "mean_precision_at_k",
        "median_lift_at_k", "total_positives_captured", "total_k",
        "folds_beating_label_rate", "publishable_rolling_candidate", "folds",
    }
    assert required <= set(data)
    df = pd.read_csv(folds)
    fold_required = {
        "cutoff_year", "validation_year", "test_feature_year",
        "test_target_year", "selected_model", "best_model_by_test",
        "test_topk_recall", "test_precision_at_k", "test_lift_at_k",
        "test_positives_captured", "test_k", "label_rate_test",
        "test_best_topk_recall", "test_best_precision_at_k",
        "test_best_lift_at_k", "test_best_positives_captured",
        "best_model_by_test_is_diagnostic",
    }
    assert fold_required <= set(df.columns)
    assert (df["validation_year"] < df["test_feature_year"]).all()
    assert (df["test_feature_year"] < df["test_target_year"]).all()


def test_evaluate_next_period_accepts_both_feature_modes():
    panel = _toy_annual_panel()
    for mode in ("operational", "full"):
        result = evaluate_next_period(panel, train_cutoff=2015, neural=False,
                                      feature_mode=mode)
        assert result["feature_mode"] == mode
        assert "error" not in result
        assert result["split"] == "strict_next_period_time"
        assert len(result["leaderboard"]) >= 6  # baselines present


def test_evaluate_next_period_invalid_feature_mode_errors():
    with pytest.raises(ValueError):
        evaluate_next_period(_toy_annual_panel(), train_cutoff=2015,
                             neural=False, feature_mode="bogus")


def test_panel_has_no_synthetic_demand():
    panel = Path(__file__).resolve().parents[1] / "artifacts" / "panel" / "panel.csv"
    if not panel.exists():
        return
    df = pd.read_csv(panel, usecols=["year", "demand_claims", "demand_fills",
                                     "demand_cost"])
    assert (df[["demand_claims", "demand_fills", "demand_cost"]].notna()).all().all()
    assert (df["demand_claims"] >= 0).all()


if __name__ == "__main__":
    test_next_period_target_is_year_plus_one()
    test_no_same_year_target_leakage()
    test_encoder_fit_mask_limits_categories()
    test_encoder_unknown_category_is_zero()
    test_evaluate_next_period_records_feature_mode()
    test_feature_columns_for_mode_operational_excludes_periodic()
    test_feature_columns_for_mode_full_equals_ablation_full()
    test_feature_columns_for_mode_invalid_mode_errors()
    test_operational_mode_contract_on_built_view()
    test_leaderboard_artifact_schema()
    test_leaderboard_has_residual_models()
    test_leaderboard_has_medicaid_ablation()
    test_calibrated_blend_records_medicaid_variant()
    test_layer_uplift_artifact_schema()
    test_shortage_risk_artifact_schema()
    test_rolling_shortage_risk_artifact_schema()
    test_evaluate_next_period_accepts_both_feature_modes()
    test_evaluate_next_period_invalid_feature_mode_errors()
    test_panel_has_no_synthetic_demand()
    print("ok")
