"""Strict next-quarter view, no-leakage, and quarterly artifact tests.

Logic tests use tiny deterministic frames (fast); artifact tests read
prebuilt model/artifacts/evaluation/quarterly_*.{csv,json} only when present.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from arkansas_pharma_signal.quarterly import (
    PUBLISHABLE_THRESHOLD,
    FLUVIEW_DEFAULT,
    WASTEWATER_DEFAULT,
    add_real_exogenous_layers,
    add_news_event_layers,
    add_nadac_price_layer,
    add_arcos_distribution_layer,
    add_weekly_surveillance_layers,
    build_quarterly_forecast_grid,
    build_quarterly_view,
    demand_state_metrics,
    evaluate_quarterly,
    evaluate_quarterly_rolling,
    _select_quarterly_transition_model_rolling,
    quarterly_fold_cutoffs,
    _normalize_ndc_identifier,
)

TARGET_COLS = {"target"}


def test_ndc_normalization_preserves_leading_zeroes():
    assert _normalize_ndc_identifier(116200116) == "00116200116"
    assert _normalize_ndc_identifier("00116-2001-16") == "00116200116"
    assert _normalize_ndc_identifier(None) == ""


def _toy_panel():
    rows = []
    for year in range(2019, 2023):
        for q in range(1, 5):
            for drug, base in (("a", 100), ("b", 50), ("c", 200), ("d", 25),
                               ("e", 300), ("f", 75), ("g", 150), ("h", 40)):
                rows.append({"year": year, "quarter": q, "drug": drug,
                             "value": base + (year - 2019) * 8 + q * 2})
    return pd.DataFrame(rows)


def _rolling_toy_panel():
    rows = []
    for year in range(2016, 2023):
        for q in range(1, 5):
            for i in range(40):
                rows.append({
                    "year": year, "quarter": q, "drug": f"drug_{i}",
                    "value": 50 + i + (year - 2016) * 3 + q,
                })
    return pd.DataFrame(rows)


def test_quarterly_target_is_next_quarter_real_value():
    panel = _toy_panel()
    view = build_quarterly_view(panel)
    assert set(view["target"]) <= set(panel["value"])


def test_quarterly_no_same_quarter_target_leakage():
    view = build_quarterly_view(_toy_panel())
    for col in view.columns:
        if col == "target":
            continue
        assert col != "target_t1", f"feature {col} leaks the next-quarter target"


def test_quarterly_view_next_step_is_value_shift():
    panel = _toy_panel()
    view = build_quarterly_view(panel)
    panel["q_id"] = (panel["year"] - 2019) * 4 + panel["quarter"] - 1
    fwd = panel[["drug", "q_id", "value"]].rename(columns={"value": "v_next"})
    fwd["q_id"] = fwd["q_id"] - 1
    view["q_id"] = (view["year"] - 2019) * 4 + view["quarter"] - 1
    chk = view.merge(fwd, on=["drug", "q_id"], how="left")
    assert chk["target"].notna().all()
    assert (chk["target"] == chk["v_next"]).all()


def test_quarterly_view_rejects_nonconsecutive_target_and_lag():
    panel = pd.DataFrame([
        {"year": 2020, "quarter": 1, "drug": "a", "value": 10},
        {"year": 2020, "quarter": 3, "drug": "a", "value": 30},
        {"year": 2020, "quarter": 4, "drug": "a", "value": 40},
    ])
    view = build_quarterly_view(panel)
    assert len(view) == 1
    row = view.iloc[0]
    assert (row["year"], row["quarter"]) == (2020, 3)
    assert pd.isna(row["value_last"])
    assert row["target"] == 40


def test_nadac_layer_preserves_ndc_and_computes_lagged_price_change(tmp_path):
    view = pd.DataFrame([
        {"year": 2020, "quarter": 1, "drug": "a", "ndc": "00116200116"},
        {"year": 2020, "quarter": 2, "drug": "a", "ndc": "00116200116"},
    ])
    source = tmp_path / "nadac.csv"
    pd.DataFrame([
        {"ndc": "116200116", "nadac_per_unit": 1.0, "as_of_date": "2020-02-01"},
        {"ndc": "00116200116", "nadac_per_unit": 1.2, "as_of_date": "2020-05-01"},
    ]).to_csv(source, index=False)
    out, cols = add_nadac_price_layer(view, source)
    assert "exo_nadac_price_qoq" in cols
    assert out["exo_nadac_price_mean"].notna().all()
    assert out.iloc[1]["exo_nadac_price_qoq"] == pytest.approx(0.2)


def test_quarterly_previous_quarter_is_value_at_t():
    view = build_quarterly_view(_toy_panel())
    panel = _toy_panel()
    panel["q_id"] = (panel["year"] - 2019) * 4 + panel["quarter"] - 1
    cur = panel[["drug", "q_id", "value"]].rename(columns={"value": "v_cur"})
    view["q_id"] = (view["year"] - 2019) * 4 + view["quarter"] - 1
    chk = view.merge(cur, on=["drug", "q_id"], how="left")
    assert (chk["value"] == chk["v_cur"]).all()
    # the previous-quarter baseline equals the target's preceding quarter
    assert (chk["value"] != chk["target"]).any()


def test_quarterly_eval_baseline_present():
    res = evaluate_quarterly(_toy_panel())
    lb = res["leaderboard"]
    assert {"previous_quarter", "ma2", "drug_mean", "global_mean",
            "drug_quarter_transition", "drug_quarter_transition_blend",
            "multi_transition_blend",
            "ma2_transition_blend", "ma2_multi_transition_blend",
            "ridge_history", "calibrated_ridge_blend"} <= set(lb["model"])
    prev = lb[lb["model"] == "previous_quarter"].iloc[0]
    assert abs(prev["improvement_vs_previous_quarter"]) < 1e-9
    assert res["publishability_threshold"] == PUBLISHABLE_THRESHOLD
    assert "selected_model" in res
    assert "validation_wape" in res["selected_model"]
    assert "best_naive_validation_wape" in res


def test_demand_state_diagnostic_uses_fixed_material_change_band():
    metrics = demand_state_metrics(
        y_true=[80.0, 100.0, 130.0],
        y_pred=[80.0, 110.0, 125.0],
        current=[100.0, 100.0, 100.0],
        band=0.20,
    )
    assert metrics["demand_state_accuracy"] == pytest.approx(1.0)
    assert metrics["demand_state_balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["demand_state_macro_f1"] == pytest.approx(1.0)
    assert metrics["demand_state_decrease_rows"] == 0
    assert metrics["demand_state_stable_rows"] == 2
    assert metrics["demand_state_increase_rows"] == 1
    with pytest.raises(ValueError):
        demand_state_metrics([1.0], [1.0], [1.0], band=1.0)


def test_quarterly_forecast_grid_contract():
    grid = build_quarterly_forecast_grid(_toy_panel(), max_rows=12)
    assert len(grid) == 12
    assert set(grid["target"]) == {"medicaid_prescription_count"}
    assert set(grid["geography_level"]) == {"state"}
    assert set(grid["geography_id"]) == {"AR"}
    assert grid["prediction"].notna().all()
    assert (grid["prediction"] >= 0).all()
    assert set(grid["model_family"]) == {"quarterly_medicaid_selected_transition"}


def test_quarterly_forecast_grid_excludes_stale_drug_histories():
    panel = _toy_panel()
    panel = panel[~((panel["drug"] == "a") & (panel["year"] == 2022))]
    grid = build_quarterly_forecast_grid(panel, max_rows=100)
    assert "a" not in set(grid["drug_key"])
    assert set(grid["forecast_date"]) == {"2023-01-01"}


def test_quarterly_forecast_selector_aggregates_chronological_validation_years():
    panel = _rolling_toy_panel()
    view = build_quarterly_view(panel)
    score = panel.sort_values(["drug", "year", "quarter"]).groupby(
        "drug", as_index=False).tail(1).copy()
    score["value_last"] = score["value"]
    score["ma2"] = score["value"]
    selected = _select_quarterly_transition_model_rolling(view, score)
    assert selected["selection_method"] == "rolling_validation_mean_wape"
    assert selected["validation_years"] == [2020, 2021, 2022]
    assert len(selected["rolling_validation_folds"]) == 3
    train_rows = [fold["train_rows"] for fold in selected["rolling_validation_folds"]]
    assert train_rows == sorted(train_rows)


def test_quarterly_rolling_folds_are_chronological():
    panel = _toy_panel()
    cutoffs = quarterly_fold_cutoffs(panel, min_train_years=2)
    assert cutoffs == [2021]
    rolling = evaluate_quarterly_rolling(panel, min_train_years=2)
    folds = rolling["folds"]
    assert len(folds) == 1
    row = folds.iloc[0]
    assert row["train_years"] == "<= 2020"
    assert row["validation_year"] == "2021"
    assert row["test_years"] == "> 2021"
    assert row["train_rows"] > 0
    assert row["validation_rows"] > 0
    assert row["test_rows"] > 0
    assert "publishable_rolling_candidate" in rolling
    assert "persistence_demand_state_balanced_accuracy" in folds.columns
    assert "mean_state_balanced_improvement_vs_persistence" in rolling
    assert rolling["requested_accuracy_threshold"] == pytest.approx(0.75)
    assert "requested_accuracy_goal_met" in rolling
    assert "promotion_candidate" in rolling


def test_quarterly_rolling_window_limits_test_years():
    panel = _toy_panel()
    rolling = evaluate_quarterly_rolling(
        panel, min_train_years=1, test_window_years=1)
    folds = rolling["folds"]
    assert rolling["test_window_years"] == 1
    assert not folds.empty
    assert "selected_model" in folds.columns
    assert "best_model_by_test" in folds.columns
    assert "selected_improvement_vs_best_naive" in folds.columns
    assert "selected_demand_state_balanced_accuracy" in folds.columns
    for _, row in folds.iterrows():
        val_year = int(row["validation_year"])
        assert row["test_years"] == f"{val_year + 1}-{val_year + 1}"


def test_quarterly_baseline_selection_is_validation_only():
    res = evaluate_quarterly(_toy_panel(), train_cutoff=2021)
    lb = res["leaderboard"]
    validation = lb[lb["model"].isin(
        ["previous_quarter", "ma2", "drug_mean", "global_mean"]
    )].set_index("model")["validation_wape"]
    assert res["best_naive"]["model"] == validation.idxmin()
    assert res["best_naive_validation_wape"] == validation.min()


def test_quarterly_seasonal_naive_is_compared():
    res = evaluate_quarterly(_toy_panel(), train_cutoff=2021)
    assert "same_target_quarter_last_year" in set(res["leaderboard"]["model"])


def test_exogenous_annual_layers_use_previous_completed_year():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        annual = Path(td) / "panel.csv"
        pd.DataFrame([
            {"year": 2020, "drug_key": "a", "demand_claims": 111,
             "ar_ili_mean": 2.0},
            {"year": 2021, "drug_key": "a", "demand_claims": 999,
             "ar_ili_mean": 9.0},
        ]).to_csv(annual, index=False)
        layered, cols = add_real_exogenous_layers(view, annual_panel_path=annual)
    assert "exo_annual_drug_demand_claims" in cols
    row = layered[(layered["drug"] == "a") & (layered["year"] == 2021)].iloc[0]
    assert row["annual_feature_year"] == 2020
    assert row["exo_annual_drug_demand_claims"] == 111
    assert row["exo_annual_drug_ar_ili_mean"] == 2.0


def test_exogenous_events_join_feature_quarter_only():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        events = Path(td) / "events.csv.gz"
        pd.DataFrame([
            {"event_type": "drug_shortage", "start_time": "2021-04-05",
             "severity": 0.7, "confidence": 1.0},
            {"event_type": "recall", "start_time": "2021-07-05",
             "severity": 0.2, "confidence": 0.5},
        ]).to_csv(events, index=False, compression="gzip")
        layered, cols = add_real_exogenous_layers(view, events_path=events)
    assert "exo_event_count" in cols
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    q3 = layered[(layered["year"] == 2021) & (layered["quarter"] == 3)]
    assert set(q2["exo_event_shortage_count"].dropna()) == {1.0}
    assert set(q3["exo_event_recall_count"].dropna()) == {1.0}


def test_news_events_join_feature_quarter_only():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        events = Path(td) / "article_events.csv.gz"
        pd.DataFrame([
            {"event_type": "shortage", "source_timestamp": "2021-04-05T00:00:00Z",
             "severity": 0.7, "confidence": 1.0},
            {"event_type": "disease_outbreak", "source_timestamp": "2021-07-05T00:00:00Z",
             "severity": 0.4, "confidence": 0.5},
        ]).to_csv(events, index=False, compression="gzip")
        layered, cols = add_news_event_layers(view, events)
    assert "exo_news_shortage_count" in cols
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    q3 = layered[(layered["year"] == 2021) & (layered["quarter"] == 3)]
    assert set(q2["exo_news_shortage_count"].dropna()) == {1.0}
    assert set(q3["exo_news_outbreak_count"].dropna()) == {1.0}


def test_arcos_layer_joins_feature_quarter_only():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        path = Path(td) / "arcos.csv"
        pd.DataFrame([
            {"year": 2020, "drug_name": "a", "quarter1_grams": "10",
             "quarter2_grams": "12", "quarter3_grams": "14", "quarter4_grams": "16"},
        ]).to_csv(path, index=False)
        layered, cols = add_arcos_distribution_layer(view, path)
    assert cols == ["arcos_grams_log", "arcos_grams_qoq"]
    row = layered[(layered["year"] == 2020) &
                  (layered["quarter"] == 2) &
                  (layered["drug"] == "a")].iloc[0]
    assert row["arcos_grams_log"] > 0
    assert row["arcos_grams_qoq"] == 0.2


def test_weekly_surveillance_joins_prior_quarter_only():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        flu = Path(td) / "flu.csv"
        # all observation weeks fall in 2021 Q1 (epiweeks 01-13; 202112 ends
        # 2021-03-28, fully inside Q1)
        pd.DataFrame([
            {"epiweek": 202101, "region": "ar", "num_ili": 10,
             "num_patients": 100, "wili": 1.5},
            {"epiweek": 202107, "region": "ar", "num_ili": 20,
             "num_patients": 200, "wili": 2.5},
            {"epiweek": 202112, "region": "ar", "num_ili": 30,
             "num_patients": 300, "wili": 3.5},
        ]).to_csv(flu, index=False)
        layered, cols = add_weekly_surveillance_layers(view, fluview_path=flu)
    assert "exo_weekly_flu_ar_num_ili" in cols
    assert "exo_weekly_flu_ar_weeks" in cols
    q1 = layered[(layered["year"] == 2021) & (layered["quarter"] == 1)]
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    q3 = layered[(layered["year"] == 2021) & (layered["quarter"] == 3)]
    # observation quarter itself and later quarters stay missing
    assert q1["exo_weekly_flu_ar_num_ili"].isna().all()
    assert q3["exo_weekly_flu_ar_num_ili"].isna().all()
    # only the immediately following feature quarter sees the summary
    assert set(q2["exo_weekly_flu_ar_num_ili"].dropna()) == {20.0}
    assert set(q2["exo_weekly_flu_ar_weeks"].dropna()) == {3}
    assert set(q2["exo_weekly_flu_ar_wili_last"].dropna()) == {3.5}
    assert set(q2["exo_weekly_flu_ar_wili_max"].dropna()) == {3.5}
    assert set(q2["exo_weekly_flu_ar_wili_delta"].dropna()) == {2.0}


def test_weekly_wastewater_layer_preserves_missingness_and_coverage():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        ww = Path(td) / "ww.csv"
        pd.DataFrame([
            {"week_end": "2021-02-01", "site": "s1", "site_wval": 2.0,
             "pathogen_target": "SARS-CoV-2"},
            {"week_end": "2021-02-08", "site": "s2", "site_wval": 4.0,
             "pathogen_target": "SARS-CoV-2"},
            {"week_end": "2021-02-15", "site": "s1", "site_wval": 6.0,
             "pathogen_target": "RSV"},
        ]).to_csv(ww, index=False)
        layered, cols = add_weekly_surveillance_layers(view, wastewater_path=ww)
    assert "exo_weekly_ww_wval_mean" in cols
    assert "exo_weekly_ww_sites" in cols
    assert "exo_weekly_ww_sars2_wval_mean" in cols
    q1 = layered[(layered["year"] == 2021) & (layered["quarter"] == 1)]
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    assert q1["exo_weekly_ww_wval_mean"].isna().all()
    assert set(q2["exo_weekly_ww_wval_mean"].dropna()) == {4.0}
    assert set(q2["exo_weekly_ww_sites"].dropna()) == {2}
    assert set(q2["exo_weekly_ww_sars2_wval_mean"].dropna()) == {3.0}


def test_fluview_revision_uses_latest_issue_only():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        flu = Path(td) / "flu.csv"
        # same (region, epiweek) republished under later issues
        pd.DataFrame([
            {"release_date": "2021-02-01", "region": "ar", "issue": 202105,
             "epiweek": 202101, "num_ili": 10, "num_patients": 100,
             "wili": 1.5},
            {"release_date": "2021-02-15", "region": "ar", "issue": 202107,
             "epiweek": 202101, "num_ili": 99, "num_patients": 900,
             "wili": 9.9},
        ]).to_csv(flu, index=False)
        layered, cols = add_weekly_surveillance_layers(view, fluview_path=flu)
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    # aggregate reflects only the latest issue, not the early snapshot
    assert set(q2["exo_weekly_flu_ar_num_ili"].dropna()) == {99.0}
    assert set(q2["exo_weekly_flu_ar_num_patients"].dropna()) == {900.0}
    assert set(q2["exo_weekly_flu_ar_weeks"].dropna()) == {1}


def test_fluview_week_crossing_quarter_boundary_joins_later_quarter():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        flu = Path(td) / "flu.csv"
        # 2021-W13: Monday 2021-03-29 (Q1), week end 2021-04-04 (Q2)
        pd.DataFrame([
            {"region": "ar", "epiweek": 202113, "num_ili": 30,
             "num_patients": 300, "wili": 3.5},
        ]).to_csv(flu, index=False)
        layered, cols = add_weekly_surveillance_layers(view, fluview_path=flu)
    q2 = layered[(layered["year"] == 2021) & (layered["quarter"] == 2)]
    q3 = layered[(layered["year"] == 2021) & (layered["quarter"] == 3)]
    # not complete before Q2 starts (week ends 2021-04-04 > 2021-04-01)
    assert q2["exo_weekly_flu_ar_num_ili"].isna().all()
    # joins the quarter after its completed week end instead
    assert set(q3["exo_weekly_flu_ar_num_ili"].dropna()) == {30.0}
    assert set(q3["exo_weekly_flu_ar_weeks"].dropna()) == {1}


def test_weekly_surveillance_missing_files_unchanged():
    view = build_quarterly_view(_toy_panel())
    layered, cols = add_weekly_surveillance_layers(
        view, fluview_path=Path("/nonexistent/flu.csv"),
        wastewater_path=Path("/nonexistent/ww.csv"))
    assert cols == []
    assert layered.equals(view)


def test_weekly_surveillance_bad_columns_graceful():
    view = build_quarterly_view(_toy_panel())
    with TemporaryDirectory() as td:
        bad = Path(td) / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(bad, index=False)
        layered, cols = add_weekly_surveillance_layers(view, fluview_path=bad)
    assert cols == []
    assert layered.equals(view)


def test_weekly_surveillance_repo_files_load():
    if not FLUVIEW_DEFAULT.exists() or not WASTEWATER_DEFAULT.exists():
        return
    view = build_quarterly_view(_toy_panel())
    layered, cols = add_weekly_surveillance_layers(
        view, fluview_path=FLUVIEW_DEFAULT, wastewater_path=WASTEWATER_DEFAULT)
    assert any(c.startswith("exo_weekly_") for c in cols)
    assert len(layered) == len(view)


def test_quarterly_leaderboard_artifact_schema():
    lb = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "quarterly_leaderboard.csv"
    if not lb.exists():
        return
    df = pd.read_csv(lb)
    required = {"model", "family", "wape", "mae", "rmse", "smape", "r2",
                "improvement_vs_previous_quarter"}
    assert required <= set(df.columns)
    assert {"previous_quarter", "ma2", "drug_mean", "global_mean",
            "drug_quarter_transition", "drug_quarter_transition_blend",
            "multi_transition_blend",
            "ma2_transition_blend", "ma2_multi_transition_blend",
            "ridge_history", "calibrated_ridge_blend"} <= set(df["model"])


def test_quarterly_metrics_artifact_schema():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "quarterly_metrics.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    for key in ("split", "best_naive", "best_model", "publishable_candidate",
                "publishability_threshold", "improvement_vs_previous_quarter",
                "publishability_reason", "selected_model"):
        assert key in d
    assert d["publishability_threshold"] == PUBLISHABLE_THRESHOLD
    assert d["best_naive"]["model"] in {"previous_quarter", "ma2",
                                        "drug_mean", "global_mean"}


if __name__ == "__main__":
    test_quarterly_target_is_next_quarter_real_value()
    test_quarterly_no_same_quarter_target_leakage()
    test_quarterly_view_next_step_is_value_shift()
    test_quarterly_previous_quarter_is_value_at_t()
    test_quarterly_eval_baseline_present()
    test_quarterly_forecast_grid_contract()
    test_quarterly_rolling_folds_are_chronological()
    test_quarterly_rolling_window_limits_test_years()
    test_exogenous_annual_layers_use_previous_completed_year()
    test_exogenous_events_join_feature_quarter_only()
    test_weekly_surveillance_joins_prior_quarter_only()
    test_fluview_revision_uses_latest_issue_only()
    test_fluview_week_crossing_quarter_boundary_joins_later_quarter()
    test_weekly_wastewater_layer_preserves_missingness_and_coverage()
    test_weekly_surveillance_missing_files_unchanged()
    test_weekly_surveillance_bad_columns_graceful()
    test_weekly_surveillance_repo_files_load()
    test_quarterly_leaderboard_artifact_schema()
    test_quarterly_metrics_artifact_schema()
    print("ok")
