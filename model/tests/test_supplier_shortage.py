"""Focused tests for FDA supplier-drug shortage event evaluation."""

import numpy as np
import pandas as pd
import pytest

from arkansas_pharma_signal.supplier_shortage import (
    load_archived_fda_shortage_panel,
    FEATURE_COLS,
    build_monthly_panel,
    build_next_month_view,
    build_supplier_shortage_scores,
    evaluate_supplier_shortage_rolling_view,
    evaluate_archived_supplier_shortage,
    load_arkansas_exposed_supplier_panel,
    evaluate_supplier_shortage_view,
    normalize_drug,
    normalize_supplier,
    supplier_shortage_rolling_cutoffs,
    _feature_matrix,
    _feature_spec,
)


def test_load_archived_fda_shortage_panel_validates_and_maps_target(tmp_path):
    path = tmp_path / "archive.csv"
    pd.DataFrame([
        {"ndc9": "001234567", "supplier": "Acme", "month": "2020-01",
         "shortage_active": 1, "right_censored": 0},
        {"ndc9": "001234567", "supplier": "Acme", "month": "2020-02",
         "shortage_active": 0, "right_censored": 0},
    ]).to_csv(path, index=False)
    out = load_archived_fda_shortage_panel(path)
    assert list(out["drug"]) == ["001234567", "001234567"]
    assert list(out["shortage_event"]) == [1, 0]


def _synthetic_records(years=(2018, 2019, 2020, 2021, 2022), seed=0):
    rng = np.random.default_rng(seed)
    suppliers = ["Hospira, Inc.", "Accord Healthcare, Inc.",
                 "Baxter Healthcare Corporation"]
    drugs = ["Furosemide Injection", "Midazolam Hydrochloride",
             "Dexmedetomidine Hydrochloride"]
    rows = []
    for supplier in suppliers:
        for drug in drugs:
            for year in years:
                for month in range(1, 13):
                    if rng.random() < 0.3:
                        rows.append({
                            "supplier": supplier,
                            "drug": drug,
                            "event_date": pd.Timestamp(year, month, 5),
                            "month": f"{year}-{month:02d}",
                        })
    return pd.DataFrame(rows)


def _panel_and_view(records):
    panel = build_monthly_panel(records)
    return panel, build_next_month_view(panel)


def test_normalization_deterministic():
    assert normalize_supplier("Hospira, Inc.") == normalize_supplier(
        "  HOSPIRA inc.  ")
    assert normalize_supplier("Baxter Healthcare Corporation") == \
        normalize_supplier("baxter healthcare corporation")
    assert normalize_drug("Furosemide Injection") == \
        normalize_drug("furosemide-injection")
    assert normalize_supplier("Hospira, Inc.") != \
        normalize_supplier("Accord Healthcare, Inc.")


def test_panel_complete_and_grouped():
    records = _synthetic_records()
    panel, _ = _panel_and_view(records)
    expected_months = list(pd.period_range("2018-01", "2022-12", freq="M").strftime("%Y-%m"))
    assert sorted(panel["month"].unique()) == expected_months
    for (supplier, drug), sub in panel.groupby(["supplier", "drug"]):
        observed = set(records[(records["supplier"] == supplier)
                               & (records["drug"] == drug)]["month"])
        pair_start = min(observed)
        pair_months = list(pd.period_range(pair_start, "2022-12", freq="M").strftime("%Y-%m"))
        assert sorted(sub["month"]) == pair_months, "incomplete pair grid"
        assert set(sub.loc[sub["shortage_event"] == 1, "month"]) == observed


def test_panel_does_not_create_leading_unobserved_months():
    records = pd.DataFrame([
        {"supplier": "early", "drug": "drug", "month": "2018-01"},
        {"supplier": "late", "drug": "drug", "month": "2020-03"},
    ])
    panel = build_monthly_panel(records)
    late = panel[(panel["supplier"] == "late") & (panel["drug"] == "drug")]
    assert late["month"].min() == "2020-03"
    assert "2018-01" not in set(late["month"])


def test_no_next_month_leakage():
    """Features at t must never use the t+1 target or the t current event."""
    records = _synthetic_records()
    panel, view = _panel_and_view(records)
    assert "current_event" not in FEATURE_COLS
    for _, row in view.iterrows():
        pair = panel[(panel["supplier"] == row["supplier"])
                     & (panel["drug"] == row["drug"])]
        t = pd.Period(row["month"], freq="M")
        current = pair.loc[pair["month"] == str(t), "shortage_event"].iloc[0]
        future = pair.loc[pair["month"] == str(t + 1), "shortage_event"]
        assert not future.empty, "target month missing from complete panel"
        assert row["target"] == pytest.approx(future.iloc[0])
        assert row["target_right_censored"] == pytest.approx(0.0)
        assert row["current_event"] == pytest.approx(current)
        pair_start = pd.Period(pair["month"].min(), freq="M")
        if t - 1 >= pair_start:
            past = pair.loc[pair["month"] == str(t - 1), "shortage_event"].iloc[0]
            assert row["lag1_event"] == pytest.approx(past)
        # The last observed month has no t+1 label and must be dropped.
    assert view["month"].max() == "2022-11"


def test_drug_global_context_is_prior_only():
    records = pd.DataFrame([
        {"supplier": "a", "drug": "drug", "month": "2020-01"},
        {"supplier": "b", "drug": "drug", "month": "2020-02"},
        {"supplier": "a", "drug": "drug", "month": "2020-03"},
    ])
    _, view = _panel_and_view(records)
    row = view[(view["supplier"] == "a") & (view["month"] == "2020-02")].iloc[0]
    assert row["drug_global_lag1_event"] == pytest.approx(1.0)
    # The event in the current month is never reflected in the drug-global
    # lagged features for the same month.
    row = view[(view["supplier"] == "b") & (view["month"] == "2020-02")].iloc[0]
    assert row["drug_global_lag1_event"] == pytest.approx(1.0)


def test_chronological_split_and_counts():
    records = _synthetic_records()
    panel, view = _panel_and_view(records)
    result = evaluate_supplier_shortage_view(view, panel)
    assert result["split"] == "strict_next_month_supplier_drug_time"
    assert set(result["candidates"]) == {
        "previous_month_persistence", "logistic_ridge"}
    assert result["selected_model"] == "logistic_ridge"
    assert result["censoring"]["policy"] == "exclude_right_censored_targets"
    assert result["n_rows"]["train"] > 0
    assert result["n_rows"]["validation"] > 0
    assert result["n_rows"]["test"] > 0
    assert result["n_pairs"] == len(panel[["supplier", "drug"]].drop_duplicates())
    ldf = result["leaderboard"]
    assert set(ldf["model"]) == {"previous_month_persistence", "logistic_ridge"}
    for _, row in ldf.iterrows():
        assert 0.0 <= row["test_auroc"] <= 1.0
        assert 0.0 <= row["test_brier"] <= 1.0
    selected = ldf[ldf["selected"]].iloc[0]
    assert selected["model"] == "logistic_ridge"
    assert selected["validation_f1"] >= 0.0


def test_right_censored_targets_are_excluded_from_scoring():
    records = _synthetic_records()
    panel = build_monthly_panel(records)
    panel.loc[panel["month"].eq("2022-05"), "right_censored"] = 1
    view = build_next_month_view(panel)
    result = evaluate_supplier_shortage_view(
        view, panel, train_cutoff=2020, val_year=2021, test_year=2022)
    assert result["censoring"]["excluded_target_rows"] > 0
    assert result["censoring"]["scored_target_rows"] < len(view)


def test_rolling_folds_at_least_three():
    records = _synthetic_records()
    _, view = _panel_and_view(records)
    months = sorted(view["month"].unique())
    cutoffs = supplier_shortage_rolling_cutoffs(months, min_train_months=24)
    assert len(cutoffs) >= 3
    rolling = evaluate_supplier_shortage_rolling_view(view, min_train_months=24)
    assert rolling["fold_count"] >= 3
    assert rolling["split"] == "rolling_origin_next_month_supplier_drug"
    assert "auroc" in rolling["logistic_ridge_mean"]


def test_rolling_folds_no_leakage():
    records = _synthetic_records()
    _, view = _panel_and_view(records)
    months = sorted(view["month"].unique())
    rolling = evaluate_supplier_shortage_rolling_view(view, min_train_months=24)
    for fold in rolling["folds"].to_dict("records"):
        assert fold["test_rows"] > 0
        assert fold["validation_rows"] > 0
        assert fold["validation_months"][1] < fold["test_months"][0]
        assert fold["fit_train_months"][1] < fold["validation_months"][0]
        train_end = months.index(fold["train_months"][1]) + 1
        assert fold["test_months"][0] == months[train_end]
        assert fold["test_positives"] > 0


def test_future_identity_is_unknown_to_training_feature_spec():
    train = pd.DataFrame({
        "supplier": ["known"], "drug": ["drug"],
        **{col: [0.0] for col in FEATURE_COLS},
    })
    test = train.copy()
    test["supplier"] = "future-only"
    spec = _feature_spec(train)
    matrix = _feature_matrix(test, spec)
    assert "supplier_future-only" not in spec
    assert matrix.shape[1] == len(spec)
    supplier_width = sum(c.startswith("supplier_") for c in spec)
    assert np.all(matrix[0, len(FEATURE_COLS):len(FEATURE_COLS) + supplier_width] == 0.0)


def test_latest_supplier_scores_are_bounded_and_use_latest_feature_month():
    records = _synthetic_records()
    scores = build_supplier_shortage_scores(records)
    assert len(scores) == len(records[["supplier", "drug"]].drop_duplicates())
    assert scores["feature_month"].eq("2022-12").all()
    assert scores["shortage_probability"].between(0.0, 1.0).all()
    assert scores["evidence_type"].eq("fda_reported_supplier_drug_event_model").all()


def test_archived_evaluator_returns_censor_safe_protocol(tmp_path):
    records = _synthetic_records()
    panel = build_monthly_panel(records)
    path = tmp_path / "archive.csv"
    panel.rename(columns={"drug": "ndc9", "shortage_event": "shortage_active"}).assign(
                 generic_name="Drug",
                 resolution_observed=0, first_posting_month=panel["month"].min(),
                 last_capture_month=panel["month"].max()).to_csv(path, index=False)
    result = evaluate_archived_supplier_shortage(
        path, train_cutoff=2020, validation_year=2021, test_year=2022,
        min_train_months=24)
    assert result["protocol"] == "public_fda_archive_supplier_drug_next_month_v1"
    assert result["rolling"]["censoring_policy"] == "exclude_right_censored_targets"


def test_arkansas_exposure_filter_normalizes_ndc9(tmp_path):
    records = _synthetic_records()
    panel = build_monthly_panel(records)
    panel["drug"] = "1"
    archive = tmp_path / "archive.csv"
    (panel.rename(columns={"drug": "ndc9", "shortage_event": "shortage_active"})
     .assign(generic_name="Drug", resolution_observed=0,
             first_posting_month=panel["month"].min(),
             last_capture_month=panel["month"].max())
     .to_csv(archive, index=False))
    exposure = tmp_path / "exposure.csv"
    pd.DataFrame({"ndc9": ["1"]}).to_csv(exposure, index=False)
    out = load_arkansas_exposed_supplier_panel(archive, exposure)
    assert not out.empty
    assert out["drug"].eq("000000001").all()


if __name__ == "__main__":
    test_normalization_deterministic()
    test_panel_complete_and_grouped()
    test_no_next_month_leakage()
    test_chronological_split_and_counts()
    test_rolling_folds_at_least_three()
    test_rolling_folds_no_leakage()
    print("ok")
