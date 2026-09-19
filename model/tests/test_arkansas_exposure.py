import pandas as pd
import numpy as np

from arkansas_pharma_signal.arkansas_exposure import (
    build_next_quarter_exposure_view,
    evaluate_next_quarter_exposure_rolling,
)
from arkansas_pharma_signal.regression import LogisticRidge


def test_next_quarter_view_drops_nonconsecutive_periods_and_lags_history():
    panel = pd.DataFrame([
        {"ndc9": "a", "year": 2019, "quarter": 1,
         "medicaid_prescriptions": 10, "fda_shortage_active": 0},
        {"ndc9": "a", "year": 2019, "quarter": 2,
         "medicaid_prescriptions": 12, "fda_shortage_active": 1},
        {"ndc9": "a", "year": 2019, "quarter": 4,
         "medicaid_prescriptions": 14, "fda_shortage_active": 0},
        {"ndc9": "b", "year": 2019, "quarter": 1,
         "medicaid_prescriptions": 8, "fda_shortage_active": 0},
        {"ndc9": "b", "year": 2019, "quarter": 2,
         "medicaid_prescriptions": 9, "fda_shortage_active": 1},
    ])
    view = build_next_quarter_exposure_view(panel)
    assert len(view) == 2
    row = view[(view.ndc9 == "a") & (view.quarter == 1)].iloc[0]
    assert row["target"] == 1
    assert row["lag1_shortage_active"] == 0


def test_exposure_view_lags_supplier_and_archive_evidence():
    panel = pd.DataFrame([
        {"ndc9": "a", "year": 2019, "quarter": 1,
         "medicaid_prescriptions": 10, "fda_shortage_active": 0,
         "fda_shortage_supplier_count": 0, "fda_archive_row_observed": 0},
        {"ndc9": "a", "year": 2019, "quarter": 2,
         "medicaid_prescriptions": 12, "fda_shortage_active": 1,
         "fda_shortage_supplier_count": 2, "fda_archive_row_observed": 1},
        {"ndc9": "a", "year": 2019, "quarter": 3,
         "medicaid_prescriptions": 14, "fda_shortage_active": 1,
         "fda_shortage_supplier_count": 3, "fda_archive_row_observed": 1},
    ])
    view = build_next_quarter_exposure_view(panel)
    row = view[view["quarter"].eq(2)].iloc[0]
    assert row["current_supplier_count"] == 2
    assert row["lag1_supplier_count"] == 0
    assert row["current_archive_row_observed"] == 1
    assert row["lag1_archive_row_observed"] == 0


def test_onset_view_excludes_already_active_rows():
    panel = pd.DataFrame([
        {"ndc9": "a", "year": 2019, "quarter": 1,
         "medicaid_prescriptions": 10, "fda_shortage_active": 0},
        {"ndc9": "a", "year": 2019, "quarter": 2,
         "medicaid_prescriptions": 12, "fda_shortage_active": 1},
        {"ndc9": "b", "year": 2019, "quarter": 1,
         "medicaid_prescriptions": 8, "fda_shortage_active": 1},
        {"ndc9": "b", "year": 2019, "quarter": 2,
         "medicaid_prescriptions": 9, "fda_shortage_active": 1},
    ])
    view = build_next_quarter_exposure_view(panel, onset_only=True)
    assert len(view) == 1
    assert view.iloc[0]["target"] == 1


def test_logistic_ridge_accepts_class_weights_for_rare_events():
    model = LogisticRidge(alpha=1.0, max_iter=20)
    model.fit(
        np.array([[0.0], [0.1], [1.0], [1.1]]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        ["x"],
        sample_weight=np.array([1.0, 1.0, 4.0, 4.0]),
    )
    assert np.isfinite(model.predict_proba(np.array([[0.5]]))).all()


def test_exposure_rolling_uses_next_year_test_folds():
    rows = []
    for year in range(2012, 2020):
        for quarter in range(1, 5):
            for ndc, offset in (("a", 0), ("b", 1), ("c", 2)):
                rows.append({
                    "ndc9": ndc, "year": year, "quarter": quarter,
                    "medicaid_prescriptions": 10 + offset,
                    "fda_shortage_active": int((year + quarter + offset) % 3 == 0),
                })
    view = build_next_quarter_exposure_view(pd.DataFrame(rows))
    result = evaluate_next_quarter_exposure_rolling(view, min_train_years=3)
    assert result["fold_count"] >= 2
    assert all(fold["test_year"] == fold["validation_year"] + 1
               for fold in result["folds"])
    assert "requested_accuracy_goal_met" in result
    assert "skill_claim_supported" in result
