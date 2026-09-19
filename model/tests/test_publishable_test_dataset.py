import pandas as pd

from arkansas_pharma_signal.publishable_test_dataset import (
    _context_columns,
    _ndc9,
    build_next_quarter_test_panel,
    build_supplier_month_test_panel,
)
from arkansas_pharma_signal.arcos_evaluation import build_next_quarter_view


def test_ndc9_preserves_canonical_nine_digit_keys():
    assert _ndc9("000021433") == "000021433"


def test_context_join_key_is_strictly_next_quarter():
    source = pd.DataFrame([
        {"year": 2020, "quarter": 1, "signal": 7.0},
        {"year": 2020, "quarter": 2, "signal": 9.0},
    ])
    out = _context_columns(source, "prior_")
    assert out["context_qid"].tolist() == [33, 34]
    assert out["prior_signal"].tolist() == [7.0, 9.0]


def test_next_quarter_view_rejects_gaps_and_shifts_targets():
    panel = pd.DataFrame([
        {"ndc9": "000000001", "year": 2020, "quarter": 1, "qid": 32,
         "medicaid_prescriptions": 10, "fda_shortage_active": 0,
         "fda_shortage_supplier_count": 0, "fda_archive_row_observed": 0,
         "shortage_right_censored": 0, "prior_context_missing": 0,
         "prior_ar_ili_mean": 1.0},
        {"ndc9": "000000001", "year": 2020, "quarter": 2, "qid": 33,
         "medicaid_prescriptions": 12, "fda_shortage_active": 1,
         "fda_shortage_supplier_count": 1, "fda_archive_row_observed": 1,
         "shortage_right_censored": 0, "prior_context_missing": 0,
         "prior_ar_ili_mean": 2.0},
        {"ndc9": "000000001", "year": 2021, "quarter": 1, "qid": 36,
         "medicaid_prescriptions": 15, "fda_shortage_active": 0,
         "fda_shortage_supplier_count": 0, "fda_archive_row_observed": 0,
         "shortage_right_censored": 0, "prior_context_missing": 0,
         "prior_ar_ili_mean": 3.0},
    ])
    out = build_next_quarter_test_panel(panel)
    assert len(out) == 1
    assert out.iloc[0]["target_next_shortage_active"] == 1
    assert pd.isna(out.iloc[0]["feature_lag1_medicaid_prescriptions"])


def test_supplier_month_view_preserves_censoring_and_exact_months():
    panel = pd.DataFrame([
        {"ndc9": "000000001", "supplier": "A", "generic_name": "Drug",
         "month": "2021-01", "shortage_active": 1, "right_censored": 0,
         "resolution_observed": 0, "first_posting_month": "2021-01",
         "last_capture_month": "2022-01"},
        {"ndc9": "000000001", "supplier": "A", "generic_name": "Drug",
         "month": "2021-02", "shortage_active": 1, "right_censored": 1,
         "resolution_observed": 0, "first_posting_month": "2021-01",
         "last_capture_month": "2022-01"},
        {"ndc9": "000000001", "supplier": "A", "generic_name": "Drug",
         "month": "2021-04", "shortage_active": 0, "right_censored": 1,
         "resolution_observed": 0, "first_posting_month": "2021-01",
         "last_capture_month": "2022-01"},
    ])
    out = build_supplier_month_test_panel(panel)
    assert len(out) == 1
    assert out.iloc[0]["target_month"] == "2021-02"
    assert out.iloc[0]["target_next_right_censored"] == 1


def test_arcos_view_preserves_exact_next_quarter_target():
    panel = pd.DataFrame([
        {"year": 2020, "quarter": 1, "period": "2020-Q1",
         "period_index": 8080, "drug_code": "1", "drug_name": "x",
         "zip3": "001", "grams": 10.0, "log_grams": 2.4},
        {"year": 2020, "quarter": 2, "period": "2020-Q2",
         "period_index": 8081, "drug_code": "1", "drug_name": "x",
         "zip3": "001", "grams": 12.0, "log_grams": 2.6},
        {"year": 2022, "quarter": 1, "period": "2022-Q1",
         "period_index": 8088, "drug_code": "1", "drug_name": "x",
         "zip3": "001", "grams": 20.0, "log_grams": 3.0},
    ])
    out = build_next_quarter_view(panel)
    assert len(out) == 1
    assert out.iloc[0]["target"] == 12.0
    assert out.iloc[0]["target_period_index"] == 8081
