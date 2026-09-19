import numpy as np
import pandas as pd

from arkansas_pharma_signal.shortage_pressure import (
    build_next_month_pressure_view, build_shortage_breadth_panel,
    build_next_month_numeric_supplier_count_view, build_pressure_panel,
    evaluate_numeric_supplier_count_rolling, evaluate_shortage_breadth_rolling,
    evaluate_pressure_rolling,
)


def test_pressure_panel_emits_five_state_contract():
    rows = []
    for month, active in [("2024-01", []), ("2024-02", ["a"]),
                          ("2024-03", ["a", "b"])]:
        for supplier in ["a", "b"]:
            rows.append({"ndc9": "1", "supplier": supplier, "month": month,
                         "shortage_active": int(supplier in active)})
    panel = build_pressure_panel(pd.DataFrame(rows))
    assert panel["pressure_state"].tolist() == [0, 1, 2]
    view = build_next_month_pressure_view(panel)
    assert view["target_pressure_state"].tolist() == [1, 2]
    assert set(view["target_pressure_state"]) == {1, 2}


def test_pressure_rolling_is_leakage_safe_and_has_state_metrics():
    rows = []
    for drug in ["1", "2", "3"]:
        for month_index in range(42):
            month = pd.Period("2020-01", freq="M") + month_index
            active = (month_index + int(drug)) % 3
            for supplier in ["a", "b"]:
                rows.append({"ndc9": drug, "supplier": supplier,
                             "month": str(month),
                             "shortage_active": int(active == 2 or (active == 1 and supplier == "a"))})
    view = build_next_month_pressure_view(build_pressure_panel(pd.DataFrame(rows)))
    result = evaluate_pressure_rolling(view, min_train_months=12,
                                       validation_months=2, test_months=2)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert 0.0 <= result["mean_model_accuracy"] <= 1.0
    assert set(result["state_definition"]) == {0, 1, 2, 3, 4}
    assert result["per_drug_count_with_minimum_rows"] == 3
    assert set(result["per_drug_accuracy"]) == {"1", "2", "3"}


def test_numeric_supplier_count_rolling_reports_numeric_metrics():
    rows = []
    for drug in ["1", "2", "3"]:
        for month_index in range(42):
            month = pd.Period("2020-01", freq="M") + month_index
            active = (month_index + int(drug)) % 3
            for supplier in ["a", "b"]:
                rows.append({"ndc9": drug, "supplier": supplier,
                             "month": str(month),
                             "shortage_active": int(active == 2 or
                                                     (active == 1 and supplier == "a"))})
    view = build_next_month_numeric_supplier_count_view(
        build_pressure_panel(pd.DataFrame(rows)))
    result = evaluate_numeric_supplier_count_rolling(
        view, min_train_months=12, validation_months=2, test_months=2)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert 0.0 <= result["mean_model_within_5_percent_error"] <= 1.0


def test_shortage_breadth_panel_deduplicates_ndcs_per_month():
    rows = []
    for month_index in range(36):
        month = pd.Period("2020-01", freq="M") + month_index
        rows.extend([
            {"ndc9": "1", "month": str(month), "shortage_active": 1},
            {"ndc9": "1", "month": str(month), "shortage_active": 1},
            {"ndc9": "2", "month": str(month), "shortage_active": int(month_index % 2)},
        ])
    panel = build_shortage_breadth_panel(pd.DataFrame(rows))
    assert panel.iloc[0]["active_ndc_count"] == 1
    result = evaluate_shortage_breadth_rolling(
        pd.DataFrame(rows), min_train_months=12, validation_months=2, test_months=2)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
