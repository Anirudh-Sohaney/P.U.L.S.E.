import pandas as pd

from arkansas_pharma_signal.public_demand_utility import (
    build_public_demand_view, evaluate_public_demand_context_families,
    evaluate_public_demand_context_utility,
)


def test_public_demand_view_requires_consecutive_next_quarter():
    frame = pd.DataFrame([
        {"ndc9": "1", "year": 2020, "quarter": 1, "medicaid_prescriptions": 10,
         "fda_shortage_active": 0, "fda_shortage_supplier_count": 1,
         "prior_ar_ili_mean": 2},
        {"ndc9": "1", "year": 2020, "quarter": 3, "medicaid_prescriptions": 20,
         "fda_shortage_active": 0, "fda_shortage_supplier_count": 1,
         "prior_ar_ili_mean": 3},
    ])
    assert build_public_demand_view(frame).empty


def test_public_demand_context_utility_reports_separate_models():
    rows = []
    for ndc in range(8):
        for year in range(2017, 2023):
            for quarter in range(1, 5):
                rows.append({
                    "ndc9": str(ndc), "year": year, "quarter": quarter,
                    "medicaid_prescriptions": float(10 + ndc + year - 2017 + quarter),
                    "fda_shortage_active": 0, "fda_shortage_supplier_count": 1,
                    "prior_ar_ili_mean": float(quarter),
                })
    result = evaluate_public_demand_context_utility(build_public_demand_view(pd.DataFrame(rows)))
    assert result["fold_count"] >= 1
    assert result["test_rows"] >= 25
    assert "mean_wape_delta_context_minus_base" in result
    families = evaluate_public_demand_context_families(
        build_public_demand_view(pd.DataFrame(rows)))
    assert set(families["families"]) == {"disease", "news"}
