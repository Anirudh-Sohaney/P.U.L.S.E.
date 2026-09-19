import pytest
import pandas as pd

pytest.importorskip("sklearn")

from arkansas_pharma_signal.nonlinear_demand_utility import (  # noqa: E402
    evaluate_nonlinear_demand_context_utility,
)
from arkansas_pharma_signal.public_demand_utility import build_public_demand_view  # noqa: E402


def test_nonlinear_demand_utility_reports_matched_families():
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
    result = evaluate_nonlinear_demand_context_utility(
        build_public_demand_view(pd.DataFrame(rows)))
    assert result["fold_count"] >= 1
    assert result["test_rows"] >= 25
    assert "mean_wape_delta_context_minus_base" in result
