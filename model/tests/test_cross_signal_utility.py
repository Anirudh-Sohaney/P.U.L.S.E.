import pandas as pd

from arkansas_pharma_signal.cross_signal_utility import (
    evaluate_shortage_recall_utility,
)


def test_cross_signal_utility_reports_external_ablation():
    rows = []
    for ndc in ["000000001", "000000002", "000000003"]:
        for index in range(60):
            month = pd.Period("2015-01", freq="M") + index
            rows.append({
                "ndc9": ndc, "month": month,
                "active_supplier_count": index % 3,
                "observed_supplier_count": 2,
                "lag1_pressure": (index - 1) % 3,
                "lag2_pressure": (index - 2) % 3,
                "rolling3_pressure": 1.0, "calendar_month": month.month,
                "recall_state": index % 3,
                "target_pressure_state": (index + 1) % 3,
            })
    result = evaluate_shortage_recall_utility(pd.DataFrame(rows))
    assert result["fold_count"] >= 1
    assert result["test_rows"] >= 25
    assert "mean_balanced_accuracy_delta" in result
    assert result["publishable_candidate"] is False

