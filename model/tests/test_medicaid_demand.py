from pathlib import Path

import pandas as pd
import pytest

from arkansas_pharma_signal.medicaid_demand import (
    evaluate_arkansas_medicaid_rolling, load_arkansas_medicaid_quarterly,
)


def test_medicaid_loader_rejects_missing_quarters(tmp_path: Path):
    path = tmp_path / "sdud.csv"
    pd.DataFrame([
        {"state": "AR", "source_year": 2020, "quarter": 1,
         "utilization_type": "FFSU", "number_of_prescriptions": 10},
        {"state": "AR", "source_year": 2020, "quarter": 3,
         "utilization_type": "FFSU", "number_of_prescriptions": 11},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing quarters"):
        load_arkansas_medicaid_quarterly(path)


def test_medicaid_evaluator_uses_next_quarter_target():
    rows = []
    for index in range(40):
        period = pd.Period("2015Q1", freq="Q") + index
        rows.append({"period": period, "prescriptions": float(1000 + index * 10),
                     "year": period.year, "quarter": period.quarter})
    result = evaluate_arkansas_medicaid_rolling(
        pd.DataFrame(rows), min_train_quarters=16, validation_quarters=4,
        test_quarters=4)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
