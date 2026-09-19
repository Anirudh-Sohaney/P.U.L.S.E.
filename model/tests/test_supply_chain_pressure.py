from pathlib import Path

import pandas as pd
import pytest

from arkansas_pharma_signal.supply_chain_pressure import (
    evaluate_gscpi_rolling, load_gscpi,
)


def test_load_gscpi_requires_complete_monthly_history(tmp_path: Path):
    path = tmp_path / "gscpi.csv"
    pd.DataFrame({"period": ["2020-01", "2020-03"], "gscpi": [0.0, 1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing monthly"):
        load_gscpi(path)


def test_gscpi_rolling_uses_fixed_three_state_bands():
    rows = []
    for index in range(60):
        period = pd.Period("2018-01", freq="M") + index
        rows.append({"period": period, "gscpi": (-0.5, 0.5, 2.0)[index % 3]})
    result = evaluate_gscpi_rolling(pd.DataFrame(rows), min_train_months=24,
                                    validation_months=3, test_months=3)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert result["state_thresholds"] == [0.0, 1.0]
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2"}
