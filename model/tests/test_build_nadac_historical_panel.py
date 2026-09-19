from pathlib import Path

import pandas as pd

from arkansas_pharma_signal.nadac_target import _ndc9
from scripts.build_nadac_historical_panel import _read_cms_snapshot


def test_cms_snapshot_normalizes_year_specific_price_column(tmp_path: Path):
    path = tmp_path / "nadac_2024.csv"
    pd.DataFrame([
        {"NDC": "123-45-6789", "NADAC Per Unit": "1.25", "As of Date": "01/03/2024"},
        {"NDC": "999999999", "NADAC Per Unit": "2.00", "As of Date": "01/03/2024"},
    ]).to_csv(path, index=False)

    result = _read_cms_snapshot(path, {_ndc9("123456789")})

    assert result.to_dict("records") == [{
        "ndc": "123456789",
        "nadac_per_unit": 1.25,
        "as_of_date": pd.Timestamp("2024-01-03"),
    }]
