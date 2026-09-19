"""Contract tests for the separate news-only SLM integration boundary."""

import pandas as pd

from arkansas_pharma_signal.config import Config
from arkansas_pharma_signal.news_only_adapter import (
    NEWS_ONLY_SIGNAL_IDS, build_news_only_annual_features,
    load_news_only_features,
)


def _source(tmp_path):
    rows = []
    for date in ["2024-01-01", "2024-02-01"]:
        rows.append({"date": date, **{signal: 1.0 for signal in NEWS_ONLY_SIGNAL_IDS}})
    path = tmp_path / "news_only.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_news_only_adapter_validates_schema_and_annualizes(tmp_path):
    cfg = Config(root=str(tmp_path))
    source = _source(tmp_path)
    monthly = load_news_only_features(cfg, source)
    annual = build_news_only_annual_features(cfg, source)
    assert monthly.shape == (2, 21)
    assert annual.shape == (1, 21)
    assert annual.loc[0, "news_only_national_policy_direction"] == 2.0


def test_news_only_signal_ids_are_unique_and_fixed():
    assert len(NEWS_ONLY_SIGNAL_IDS) == 20
    assert len(set(NEWS_ONLY_SIGNAL_IDS)) == 20
