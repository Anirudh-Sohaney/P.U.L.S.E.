from pathlib import Path

import numpy as np
import pandas as pd

from arkansas_pharma_signal.historical_metric_context import (
    METRIC_CONTEXT_FEATURES, build_historical_metric_context,
)


def test_historical_context_is_explicitly_missing_when_sources_are_unavailable(tmp_path):
    panel = pd.DataFrame([{"year": 2020, "quarter": 1, "drug": "x", "ndc": ""}])
    contexts, report = build_historical_metric_context(tmp_path, panel)
    vector = contexts[(2020, 1, "x")]
    assert vector.shape == (16,)
    assert np.all(vector[:8] == 0)
    assert np.all(vector[8:] == 1)
    assert report["current_forecast_artifact_used"] is False
    assert report["unavailable_features_remain_missing"] is True


def test_historical_context_uses_only_latest_observation_before_origin(tmp_path):
    source = tmp_path / "data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame([
        {"ndc9": "123456789", "month": "2019-12", "supplier": "a", "shortage_active": 1},
        {"ndc9": "123456789", "month": "2020-02", "supplier": "a", "shortage_active": 1},
        {"ndc9": "123456789", "month": "2020-02", "supplier": "b", "shortage_active": 1},
        {"ndc9": "123456789", "month": "2020-04", "supplier": "a", "shortage_active": 1},
    ]).to_csv(source, index=False)
    panel = pd.DataFrame([{"year": 2020, "quarter": 1, "drug": "x", "ndc": "123456789"}])
    contexts, report = build_historical_metric_context(tmp_path, panel)
    vector = contexts[(2020, 1, "x")]
    assert vector[0] == 2.0
    assert vector[8] == 0.0
    assert report["coverage_rows"][METRIC_CONTEXT_FEATURES[0]] == 1


def test_historical_context_joins_nadac_at_or_before_origin(tmp_path):
    source = tmp_path / "data/targeted_additions/cms_nadac_historical/data/nadac_arkansas_exposed_2021_2025.csv.gz"
    source.parent.mkdir(parents=True)
    pd.DataFrame([
        {"ndc": "123456789", "as_of_date": "2019-12-18", "nadac_per_unit": 1.0},
        {"ndc": "123456789", "as_of_date": "2020-02-05", "nadac_per_unit": 2.0},
        {"ndc": "123456789", "as_of_date": "2020-04-01", "nadac_per_unit": 99.0},
    ]).to_csv(source, index=False, compression="gzip")
    panel = pd.DataFrame([{"year": 2020, "quarter": 1, "drug": "x", "ndc": "123456789"}])
    contexts, report = build_historical_metric_context(tmp_path, panel)
    vector = contexts[(2020, 1, "x")]
    assert vector[1] == 2.0
    assert vector[9] == 0.0
    assert report["coverage_rows"][METRIC_CONTEXT_FEATURES[1]] == 1


def test_historical_context_joins_point_in_time_nssp_state(tmp_path):
    source = tmp_path / "data/targeted_additions/cdc_nssp_ar_current/data/cdc_nssp_ar_weekly.json"
    source.parent.mkdir(parents=True)
    dates = pd.date_range("2020-01-04", periods=60, freq="7D")
    pd.DataFrame({
        "week_end": dates, "county": "All",
        "percent_visits_influenza": np.arange(60, dtype=float),
    }).to_json(source, orient="records")
    panel = pd.DataFrame([{"year": 2021, "quarter": 1, "drug": "x", "ndc": ""}])
    contexts, report = build_historical_metric_context(tmp_path, panel)
    vector = contexts[(2021, 1, "x")]
    assert vector[6] in {0.0, 1.0, 2.0, 3.0, 4.0}
    assert vector[14] == 0.0
    assert report["coverage_rows"][METRIC_CONTEXT_FEATURES[6]] == 1


def test_historical_context_joins_point_in_time_respnet_state(tmp_path):
    source = tmp_path / "data/targeted_additions/cdc_respnet_rsv_current/data/respnet_rsv_overall_weekly.json"
    source.parent.mkdir(parents=True)
    dates = pd.date_range("2020-01-04", periods=60, freq="7D")
    pd.DataFrame({
        "date": dates, "estimate": np.arange(60, dtype=float) + 1,
    }).to_json(source, orient="records")
    panel = pd.DataFrame([{"year": 2021, "quarter": 1, "drug": "x", "ndc": ""}])
    contexts, report = build_historical_metric_context(tmp_path, panel)
    vector = contexts[(2021, 1, "x")]
    assert vector[7] in {0.0, 1.0, 2.0, 3.0, 4.0}
    assert vector[15] == 0.0
    assert report["coverage_rows"][METRIC_CONTEXT_FEATURES[7]] == 1
