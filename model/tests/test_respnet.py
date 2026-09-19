import pandas as pd
import pytest

from arkansas_pharma_signal.respnet import (
    evaluate_respnet_rsv_state,
    load_respnet_rsv_weekly,
)


def _panel():
    dates = pd.date_range("2018-10-06", periods=8 * 52, freq="7D")
    return pd.DataFrame({
        "date": dates,
        "estimate": [float((index % 8) + 1) for index in range(len(dates))],
    })


def test_respnet_loader_builds_consecutive_week_features():
    result = load_respnet_rsv_weekly(_panel())
    assert len(result) == len(_panel()) - 1
    assert result["week_end"].diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert result["target"].notna().all()
    assert set(("current_value", "lag1_value", "rolling4_value")).issubset(result)
    assert result["target_week_end"].iloc[0] > result["week_end"].iloc[0]
    assert (result["target_year"] >= result["year"]).all()


def test_respnet_loader_rejects_mixed_dimension_rows():
    panel = _panel()
    panel["state"] = "Overall"
    panel["surveillance_network"] = "RSV-NET"
    panel.loc[0, "state"] = "Arkansas"
    with pytest.raises(ValueError, match="unfiltered state"):
        load_respnet_rsv_weekly(panel)


def test_respnet_state_evaluation_is_five_state_and_event_scored():
    result = evaluate_respnet_rsv_state(load_respnet_rsv_weekly(_panel()))
    assert result["state_count"] == 5
    assert result["test_rows"] >= 25
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
    assert result["mean_model_balanced_accuracy"] == pytest.approx(
        sum(row["test_balanced_accuracy"] for row in result["folds"])
        / len(result["folds"])
    )
