import numpy as np
import pandas as pd

from arkansas_pharma_signal.apcd_claim_counts import (
    evaluate_apcd_claim_activity, load_apcd_claim_counts,
)


def _source():
    rows = []
    for submitter in ["A", "B", "C"]:
        for period in pd.period_range("2018-01", "2022-12", freq="M"):
            rows.append({
                "submitter_id": submitter,
                "submitter_name": submitter,
                "year": period.year,
                "month": period.month,
                "claim_count": 1000 + 10 * period.month + (period.year - 2018) * 100,
            })
    return pd.DataFrame(rows)


def test_apcd_loader_builds_consecutive_entity_transitions():
    view = load_apcd_claim_counts(_source())
    assert not view.empty
    assert view["next_period"].eq(view["period"] + 1).all()
    assert view["submitter_id"].nunique() == 3


def test_apcd_activity_evaluation_has_five_states_and_event_route():
    result = evaluate_apcd_claim_activity(load_apcd_claim_counts(_source()))
    assert result["state_count"] == 5
    assert result["test_rows"] >= 25
    assert result["event_metrics"]["event_definition"]["event_states"] == [3, 4]
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2", "3", "4"}
