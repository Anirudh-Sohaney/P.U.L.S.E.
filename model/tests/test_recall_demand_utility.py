import pandas as pd

from arkansas_pharma_signal.recall_demand_utility import build_recall_demand_view


def test_recall_context_join_preserves_ndc9_and_prior_quarter_semantics():
    demand = pd.DataFrame([
        {"ndc9": "000021433", "year": year, "quarter": quarter,
         "medicaid_prescriptions": 10 + index,
         "fda_shortage_active": 0, "fda_shortage_supplier_count": 0}
        for index, (year, quarter) in enumerate(
            [(2019, 1), (2019, 2), (2019, 3), (2019, 4), (2020, 1)])
    ])
    recalls = pd.DataFrame([{
        "ndc": "000021433", "start": pd.Period("2019-01", freq="M"),
        "end": pd.Period("2019-01", freq="M"), "severity": 2,
        "event_id": "event-1",
    }])

    result = build_recall_demand_view(demand, recalls)

    assert result["ndc9"].eq("000021433").all()
    q1 = result[result["quarter"].eq(1) & result["year"].eq(2019)].iloc[0]
    q2 = result[result["quarter"].eq(2) & result["year"].eq(2019)].iloc[0]
    assert q1["recall_current_observed"] == 2
    assert q2["recall_lagged_observed"] == 2
