import pandas as pd

from arkansas_pharma_signal.recall_pressure import (
    build_recall_panel, build_supplier_recall_panel, evaluate_recall_numeric_rolling,
    evaluate_recall_rolling,
)


def test_recall_panel_carries_ongoing_events_and_uses_three_states():
    events = pd.DataFrame([
        {"ndc": "1", "start": pd.Period("2020-01", freq="M"),
         "end": pd.Period("2020-02", freq="M"), "severity": 1},
        {"ndc": "1", "start": pd.Period("2020-04", freq="M"),
         "end": pd.NaT, "severity": 2},
    ])
    panel = build_recall_panel(events)
    assert panel.loc[panel.month.eq(pd.Period("2020-01")), "recall_state"].iloc[0] == 1
    assert panel.loc[panel.month.eq(pd.Period("2020-02")), "recall_state"].iloc[0] == 1
    assert panel.loc[panel.month.eq(pd.Period("2020-03")), "recall_state"].iloc[0] == 0
    assert panel.loc[panel.month.eq(pd.Period("2020-03")), "target_state"].iloc[0] == 2


def test_recall_rolling_has_chronological_test_rows():
    rows = []
    for ndc in ["1", "2", "3"]:
        for month_index in range(48):
            month = pd.Period("2018-01", freq="M") + month_index
            rows.append({"ndc": ndc, "month": month, "recall_state": month_index % 3,
                         "next_month": month + 1, "target_state": (month_index + 1) % 3,
                         "lag1_state": (month_index - 1) % 3, "lag2_state": (month_index - 2) % 3,
                         "rolling3_state": 1.0, "calendar_month": month.month})
    result = evaluate_recall_rolling(pd.DataFrame(rows), min_train_months=12,
                                     validation_months=2, test_months=2)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25


def test_supplier_recall_panel_preserves_supplier_and_three_states():
    events = pd.DataFrame([
        {"supplier": "Acme", "ndc": "1", "start": pd.Period("2020-01", freq="M"),
         "end": pd.Period("2020-01", freq="M"), "severity": 1},
        {"supplier": "Acme", "ndc": "1", "start": pd.Period("2020-03", freq="M"),
         "end": pd.Period("2020-03", freq="M"), "severity": 2},
        {"supplier": "Beta", "ndc": "1", "start": pd.Period("2020-02", freq="M"),
         "end": pd.Period("2020-02", freq="M"), "severity": 1},
        {"supplier": "Beta", "ndc": "1", "start": pd.Period("2020-04", freq="M"),
         "end": pd.Period("2020-04", freq="M"), "severity": 1},
    ])
    panel = build_supplier_recall_panel(events)
    assert set(panel["supplier"]) == {"Acme", "Beta"}
    assert set(panel["recall_state"]) == {0, 1, 2}
    assert panel[["supplier", "ndc", "month"]].duplicated().sum() == 0


def test_recall_numeric_rolling_reports_numeric_error_metrics():
    rows = []
    for ndc in ["1", "2", "3"]:
        for month_index in range(48):
            month = pd.Period("2018-01", freq="M") + month_index
            rows.append({"ndc": ndc, "month": month, "recall_state": month_index % 3,
                         "next_month": month + 1, "target_state": (month_index + 1) % 3,
                         "lag1_state": (month_index - 1) % 3, "lag2_state": (month_index - 2) % 3,
                         "rolling3_state": 1.0, "calendar_month": month.month})
    result = evaluate_recall_numeric_rolling(
        pd.DataFrame(rows), min_train_months=12, validation_months=2, test_months=2)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert 0.0 <= result["mean_model_within_5_percent_error"] <= 1.0
    assert "mean_model_wape" in result
