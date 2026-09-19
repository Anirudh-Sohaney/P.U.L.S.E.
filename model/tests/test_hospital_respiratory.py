import pandas as pd

from arkansas_pharma_signal.hospital_respiratory import (
    evaluate_hospital_respiratory_numeric, evaluate_hospital_respiratory_state,
    load_hospital_respiratory_weekly,
)


def _frame():
    dates = pd.date_range("2020-01-04", periods=160, freq="7D")
    values = [1 + ((i // 8) % 3) * 10 + (i % 3) for i in range(160)]
    return pd.DataFrame({
        "weekendingdate": dates,
        "jurisdiction": "AR",
        "totalconfc19newadm": values,
        "totalconfflunewadm": values,
        "totalconfrsvnewadm": values,
    })


def test_hospital_loader_preserves_consecutive_pathogen_rows():
    view = load_hospital_respiratory_weekly(_frame())
    assert set(view["pathogen"]) == {"covid", "influenza", "rsv"}
    assert len(view) == 3 * 159
    assert view["week_end"].diff().dropna().abs().min() > pd.Timedelta(0)


def test_hospital_state_evaluator_has_three_states_and_folds():
    view = load_hospital_respiratory_weekly(_frame())
    result = evaluate_hospital_respiratory_state(
        view, pathogen="influenza",
        folds=((2020, 2020, 2021), (2021, 2021, 2022)),
    )
    assert result["fold_count"] == 2
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2"}
    assert result["event_metrics"]["event_definition"]["event_states"] == [1, 2]
    assert result["event_metrics"]["true_positive_precision"] is not None


def test_hospital_numeric_evaluator_reports_numeric_error():
    view = load_hospital_respiratory_weekly(_frame())
    result = evaluate_hospital_respiratory_numeric(
        view, pathogen="influenza",
        folds=((2020, 2020, 2021), (2021, 2021, 2022)))
    assert result["fold_count"] == 2
    assert 0.0 <= result["mean_model_within_5_percent_error"] <= 1.0
    assert "mean_model_wape" in result
