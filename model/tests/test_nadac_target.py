import pandas as pd

from arkansas_pharma_signal.nadac_target import (
    evaluate_nadac_change_state_rolling,
    evaluate_nadac_rolling,
    evaluate_nadac_transitions,
    load_nadac_transitions,
)


def test_nadac_change_state_screen_rejects_collapsed_quantiles():
    dates = pd.date_range("2021-01-01", "2025-12-26", freq="7D")
    rows = [{"date": date, "price": 1.0, "target_price": 1.0,
             "log_price_delta": 0.0} for date in dates]
    result = evaluate_nadac_change_state_rolling(pd.DataFrame(rows))
    assert result["rejected"] is True
    assert result["state_count"] == 5
    assert result["fold_count"] >= 3


def test_load_nadac_transitions_requires_exact_week(tmp_path):
    path = tmp_path / "nadac.csv"
    pd.DataFrame([
        {"ndc": "12345678901", "nadac_per_unit": 1.0, "as_of_date": "2021-01-01"},
        {"ndc": "12345678901", "nadac_per_unit": 1.1, "as_of_date": "2021-01-08"},
        {"ndc": "12345678901", "nadac_per_unit": 1.2, "as_of_date": "2021-01-24"},
    ]).to_csv(path, index=False)
    out = load_nadac_transitions(path)
    assert len(out) == 1
    assert out.iloc[0]["target_price"] == 1.1


def test_nadac_evaluation_is_chronological():
    dates = pd.date_range("2021-01-01", "2022-03-25", freq="7D")
    rows = []
    for ndc, base in [("001", 1.0), ("002", 2.0)]:
        for i, date in enumerate(dates):
            price = base + i * 0.01
            rows.append({"ndc": ndc, "date": date, "price": price,
                         "target_price": price + 0.01,
                         "log_price": __import__("numpy").log1p(price),
                         "log_price_delta": 0.0, "price_obs": 1.0})
    result = evaluate_nadac_transitions(pd.DataFrame(rows))
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert 0 <= result["model"]["under_5_percent_error"] <= 1


def test_nadac_rolling_selects_without_test_labels():
    dates = pd.date_range("2021-01-01", "2022-12-30", freq="7D")
    rows = []
    for i, date in enumerate(dates):
        price = 1.0 + i * 0.01
        rows.append({"ndc": "001", "date": date, "price": price,
                     "target_price": price + 0.01,
                     "log_price": __import__("numpy").log1p(price),
                     "log_price_delta": 0.0, "price_obs": 1.0})
    result = evaluate_nadac_rolling(
        pd.DataFrame(rows),
        folds=(("2021-06-30", "2021-12-31", "2022-06-30"),))
    assert result["fold_count"] == 1
    assert result["folds"][0]["test_rows"] > 0
    assert result["publishable_rolling_candidate"] is False
