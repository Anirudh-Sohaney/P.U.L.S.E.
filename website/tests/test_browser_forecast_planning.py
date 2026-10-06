import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "frontend/public/browser_forecast.py"
SPEC = importlib.util.spec_from_file_location("browser_forecast_planning", MODULE_PATH)
forecast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forecast)


def test_browser_csv_dates_require_unambiguous_calendar_days():
    sales = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=130),
                          "drug_name": "Drug A", "units_sold": 2})
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
    sales.loc[10, "date"] = "02/03/2026"
    with pytest.raises(ValueError, match="Sales date must use YYYY-MM-DD"):
        forecast._read_sales(sales.to_csv(index=False))

    sales.loc[10, "date"] = "2026-01-11"
    valid_sales = forecast._read_sales(sales.to_csv(index=False))
    inventory = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=130),
                              "drug_name": "Drug A", "on_hand_units": 10,
                              "expected_arrival_date": ""})
    inventory["date"] = inventory["date"].dt.strftime("%Y-%m-%d")
    inventory.loc[12, "date"] = "2026-01-13T10:00:00"
    with pytest.raises(ValueError, match="Inventory date must use YYYY-MM-DD"):
        forecast._read_inventory(inventory.to_csv(index=False), valid_sales)
    inventory.loc[12, "date"] = "2026-01-13"
    inventory.loc[129, "expected_arrival_date"] = "02/03/2026"
    with pytest.raises(ValueError, match="Expected arrival date must use YYYY-MM-DD"):
        forecast._read_inventory(inventory.to_csv(index=False), valid_sales)


@pytest.mark.parametrize("column,bad_value,error", [
    ("unit_price_usd", "unknown", "unit_price_usd"),
    ("unit_price_usd", -1, "unit_price_usd"),
    ("stockout_flag", "unknown", "stockout_flag"),
    ("stockout_flag", 2, "stockout_flag"),
])
def test_browser_training_rejects_invalid_optional_sales_features(column, bad_value, error):
    days = pd.date_range("2026-01-01", periods=130)
    sales = pd.DataFrame({"date": days, "drug_name": "Drug A", "units_sold": 2,
                          column: ["0"] * len(days)})
    sales.loc[10, column] = bad_value
    with pytest.raises(ValueError, match=error):
        forecast._read_sales(sales.to_csv(index=False))


def test_browser_training_rejects_future_dated_private_records():
    sales = pd.DataFrame({"date": pd.date_range(end="2026-10-04", periods=130),
                          "drug_name": "Drug A", "units_sold": 2})
    with pytest.raises(ValueError, match="Sales CSV contains a date after today"):
        forecast.train(json.dumps({"sales_csv": sales.to_csv(index=False),
                                   "inventory_csv": "", "local_today": "2026-10-03"}))
    sales["date"] -= timedelta(days=1)
    valid_sales = forecast._read_sales(sales.to_csv(index=False), today="2026-10-03")
    inventory = "date,drug_name,on_hand_units\n2026-10-04,Drug A,10\n"
    with pytest.raises(ValueError, match="Inventory CSV contains a date after today"):
        forecast._read_inventory(inventory, valid_sales, today="2026-10-03")


def test_undated_or_duplicate_inventory_cannot_be_used_as_current_stock():
    days = pd.date_range("2026-01-01", periods=130)
    sales = forecast._read_sales(pd.DataFrame({
        "date": days, "drug_name": "Drug A", "units_sold": 2,
    }).to_csv(index=False))
    with pytest.raises(ValueError, match="requires date"):
        forecast._read_inventory("drug_name,on_hand_units\nDrug A,10\n", sales)
    duplicate = "date,drug_name,on_hand_units\n2026-05-10,Drug A,10\n2026-05-10,Drug A,20\n"
    with pytest.raises(ValueError, match="duplicate"):
        forecast._read_inventory(duplicate, sales)
    future = "date,drug_name,on_hand_units\n2026-05-11,Drug A,10\n"
    with pytest.raises(ValueError, match="must match its last sales date"):
        forecast._read_inventory(future, sales)
    older = "date,drug_name,on_hand_units\n2026-05-09,Drug A,10\n"
    with pytest.raises(ValueError, match="must match its last sales date"):
        forecast._read_inventory(older, sales)


def test_incoming_stock_only_covers_days_after_dated_arrival():
    dates = [pd.Timestamp("2026-10-01") + timedelta(days=day) for day in range(1, 15)]
    trailing = pd.Series(np.ones(28))
    latest = SimpleNamespace(date=pd.Timestamp("2026-10-01"), on_hand_units=2, on_order_units=20,
                             expected_arrival_date=pd.Timestamp("2026-10-10"))
    plan = forecast._plan_quantities(14, np.ones(14), trailing, latest, dates)
    assert plan["days_until_stockout"] == 3
    assert plan["minimum_buy_7d_units"] == 5
    assert plan["minimum_buy_14d_units"] == 7
    assert plan["recommended_order_units"] >= plan["minimum_buy_14d_units"]
    assert plan["incoming_counted_by_day_14"] is True

    latest.expected_arrival_date = pd.NaT
    undated = forecast._plan_quantities(14, np.ones(14), trailing, latest, dates)
    assert undated["minimum_buy_14d_units"] == 12
    assert undated["incoming_counted_by_day_14"] is False
    assert undated["recommended_order_units"] >= undated["minimum_buy_14d_units"]

    latest.expected_arrival_date = pd.Timestamp("2026-09-30")
    overdue = forecast._plan_quantities(14, np.ones(14), trailing, latest, dates)
    assert overdue["incoming_eta_overdue"] is True
    assert overdue["incoming_counted_by_day_14"] is False
    assert overdue["minimum_buy_14d_units"] == 12


def test_browser_model_does_not_credit_undated_large_order():
    days = pd.date_range("2026-01-01", periods=180)
    sales = pd.DataFrame({"date": days, "drug_name": "Drug A",
                          "units_sold": [5 + index % 7 for index in range(len(days))]})
    inventory = pd.DataFrame({"date": days, "drug_name": "Drug A",
                              "on_hand_units": [index % 11 for index in range(len(days))],
                              "on_order_units": [0] * (len(days) - 1) + [1000]})
    inventory.loc[inventory.index[-1], "on_hand_units"] = 0
    result = json.loads(forecast.train(json.dumps({
        "sales_csv": sales.to_csv(index=False),
        "inventory_csv": inventory.to_csv(index=False),
        "signal_rows": [], "signal_years": 0,
    })))
    plan = result["items"][0]
    assert result["model"] == "browser_xgboost_direct_14d_v5"
    assert plan["on_order_units"] == 1000
    assert plan["incoming_counted_by_day_14"] is False
    assert plan["minimum_buy_1d_units"] > 0
    assert plan["recommended_order_units"] >= plan["minimum_buy_14d_units"]


def test_browser_training_requires_dated_inventory_history():
    days = pd.date_range("2026-01-01", periods=180)
    sales = pd.DataFrame({"date": days, "drug_name": "Drug A", "units_sold": 2})
    inventory = pd.DataFrame({"date": [days[-1]], "drug_name": ["Drug A"],
                              "on_hand_units": [10]})
    with pytest.raises(ValueError, match="at least 130 sales days"):
        forecast.train(json.dumps({"sales_csv": sales.to_csv(index=False),
                                   "inventory_csv": inventory.to_csv(index=False),
                                   "signal_years": 0}))


def test_browser_model_uses_prior_inventory_without_future_stock_leakage():
    days = pd.date_range("2026-01-01", periods=130)
    sales = pd.DataFrame({"date": days, "drug_name": "Drug A", "units_sold": 2})
    inventory = pd.DataFrame({"date": days, "drug_name": "Drug A",
                              "on_hand_units": list(range(130))})
    work, columns = forecast._features(sales.merge(inventory, on=["date", "drug_name"]))
    assert "inventory_lag1" in columns
    assert pd.isna(work.loc[0, "inventory_lag1"])
    assert work.loc[1, "inventory_lag1"] == 0
    assert work.loc[129, "inventory_lag1"] == 128


def test_signal_history_uses_revision_known_at_each_feature_date():
    dates = pd.date_range("2026-02-01", "2026-04-05")
    rows = [
        {"id": "sig_example", "observation_date": "2026-01-31",
         "source_timestamp": "2026-02-01T10:00:00+00:00", "value": 1},
        {"id": "sig_example", "observation_date": "2026-02-28",
         "source_timestamp": "2026-03-01T10:00:00+00:00", "value": 2},
        {"id": "sig_example", "observation_date": "2026-01-31",
         "source_timestamp": "2026-03-10T10:00:00+00:00", "value": 9},
        {"id": "sig_example", "observation_date": "2026-02-28",
         "source_timestamp": "2026-03-15T10:00:00+00:00", "value": 3},
    ]
    series = forecast._signal_series(rows, dates)["sig_example"]
    assert series[dates.get_loc("2026-02-20")] == 1
    assert series[dates.get_loc("2026-03-05")] == 2
    assert series[dates.get_loc("2026-03-12")] == 2
    assert series[dates.get_loc("2026-03-20")] == 3


@pytest.mark.parametrize("signal_is_worse", [True, False])
def test_public_signal_is_kept_only_when_purged_holdout_improves(monkeypatch, signal_is_worse):
    days = pd.date_range("2026-01-01", periods=180)
    sales = pd.DataFrame({"date": days, "drug_name": "Drug A",
                          "units_sold": [5 + index % 7 + index // 30 for index in range(180)]})
    inventory = pd.DataFrame({"date": days, "drug_name": "Drug A",
                              "on_hand_units": 20})
    monkeypatch.setattr(forecast, "_signal_series", lambda rows, dates: {
        "sig_public": np.arange(len(dates), dtype=float)})

    class ControlledModel:
        def fit(self, features, target):
            self.mean = float(target.mean())
            self.has_signal = any(column.startswith("signal_") for column in features)
            return self

        def predict(self, features):
            penalized = self.has_signal if signal_is_worse else not self.has_signal
            return np.full(len(features), self.mean + (1000 if penalized else 0))

    monkeypatch.setattr(forecast, "_new_model", ControlledModel)
    result = json.loads(forecast.train(json.dumps({
        "sales_csv": sales.to_csv(index=False),
        "inventory_csv": inventory.to_csv(index=False),
        "signal_rows": [{"observation_date": "2026-01-01"}],
        "signal_years": 1,
        "local_today": "2026-10-05",
    })))
    chosen = result["per_drug"][0]
    assert chosen["signals_used"] == ([] if signal_is_worse else ["sig_public"])
    if signal_is_worse:
        assert chosen["wape"] == chosen["private_only_wape"]
    else:
        assert chosen["wape"] < chosen["private_only_wape"]
