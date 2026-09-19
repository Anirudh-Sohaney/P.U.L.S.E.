import numpy as np
import pandas as pd

from arkansas_pharma_signal.arcos_evaluation import (
    build_latest_scoring_view, build_next_quarter_view, forecast_latest_arcos,
    load_arcos, evaluate_arcos_state_rolling_view, evaluate_arcos_view,
)
from arkansas_pharma_signal.features import build_arcos_annual_features


def _raw():
    return pd.DataFrame({
        "year": [2020, 2020, 2022, 2022], "drug_code": ["1"] * 4,
        "drug_name": ["x"] * 4, "zip3": ["001", "001", "001", "001"],
        "quarter1_grams": ["1,000", "", "2,000", "2,100"],
        "quarter2_grams": ["1,100", "1,200", "2,100", "2,200"],
        "quarter3_grams": ["1,200", "1,300", "2,200", "2,300"],
        "quarter4_grams": ["1,300", "1,400", "2,300", "2,400"],
    })


def test_load_parses_commas_and_preserves_missing_year_quarters(tmp_path):
    path = tmp_path / "arcos.csv"
    _raw().to_csv(path, index=False)
    out = load_arcos(path)
    assert out.loc[out["period"] == "2020-Q1", "grams"].iloc[0] == 1000
    assert "2021-Q1" not in set(out["period"])


def test_annual_features_join_by_drug_and_use_prior_growth(tmp_path):
    path = tmp_path / "arcos.csv"
    pd.DataFrame({
        "year": [2020, 2020, 2021], "drug_name": ["Drug A"] * 3,
        "zip3": ["001", "002", "001"], "total_grams": ["1,000", "500", "3,000"],
    }).to_csv(path, index=False)

    class Config:
        arcos_retail_summary = "arcos.csv"
        def data_path(self, value):
            return path

    out = build_arcos_annual_features(Config())
    row_2020 = out[out["year"].eq(2020)].iloc[0]
    row_2021 = out[out["year"].eq(2021)].iloc[0]
    assert row_2020["arcos_distribution_grams"] == 1500
    assert row_2020["arcos_distribution_zip3_count"] == 2
    assert pd.isna(row_2020["arcos_distribution_growth"])
    assert row_2021["arcos_distribution_growth"] == 1.0


def test_target_requires_exact_next_quarter():
    panel = pd.DataFrame({
        "year": [2020, 2020, 2022], "quarter": [1, 2, 1],
        "period": ["2020-Q1", "2020-Q2", "2022-Q1"],
        "period_index": [8080, 8081, 8088], "drug_code": ["1"] * 3,
        "drug_name": ["x"] * 3, "zip3": ["001"] * 3,
        "grams": [1.0, 2.0, 3.0], "log_grams": np.log1p([1.0, 2.0, 3.0]),
    })
    view = build_next_quarter_view(panel)
    assert list(view["period"]) == ["2020-Q1"]
    assert view.iloc[0]["target"] == 2.0


def test_realistic_evaluation_returns_metrics():
    rows = []
    for year in range(2013, 2022):
        for quarter in range(1, 5):
            rows.append({"year": year, "quarter": quarter,
                         "period": f"{year}-Q{quarter}",
                         "period_index": year * 4 + quarter - 1,
                         "drug_code": "1", "drug_name": "x", "zip3": "001",
                         "grams": float(year * 10 + quarter),
                         "log_grams": np.log1p(float(year * 10 + quarter))})
    view = build_next_quarter_view(pd.DataFrame(rows))
    result = evaluate_arcos_view(view, train_end_year=2017)
    assert result["n_rows"]["test"] > 0
    assert set(result["leaderboard"]["model"]) == {
        "ridge_log1p", "previous_quarter", "seasonal_quarter"}


def test_rolling_selects_only_from_validation_candidates():
    rows = []
    for year in range(2013, 2022):
        for quarter in range(1, 5):
            value = float(year * 10 + quarter)
            rows.append({"year": year, "quarter": quarter,
                         "period": f"{year}-Q{quarter}",
                         "period_index": year * 4 + quarter - 1,
                         "drug_code": "1", "drug_name": "x", "zip3": "001",
                         "grams": value, "log_grams": np.log1p(value)})
    from arkansas_pharma_signal.arcos_evaluation import evaluate_arcos_rolling_view
    out = evaluate_arcos_rolling_view(build_next_quarter_view(pd.DataFrame(rows)))
    assert out["fold_count"] > 0
    assert set(out["folds"]["selected_model"]).issubset({
        "ridge_log1p", "previous_quarter", "seasonal_quarter"})


def test_arcos_state_rolling_evaluation_supports_legacy_three_states_and_fit_thresholds():
    rows = []
    for drug_index in range(4):
        for quarter_index in range(40):
            year = 2010 + quarter_index // 4
            quarter = quarter_index % 4 + 1
            value = float((drug_index + 1) * (quarter_index + 2) ** 2)
            rows.append({
                "year": year, "quarter": quarter,
                "period": f"{year}-Q{quarter}",
                "period_index": year * 4 + quarter - 1,
                "drug_code": str(drug_index), "drug_name": "x",
                "zip3": f"72{drug_index}", "grams": value,
                "log_grams": np.log1p(value),
            })
    result = evaluate_arcos_state_rolling_view(build_next_quarter_view(
        pd.DataFrame(rows)), state_count=3)
    assert result["fold_count"] >= 3
    assert result["test_rows"] >= 25
    assert set(result["state_counts_in_scored_rows"]) == {"0", "1", "2"}
    assert all(len(fold["thresholds_log1p"]) == 2 for fold in result["folds"])
    assert all(fold["selected_model"] in {"persistence", "logistic_one_vs_rest"}
               for fold in result["folds"])


def test_latest_arcos_forecast_has_next_period_and_selected_model():
    rows = []
    for quarter in range(1, 9):
        year = 2020 + (quarter - 1) // 4
        q = (quarter - 1) % 4 + 1
        rows.append({"year": year, "quarter": q, "period": f"{year}-Q{q}",
                     "period_index": year * 4 + q - 1, "drug_code": "1",
                     "drug_name": "ALPHA", "zip3": "722",
                     "grams": float(quarter), "log_grams": float(quarter)})
    panel = pd.DataFrame(rows)
    view = build_next_quarter_view(panel)
    assert len(build_latest_scoring_view(panel)) == 1
    forecast = forecast_latest_arcos(view, panel, validation_quarters=2)
    assert len(forecast) == 1
    assert forecast.iloc[0]["forecast_period"] == "2022-Q1"
    assert forecast.iloc[0]["selected_model"] in {
        "ridge_log1p", "previous_quarter", "seasonal_quarter"
    }


def test_latest_arcos_forecast_drops_stale_pairs():
    rows = []
    for period_index, value in [(8080, 1.0), (8081, 2.0), (8084, 4.0),
                                (8085, 5.0), (8086, 6.0), (8087, 7.0)]:
        year, q = divmod(period_index, 4)
        rows.append({"year": year, "quarter": q + 1,
                     "period": f"{year}-Q{q + 1}", "period_index": period_index,
                     "drug_code": "1", "drug_name": "ALPHA", "zip3": "722",
                     "grams": value, "log_grams": value})
    for period_index, value in [(8080, 1.0), (8081, 2.0), (8082, 3.0),
                                (8083, 4.0), (8084, 5.0)]:
        year, q = divmod(period_index, 4)
        rows.append({"year": year, "quarter": q + 1,
                     "period": f"{year}-Q{q + 1}", "period_index": period_index,
                     "drug_code": "2", "drug_name": "BETA", "zip3": "723",
                     "grams": value, "log_grams": value})
    panel = pd.DataFrame(rows)
    forecast = forecast_latest_arcos(build_next_quarter_view(panel), panel,
                                     validation_quarters=2)
    assert set(forecast["drug_code"]) == {"1"}
    assert forecast["forecast_period"].eq("2022-Q1").all()
