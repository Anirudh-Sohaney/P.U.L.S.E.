"""Final untouched-test benchmark for the selected per-drug signal model.

The selection rule was chosen on the validation period: select exactly five
catalog signals per drug by absolute training-period correlation, then use
their 1-, 7-, and 14-day lags in a direct 14-day XGBoost forecast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

TEST = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST))
from run_xgb_experiment import make_frame, metrics  # noqa: E402
from run_xgb_signals import signal_matrix, select_top15  # noqa: E402

OUT_JSON = TEST / "publishable_benchmark_metrics.json"
OUT_MD = TEST / "publishable_benchmark_results.md"
SELECTED_COUNT = 5


def model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=220, max_depth=2, learning_rate=0.04,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.5,
        reg_lambda=10, objective="reg:squarederror", tree_method="hist",
        n_jobs=2, random_state=20250915,
    )


def base_columns(frame: pd.DataFrame) -> list[str]:
    historical = [c for c in frame if c.startswith(("lag_", "mean_", "std_"))]
    historical = [c for c in historical if int(c.split("_")[1]) <= 56]
    return historical + ["dow", "weekofyear", "month", "sin_year", "cos_year", "price_lag1", "stockout_lag1"]


def lagged_signal_features(part: pd.DataFrame, selected: list[str]) -> tuple[pd.DataFrame, list[str]]:
    lagged = {
        f"{signal}__lag{lag}": part[signal].shift(lag)
        for signal in selected for lag in (1, 7, 14)
    }
    return pd.concat([part, pd.DataFrame(lagged, index=part.index)], axis=1), list(lagged)


def main() -> None:
    sales = make_frame(False)
    dates = pd.DatetimeIndex(sorted(sales.date.unique()))
    signals, signal_rows = signal_matrix(dates)
    frame = sales.set_index("date").join(signals, how="left").reset_index()
    frame[signals.columns] = frame[signals.columns].fillna(0.0)
    base, candidates = base_columns(frame), list(signals.columns)
    first, last = dates.min(), dates.max()
    validation_end = first + (last - first) * 0.80

    actual, selected_predictions, sales_only_predictions = [], [], []
    selections = {}
    for drug, part in frame.groupby("drug_name", sort=True):
        usable = part.dropna(subset=base + ["target_14d"])
        train = usable[usable.date < validation_end]
        selected = [item[0] for item in select_top15(train, train.target_14d, candidates)[:SELECTED_COUNT]]
        selections[drug] = selected
        work, signal_columns = lagged_signal_features(usable, selected)
        work = work.dropna(subset=signal_columns + ["target_14d"])
        train, test = work[work.date < validation_end], work[work.date >= validation_end]
        fitted = model().fit(train[base + signal_columns], train.target_14d, verbose=False)
        baseline = model().fit(train[base], train.target_14d, verbose=False)
        actual.extend(test.target_14d.to_numpy())
        selected_predictions.extend(fitted.predict(test[base + signal_columns]))
        sales_only_predictions.extend(baseline.predict(test[base]))

    selected_score = metrics(actual, selected_predictions)
    baseline_score = metrics(actual, sales_only_predictions)
    result = {
        "date_range": [str(first.date()), str(last.date())],
        "final_test_start": str(validation_end.date()),
        "signal_candidate_columns": len(candidates),
        "source_signal_rows": int(len(signal_rows)),
        "selection": {
            "rule": "top_5_absolute_training_correlation",
            "signals_per_drug": SELECTED_COUNT,
            "lags_days": [1, 7, 14],
            "selected_drugs": len(selections),
        },
        "untouched_test": {"selected_signals": selected_score, "sales_only": baseline_score},
        "wape_improvement_vs_sales_only": float(1 - selected_score["wape"] / baseline_score["wape"]),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT_MD.write_text(f"""# Final per-drug signal benchmark

This is a synthetic-sales benchmark, not evidence from observed pharmacy sales.
The fixed selection rule was chosen on the earlier validation period: select
exactly five signals per drug by absolute correlation with its training demand,
then add each selected signal at 1-, 7-, and 14-day lags. The final period from
**{validation_end.date()}** onward was not used to choose that rule.

| Metric | Selected signals | Sales-only |
|---|---:|---:|
| MAE | {selected_score['mae']:.3f} | {baseline_score['mae']:.3f} |
| RMSE | {selected_score['rmse']:.3f} | {baseline_score['rmse']:.3f} |
| WAPE | {selected_score['wape']:.2%} | {baseline_score['wape']:.2%} |
| sMAPE | {selected_score['smape']:.2%} | {baseline_score['smape']:.2%} |
| Within 20% | {selected_score['within_20pct']:.2%} | {baseline_score['within_20pct']:.2%} |
| Test origins | {selected_score['n']:,} | {baseline_score['n']:,} |

The selected-signal model reduced WAPE by
**{result['wape_improvement_vs_sales_only']:.1%}** against the same per-drug
sales-only XGBoost baseline. The complete machine-readable result is in
`publishable_benchmark_metrics.json`.
""", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
