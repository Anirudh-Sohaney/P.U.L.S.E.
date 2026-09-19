"""Per-drug untouched-test comparison using the repository benchmark setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = ROOT / "test"
sys.path.insert(0, str(TEST))

from run_xgb_experiment import make_frame, metrics  # noqa: E402
from run_xgb_signals import select_top15, signal_matrix  # noqa: E402
from run_publishable_benchmark import base_columns, model  # noqa: E402


def main() -> None:
    sales = make_frame(False)
    dates = pd.DatetimeIndex(sorted(sales.date.unique()))
    signals, _ = signal_matrix(dates)
    frame = sales.set_index("date").join(signals, how="left").reset_index()
    frame[signals.columns] = frame[signals.columns].fillna(0.0)
    base = base_columns(frame)
    train_end = dates.min() + (dates.max() - dates.min()) * .60
    test_start = dates.min() + (dates.max() - dates.min()) * .80
    rows = []
    for drug, part in frame.groupby("drug_name", sort=True):
        work = part.dropna(subset=base + ["target_14d"]).copy()
        selected = [x[0] for x in select_top15(
            work[work.date < train_end],
            work.loc[work.date < train_end, "target_14d"], list(signals.columns),
        )][:5]
        lagged = []
        for signal in selected:
            for lag in (1, 7, 14):
                col = f"{signal}__lag{lag}"
                work[col] = work[signal].shift(lag)
                lagged.append(col)
        work = work.dropna(subset=lagged)
        train, test = work[work.date < test_start], work[work.date >= test_start]
        base_score = metrics(test.target_14d, model().fit(
            train[base], train.target_14d, verbose=False
        ).predict(test[base]))
        signal_score = metrics(test.target_14d, model().fit(
            train[base + lagged], train.target_14d, verbose=False
        ).predict(test[base + lagged]))
        rows.append({
            "drug_name": drug,
            "sales_only_wape": base_score["wape"],
            "signal_wape": signal_score["wape"],
            "wape_reduction": 1 - signal_score["wape"] / base_score["wape"],
            "sales_only_mae": base_score["mae"],
            "signal_mae": signal_score["mae"],
        })
    path = HERE / "per_drug_benchmark_metrics.csv"
    pd.DataFrame(rows).sort_values("wape_reduction", ascending=False).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
