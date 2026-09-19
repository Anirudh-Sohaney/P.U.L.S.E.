"""Record the training-period signals selected by the unchanged benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = ROOT / "test"
sys.path.insert(0, str(TEST))

from run_xgb_experiment import make_frame  # noqa: E402
from run_xgb_signals import select_top15, signal_matrix  # noqa: E402
from run_publishable_benchmark import base_columns  # noqa: E402


def main() -> None:
    sales = make_frame(False)
    dates = pd.DatetimeIndex(sorted(sales.date.unique()))
    signals, _ = signal_matrix(dates)
    frame = sales.set_index("date").join(signals, how="left").reset_index()
    frame[signals.columns] = frame[signals.columns].fillna(0.0)
    base = base_columns(frame)
    train_end = dates.min() + (dates.max() - dates.min()) * .60
    rows = []
    for drug, part in frame.groupby("drug_name", sort=True):
        train = part.dropna(subset=base + ["target_14d"])
        train = train[train.date < train_end]
        for rank, (signal, correlation, _) in enumerate(
            select_top15(train, train.target_14d, list(signals.columns)), 1
        ):
            rows.append({
                "drug_name": drug,
                "rank": rank,
                "signal": signal,
                "train_correlation": correlation,
            })
    path = HERE / "selected_signal_diagnostics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
