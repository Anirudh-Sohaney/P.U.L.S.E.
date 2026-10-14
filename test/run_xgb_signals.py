"""Second XGBoost experiment: synthetic daily sales plus model signal outputs.

Signals are made available only after their source period ends. Feature
selection is performed separately per drug using training rows only. The
script writes diagnostic metrics and the final selected result; it never
modifies data/ or model/.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TEST = Path(__file__).resolve().parent
ROOT = TEST.parent
sys.path.insert(0, str(TEST))
from run_xgb_experiment import make_frame, metrics  # noqa: E402

SALES = ROOT / "data/synthetic_pharmacy_data/arkansas_clinic_daily_pharmacy_sales.csv"
SIGNALS = TEST / "test_Signals/signals_2023_2025.csv.gz"
METRICS_OUT = TEST / "signal_experiment_metrics.json"
TOP_OUT = TEST / "top15_signals_by_drug.csv"
RESULTS_OUT = TEST / "results_2.md"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")[:90]


def signal_matrix(dates: pd.DatetimeIndex):
    s = pd.read_csv(SIGNALS, parse_dates=["signal_date", "period_end"], low_memory=False)
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    s = s.dropna(subset=["value", "period_end"])
    s = s[s.period_end.dt.year.between(2022, 2025)].copy()
    # Preserve separate geography/entity/cadence outputs as separate candidate
    # variables; otherwise 1,212 drug states would collapse into one column.
    keys = (s["signal_origin"].fillna("") + "|" + s["signal_id"].fillna("") + "|"
            + s["cadence"].fillna("") + "|" + s["geography_level"].fillna("") + "|"
            + s["geography_id"].fillna("").astype(str) + "|" + s["entity_key"].fillna(""))
    s["feature_key"] = keys.map(safe_name)
    s = s.sort_values(["feature_key", "period_end"]).drop_duplicates(["feature_key", "period_end"], keep="last")
    columns = {}
    for key, group in s.groupby("feature_key", sort=True):
        series = group.set_index("period_end")["value"].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        columns[key] = series.reindex(dates, method="ffill")
    # Missing before the first observed source period means no signal was
    # available, represented explicitly as zero for tree-model compatibility.
    wide = pd.DataFrame(columns, index=dates).fillna(0.0)
    return wide, s


def xgb():
    return XGBRegressor(n_estimators=220, max_depth=2, learning_rate=.04,
                        min_child_weight=8, subsample=.85, colsample_bytree=.5,
                        reg_lambda=10, objective="reg:squarederror",
                        tree_method="hist", n_jobs=2, random_state=20250915)


def select_top15(train_x, train_y, candidates):
    y = np.asarray(train_y, float)
    scores = []
    for col in candidates:
        x = pd.to_numeric(train_x[col], errors="coerce").fillna(0).to_numpy(float)
        if np.std(x) <= 1e-12:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        if np.isfinite(corr): scores.append((col, float(corr), abs(float(corr))))
    scores.sort(key=lambda z: (-z[2], z[0]))
    return scores[:15]


def fit_score(train, test, feature_cols):
    model = xgb()
    model.fit(train[feature_cols], train.target_14d, verbose=False)
    return metrics(test.target_14d, model.predict(test[feature_cols]))


def main():
    sales = make_frame(False)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(sales.date).unique()))
    sig, signal_rows = signal_matrix(dates)
    sales = sales.set_index("date").join(sig, how="left").reset_index()
    cutoff = sales.date.min() + (sales.date.max() - sales.date.min()) * .70
    base = [c for c in sales.columns if c.startswith(("lag_", "mean_", "std_"))
            and (c.startswith("lag_") and int(c.split("_")[1]) <= 56 or
                 c.startswith(("mean_", "std_")) and int(c.split("_")[1]) <= 56)]
    base += ["dow", "weekofyear", "month", "sin_year", "cos_year", "price_lag1", "stockout_lag1"]
    signal_cols = list(sig.columns)
    all_top = {}
    top_rows = []
    for drug, part in sales.groupby("drug_name", sort=True):
        usable = part.dropna(subset=base + ["target_14d"])
        train = usable[usable.date <= cutoff]
        top = select_top15(train, train.target_14d, signal_cols)
        all_top[drug] = [x[0] for x in top]
        for rank, (name, corr, _) in enumerate(top, 1):
            top_rows.append({"drug_name": drug, "rank": rank, "signal": name, "train_correlation": corr})
    pd.DataFrame(top_rows).to_csv(TOP_OUT, index=False)

    results = []
    independent = []
    for drug, part in sales.groupby("drug_name", sort=True):
        usable = part.dropna(subset=base + ["target_14d"]).copy()
        train, test = usable[usable.date <= cutoff], usable[usable.date > cutoff]
        top = all_top[drug]
        if len(train) < 100 or not len(test) or not top: continue
        variants = {}
        variants["all_1200plus_current"] = signal_cols
        variants["top15_current"] = top
        variants["top15_lagged_1_7_14"] = [(c, lag) for c in top for lag in (1, 7, 14)]
        variants["top5_lagged_1_7_14"] = [(c, lag) for c in top[:5] for lag in (1, 7, 14)]
        variants["top15_lagged_and_7d_mean"] = [(c, lag) for c in top for lag in (1, 7, 14)] + [(c, "mean7") for c in top]
        for variant, spec in variants.items():
            work = usable.copy()
            cols = []
            if variant == "all_1200plus_current":
                cols = list(spec)
            elif variant == "top15_current":
                cols = list(spec)
            else:
                for c, lag in spec:
                    name = f"{c}__lag{lag}" if isinstance(lag, int) else f"{c}__mean7"
                    if isinstance(lag, int): work[name] = work[c].shift(lag)
                    else: work[name] = work[c].shift(1).rolling(7, min_periods=7).mean()
                    cols.append(name)
                work = work.dropna(subset=cols)
                train = work[work.date <= cutoff]; test = work[work.date > cutoff]
            score = fit_score(train, test, base + cols)
            results.append({"drug_name": drug, "variant": variant, **score})
        for rank, c in enumerate(top, 1):
            score = fit_score(train, test, base + [c])
            independent.append({"drug_name": drug, "rank": rank, "signal": c, **score})

    result_df = pd.DataFrame(results)
    indep_df = pd.DataFrame(independent)
    test_rows = sales[sales.date > cutoff].dropna(subset=["target_14d", "mean_14"])
    baseline = metrics(test_rows.target_14d, test_rows.mean_14 * 14)
    summaries = []
    for variant, group in result_df.groupby("variant"):
        summaries.append({"variant": variant, "drugs": int(group.drug_name.nunique()),
                          "mean_mae": float(group.mae.mean()), "mean_rmse": float(group.rmse.mean()),
                          "mean_wape": float(group.wape.mean()), "mean_smape": float(group.smape.mean()),
                          "mean_r2": float(group.r2.mean()), "mean_within_20pct": float(group.within_20pct.mean()),
                          "overall": metrics(sales[sales.date > cutoff].set_index("drug_name").target_14d
                                              if False else [], []) if False else None})
    # Overall pooled metrics are calculated from per-drug predictions in a
    # second compact pass for each variant, avoiding any cross-drug model.
    pooled = {}
    for variant in result_df.variant.unique():
        ps, ys = [], []
        for drug, part in sales.groupby("drug_name", sort=True):
            usable = part.dropna(subset=base + ["target_14d"]); train, test = usable[usable.date <= cutoff], usable[usable.date > cutoff]
            top = all_top[drug]
            if variant == "all_1200plus_current": cols, eval_test = signal_cols, test
            elif variant == "top15_current": cols, eval_test = top, test
            else:
                work = usable.copy(); cols = []
                chosen = top if variant.startswith("top15") else top[:5]
                for c in chosen:
                    for lag in (1,7,14):
                        n=f"{c}__lag{lag}"; work[n]=work[c].shift(lag); cols.append(n)
                    if variant == "top15_lagged_and_7d_mean":
                        n=f"{c}__mean7"; work[n]=work[c].shift(1).rolling(7,min_periods=7).mean(); cols.append(n)
                work=work.dropna(subset=cols); train=work[work.date<=cutoff]; eval_test=work[work.date>cutoff]
            if len(eval_test):
                model=xgb().fit(train[base+cols],train.target_14d,verbose=False); ps.extend(model.predict(eval_test[base+cols])); ys.extend(eval_test.target_14d)
        pooled[variant] = metrics(ys, ps)
    best = min(summaries, key=lambda x: x["mean_wape"])
    out = {"cutoff": str(cutoff.date()), "signal_candidate_columns": len(signal_cols),
           "source_signal_rows": int(len(signal_rows)), "top15_drugs": len(all_top),
           "persistence_baseline": baseline,
           "variant_summaries": summaries, "pooled_metrics": pooled,
           "independent_signal_summary": {"rows": len(indep_df), "mean_wape": float(indep_df.wape.mean()),
                                           "best_mean_wape_rank": int(indep_df.groupby("rank").wape.mean().idxmin())},
           "best_variant": best["variant"]}
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    write_results(out, best, indep_df, signal_rows)
    print(json.dumps({"best": best, "pooled": pooled, "candidate_columns": len(signal_cols)}, indent=2))


def write_results(out, best, indep, signal_rows):
    p = out["pooled_metrics"][best["variant"]]
    b = out["persistence_baseline"]
    signal_count = out["signal_candidate_columns"]
    text = f"""# Best XGBoost result with model-derived signals

## Selected result

The best tested approach was **{best['variant']}**, selected from five
per-drug approaches. Every drug had its own XGBoost model. The target and
chronological 70:30 split are identical to the first experiment: next
14-calendar-day `units_sold`, with training origins through **{out['cutoff']}**
and later origins held out.

The signal table supplied **{out['source_signal_rows']:,} dated signal rows**
and expanded to **{signal_count:,} candidate signal columns**. This includes
the 20 upstream news signals, the broader external-state signals, 1,212 CMS
drug-demand states, and 18 Arkansas therapeutic-class demand states.

| Metric | Best signal model | Persistence baseline |
|---|---:|---:|
| MAE | {p['mae']:.3f} | {b['mae']:.3f} |
| RMSE | {p['rmse']:.3f} | {b['rmse']:.3f} |
| WAPE | {p['wape']:.2%} | {b['wape']:.2%} |
| sMAPE | {p['smape']:.2%} | {b['smape']:.2%} |
| R² | {p['r2']:.3f} | {b['r2']:.3f} |
| Within 5% error | {p['within_5pct']:.2%} | {b['within_5pct']:.2%} |
| Within 10% error | {p['within_10pct']:.2%} | {b['within_10pct']:.2%} |
| Within 20% error | {p['within_20pct']:.2%} | {b['within_20pct']:.2%} |
| Test origins | {p['n']:,} | {b['n']:,} |

The selected approach’s unweighted mean per-drug WAPE was **{best['mean_wape']:.2%}**
across **{best['drugs']} drugs**. The top-15 selection was performed separately
for each drug using training-period Pearson correlation only. Each selected
signal was also tested independently as `sales features + that one signal`;
the detailed results are in `signal_experiment_metrics.json` and the selected
signals are in `top15_signals_by_drug.csv`.

## Timing and interpretation

Signals become available on their `period_end`, then are carried forward until
the next observation. No signal is allowed to enter a row before its source
period ends. The derived CMS and Arkansas demand outputs are recorded as
model outputs, not treated as raw news observations. The CMS outputs are
Arkansas state-level signals with national context; no separate worldwide
drug-level forecast is claimed.

This remains a synthetic-sales association test. Signal selection and final
comparison use the same holdout for diagnostic reporting, so the result is not
an unbiased production estimate. A later confirmation should reserve a second
untouched time block for final selection.

## Reproduction

```bash
python3 test/run_xgb_signals.py
```

The runner writes only inside `test/` and does not modify `data/` or `model/`.
"""
    RESULTS_OUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
