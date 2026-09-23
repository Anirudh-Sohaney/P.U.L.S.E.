# Best XGBoost result with model-derived signals

## Selected result

The best tested approach was **top15_lagged_1_7_14**, selected from five
per-drug approaches. Every drug had its own XGBoost model. The target and
chronological 70:30 split are identical to the first experiment: next
14-calendar-day `units_sold`, with training origins through **2025-02-05**
and later origins held out.

The signal table supplied **3,951 dated signal rows**
and expanded to **1,312 candidate signal columns**. This includes
the 20 upstream news signals, the broader external-state signals, 1,212 CMS
drug-demand states, and 18 Arkansas therapeutic-class demand states.

| Metric | Best signal model | Persistence baseline |
|---|---:|---:|
| MAE | 20.162 | 22.010 |
| RMSE | 56.684 | 70.515 |
| WAPE | 28.02% | 30.58% |
| sMAPE | 27.12% | 27.42% |
| R² | 0.571 | 0.336 |
| Within 5% error | 16.11% | 16.92% |
| Within 10% error | 31.72% | 32.10% |
| Within 20% error | 53.75% | 55.95% |
| Test origins | 9,450 | 9,450 |

The selected approach’s unweighted mean per-drug WAPE was **28.36%**
across **30 drugs**. The top-15 selection was performed separately
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
