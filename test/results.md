# Current test index

All metrics in this directory were regenerated on the current deterministic
synthetic pharmacy dataset. Earlier logged results have been superseded.

## Authoritative untouched-test result

The chronological 60%/20%/20% benchmark, with training-only signal
selection, is recorded in `publishable_benchmark_results.md` and
`publishable_benchmark_metrics.json`.

| Metric | Sparse signal model | Sales-only XGBoost |
|---|---:|---:|
| WAPE | 29.40% | 32.35% |
| MAE | 21.258 | 23.396 |
| RMSE | 68.306 | 65.027 |
| Test origins | 6,180 | 6,180 |

The deduplicated sparse check gives 29.03% WAPE.

## Additional current outputs

- `experiment_metrics.json`: native sales-feature experiment output.
- `results_2.md` and `signal_experiment_metrics.json`: native 1,312-signal
  diagnostic run with its chronological 70:30 split.
- `top15_signals_by_drug.csv`: training-selected diagnostic signals.
- `signal_alignment_audit.md`: refreshed event-tag and signal-availability
  audit.

## Reproduction

```bash
python3 test/run_xgb_experiment.py
python3 test/run_xgb_signals.py
python3 test/run_publishable_benchmark.py
python3 test/audit_signal_alignment.py
```
