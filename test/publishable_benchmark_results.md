# Final per-drug signal benchmark

This is a synthetic-sales benchmark, not evidence from observed pharmacy sales.
The fixed selection rule was chosen on the earlier validation period: select
exactly five signals per drug by absolute correlation with its training demand,
then add each selected signal at 1-, 7-, and 14-day lags. The final period from
**2025-05-26** onward was not used to choose that rule.

| Metric | Selected signals | Sales-only |
|---|---:|---:|
| MAE | 21.296 | 22.981 |
| RMSE | 69.154 | 63.455 |
| WAPE | 29.45% | 31.78% |
| sMAPE | 27.17% | 28.04% |
| Within 20% | 57.20% | 57.41% |
| Test origins | 6,180 | 6,180 |

The selected-signal model reduced WAPE by
**7.3%** against the same per-drug
sales-only XGBoost baseline. The complete machine-readable result is in
`publishable_benchmark_metrics.json`.
