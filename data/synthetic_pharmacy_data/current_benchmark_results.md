# Untouched-test benchmark

This is a transparent synthetic stress test, not observed pharmacy data.
Signal selection used only the first 60% of dates. The middle 20% is a
validation period. The final 20% beginning **2025-05-26** was not
used for signal selection or model choice.

The pre-registered primary signal approach was `top5_lagged`: the five
training-selected signals for each drug, delayed by 1, 7, and 14 days. The
sales-only comparison uses the same XGBoost family and the same dates.

| Metric | Primary signals | Sales-only |
|---|---:|---:|
| MAE | 21.258 | 23.396 |
| RMSE | 68.306 | 65.027 |
| WAPE | 29.40% | 32.35% |
| sMAPE | 26.23% | 28.02% |
| R² | 0.517 | 0.562 |
| Within 20% | 58.03% | 57.09% |
| Test origins | 6,180 | 6,180 |

The complete validation and untouched-test results are in
`current_benchmark_metrics.json`. The generator, event tags, and
provenance are documented in this directory's `synthetic_guide.md` and
`benchmark_protocol.md`.

As a robustness check, near-static and exact-duplicate signal columns were
removed, leaving **74** columns. The resulting sparse model
had **29.03% WAPE** on the same untouched test period.
