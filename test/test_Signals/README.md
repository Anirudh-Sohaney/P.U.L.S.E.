# 2023–2025 model signals for synthetic-sales testing

`signals_2023_2025.csv.gz` is a tidy, long-format signal table. It combines:

1. the validated 20-column monthly news-only model output; and
2. state and national rows from the model's dated external-state feature
   store, restricted to native weekly, monthly, and annual frequencies.

Each row preserves the signal date, period end, cadence, signal ID, value,
geography, source, source timestamp, and missingness metadata. The data is
limited to 2023–2025 so it can be joined to the synthetic clinic sales by
date/month. Weekly values are not artificially expanded into daily rows.
`source_release_frequency` retains the original model-store frequency when
the standardized test cadence groups `daily_or_weekly` with weekly signals.

The file is intentionally long-format so a tester can select a signal by
`signal_id` and join it according to its `cadence`. `signal_origin` separates
the news-model output from the broader external-state feature surface.

The file also records the downstream outputs in `model/final_predictions.json`:
1,212 annual CMS Part D drug-demand states (`entity_type=drug`) and 18 monthly
Arkansas ATC therapeutic-class demand states (`entity_type=therapeutic_class`).
These rows are marked `signal_origin=derived_demand_output` and retain their
entity keys and state definitions. The CMS drug outputs are Arkansas
state-level signals; the model bundle does not contain a separate worldwide
drug-level demand output, so national/worldwide context is not mislabeled as a
global drug forecast.

Rebuild with:

```bash
python3 test/test_Signals/build_signal_csv.py
```
