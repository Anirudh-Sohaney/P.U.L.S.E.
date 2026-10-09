# Model comparison: rebuilt demand models

## Protocol

The grid has 10 feature-input variants × 9 XGBoost profiles = 90 configurations. Each of 30 synthetic drug series gets its own model. The first 60% of dates select signals and fit validation models; the next 20% selects the configuration by unweighted mean per-drug WAPE. The validation winner is then refit on the first 80% and evaluated once on the final 20%. A 14-day purge before each boundary prevents the next-14-day training labels from overlapping validation or final-test outcomes. No configuration was selected from final-test results.

The synthetic data covers 2023-01-01 through 2025-12-31. Validation begins 2024-10-19; final test begins 2025-05-26. The signal table expands to 1,312 dated candidate columns with 6,484 source rows.

Signal rankings use only training dates and absolute Pearson correlation between each candidate’s one-day lag and the next-14-day demand target, except the explicitly named legacy-rank variant, which ranks raw values to reproduce the earlier benchmark's rule. Signal lags are 1, 7, 14, and optionally 28 days. The all-signals row uses 1,312 lagged features. All signal inputs are forward-filled only after their recorded period end and begin at zero before the first observation.

All profiles use subsample=0.85, histogram tree building, and one worker. The grid includes squared-error, Poisson count, Tweedie, and pseudo-Huber objectives alongside five tree-depth/regularization/estimator profiles. The historical profile retains seed 20250915; all new profiles use 20261008. Sales features are prior-day sales lags through 56 days, rolling means and standard deviations through 56 days, calendar terms, and lagged price/stockout indicators.

Metrics are aggregated over forecast origins; `mean drug WAPE` is the unweighted average of per-drug WAPE. `pooled WAPE` weights errors by total demand. Predictions are clamped at zero. Synthetic event tags are never model features.

## 90 configuration results

| Configuration | Signal source | Count | Signal lags | Rank lag | Depth | Trees | Learning rate | Min child weight | Lambda | Column sample | Objective | Tweedie power | Max delta step | Seed | Validation mean drug WAPE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| sales_only__shallow_regularized | none | 0 | none | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 34.84% |
| sales_only__shallow_more_trees | none | 0 | none | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.16% |
| sales_only__medium_balanced | none | 0 | none | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.88% |
| sales_only__medium_less_regularized | none | 0 | none | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 39.66% |
| sales_only__deep_regularized | none | 0 | none | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.33% |
| sales_only__shallow_poisson | none | 0 | none | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 30.26% |
| sales_only__shallow_tweedie | none | 0 | none | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 31.82% |
| sales_only__shallow_pseudo_huber | none | 0 | none | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 275355.46% |
| sales_only__legacy_published_recipe | none | 0 | none | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 36.19% |
| all_1312_lag1__shallow_regularized | all | 1312 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 35.23% |
| all_1312_lag1__shallow_more_trees | all | 1312 | 1 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 36.56% |
| all_1312_lag1__medium_balanced | all | 1312 | 1 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.81% |
| all_1312_lag1__medium_less_regularized | all | 1312 | 1 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 39.34% |
| all_1312_lag1__deep_regularized | all | 1312 | 1 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.52% |
| all_1312_lag1__shallow_poisson | all | 1312 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 28.62% |
| all_1312_lag1__shallow_tweedie | all | 1312 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 29.65% |
| all_1312_lag1__shallow_pseudo_huber | all | 1312 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 275331.05% |
| all_1312_lag1__legacy_published_recipe | all | 1312 | 1 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 37.94% |
| top5_all_lag1__shallow_regularized | all | 5 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 29.97% |
| top5_all_lag1__shallow_more_trees | all | 5 | 1 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 31.99% |
| top5_all_lag1__medium_balanced | all | 5 | 1 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 33.33% |
| top5_all_lag1__medium_less_regularized | all | 5 | 1 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 33.96% |
| top5_all_lag1__deep_regularized | all | 5 | 1 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 32.70% |
| top5_all_lag1__shallow_poisson | all | 5 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 24.61% |
| top5_all_lag1__shallow_tweedie | all | 5 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 25.65% |
| top5_all_lag1__shallow_pseudo_huber | all | 5 | 1 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 275331.19% |
| top5_all_lag1__legacy_published_recipe | all | 5 | 1 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 31.50% |
| top5_all_lags_1_7_14__shallow_regularized | all | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 32.58% |
| top5_all_lags_1_7_14__shallow_more_trees | all | 5 | 1,7,14 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 35.31% |
| top5_all_lags_1_7_14__medium_balanced | all | 5 | 1,7,14 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 35.16% |
| top5_all_lags_1_7_14__medium_less_regularized | all | 5 | 1,7,14 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 35.20% |
| top5_all_lags_1_7_14__deep_regularized | all | 5 | 1,7,14 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 34.03% |
| top5_all_lags_1_7_14__shallow_poisson | all | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 25.57% |
| top5_all_lags_1_7_14__shallow_tweedie | all | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 26.19% |
| top5_all_lags_1_7_14__shallow_pseudo_huber | all | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274904.49% |
| top5_all_lags_1_7_14__legacy_published_recipe | all | 5 | 1,7,14 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 34.00% |
| top5_all_lags_1_7_14_28__shallow_regularized | all | 5 | 1,7,14,28 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 39.97% |
| top5_all_lags_1_7_14_28__shallow_more_trees | all | 5 | 1,7,14,28 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 44.09% |
| top5_all_lags_1_7_14_28__medium_balanced | all | 5 | 1,7,14,28 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 47.70% |
| top5_all_lags_1_7_14_28__medium_less_regularized | all | 5 | 1,7,14,28 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 48.15% |
| top5_all_lags_1_7_14_28__deep_regularized | all | 5 | 1,7,14,28 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 46.33% |
| top5_all_lags_1_7_14_28__shallow_poisson | all | 5 | 1,7,14,28 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 28.23% |
| top5_all_lags_1_7_14_28__shallow_tweedie | all | 5 | 1,7,14,28 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 28.72% |
| top5_all_lags_1_7_14_28__shallow_pseudo_huber | all | 5 | 1,7,14,28 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274025.31% |
| top5_all_lags_1_7_14_28__legacy_published_recipe | all | 5 | 1,7,14,28 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 40.15% |
| top10_all_lags_1_7_14__shallow_regularized | all | 10 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 42.33% |
| top10_all_lags_1_7_14__shallow_more_trees | all | 10 | 1,7,14 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 45.57% |
| top10_all_lags_1_7_14__medium_balanced | all | 10 | 1,7,14 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 44.29% |
| top10_all_lags_1_7_14__medium_less_regularized | all | 10 | 1,7,14 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 42.45% |
| top10_all_lags_1_7_14__deep_regularized | all | 10 | 1,7,14 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 44.85% |
| top10_all_lags_1_7_14__shallow_poisson | all | 10 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 30.73% |
| top10_all_lags_1_7_14__shallow_tweedie | all | 10 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 32.49% |
| top10_all_lags_1_7_14__shallow_pseudo_huber | all | 10 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274905.09% |
| top10_all_lags_1_7_14__legacy_published_recipe | all | 10 | 1,7,14 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 43.20% |
| top5_news20_lags_1_7_14__shallow_regularized | model_news_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 32.29% |
| top5_news20_lags_1_7_14__shallow_more_trees | model_news_output | 5 | 1,7,14 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 34.01% |
| top5_news20_lags_1_7_14__medium_balanced | model_news_output | 5 | 1,7,14 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.48% |
| top5_news20_lags_1_7_14__medium_less_regularized | model_news_output | 5 | 1,7,14 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 39.76% |
| top5_news20_lags_1_7_14__deep_regularized | model_news_output | 5 | 1,7,14 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 36.74% |
| top5_news20_lags_1_7_14__shallow_poisson | model_news_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 27.03% |
| top5_news20_lags_1_7_14__shallow_tweedie | model_news_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 26.67% |
| top5_news20_lags_1_7_14__shallow_pseudo_huber | model_news_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274903.21% |
| top5_news20_lags_1_7_14__legacy_published_recipe | model_news_output | 5 | 1,7,14 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 33.15% |
| top5_derived_lags_1_7_14__shallow_regularized | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 34.84% |
| top5_derived_lags_1_7_14__shallow_more_trees | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.16% |
| top5_derived_lags_1_7_14__medium_balanced | derived_demand_output | 5 | 1,7,14 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.88% |
| top5_derived_lags_1_7_14__medium_less_regularized | derived_demand_output | 5 | 1,7,14 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 39.66% |
| top5_derived_lags_1_7_14__deep_regularized | derived_demand_output | 5 | 1,7,14 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 37.33% |
| top5_derived_lags_1_7_14__shallow_poisson | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 30.26% |
| top5_derived_lags_1_7_14__shallow_tweedie | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 31.82% |
| top5_derived_lags_1_7_14__shallow_pseudo_huber | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 275355.46% |
| top5_derived_lags_1_7_14__legacy_published_recipe | derived_demand_output | 5 | 1,7,14 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 36.19% |
| top5_external_lags_1_7_14__shallow_regularized | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 44.49% |
| top5_external_lags_1_7_14__shallow_more_trees | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 46.80% |
| top5_external_lags_1_7_14__medium_balanced | model_external_state_feature | 5 | 1,7,14 | 1 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 45.42% |
| top5_external_lags_1_7_14__medium_less_regularized | model_external_state_feature | 5 | 1,7,14 | 1 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 46.14% |
| top5_external_lags_1_7_14__deep_regularized | model_external_state_feature | 5 | 1,7,14 | 1 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 46.20% |
| top5_external_lags_1_7_14__shallow_poisson | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 34.88% |
| top5_external_lags_1_7_14__shallow_tweedie | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 37.60% |
| top5_external_lags_1_7_14__shallow_pseudo_huber | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274905.32% |
| top5_external_lags_1_7_14__legacy_published_recipe | model_external_state_feature | 5 | 1,7,14 | 1 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 45.37% |
| top5_all_lags_1_7_14_legacy_rank__shallow_regularized | all | 5 | 1,7,14 | 0 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:squarederror | n/a | 0 | 20261008 | 31.33% |
| top5_all_lags_1_7_14_legacy_rank__shallow_more_trees | all | 5 | 1,7,14 | 0 | 2 | 300 | 0.030 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 33.31% |
| top5_all_lags_1_7_14_legacy_rank__medium_balanced | all | 5 | 1,7,14 | 0 | 3 | 220 | 0.040 | 5 | 5 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 34.42% |
| top5_all_lags_1_7_14_legacy_rank__medium_less_regularized | all | 5 | 1,7,14 | 0 | 3 | 350 | 0.050 | 2 | 1 | 0.90 | reg:squarederror | n/a | 0 | 20261008 | 34.71% |
| top5_all_lags_1_7_14_legacy_rank__deep_regularized | all | 5 | 1,7,14 | 0 | 5 | 220 | 0.030 | 8 | 10 | 0.80 | reg:squarederror | n/a | 0 | 20261008 | 33.46% |
| top5_all_lags_1_7_14_legacy_rank__shallow_poisson | all | 5 | 1,7,14 | 0 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | count:poisson | n/a | 0.7 | 20261008 | 24.86% |
| top5_all_lags_1_7_14_legacy_rank__shallow_tweedie | all | 5 | 1,7,14 | 0 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:tweedie | 1.3 | 0 | 20261008 | 25.69% |
| top5_all_lags_1_7_14_legacy_rank__shallow_pseudo_huber | all | 5 | 1,7,14 | 0 | 2 | 120 | 0.040 | 8 | 10 | 0.70 | reg:pseudohubererror | n/a | 0 | 20261008 | 274903.91% |
| top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe | all | 5 | 1,7,14 | 0 | 2 | 220 | 0.040 | 8 | 10 | 0.50 | reg:squarederror | n/a | 0 | 20250915 | 32.95% |

## Selected model and per-drug results

Validation selected `top5_all_lag1__shallow_poisson` at 24.61% mean per-drug WAPE. It achieved 30.66% mean per-drug WAPE and 32.87% pooled WAPE on the final period. The same-profile sales-only XGBoost scored 29.85% pooled WAPE; selected signals changed pooled WAPE by -10.14%. Seasonal-naive scored 33.55% pooled WAPE.

The repository already contains an earlier report in `../data/synthetic_pharmacy_data/current_benchmark_results.md`: top-five signals at 1-, 7-, and 14-day lags were reported at 29.40% WAPE versus 32.35% for sales-only on this synthetic final period. That runner did not purge the 14-day target horizon at the train/test boundary, so its numbers are exploratory and may be optimistic; its XGBoost recipe and selection implementation also differ. This purged expanded grid is reported below and its validation winner is compared with a matched sales-only baseline on final dates.
The prior fixed recipe is also rerun directly with a 14-day purge: top-five raw-value correlation selection, 1/7/14-day signal lags, and the original 220-tree depth-2 XGBoost settings. It scores 29.27% pooled WAPE versus 31.96% for the same-profile sales-only baseline. Its configuration is included in the comparison table; this isolates the effect of the boundary purge from the newer grid's tuning choices.
The historical recipe is saved as a separate inference option in `rebuilt_demand/models/legacy_recipe/` with its own feature manifest, `rebuilt_demand/legacy_recipe_demand_model.json`. Its paired per-drug results are saved in `rebuilt_demand/legacy_recipe_per_drug_test.csv`.

| Drug | Historical recipe WAPE | Matched sales-only WAPE | WAPE reduction |
|---|---:|---:|---:|
| Acetaminophen 500 mg tablet | 44.89% | 50.98% | 11.93% |
| Albuterol HFA inhaler | 56.71% | 50.02% | -13.37% |
| Amlodipine 5 mg tablet | 9.42% | 7.19% | -30.97% |
| Amoxicillin 500 mg capsule | 31.37% | 38.40% | 18.31% |
| Amoxicillin-clavulanate 875/125 mg tablet | 31.30% | 46.80% | 33.11% |
| Atorvastatin 20 mg tablet | 12.12% | 9.95% | -21.79% |
| Azithromycin 250 mg tablet | 20.41% | 50.99% | 59.97% |
| Cephalexin 500 mg capsule | 30.60% | 53.56% | 42.86% |
| Cetirizine 10 mg tablet | 11.03% | 11.32% | 2.55% |
| Doxycycline 100 mg capsule | 28.40% | 72.62% | 60.89% |
| Famotidine 20 mg tablet | 12.23% | 14.79% | 17.31% |
| Fluticasone nasal spray | 14.70% | 16.46% | 10.67% |
| Gabapentin 300 mg capsule | 12.83% | 12.37% | -3.73% |
| Glipizide 5 mg tablet | 12.98% | 14.09% | 7.90% |
| Hydrochlorothiazide 25 mg tablet | 10.23% | 10.86% | 5.82% |
| Ibuprofen 200 mg tablet | 47.18% | 45.29% | -4.16% |
| Insulin glargine 100 units/mL pen | 30.71% | 19.65% | -56.28% |
| Levothyroxine 50 mcg tablet | 13.20% | 14.80% | 10.81% |
| Lisinopril 10 mg tablet | 10.35% | 9.86% | -5.06% |
| Losartan 50 mg tablet | 9.22% | 9.88% | 6.65% |
| Metformin 500 mg tablet | 16.36% | 8.20% | -99.55% |
| Naloxone 4 mg nasal spray | 42.28% | 39.49% | -7.07% |
| Naproxen 500 mg tablet | 41.30% | 41.42% | 0.29% |
| Nirmatrelvir/ritonavir 300/100 mg pack | 72.75% | 73.42% | 0.91% |
| Omeprazole 20 mg capsule | 10.42% | 9.95% | -4.77% |
| Oseltamivir 75 mg capsule | 65.60% | 55.39% | -18.43% |
| Prednisone 20 mg tablet | 51.05% | 46.25% | -10.38% |
| Semaglutide 0.25/0.5 mg pen | 73.50% | 76.36% | 3.74% |
| Sertraline 50 mg tablet | 12.24% | 13.55% | 9.68% |
| Simvastatin 40 mg tablet | 12.95% | 13.57% | 4.56% |

| Drug | Signal model WAPE | Sales-only WAPE | WAPE reduction | MAE | RMSE | Origins |
|---|---:|---:|---:|---:|---:|---:|
| Acetaminophen 500 mg tablet | 51.13% | 49.62% | -3.04% | 152.238 | 276.404 | 206 |
| Albuterol HFA inhaler | 59.91% | 51.77% | -15.73% | 78.567 | 133.777 | 206 |
| Amlodipine 5 mg tablet | 8.67% | 7.20% | -20.54% | 7.391 | 9.064 | 206 |
| Amoxicillin 500 mg capsule | 45.56% | 31.10% | -46.50% | 48.592 | 58.226 | 206 |
| Amoxicillin-clavulanate 875/125 mg tablet | 45.96% | 40.15% | -14.48% | 28.170 | 36.391 | 206 |
| Atorvastatin 20 mg tablet | 11.28% | 9.35% | -20.63% | 9.356 | 11.258 | 206 |
| Azithromycin 250 mg tablet | 37.79% | 44.01% | 14.12% | 21.956 | 28.773 | 206 |
| Cephalexin 500 mg capsule | 40.27% | 34.53% | -16.63% | 21.463 | 25.695 | 206 |
| Cetirizine 10 mg tablet | 11.72% | 11.16% | -5.00% | 6.109 | 7.083 | 206 |
| Doxycycline 100 mg capsule | 46.25% | 49.93% | 7.37% | 23.903 | 30.676 | 206 |
| Famotidine 20 mg tablet | 11.91% | 14.26% | 16.49% | 4.643 | 6.226 | 206 |
| Fluticasone nasal spray | 14.62% | 15.42% | 5.21% | 5.771 | 7.197 | 206 |
| Gabapentin 300 mg capsule | 12.03% | 12.13% | 0.81% | 6.311 | 7.893 | 206 |
| Glipizide 5 mg tablet | 12.40% | 13.34% | 7.02% | 3.997 | 4.889 | 206 |
| Hydrochlorothiazide 25 mg tablet | 10.52% | 9.92% | -6.05% | 6.195 | 7.600 | 206 |
| Ibuprofen 200 mg tablet | 50.60% | 43.71% | -15.75% | 115.701 | 198.949 | 206 |
| Insulin glargine 100 units/mL pen | 27.13% | 19.62% | -38.29% | 4.308 | 5.104 | 206 |
| Levothyroxine 50 mcg tablet | 13.45% | 15.38% | 12.57% | 8.851 | 10.001 | 206 |
| Lisinopril 10 mg tablet | 9.15% | 9.19% | 0.44% | 8.832 | 10.337 | 206 |
| Losartan 50 mg tablet | 8.95% | 9.43% | 5.12% | 5.781 | 7.129 | 206 |
| Metformin 500 mg tablet | 13.02% | 8.41% | -54.81% | 15.171 | 17.612 | 206 |
| Naloxone 4 mg nasal spray | 40.60% | 37.24% | -9.00% | 1.971 | 2.319 | 206 |
| Naproxen 500 mg tablet | 46.10% | 42.43% | -8.67% | 44.222 | 74.368 | 206 |
| Nirmatrelvir/ritonavir 300/100 mg pack | 72.91% | 69.95% | -4.23% | 7.754 | 12.618 | 206 |
| Omeprazole 20 mg capsule | 9.21% | 8.85% | -4.03% | 6.941 | 8.260 | 206 |
| Oseltamivir 75 mg capsule | 64.82% | 58.68% | -10.46% | 10.534 | 16.422 | 206 |
| Prednisone 20 mg tablet | 55.82% | 47.26% | -18.10% | 45.478 | 82.360 | 206 |
| Semaglutide 0.25/0.5 mg pen | 64.42% | 67.38% | 4.39% | 2.461 | 2.962 | 206 |
| Sertraline 50 mg tablet | 10.78% | 12.17% | 11.44% | 6.350 | 7.839 | 206 |
| Simvastatin 40 mg tablet | 12.88% | 12.82% | -0.50% | 4.117 | 5.061 | 206 |

The full configuration objects and per-split metrics are in `rebuilt_demand/comparison_results.json`; the compact table is in `rebuilt_demand/comparison_results.csv`. Per-drug test metrics for the validation-selected model and fixed historical recipe are in `rebuilt_demand/selected_model_per_drug_test.csv` and `rebuilt_demand/legacy_recipe_per_drug_test.csv`.

## Initial 20-signal replacement

The original article-to-signal runner and weights were not found in the inspected Git history. The available 20-column table is a dated output artifact, not per-article training labels. The 3DLNews `.gz` files are Git LFS pointers, but a separate GDELT article corpus is present: 354 records, 292 with full text, across only 10 distinct publication months from 2023-10 through 2025-12. Those sparse month aggregates are insufficient for a defensible article-conditioned model over the 97 monthly target rows. The replacement therefore forecasts each of the same 20 output columns one month ahead from their own historical lags and calendar features. It does not claim to recover news understanding or reproduce the original model.

The historical traces support that boundary: commit `d5e11df` contains a downstream `prod_pipeline.py` that consumes dated 20-signal outputs; `d6ca4b7` archives a README, completion gate, and monthly signal table without an upstream runner or weights; and `6d13c14` summarizes already-extracted `lm_event`/`lm_stage`/`lm_direction` fields. The local news artifact metadata mentions GDELT and FLAN-T5-small, but no matching inference implementation or weights were found. The investigation is detailed in `docs/NEWS_ONLY_INTEGRATION.md`.

It trains on 97 monthly output rows from 2018-01-01 to 2026-01-01 and evaluates the last 20 rows beginning 2024-06-01. Mean holdout MAE is 6.3771 versus 8.9300 for persistence. Per-signal metrics and the 20 next-month output values are saved in `rebuilt_demand/initial_20_metrics.json` and `rebuilt_demand/models/initial_20_signals/next_month_20_signals.csv`.

A separate text-to-output experiment is in `rebuilt_demand/train_news_text.py`. It uses the local GDELT articles and same-month archived outputs; only 10 months overlap (six train, two validation, two test). Its two-month test pooled WAPE was 11.91% versus 87.07% for persistence, with pooled MAE 0.944 versus 6.9. Because the test covers only two months and uses same-month article aggregates, the result is exploratory and not a reliable ahead-of-time forecast. Reproduction outputs are `rebuilt_demand/article_text_20_signal_metrics.json`, `rebuilt_demand/article_text_20_signal_test_predictions.csv`, and `rebuilt_demand/models/initial_20_signals/article_text_model.joblib`.

## Interpretation and limitations

All demand scores are on the repository's deterministic single-site synthetic panel. Performance does not establish accuracy on observed pharmacy sales. The 1,312 table includes model-derived outputs, so the benchmark measures their conditional association with synthetic demand, not independent external predictive value. The historical and rebuilt WAPE results use the same documented final date window and WAPE definition, but differ in feature-selection details and XGBoost settings; compare the recipes before attributing the score difference to one change.
