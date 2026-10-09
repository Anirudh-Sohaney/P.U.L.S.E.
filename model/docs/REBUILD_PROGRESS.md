# Model rebuild progress

**Last updated:** 2026-10-09  
**Status:** Local synthetic proof of concept is runnable. Article-text 20-signal inference is wired to the backend as a quarantined shadow path; real-world production validation is outstanding.

## Current result

The local demand forecaster predicts 14-day unit totals for 30 synthetic drug series from sales history and dated catalog signals. The catalog has 1,312 signal columns. The default saved recipe is the fixed historical top-five signal approach, rebuilt with a 14-day purge between training labels and the test period.

| Evaluation | Signal model | Matched sales-only | Result |
|---|---:|---:|---|
| Fixed historical recipe, purged final period | 29.27% pooled WAPE | 31.96% pooled WAPE | 8.43% relative WAPE reduction; 18 of 30 drugs improved |
| 90-grid validation winner, purged final period | 32.87% pooled WAPE | 29.85% pooled WAPE | 10.14% relative WAPE regression |
| Archived 20-output temporal model, final 20 months | 6.3771 mean MAE | 8.9300 persistence MAE | Lower error on the archived monthly values |
| Article text to archived 20 outputs, two-month holdout | 11.91% pooled WAPE | 87.07% persistence WAPE | Exploratory only; only 10 months matched labels |

The purged historical recipe closely reproduces the earlier synthetic result of 29.40% versus 32.35%. Its comparison uses the repository's existing final date window and synthetic data; it is not evidence on observed pharmacy sales.

## Work completed

- On 2026-10-09, corrected the separate annual city/drug model's blend calibration in `arkansas_pharma_signal/cli.py`. The old training command used a full-data ridge to select its last-year validation weight, leaking those labels. The new command freezes identity categories before the validation feature year and selects the weight from a ridge fit on prior years only. On the local 544,070-row 2013–2024 panel, the corrected 2023 feature-year validation chose ridge weight 0.00 at WAPE 0.14846; the old 0.70 weight and 0.14609 score were contaminated. The regenerated artifact still fails the operational input gate for 56 source variables. This calibration repair does not validate source release timing or make the annual model live-ready.

- Inspected current model code, local archived outputs, synthetic benchmark procedures, and reachable Git history. The history shows downstream code consuming the 20 monthly outputs, but no recovered upstream FLAN-T5 inference runner or matching checkpoint.
- Found 97 archived monthly rows for the 20 outputs (2018-01 through 2026-01). Trained 20 temporal XGBoost models on their own output lags and calendar features. These preserve the signal IDs and output schema; they do not reproduce article understanding.
- Found a separate GDELT article corpus with 354 records, 292 containing full text, but only 10 distinct publication months matching the archived labels. Trained a small TF-IDF/Ridge text experiment and recorded its metrics without promoting it.
- Exported that locally saved TF-IDF/Ridge model into a checksum-pinned JSON artifact and added a NumPy-only backend loader. A parity replay over all ten matched text months reproduced the saved sklearn outputs to a maximum absolute error of `3.38e-14` after the same nonnegative clipping.
- Added a daily shadow refresh for the latest closed UTC month. It uses only captured article titles and summaries, preserves the 20 catalog IDs, records model estimates as unvalidated/unusable observations, and leaves the original FLAN-T5 path distinct. The local source database currently has only five article versions from 2026-10-05 through 2026-10-08 and no captured text for the previous closed month, so it cannot yet generate a current monthly estimate.
- Trained and compared 90 configurations across 30 synthetic drug series. The grid varies signal groups/counts and lags, correlation-ranking alignment, tree depth, estimator count, learning rate, regularization, column sampling, and squared-error, Poisson, Tweedie, and pseudo-Huber objectives.
- Kept the validation-selected grid winner separate from the fixed historical recipe. Both model sets have saved weights, feature manifests, and a selectable local inference path.
- Verified Python compilation, report regeneration, and local inference for both demand recipes. Each inference run emitted 30 forecasts. `git diff --check` passed at the last code review.

## Main artifacts

- Training and report generation: [`../rebuilt_demand/train.py`](../rebuilt_demand/train.py)
- Local demand inference: [`../rebuilt_demand/predict.py`](../rebuilt_demand/predict.py)
- Article-text experiment: [`../rebuilt_demand/train_news_text.py`](../rebuilt_demand/train_news_text.py)
- Human-readable experiment comparison: [`../comparison.md`](../comparison.md)
- Folder guide and run commands: [`../README.md`](../README.md)
- Historical news-source findings: [`NEWS_ONLY_INTEGRATION.md`](NEWS_ONLY_INTEGRATION.md)

## Remaining gaps

1. The exact historical FLAN-T5 article-to-20-signal runner/checkpoint remains unavailable. The local TF-IDF/Ridge replacement is wired only as a shadow and does not claim to reproduce it.
2. Ten matched article months are not enough to evaluate a reliable text-to-signal model. More dated, legally usable article text and signal labels are needed; the current feed summary input also differs from the archived full-text training corpus.
3. Demand accuracy has only been evaluated on the deterministic single-site synthetic panel. Evaluation against observed pharmacy sales, inventory, and stockout outcomes is required before operational use.
4. The 1,312 catalog includes derived/model outputs; the synthetic study measures their association with generated demand and does not establish independent real-world predictive value.

## Update rule

After each substantive model or data change, add the date, changed artifact, evaluation protocol, measured result, and any newly resolved or remaining limitation here. Keep `comparison.md` as the detailed per-configuration scorecard and this file as the running project log.
