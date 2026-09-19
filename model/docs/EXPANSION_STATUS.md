# Multi-Drug / Disease Expansion Status

Updated 2026-09-14.

## Strongest current path

The current strongest target family is annual Arkansas statewide Medicare Part
D demand by generic drug. The real local panel contains 14,275 rows covering
1,548 drugs from 2013 through 2024. The existing rolling-origin evaluator uses
year `t` to predict the observed target in `t+1`; thresholds are learned only
from the training partition and the output has five ordered demand states.

The rerun produced six chronological folds, 6,998 held-out rows, 89.09%
mean exact accuracy, 89.35% balanced accuracy, and 95.30% precision in the
elevated states. Per-drug accounting now shows 856 drugs with all six scored
test rows and at least 80% exact accuracy; 905 drugs meet the same accuracy
threshold with at least five scored rows. This is enough to support a
10–15-drug prototype surface, subject to reporting the selected drug names,
row counts, fold scores, and target definition.

The artifact is
`model/artifacts/evaluation/arkansas_state_partd_demand_five_state_metrics.json`.
The evaluator now writes `per_drug` results in
`arkansas_pharma_signal.regional_demand_state`.

## What this does and does not prove

This is a real, reproducible Arkansas demand proxy, not pharmacy inventory,
all-payer demand, supply, diagnosis, or county demand. It is annual, so it is
not yet a suitable weekly or monthly replenishment forecast. Its current
features are historical demand features and persistence-selected models; the
annual evaluator has not yet established that the dated news/NLP layer adds
value. The next experiment must join the available dated news features to the
same chronological folds and compare history-only versus history-plus-news
for a fixed, preselected set of drugs.

That ablation has now been run for a fixed 15-drug set. The news overlap only
supports three annual test transitions per drug (45 rows total), so it is not
a final promotion test. History-only scored 80.00% exact accuracy and 75.71%
balanced accuracy; adding the annual news features scored 62.22% exact and
58.66% balanced. The result is rejected for promotion and is evidence that
the news layer needs better temporal coverage, target alignment, or feature
training before it can be claimed to improve drug-demand prediction. It is
saved in `model/artifacts/evaluation/annual_drug_news_ablation.json` and can
be reproduced with `model/scripts/evaluate_annual_drug_news.py`.

## Bounded multi-drug supply screen

To test the supply branch at the requested breadth without counting multiple
NDC packages as separate drugs, the FDA shortage archive was aggregated to
12 distinct generic/formulation labels. Selection used only observed months
through 2015-12, and scoring began after that cutoff. Across 124 rolling
monthly folds and 4,245 held-out next-month transitions, the existing
supplier-count model achieved 98.39% within 5% for all 12 labels. This is a national FDA supplier-count proxy
restricted to NDC evidence exposed to Arkansas, not local inventory or
pharmacy dispensing.

A bounded news ablation added five dated news features to the same history
features. It also scored 98.39% within 5%, with no measured lift: validation
selected persistence in every fold. This supports a 12-label supply-proxy
prototype surface, but not a claim that news improves supply prediction. The
labels include related formulations, so they should not be described as 12
independent active ingredients.

The same folds also exercised the existing NumPy MLP using the five news
features and supply-history features. Its validation-selected output scored
98.39% within 5%, identical to the persistence-selected regression result.
The neural path is therefore implemented and tested, but it has not shown
incremental value on this stable supplier-count target.

For a direct learned-model audit, the evaluator also reports the ridge model
without the persistence fallback. The direct history-only ridge scored 83.85%
within 5%, while direct ridge with the five news features scored 82.34% across
the same 4,245 held-out transitions. Thus the learned news model clears the
70% numeric prototype floor, but news slightly reduces accuracy here and the
neural model does not clear that floor.

The deterministic chronology-safe artifact is
`model/artifacts/evaluation/chronology_safe_12_generic_supply_news_screen.json`.
It is reproducible with `PYTHONPATH=model python3
model/scripts/evaluate_generic_supply_news.py`.

The evaluator also materializes 12 prototype signals in
`model/artifacts/forecasts/chronology_safe_12_generic_supply_news_next_month.csv.gz`.
Each row records the latest observed month, forecast month, predicted active
supplier count, and five lagged news features. Some labels have no observation
in the newest archive month, so their forecast periods differ; this is exposed
in `feature_period` and is not presented as one common live forecast.
The same rows now also expose `direct_ridge_news_prediction`, fit on all
observed transitions, so consumers can compare the learned news signal with
the validation-selected persistence signal.

No disease family currently has equally strong, independently validated
5–8-disease coverage in the repository. FluView is useful as a disease
pressure proxy, but the available Arkansas disease labels do not yet provide
the required multi-disease proof under the current gates.

As a bounded check, the existing CDC FluView weekly panel was evaluated for
Arkansas and national ILI/WILI next-week targets. The national ILI signal
reached 91.51% exact / 80.72% balanced three-state accuracy, and national WILI
reached 86.91% / 77.82%, across seven chronological year folds. These are two
measures of influenza pressure, not two diseases, so they do not satisfy the
5–8-disease requirement and are not promoted as such. The numeric within-5%
route was weak (16.68%–19.32% nationally).

A further national NNDSS screen does not change that conclusion. Twelve
labels with sufficient 2022–2023 training observations were tested on 2024
weekly `m1` counts. Prior-week history plus lagged monthly news produced no
label meeting the 80% three-state gate. The best exact result was 84.31% for
Shigellosis, but its balanced accuracy was only 50.30%; the strongest
history-only exact result, Campylobacteriosis at 84.31%, had 61.05% balanced
accuracy. The complete screen is logged as Pass 426 in `REVIEW_LOG.md`.

## Statewide ARCOS drug/news check

To test a less-stable supply-side target, the real DEA ARCOS Arkansas retail
distribution summary was aggregated across ZIP3 and evaluated for the 15
best-covered controlled-substance codes. The evaluator
`model/scripts/evaluate_arcos_state_news.py` uses training-defined per-drug
low/mid/high terciles, prior-quarter distribution history, and quarterly
Arkansas 3DLNews keyword counts to predict the next calendar quarter. Drug
selection is restricted to observations available through 2018, and scoring
is restricted to 2013-Q1 through 2022-Q4, the dated-news coverage window.

Across 9 chronological four-quarter folds, the selected system scored 61.14%
exact / 50.83% balanced accuracy. The news logistic candidate scored 62.76%
exact / 53.89% balanced accuracy, versus 43.40% / 33.33% for persistence.
News therefore adds a measurable but insufficient lift on this target; it
does not meet the 80% three-state requirement. The result is stored in
`model/artifacts/evaluation/arcos_state_news_metrics.json`. ARCOS grams remain
a statewide distribution proxy, not pharmacy inventory or dispensing.

## Monthly news-aligned check

The existing monthly HHS Arkansas pharmacy-NDC panel was tested with a fixed
15-drug selection, month `t` features predicting month `t+1`, and a separate
history-only versus history-plus-news comparison. The training period was
2018-01 through 2021-12 and the test period was 2022-01 through 2024-12.

After adding a 2021 validation year for regularization selection, restricting
drug selection to the training period, and learning one shared low/medium/high
scale across the selected drugs, 10 drugs had at least 30 training rows, 10
validation rows, and 25 test rows. Across 344 held-out drug-month rows,
history-only scored 68.60% exact / 65.67% balanced accuracy. After selecting
the number of news variables on the validation year using training-only
rankings, history-plus-news scored 65.12% exact / 63.45% balanced accuracy.
This reaches the exploratory 10-drug count but fails the 80% three-state
requirement, and news reduces accuracy in this configuration. It is not
promoted. The artifact is
`model/artifacts/evaluation/monthly_drug_news_ablation.json`; the runner is
`model/scripts/evaluate_monthly_drug_news.py`.

## Extended Arkansas newspaper check

The local 3DLNews2 newspaper panel provides 16 documented keyword groups and
coverage from 2013–2022, although its pre-2016 coverage is sparse. A fixed
15-drug CMS annual demand experiment used 2013–2018 for training, 2019 for
validation context, and 2020–2022 target years for testing (45 held-out
drug-year rows). History-only scored 97.78% exact / 98.25% balanced accuracy;
adding 3DLNews keyword counts scored 77.78% exact / 77.22% balanced accuracy.
This is not promoted: it has only three test transitions per drug and news
reduces performance. The result is saved in
`model/artifacts/evaluation/annual_3dlnews_drug_ablation.json` and can be
reproduced with `model/scripts/evaluate_annual_3dlnews_drug.py`.

A full-panel check used all 1,548 statewide generic drugs and five held-out
years (2018--2022), with raw annual news counts and validation-selected
regularization. History averaged 77.56% exact / 78.02% balanced five-state
accuracy; the news model averaged 42.25% / 42.29%. Raw annual counts therefore
overfit this panel and are rejected. Future news work must normalize or model
news residuals rather than append these counts directly.

## Full CMS panel with restored NLP signals

The existing dated 20-column NLP signal surface was aggregated by year with a
log1p transform and evaluated on the full CMS panel. Training and validation
were chronological, with feature year *t* predicting demand in *t+1*; the
three test feature years were 2021--2023 (targets 2022--2024). Across 1,368
generic-drug targets and 3,575 held-out rows, history-only scored 68.50%
exact / 69.25% balanced five-state accuracy. The news model scored 80.64% /
80.96%, using validation-selected regularization. This is a qualifying
prototype result and demonstrates incremental news value, although the target
is annual CMS Part D demand rather than direct pharmacy dispensing or
inventory. The reproducible artifact is
`model/artifacts/evaluation/cms_partd_full_news_metrics.json`; the runner is
`model/scripts/evaluate_cms_partd_news.py`.

The same runner materializes the next-year signal surface at
`model/artifacts/forecasts/cms_partd_full_news_next_year.csv.gz`: 1,324 rows
use 2024 observed demand and dated NLP features to predict 2025 demand states.
Rows include all 20 news columns, the forecast state, and model/source fields.
This is a research signal surface; it is not a direct pharmacy inventory
forecast.

## Supply-side audit

The FDA archive is a useful longer-history source, but the candidate
per-NDC histories have only two observed supplier-pressure states in the
pre-test window after the news overlap is applied. A per-drug five-state news
model therefore cannot be fit without inventing labels or using the test
period to select targets. No supply-plus-news result was promoted, and no
temporary runner was retained.

An NDC9-level variant was also screened to test whether the 12-label breadth
could represent independent products. It scored 98.19% within 5% for the
validation-selected system across 4,464 held-out transitions, but the selected
NDC9s collapse to only three active ingredients (leucovorin, fentanyl, and
atropine). It is therefore not promoted as a 12-drug result.

The related observed-supplier-count proxy was also screened for the same
12 chronology-safe labels. It produced only 22 non-equal transitions across
1,997 label-month rows (1.10%), compared with 25 active-supplier transitions
(1.25%). It is therefore not a more eventful or defensible news target.

## Aggregate pharmacy utility check

The existing context-utility evaluator was rerun on the real HHS Arkansas
pharmacy panel joined to FDA shortage, Arkansas news, FluView, and RESP-NET
inputs. It used 59 chronological folds and 13,552 held-out rows across 1,619
drugs. The history-only base had mean WAPE 0.5933; adding the public context
bundle had mean WAPE 0.5996, a deterioration of 0.0063. Within-5% accuracy
was 8.25% for the base and 8.32% with context, and context did not improve
every fold. This does not demonstrate incremental news value and is not
promoted. The artifact is
`model/artifacts/evaluation/hhs_monthly_context_utility_metrics.json`.

## Signal-surface materialization

The existing qualified forecast builder was rerun using the available county
demand test panel and current public source files. It produced 43,366 rows
across 14 target types with zero duplicate keys, including 1,212 statewide
annual drug-demand rows and 3,303 regional annual drug-demand rows. The
materialized file is
`model/artifacts/forecasts/qualified_metric_forecasts.csv.gz`.

This is a consumable proxy-signal surface, not proof that the learned news
model meets the active accuracy requirement. The project audit still reports
the learned architecture and qualified metric library as incomplete.

## Disease-data audit

The local NNDSS file cannot currently support the requested Arkansas disease
forecast. Although its source metadata says 2022–present, the Arkansas rows
in the checked-in extract contain only 2025 and 2026 observations; 2022–2024
rows are national. Therefore there is no Arkansas training period for a
chronological five-to-eight-disease test. No disease evaluator or result was
retained from this check.

After regenerating the metric-library audit, the project audit now recognizes
1 qualified proxy metric and passes target-validity and operational-surface
checks. It still correctly fails proxy coverage breadth and learned
end-to-end accuracy, so this is progress in artifact completeness, not a
completion claim.
