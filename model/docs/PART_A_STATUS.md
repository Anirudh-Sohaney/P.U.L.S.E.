# Part A Prototype Status

## Objective

Part A is the larger external-signal model. It should use news/NLP and other
public inputs to produce pharmacy-useful signals. For this prototype, a signal
passes when its held-out prediction is within 5% for numeric targets, or is an
exact/true-positive match for state targets, at 70% or higher.

## Current evidence

The dated 20-signal news output has been restored under
`existing_models/news_signal_model/data/derived/signals_monthly.csv`. It has
97 monthly rows from 2018-01 through 2026-01 and no missing values. Only the
derived table and historical completion metadata were available; the original
FLAN-T5 inference code, weights, and validation artifacts are not in this
checkout.

The repository already contains publishable chronological evaluations for more
than ten external signal heads above 70%:

| Signal head | Held-out result | Source type | Status |
| --- | ---: | --- | --- |
| NADAC acquisition cost | 90.08% within 5% | CMS | passes |
| FDA shortage supplier count | 99.18% within 5% | FDA | passes |
| FDA shortage-pressure state | 92.09% exact | FDA | passes |
| National respiratory state | 84.08% exact | CDC | passes |
| Arkansas NSSP influenza state | 73.08% exact | CDC | passes |
| Arkansas regional demand state | 84.18% exact | CMS | passes |
| Arkansas county demand state | 74.33% exact | CMS | passes |
| Arkansas statewide Part D state | 89.09% exact | CMS | passes |
| FDA NDC recall severity | 99.98% within 5% | FDA | passes |
| FDA supplier/NDC recall severity | 99.99% within 5% | FDA | passes |
| Arkansas APCD activity state | 91.83% exact | Arkansas APCD | passes |
| Arkansas ATC demand state | 73.84% exact | HHS/RxNorm | passes |

These results are documented in `TARGET_AVAILABILITY.md`,
`TESTING_PROTOCOL.md`, and `REVIEW_LOG.md`. They use real public data and
chronological held-out folds. They are external proxies, not direct pharmacy
inventory observations.

The current news/event transfer check is weaker: on 2,670 expert-labeled
animal-health article/sentence units, the generic outbreak trigger reached
50.60% exact accuracy, 59.82% balanced accuracy, and 83.43% precision. This
is evaluation-only transfer evidence, not human Arkansas pharmacy truth, and
it does not pass the 70% exact-accuracy requirement.

The first unified news-head experiment is reproducible with
`model/scripts/evaluate_part_a_news.py`. It uses the previous month's 20 news
signals and later HHS, FDA, and CDC observations. The news-only run produced
96 joined months, 18 evaluable nonconstant targets, and 5 promotion candidates
after majority-baseline comparison. It therefore does not independently meet
the ten-head requirement.

An augmented run added each target's previous-month observed value as a
structured input, while keeping news one month behind. After correcting the
CDC period field to use `epiweek` rather than its release `issue` code, the
run evaluated 18 heads and produced **11 promotion candidates**. Four were
CDC FluView level heads, five were national CDC age-band heads, and two were
HHS drug-activity heads. This meets the prototype's 10-head threshold for a
combined external-input layer. The gain is persistence-heavy: the lagged
input is a strong contributor, so this does not prove that news alone caused
the result.

For clarity, these 11 promotion candidates are output heads, not 11
independent news signals. The four CDC level heads are `flu_ar_ili`,
`flu_ar_wili`, `flu_national_ili`, and `flu_national_wili`; the five CDC age
heads are `flu_national_age_0`, `flu_national_age_1`,
`flu_national_age_3`, `flu_national_age_4`, and `flu_national_age_5`; and the
two HHS heads are `hhs_drug_16571040250_claim_lines` and
`hhs_drug_51672407008_claim_lines`. The protocol uses news and observed
history in month *t* to predict the next observed month (*t+1*). Training ends
in 2021-12. CDC heads are evaluated from 2022-01 through 2026-01 (49
months); HHS heads are evaluated through 2024-12 (35--36 months, depending on
available rows). Only 8 of the 11 candidates reach 70% exact accuracy; three
CDC age heads are candidates under the broader precision/majority-gain gate
but score approximately 65--67% exact accuracy. This is therefore not yet
evidence of ten news-generated signals meeting the 70% accuracy goal.

The article-level ten-head NLP experiment is also now reproducible with
`model/scripts/evaluate_expert_news_heads.py`. It trained on 61 articles and
tested on 27 unseen articles (923 train sentences, 321 test sentences). It
evaluated ten expert labels covering current events, risk events, and six
information types. Raw exact accuracy exceeded 70% for several rare labels,
but none beat its majority baseline while also meeting the 70% accuracy or
70% true-positive-precision rule. Its promotion count is **0/10**. Because
the CIRAD corpus has no usable publication timestamps, this is a grouped
article-disjoint NLP transfer test, not a chronological pharmacy forecast.
The full result is stored in `model/artifacts/evaluation/expert_news_heads.json`.

## What is not proven yet

The passing table does not yet prove the intended larger-model design. Most
rows are produced by signal-specific evaluators or persistence-selected
forecasts. The news layer currently performs auditable event extraction and
disease-state aggregation, and a small NumPy neural model exists. The restored
20-column news output is usable as a dated input surface, but there is still no
held-out report showing ten news/NLP/neural-generated signals each meeting 70%
accuracy. Historical completion metadata is retained for provenance only and
is not accepted as current evidence.

Therefore Part A is **candidate-signal ready but architecture-evidence
incomplete**. The next required experiment is a single frozen external-input
layer that emits named signals, followed by one chronological evaluation table
with per-signal targets, baselines, fold scores, and promotion decisions.

## Key limitation

High scores for shortage and recall signals are partly persistence-driven and
should not be presented as proof that news predicts unexpected events. News
must be evaluated against independently observed dated outcomes, with strict
publication-time cutoffs.
