# Historical news-only output

The repository contains a dated 20-column output table from an earlier
FLAN-T5-small news model. The copy used by clean checkouts is
`website/catalog/news_only_catalog_features.csv.gz`; the model adapter can
materialize it under `model/artifacts/news/`. It has 97 monthly rows from
2018-01 through 2026-01. The adapter validates its exact schema, dates, and
numeric values before loading it.

The current `model/arkansas_pharma_signal/` package contains substantial
working model code: source adapters, learned text-relevance and event-state
components, demand and shortage models, chronological evaluators, and forecast
contracts. The exact earlier FLAN-T5 20-column generator is a separate model
path. In the inspected history, `existing_models/news_signal_model/` contains
only its README, a completion-gate report, and the dated CSV output; its
inference runner, model-specific acquisition manifest, and row-level
validation artifacts are not present. Historical values can be queried with
their recorded dates, but the original FLAN-T5 outputs cannot be regenerated
from this checkout.

The older `existing_models/pharmacy_architecture/` variant is also real model
code, but it is not that generator: its `UnifiedPharmacyArchitecture` combines
numeric features with four summarized SLM labels, and its command-line entry
point only prints the accepted-metric inventory. The historic
`model/prod_pipeline.py` is not a safe substitute. Its news loader reads the
dated CSV regardless of the `NEWS_API_KEY`, and its removed ATC inference path
used an all-zero feature vector. The current script fails closed for those
reasons. The separate `news_signals.py` event-state layer is another valid
transformation, but it does not reproduce the legacy 20 columns.

The history review also checked the original `6d13c14` architecture snapshot:
`existing_models/pharmacy_architecture/news.py` only summarizes input rows that
already contain `lm_event`, `lm_stage`, and `lm_direction`; it has no GDELT
client or FLAN-T5 call. The later `d6ca4b7` snapshot adds only the news-model
README, completion-gate metadata, and dated 20-column CSV. The pre-reset Git
bundle points to the same `d5e11df` tree, and the pre-reset workspace archive
contains the same project snapshot without another inference implementation.
No matching model checkpoint is tracked in Git LFS or present among the local
serialized model artifacts. This evidence establishes what is available in
these local and configured Git recovery sources; it does not rule out a copy
in another repository or external archive.

The integrated runner itself is present in the historical Git commit
`d5e11df` (`feat: Add unified prod pipeline...`). It trained downstream heads
from the 20-column dated news table, plus Arkansas ATC and CMS Part D inputs.
That proves the combined pipeline existed; it does not recover the upstream
FLAN-T5 signal-generation model or a serialized checkpoint. Its checked-in
`final_predictions.json` reports 1,250 outputs (20 news, 18 ATC, and 1,212 CMS
drug rows), despite the commit title's claim of 1,406. The historical code
also trains on all available rows and predicts ATC output from a zero vector,
so neither the runner nor its bundled predictions are valid live model output.
An additional `model_report.md` survives only in an unreachable Git tree
(`9bf3831da6654d3632dbb1287df2066c97cfb381`); it repeats the 1,406-output claim
and reports 80.64% exact / 80.96% balanced CMS accuracy. The report's referenced
validation files are absent, and its count conflicts with the 1,250 rows in the
archived JSON. Treat those metrics as unverified historical claims, not a
production evaluation.
The current `model/prod_pipeline.py` therefore remains fail-closed while the
live-input and point-in-time path is rebuilt around the available model code.

This distinction is important for deployment: model code in `model/` is
present and should be reused for its declared targets; only the exact historical
20-column FLAN-T5 inference path is unavailable. Do not relabel outputs from a
different model path as those legacy signal IDs.

## Reproducible historical checks

From the repository root, after hydrating the HHS, FDA, and CDC evaluation
inputs from Git LFS:

```bash
PYTHONPATH=model website/.venv/bin/python model/scripts/evaluate_part_a_news.py --root . --news-only
PYTHONPATH=model website/.venv/bin/python model/scripts/evaluate_part_a_news.py --root .
```

The first command uses only previous-month news columns; the second also uses
the previous observed value of each target. On 2026-10-05, the news-only run
produced five promotion candidates among 18 evaluated public proxy targets,
but only one candidate reached 70% exact accuracy. The augmented run produced
11 candidates, eight of which reached 70% exact accuracy. The difference is
largely persistence from observed target history, not proof of news-only
predictive value. Neither experiment validates a live 20-signal generator or
individual pharmacy inventory demand. See `PART_A_STATUS.md` for the target
definitions and limitations.

`arkansas_pharma_signal.news_only_adapter` can annualize the bundled monthly
history for research panels. The point-in-time boundary remains the dated
historical output; unknown article publication times cannot be inferred from
the monthly rows. A live replacement requires a versioned acquisition and
inference pipeline, source timestamps, and chronological evaluation before it
can be promoted to the public latest endpoint.

The rebuild under `model/rebuilt_demand/` adds a local temporal model that
forecasts the same 20 IDs from their archived histories. Its values can be
merged into the 1,312-signal catalog at their declared month end for the local
synthetic-demand forecast. This is a schema-compatible replacement, not the
historical article-to-signal generator. A separate GDELT corpus is available
under `data/targeted_additions/news_article_corpus/`: it has 354 records, 292
full-text records, but only 10 distinct publication months from 2023-10 to
2025-12. The monthly 20-column artifact has 97 rows and no per-article target
labels, so the text corpus is too sparse to support a defensible news-based
replacement without more monthly coverage and stronger signal-label
provenance. The 14-day pharmacy-demand benchmark and its current holdout
results are documented in `model/comparison.md`.

The GDELT DOC API is not a reliable way to fill the missing article-text
months by requesting a long article list: GDELT's published update says the
long historical search window is available for timeline modes, while other
output modes (including article lists) are restricted to the most recent
three months of a specified search window. See the [GDELT DOC API update](https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/).
The GDELT project separately documents historical bulk data and BigQuery
access, but those contain different records and do not automatically recreate
the full article text used by the archived model. A backfill therefore needs
an independently sourced, legally usable article archive with dated text and
provenance; current-page re-fetches of old URLs are not equivalent training
evidence.

The rebuild also records an exploratory TF-IDF/Ridge article-to-output fit in
`model/rebuilt_demand/train_news_text.py`. It scored 11.91% pooled WAPE on its
last two matched months versus 87.07% for persistence, but only 10 monthly
examples were available and the text is aggregated through month end. The
saved fit is exported to `website/catalog/news_text_20_model.json.gz`. The
worker prefers captured title/summary text for the latest closed month and
falls back to an explicitly tagged current-month-to-date estimate targeted at
month end. This gives the backend a real, reproducible shadow inference path;
it does not recover the historical FLAN-T5 runner. The output uses the same
IDs but remains excluded from usable latest values because the validation
sample is too small and live summaries do not match its full-text training
corpus. On 2026-10-09 it wrote 20 shadow rows from five captured FDA articles
for the October month-end target; these are estimates, not October observations.
