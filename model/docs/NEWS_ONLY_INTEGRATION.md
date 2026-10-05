# Historical news-only output

The repository contains a dated 20-column output table from an earlier
FLAN-T5-small news model. The copy used by clean checkouts is
`website/catalog/news_only_catalog_features.csv.gz`; the model adapter can
materialize it under `model/artifacts/news/`. It has 97 monthly rows from
2018-01 through 2026-01. The adapter validates its exact schema, dates, and
numeric values before loading it.

The original `existing_models/news_signal_model` directory, inference code,
weights, article acquisition manifests, and validation artifacts are absent
from this checkout and its inspected Git history. Historical values can be
queried with their recorded dates, but the current daily worker cannot
regenerate them. They must not be published as current news-model signals.
The `news_signals.py` event feature layer is a different transformation and
does not reproduce these 20 columns.

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
