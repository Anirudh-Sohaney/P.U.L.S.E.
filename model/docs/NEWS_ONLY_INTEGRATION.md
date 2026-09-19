# News-only SLM integration

The separate news-only model is integrated as an upstream feature source; it
is not copied into or entangled with the existing event-extraction code. The
boundary is `arkansas_pharma_signal.news_only_adapter`.

## Data boundary and provenance

The source feature table is:

```text
existing_models/news_signal_model/data/derived/signals_monthly.csv
```

It contains dated monthly output from the FLAN-T5-small extraction pipeline:
20 named news signals, no pharmacy target columns, and no future target
values. The adapter validates the exact schema, date uniqueness, numeric
types, and signal count before use. Its materialized architecture artifact is
generated at:

```text
model/artifacts/news/news_only_catalog_features.csv.gz
model/artifacts/news/news_only_catalog_features.json
```

The JSON metadata records source/output SHA-256 hashes, signal IDs, row count,
and citations. The upstream model's acquisition manifests retain article
URLs, retrieval dates, and source hashes. Source citations are GDELT
([gdeltproject.org](https://www.gdeltproject.org/)) and
FLAN-T5-small ([Hugging Face model card](https://huggingface.co/google/flan-t5-small));
its independent CDC validation is documented separately in
`existing_models/news_signal_model/docs/RESULTS.md`.

## Execution order

Run from the repository root:

```bash
# 1. Rebuild the independent news-only monthly/quarterly output when inputs change.
cd existing_models/news_signal_model
/tmp/newsenv/bin/python run_extract.py

# 2. Return to the repository root and materialize the validated bridge.
cd ../..
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . build-news-only-features

# 3. Build the existing annual panel. The bridge is joined by feature year.
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . build-panel

# 4. Retrain existing demand/risk models so their saved feature contract
#    includes the news-only columns, then generate forecasts.
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . train
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . forecast --max-rows 10000
```

`build-news-only-features` fails closed if the source is absent or malformed.
`build-panel` remains backwards-compatible when the optional source is absent.
The adapter annualizes monthly counts by sum; the existing chronological
evaluators control the feature/target time barrier. News-only columns are
classified as near-real-time inputs and appear in the existing model's news
ablation and driver-attribution surfaces under the `news_only_` prefix.

## Interpretation

The merged model can use the news-only SLM alongside the existing disease,
weather, economic, shortage, supplier, and historical-demand layers. Its
signals remain indirect demand/supply proxies. They must not be described as
individual-pharmacy stock-on-hand observations, and model retraining is
required after changing the bridge because serialized feature contracts are
intentionally immutable.
