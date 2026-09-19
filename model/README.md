# Arkansas Pharmaceutical Signal Model

This directory contains the model architecture, data contracts, training code, and review history for an Arkansas-first pharmaceutical demand, supply-risk, and shortage-risk forecasting system. Arkansas remains the primary deployment test bed; national, neighboring-state, and global inputs are added as contextual signals rather than silently treated as Arkansas observations.

## Current status (September 2026)

The repository has a working, provenance-aware signal and forecasting stack,
but it is still a research system. It currently supports demand, disease,
news, weather, recall, shortage, supplier, distribution, price, economic, and
trade-related features at several time and geography levels. The strongest
near-term path is the quarterly Medicaid demand proxy. The large multimodal
neural model is implemented and testable, but it has not earned promotion over
the simpler time-split baselines.

The current evidence does not prove pharmacy inventory improvement. Public
targets are proxies because the local data does not observe on-hand inventory,
stockouts, backorders, or wholesaler allocations. Every new signal therefore
needs a point-in-time-safe target, a persistence or seasonal baseline, enough
chronological held-out observations, and a reported promotion decision.

The latest focused local audit passes the expert-gold, input-contract, and
ARCOS evaluation tests (`52 passed`, including CLI coverage). The full suite and forecast-quality
gates still need to be run after each expansion. Passing tests are not evidence
that the neural model is accurate; do not describe the project as
production-ready until the forecast gates and pharmacy-impact evidence are
resolved.

The latest strict operational annual evaluation used 173 features and selected
the calibrated ridge bridge. Its test WAPE was `0.145785` versus `0.146271` for
the strongest naive baseline, only `0.33%` improvement. The new ARCOS context
is therefore integrated and measurable, but it is not yet a qualified
inventory-impact improvement.

## Universal intelligence layer

The current deployable upstream layer is exposed through three additional
commands:

```bash
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-graph --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-news-corpus --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli forecast-universal --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-geography --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-events --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-news-signals --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-supplier-hierarchy --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli validate-external-reference --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-expert-event-gold --root .
PYTHONPATH=model python -m arkansas_pharma_signal.cli audit-publishability --root .
```

`build-graph` writes a versioned entity graph with source, effective dates,
confidence, transformation method, and direct-versus-inferred evidence on
every edge. `build-news-corpus` restores available historical JSONL article
text while retaining metadata-only records with `is_full_text=false`.
`forecast-universal` writes `artifacts/forecasts/universal_forecast.csv` at
the county × drug × supplier contract. Unresolved county, parent, factory,
and API mappings stay explicit and lower confidence; they are never silently
filled from a city or labeler proxy.
FDA establishment matches are available separately in
`artifacts/suppliers/establishment_context.csv.gz` as supplier-level context;
they are not promoted to product-specific factory/API edges.
`build-expert-event-gold` creates an evaluation-only transfer benchmark from
public animal-health expert annotations. `audit-publishability` is the final
fail-closed acceptance report; it does not promote the model while rolling-
origin gates are missing.

The historical universal-grid implementation contains a binary `shortage_state`
training diagnostic whose threshold is selected on the strict validation year
and persisted in `artifacts/trained/models.json`. It is not emitted on the
current forecast surface and cannot satisfy the revised final-metric contract;
promoted shortage outputs must use a numeric value or at least five ordered
states. Evaluation still reports raw and balanced accuracy, precision, recall,
and F1 for this legacy diagnostic because its underlying shortage label is
sparse.
It also emits observed-event-only `neighbor_state_supply_event_risk` rows for
the six states bordering Arkansas; these rows carry no copied Arkansas demand
label.

Layer 1 is materialized by `build-news-signals` as
`artifacts/news/layer1_news_state_features.csv.gz`. It emits 300 variables for
each observed date and geography: 50 event-family states plus four
presence/outbreak-risk/spread-rate/uncertainty states for 62 named diseases. These are
news-derived external variables, not final forecasts, and the time barrier
excludes future articles during replay.

The model is designed to produce a large, filterable forecast table rather than a single forecast. Each output row is keyed by:

- forecast date and horizon,
- Arkansas geography: county, region, city, ZIP3, HSA/HRR where available,
- drug ingredient or generic drug,
- supplier/labeler/manufacturer when a public mapping exists,
- disease or event driver when attributable,
- forecast target: demand, demand shock, supply risk, shortage risk, or shortage impact.

The implementation must use real local or public data only. It must not generate synthetic training data.

## News-only SLM integration

The separate news-only FLAN-T5 signal model is connected through
`news_only_adapter.py`. Its 20 validated monthly signals are annualized and
added to the main panel with the `news_only_` prefix. The adapter is optional
and fail-closed: schema, date, duplicate, and numeric-value errors stop the
build rather than silently changing the feature contract.

Build the bridge artifact before rebuilding the main panel:

```bash
cd existing_models/news_signal_model
/tmp/newsenv/bin/python run_extract.py
cd ../..
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . build-news-only-features
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . build-panel
```

See [News-only integration](docs/NEWS_ONLY_INTEGRATION.md) for the source,
citations, validation boundary, and retraining requirements. These signals are
indirect public-news proxies, not direct pharmacy inventory observations.

## Core Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Research Basis](docs/RESEARCH_BASIS.md)
- [Implementation Contract](docs/IMPLEMENTATION_CONTRACT.md)
- [Testing Protocol](docs/TESTING_PROTOCOL.md)
- [Codebase Guide](docs/CODEBASE_GUIDE.md)
- [Review Log](docs/REVIEW_LOG.md)
- [Target Availability](docs/TARGET_AVAILABILITY.md)
- [Input Variable Audit](docs/INPUT_VARIABLE_AUDIT.md)
- [Metric Research Matrix](docs/METRIC_RESEARCH_MATRIX.md)

## Expansion direction

New work should preserve the existing Arkansas models and add reusable signal
adapters. The preferred order is: refresh freely available weekly or monthly
data; add national and neighboring-state context; join signals by drug,
supplier, geography, and publication date; then compare each addition with a
simple baseline. News remains a supporting feature because structured CDC,
FDA, CMS, DEA, weather, and supply-chain data are easier to validate.

## Universal filtered forecast

The deployable county/drug/supplier surface is generated with:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli forecast-universal \
  --root . --max-rows 10000 \
  --county-fips 05001 05003 --region central --supplier "Example Labeler"
```

Filters are applied before grid expansion. When `model/artifacts/trained/models.json`
exists, demand and shortage-risk predictions use the saved validated models;
otherwise the output identifies itself as an exposure heuristic. County and
supplier hierarchy gaps remain explicit in `confidence`, `evidence_type`, and
the empty hierarchy fields.

The universal output also includes exact statewide supplier-drug context rows
under `geography_level=arkansas_supplier_drug`. These rows preserve supplier
and normalized-drug identity without inventing a county allocation; their low
confidence and unverified Arkansas allocation are explicit. The `--supplier`
filter applies to both supplier context surfaces.

For controlled-substance regional distribution, the output also includes
`regional_distribution_pressure_forecast` ZIP3 rows at a 91-day horizon. These
rows use the separately evaluated ARCOS model and remain explicitly labeled as
distribution proxies, not direct pharmacy demand or inventory observations.

The current artifacts remain fail-closed for production publication because
the strict rolling-origin demand and modular-model gates are not yet met. The
external expert event benchmark now contains 2,670 explicitly typed transfer
units; it is not Arkansas human-pharmacy truth. BAND provides independent
human-health recall evidence:
the current trigger recalls 96.67% of its 150 positive outbreak contexts and
the bounded disease lexicon matches 100% of its 271 supported expert disease
mentions; this is recall-only evidence, not a specificity or accuracy claim.
The independent DAnIEL English token test also recovers 20/20 supported disease
spans; its four unsupported spans are reported without relabeling them.
The held-out shortage boolean evaluation reports 83.56% raw accuracy and
71.05% balanced accuracy on the refreshed strict test split. Because the label
is sparse and rolling logistic performance does not beat persistence, this is
not treated as sufficient supplier-event evidence for publication.
The supplier research view also exposes prior-only drug-wide FDA event lags and
rolling counts; these were retained for reproducibility but did not beat the
persistence baseline in rolling evaluation.
The numeric demand target is not claimed: under the strict consecutive-
quarter contract, the refreshed pointwise quarterly result is 9.40% WAPE
and 27.6% within 5%. The six-fold one-year rolling mean improvement is 17.6%,
but the unrestricted future-horizon rolling mean is 9.0%, below the 10% gate.
The qualified quarterly input set now also includes exact-NDC CMS NADAC
weekly price pressure; its measured uplift is neutral and it remains an
auditable input rather than a claimed performance improvement.

`forecast-quarterly --live-sdud-url <official-CMS-CSV>` refreshes the current
target in memory without changing the historical cache. Repeat the flag for
multiple annual CMS files when building a recent target panel. The source URLs
and retrieval dates are recorded in
`model/artifacts/metadata/quarterly_forecast.json`.

For one fail-closed report covering the public suite and, optionally, the
end-to-end layered demand head, run:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_whole_system.py \
  --end-to-end --rolling
```
