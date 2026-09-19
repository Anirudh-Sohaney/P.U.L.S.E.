# Codebase guide

## Scope

`arkansas_pharma_signal` builds provenance-bearing public signals for downstream
pharmacy demand, supply-risk, disease, and distribution models. Public inputs do
not observe pharmacy inventory, fill availability, wholesaler allocation, or
backorder status. Forecast rows must preserve this distinction in target
semantics and evidence fields.

## Execution graph

```text
raw public data
  -> normalized data/ artifacts
  -> entity and geography mappings
  -> time-safe feature layers
  -> chronological evaluation
  -> qualified metric serializer
  -> exact-grain feature store
  -> forecast surfaces
  -> publishability audit
```

A target enters downstream modeling only after chronological evaluation and
promotion-gate validation. Missing geography, supplier, factory, API, or county
mappings remain explicit; joins must not broadcast a proxy across an undeclared
grain.

## Package map

| Module | Responsibility |
|---|---|
| `config.py` | Repository paths and artifact directories |
| `io.py` | Checked CSV/JSON reads, writes, and metadata |
| `entities.py` | Drug normalization and FDA identity mapping |
| `canonical_graph.py` | Entity nodes, relationships, and edge provenance |
| `features.py`, `layers.py` | Annual and cross-source feature construction |
| `input_sources.py`, `live_inputs.py` | Public-source manifests and live/refreshable input boundaries |
| `disease_*`, `wastewater_pressure.py`, `respnet.py`, `hospital_respiratory.py` | Disease, respiratory, wastewater, and hospital-context signals |
| `shortage_pressure.py`, `recall_pressure.py`, `supplier_shortage.py` | FDA shortage, recall, and supplier-risk features and evaluators |
| `arcos_evaluation.py`, `supply_chain_pressure.py` | Controlled-substance distribution and supply-chain context |
| `regional_demand_state.py`, `county_demand_state.py`, `therapeutic_class_demand.py` | Arkansas geography and therapeutic-class targets |
| `training.py`, `regression.py`, `neural.py`, `modular_model.py` | Baselines, optional neural components, and the multimodal research path |
| `news_only_adapter.py` | Validated boundary from the separate news-only SLM into annual main-panel features |
| `datasets.py` | Panel construction, encoding, and feature contracts |
| `quarterly.py` | Medicaid quarterly target and next-quarter evaluation |
| `forecast.py` | Legacy city × drug forecast schema |
| `universal_forecast.py` | County × drug × supplier forecast contract |
| `event_schema.py`, `event_extraction.py` | Structured article events and time barriers |
| `metric_audit.py`, `external_metric_output.py` | Qualification and promotion controls |
| `metric_feature_store.py` | Exact-grain or declared-context metric attachment |
| `publishability.py` | Fail-closed publication audit |
| `cli.py` | User-facing pipeline commands |

## Operational commands

Run from the repository root. The package is imported through `PYTHONPATH` or
an editable installation:

```bash
PYTHONPATH=model python -m arkansas_pharma_signal.cli build
PYTHONPATH=model python -m arkansas_pharma_signal.cli train
PYTHONPATH=model python -m arkansas_pharma_signal.cli evaluate
PYTHONPATH=model python -m arkansas_pharma_signal.cli forecast
PYTHONPATH=model python -m arkansas_pharma_signal.cli forecast-universal
PYTHONPATH=model python -m arkansas_pharma_signal.cli audit-publishability
PYTHONPATH=model python -m arkansas_pharma_signal.cli build-news-only-features
```

Additional commands build the graph, corpus, event layer, geography, weather,
county outcomes, external-reference evaluations, and supplier hierarchy. The
many `evaluate_*` scripts are deliberately separate research screens so a
strong result in one signal family cannot hide a weak result in another.
`cli.py` is the command registry; domain calculations belong in focused
modules.

`build-news-only-features` validates the separate monthly signal artifact,
writes a hashed bridge under `model/artifacts/news/`, and exposes annual
`news_only_` columns to panel construction. Rebuilding the panel and retraining
are required before these columns can affect forecasts.

## Maintenance rules

1. Preserve source URLs, retrieval timestamps, effective dates, and
   transformation methods.
2. Keep feature time barriers earlier than target availability.
3. Add tests for schema, grain, missingness, and chronological behavior when
   changing a pipeline boundary.
4. Treat unused imports and unreachable private helpers as removable only after
   repository-wide reference search.
5. Do not delete artifacts or data directories during source cleanup.
6. Update the relevant contract document when a target, output schema, or
   promotion rule changes.
