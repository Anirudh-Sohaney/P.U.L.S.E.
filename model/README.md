# Arkansas Pharmaceutical Signal Model

This directory contains the code, tests, research notes, and generated artifacts for an Arkansas-first system that builds public-data signals and evaluates forecasts related to pharmaceutical demand, disease pressure, drug shortages, recalls, and distribution. It combines data adapters, identity and geography mapping, time-aware feature construction, baseline and research models, chronological evaluation, and guarded output contracts.

The system is designed as a contextual signal layer for pharmacy and public-health analysis. It is **not** a source of observed pharmacy stock-on-hand, wholesaler allocation, backorder, or stockout truth: the public datasets used here generally do not record those outcomes. Any “demand,” “supply risk,” “shortage pressure,” or similar value must be interpreted according to the target and evidence fields attached to it. A metric being forecastable does not establish that it improves private pharmacy inventory decisions.

This README is the entry point to the codebase. The detailed evidence, source citations, target definitions, and contracts remain in `docs/` and in the data-source manifests under `../data/`.

## What the system does

At a high level, the code turns dated public observations into traceable, model-ready signals and then into evaluated, filterable forecast surfaces:

```text
public/local source files
        │
        ▼
source adapters, schema checks, and provenance
        │
        ├── drug/entity and geography resolution
        ├── structured event and news-signal extraction
        └── cadence-specific target and feature panels
        │
        ▼
point-in-time-safe joins and historical features
        │
        ├── simple persistence / seasonal baselines
        ├── regression and state models
        └── optional multimodal neural research model
        │
        ▼
chronological holdouts, rolling-origin evaluation, and ablations
        │
        ▼
qualified metric rows, feature store, forecasts, and publishability audit
```

Arkansas is the primary local evaluation setting. National, neighboring-state, and global observations may provide context, but the code should preserve their original geography and only project them when an explicit, auditable mapping permits it. Unresolved product, supplier, county, factory, API, or parent-company links are kept unresolved rather than silently inferred.

## Folder architecture

| Path | What it contains |
|---|---|
| `arkansas_pharma_signal/` | Main Python package. Source adapters, feature and panel builders, target-specific evaluators, models, output validators, and the command-line interface live here. |
| `tests/` | Unit and contract tests for schemas, chronology, joins, model behavior, target evaluation, output validation, and CLI-related behavior. |
| `docs/` | Design and research documentation: architecture, data sources, implementation and target contracts, testing protocol, availability limitations, status, and review history. |
| `scripts/` | Standalone audit and reproducibility helpers, including metric-library and project-status checks. These complement, but do not replace, the package CLI and pytest suite. |
| `artifacts/` | Generated local outputs, including panels, trained model metadata, forecast tables, news artifacts, evaluation reports, and run metadata. Contents depend on which commands have been run and are not all source-controlled or reproducible without the corresponding input data. |
| `prod_pipeline.py` | A separate legacy historical-fit script. Its forecast and CLI emission paths fail closed because current news-model and ATC inputs are unavailable. It is not the same pipeline as the package CLI. See [Legacy JSON output](#legacy-json-output). |
| `final_predictions.json` | Checked-in legacy example; a compact collection of categorical signal predictions, not the canonical validated forecast table. The legacy script no longer overwrites it. |
| `README.md` | This overview and operating guide. |
| `docs.md` | Short additional documentation entry point; the detailed documents are in `docs/`. |
| `pyproject.toml` | Python package metadata, required and optional dependencies, and pytest configuration. |
| `.gitignore` | Local artifact/cache exclusions. |

### Package module guide

The package is deliberately split by data boundary and responsibility. Modules are grouped below by their role; several specialized adapters have their own files and tests.

#### Execution, configuration, and data contracts

| Module(s) | Responsibility |
|---|---|
| `cli.py` | Command registry and orchestration for panel builds, training, forecasts, evaluations, signal construction, and audits. |
| `config.py` | Repository-root resolution and canonical locations for inputs and generated artifact directories. |
| `io.py` | Checked CSV/JSON reading, writing, and metadata helpers. |
| `input_contract.py` | Input coverage, disposition, and eligibility summaries used to make input scope explicit. |
| `datasets.py` | Main panel/next-period dataset preparation, feature-mode selection, and ablation feature selection. |
| `features.py`, `layers.py` | Annual feature and cross-source feature construction. |
| `entities.py` | Drug-name normalization and product/ingredient identity helpers. |
| `geography.py` | Arkansas geography crosswalks, county resolution, and mapping utilities. |
| `canonical_graph.py` | Versionable entity graph with typed relationships and edge-level provenance. |
| `event_schema.py` | Schemas and semantics for structured events. |
| `publishability.py`, `external_metric_output.py` | Fail-closed checks for whether metrics and forecast rows meet their declared contracts. |

#### Public data adapters and contextual signal families

| Module(s) | Responsibility |
|---|---|
| `input_sources.py`, `live_inputs.py` | Source definitions and explicitly refreshable public inputs, including selected CMS, FDA, and weather feeds. |
| `news_corpus.py`, `event_extraction.py`, `event_schema.py` | Load available article text and metadata, extract typed events, and retain date/evidence boundaries. |
| `news_signals.py`, `news_relevance.py`, `text_signals.py`, `expert_news_heads.py`, `learned_event_state.py` | News relevance, disease/event signal construction, text features, and associated experimental/evaluation paths. |
| `news_only_adapter.py` | Validated bridge for the independent FLAN-T5-small news-only monthly signal table. It has a strict schema boundary and prefixes bridged features with `news_only_`. |
| `weekly_health_proxy.py`, `nssp_respiratory.py`, `respnet.py`, `hospital_respiratory.py`, `wastewater_pressure.py`, `regional_wastewater.py`, `overdose_pressure.py` | Disease, respiratory, wastewater, hospital-utilization, and overdose context adapters/evaluators. These are health-pressure proxies, not medication inventory labels. |
| `historical_weather.py` | Historical weather loading and aggregation with observed coverage retained. |
| `shortage_pressure.py`, `supplier_shortage.py`, `recall_pressure.py`, `arkansas_exposure.py` | FDA shortage, supplier-event, recall, and Arkansas exposure signals and evaluation. |
| `arcos_evaluation.py`, `supply_chain_pressure.py`, `supplier_hierarchy.py` | Controlled-substance distribution, supply-chain context, and supplier hierarchy construction. |
| `nadac_target.py`, `demand_price`-related adapters | Drug price and acquisition-cost targets/context where supported by the local data. Price is not treated as demand or shortage truth. |
| `medicaid_demand.py`, `monthly_pharmacy_demand.py`, `quarterly.py`, `cms_partd_quarterly.py`, `county_demand_state.py`, `regional_demand_state.py`, `therapeutic_class_demand.py`, `county_outcomes.py` | Arkansas/public demand proxies at their native monthly, quarterly, annual, county, region, or therapeutic-class grains. |
| `apcd_claim_counts.py`, `historical_metric_context.py`, `metric_feature_store.py` | Claim-activity and historical metric-context experiments, plus controlled attachment of qualified metric rows to downstream features. |

#### Models, evaluation, and reporting

| Module(s) | Responsibility |
|---|---|
| `regression.py`, `training.py`, `forecast.py`, `evaluate.py` | Baseline and regression training, operational annual forecast construction, and strict/rolling evaluation. |
| `neural.py` | Smaller neural/blending components used by the existing evaluation/training path where enabled. |
| `modular_model.py` | Larger modular multimodal research architecture (described below); its implementation is not, by itself, evidence of forecast quality or production promotion. |
| `universal_forecast.py` | Broad county × drug × supplier forecast contract with evidence, grain, horizon, confidence/uncertainty, and promotion metadata. |
| `publishable_evaluation.py`, `publishable_model_evaluation.py`, `publishable_test_dataset.py`, `external_validation.py` | Evaluation datasets and checks for chronological performance, external references, and publication eligibility. |
| `metric_audit.py`, `qualified_metric_forecasts.py`, `metric_feature_store.py`, `cross_signal_utility.py`, `signal_utility.py` | Metric qualification, utility/ablation analysis, and safe conversion/attachment of qualified forecast metrics. |
| `project_status.py`, `annotation.py`, `expert_gold.py`, `event_accuracy.py` | Status, annotation, expert-gold, and event extraction evaluation/reporting. |

The source tree contains additional target-specific modules beyond the table. For the authoritative current package inventory, inspect `arkansas_pharma_signal/` and use `rg --files model/arkansas_pharma_signal` from the repository root.

## How the modeling pipeline works

### 1. Inputs and provenance

The main inputs are local datasets under `../data/`, including the combined external-state feature store, event and entity tables, and selected additions from CMS/HHS, FDA, CDC, DEA, weather, news, and other documented public sources. Exact paths vary by task; the source catalog is in [Data sources](docs/DATA_SOURCES.md), and individual source directories under `../data/` carry their own manifests and notes.

Source records can differ in date cadence, publication lag, geography, drug identity, suppression, and coverage. Adapters and contracts are intended to preserve these characteristics, not erase them. An absent observation is not automatically zero; a source release date is not automatically the event date; and a national statistic is not automatically a county or pharmacy observation.

Some source datasets or large artifacts may be stored through Git LFS. A clone with pointer files rather than payloads cannot build every target. Retrieve tracked LFS objects when applicable and check each source manifest before interpreting a missing file as a model problem.

### 2. Identity and geography resolution

Drug and supplier names are normalized and mapped where evidence supports it, using sources such as FDA NDC records, RxNorm-related mappings, and documented crosswalks. Geography resolution uses explicit county/region/city/ZIP relationships when available. The entity graph records relationship provenance and effective dates for graph-based consumers.

Mapping coverage is not assumed complete. County allocations, supplier parents, API manufacturers, factories, or product-to-establishment links must not be invented to fill gaps. Output consumers should inspect identity fields, `confidence`, `evidence_type`, and missingness/availability fields where present.

### 3. Feature and target construction

Feature builders combine source-native observations at compatible time and geography grains. Examples include prior demand, seasonal or rolling summaries, disease surveillance, news-derived event context, weather, shortage/recall counts, supplier exposure, price, and trade/supply-chain indicators. The exact feature set depends on the task and serialized model contract; there is no universal fixed set of “all signals.”

Targets are also source- and cadence-specific. Examples include annual CMS Part D claim proxies, quarterly Arkansas Medicaid SDUD prescription counts, monthly HHS Medicaid provider/NDC claim activity, shortage-supplier counts, recall severity, respiratory pressure, or ARCOS ZIP3 distribution pressure. These targets are not interchangeable, and their forecast rows retain their defined grain and meaning.

For models that use future outcomes, chronological evaluation separates feature history from the target period. Point-in-time rules and source availability matter: only information available at or before the feature origin can be used for that forecast. Some current snapshots have no historical publication vintages, so they can be useful descriptive context but cannot support an honest historical replay as if they had been known then.

### 4. Model families

The package favors measured, interpretable baselines and target-specific models over assuming that a larger model is better. Depending on the target, the code includes persistence/seasonal comparators, ridge-style regression, logistic/state models, feature blends, and specialized source-family evaluators. Training, validation, and feature selection belong inside the appropriate chronological split; a result must be compared with a meaningful baseline on the same held-out periods.

The optional modular research model in `modular_model.py` is a multimodal architecture with six code-level stages:

1. `NewsDocumentEncoder` represents timestamped news documents and handles empty-document inputs.
2. `TypedGraphReasoner` performs masked message passing over typed entities and relations.
3. `TemporalFusionReasoner` encodes causal sequences of historical demand and external state.
4. `CrossModalInteractionReasoner` exchanges information across news, graph/geography, and temporal states.
5. `DeepConnectionReasoner` applies a further typed interaction stage across those modalities.
6. `ArkansasPredictionHeads` emits numeric/quantile and target-state outputs plus driver contributions under the configured contracts.

The model and its tests establish an implementation contract; they do not establish that this architecture beats simpler models. It remains a research path unless the appropriate leakage-safe rolling-origin gates and publication criteria are met. See [Architecture](docs/ARCHITECTURE.md), [Project goal](docs/PROJECT_GOAL.md), and [Testing protocol](docs/TESTING_PROTOCOL.md) for its exact design and evaluation requirements.

### 5. Evaluation and promotion

Evaluation should use chronological holdouts and, where enough history exists, multiple rolling-origin folds. Reports may include WAPE, MAE, error/skill relative to persistence, exact or balanced state accuracy, event precision/recall, within-tolerance coverage, fold coverage, and per-drug or per-region coverage. Metric choice depends on target semantics; accuracy on an imbalanced binary label can be misleading.

The code separates several claims that are easy to conflate:

- **A source is integrated:** it can be loaded and represented as a feature.
- **A proxy is predictable:** a historical proxy target can be forecast under a specified test contract.
- **A feature adds utility:** a matched ablation improves the same held-out folds over an otherwise identical baseline.
- **A metric is qualified:** it passes the project’s defined evidence and coverage gates.
- **A pharmacy outcome improves:** it helps predict direct private outcomes such as inventory or stockouts. The public datasets do not currently establish this claim.

`audit-publishability` and the metric serializers are intentionally fail-closed. An unqualified proxy, stale/legacy state definition, malformed row, unsupported geography, or missing required evidence must not silently become a publishable forecast. Consult the generated audit report and its source metadata instead of inferring promotion from the existence of an output file.

## Main output surfaces

Generated files live under `artifacts/` (unless a command explicitly documents another path). The exact set is run-dependent; common contracts include:

| Artifact family | Typical output | Meaning |
|---|---|---|
| Main demand panel | `artifacts/panel/panel.csv` and `artifacts/metadata/panel.json` | Joined model-ready historical panel, with build metadata. |
| Trained models | `artifacts/trained/models.json` plus metadata | Serialized model parameters/contracts used by supported forecast commands. Presence alone does not imply passing forecast gates. |
| Standard forecast | `artifacts/forecasts/forecast.csv` (name is defined by the CLI implementation) and `artifacts/metadata/forecast.json` | Forecast grid from the standard annual/city-drug path, bounded by the requested row budget. |
| Quarterly forecast | `artifacts/forecasts/quarterly_forecast.csv` and quarterly metadata | Arkansas Medicaid quarterly demand proxy forecast, optionally refreshed from official CMS SDUD input in memory. |
| Universal forecast | `artifacts/forecasts/universal_forecast.csv` and `artifacts/metadata/universal_forecast.json` | Filterable county × drug × supplier-oriented output, with unresolved mappings and evidence type explicit. If trained models are unavailable, the implementation may label rows as heuristics rather than trained predictions. |
| Qualified metric surface | `artifacts/forecasts/qualified_metric_forecasts.csv.gz` | Rows for metrics that passed their metric-level contract; not a blanket pharmacy inventory forecast. |
| Feature store | `artifacts/forecasts/qualified_metric_feature_store.csv.gz` with JSON metadata | Grain-preserving values/availability metadata intended for compatible downstream joins. Never assume it can be broadcast to a finer geography or pharmacy. |
| News artifacts | `artifacts/news/` | Historical corpus, Layer 1 news-state features, relevance outputs, and optional validated news-only bridge. These are intermediate features, not final forecasts. |
| Evaluation | `artifacts/evaluation/` | Point-in-time and rolling metrics, ablations, coverage reports, status reports, external-reference checks, and publishability results. |
| Other intermediate artifacts | `artifacts/entities/`, `geography/`, `events/`, `suppliers/`, `weather/`, `outcomes/`, `metadata/` | Built mappings, graph, structured events, weather panels, supplier context, county outcomes, and command/run metadata where generated. |

### Forecast row interpretation

Forecast output is generally a table of many rows rather than a single universal prediction. A row should be read together with its target name and semantics, forecast period/horizon, observation period, geography level and identifiers, drug identity, supplier resolution, evidence/provenance, prediction method, and uncertainty/calibration/promotion fields when available. Not every surface contains every key.

In particular:

- A county × drug or supplier × drug row is only as specific as the underlying observed data and mapping evidence.
- A supplier or labeler key is not necessarily a confirmed manufacturer, parent, API source, or local distributor.
- An exposure-based or event-based risk score is not observed on-hand inventory and should not be presented as a calibrated probability unless its contract says it is calibrated.
- `confidence`, risk values, interval bounds, or state labels must be interpreted using their associated status/semantics fields. Missing uncertainty is not certainty.
- A feature store is a join interface, not proof that a source improves a downstream model.

## Install and run

Run commands from the repository root. The package metadata requires Python 3.9+ and the core dependencies `pandas` and `numpy`; scikit-learn and pytest are included in the `dev` optional extra, and PyTorch is in the `deep` extra. Individual source adapters may need additional packages or data files. The repository may use a project-local environment such as `data/.venv`; substitute the correct interpreter for your machine.

Example setup:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e "./model[dev]"
```

On Windows PowerShell, activation is typically `\.venv\Scripts\Activate.ps1`. If not installing the package, set `PYTHONPATH=model` when invoking it.

### Core workflow

```bash
# Build the main annual panel from local data.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-panel

# Train supported demand/risk models and save their feature contracts.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . train

# Run strict next-period evaluation.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . evaluate --no-neural

# Generate the standard forecast surface.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . forecast --max-rows 10000

# Run the final fail-closed publication checks.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . audit-publishability
```

`--no-neural` skips the more expensive neural blend during the standard evaluator; omit it when you want that path evaluated and have the required dependencies/resources. Run `... cli --help` and `... cli <command> --help` for current flags. A successful command means its code completed; use the output metrics and status artifact to decide whether a forecast or metric qualified.

### Other useful workflows

```bash
# Strict quarterly Medicaid SDUD demand evaluation, optionally rolling-origin.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . evaluate-quarterly --rolling

# Generate the quarterly forecast; repeat --live-sdud-url for multiple annual CMS files.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . forecast-quarterly --max-rows 10000

# Build the entity graph, article corpus, event layer, and news signals.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-graph
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-news-corpus
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-events
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-news-signals

# Build the county × drug × supplier-oriented forecast and apply filters.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . forecast-universal \
  --max-rows 10000 --county-fips 05001 05003 --region central

# Build/calculate other supported surfaces and audits.
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-geography
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-supplier-hierarchy
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . validate-external-reference
```

The CLI also includes commands for weather refresh/history, county outcomes, event annotation/gold benchmarks, supplier shortage evaluation, ARCOS evaluation, and target-specific utility experiments. These are research and data-build tools; run the command’s help and consult its matching docs before interpreting its outputs.

### Separate news-only model integration

The 20-signal FLAN-T5-small news pipeline is maintained outside this package. When its source output is available, the adapter validates and materializes it as a dated feature bridge. The bridge does not directly forecast pharmacy availability. If it changes, rebuild the bridge and panel and retrain models so the saved feature contract matches:

The historical 20-signal table is bundled at
`website/catalog/news_only_catalog_features.csv.gz` and runs through January
2026. The upstream `run_extract.py` and model-specific inference checkpoint
are absent, so the table supports downstream research but not live news-model
inference. The web service imports its verified dated values without
promoting them as current outputs.

```bash
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-news-only-features
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-panel
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . train
```

Full source and execution details are in [News-only integration](docs/NEWS_ONLY_INTEGRATION.md). This adapter is optional; the base package does not silently fabricate the missing news model output.

## Legacy JSON output

`prod_pipeline.py` is a separate, older research workflow. Its historical fit code reads the bundled monthly news-model signal table plus particular CMS/HHS and RxNorm/ATC files. The upstream news-model inference runner and model-specific checkpoint are absent. The script now exits without emitting forecasts, and `ProductionModel.predict()` raises a clear error. The checked-in `model/final_predictions.json` remains an archival example with `news_signals_20`, `arkansas_atc_signals_18`, and `cms_part_d_signals_1300` keys.

This file is **not** the same as the package CLI’s artifact set: it does not have the canonical row-level forecast contract, and its payload uses categorical demand states. The former zero-vector ATC inference path was removed because it could fabricate a live-looking output. The news loader only reads a local historical CSV; no API key can turn it into a live runner. Treat `final_predictions.json` as a legacy/demo signal bundle unless a new runner is independently implemented, evaluated, and reconciled with the current contracts. Do not cite it as proof of live inference, inventory prediction, or the latest qualified model output.

## Testing and quality checks

The test suite is organized around the package’s functional boundaries and contracts. It covers (among other areas) schemas and input eligibility, date/grain-safe joins, demand and shortage targets, news relevance and extraction, geography, supplier identity, source-specific adapters, chronological evaluation, learned model structures, metric promotion, feature-store attachment, and forecast serialization.

Run tests from the repository root:

```bash
PYTHONPATH=model python -m pytest model/tests
```

For a focused test, pass its path, for example `model/tests/test_modular_model.py`. The authoritative list is `model/tests/`. Some integration tests require local data artifacts and optional libraries. A passing unit suite verifies tested implementation behavior; it does not demonstrate forecast superiority or readiness for pharmacy deployment.

The standalone audit helpers include:

```bash
PYTHONPATH=model python model/scripts/audit_metric_library.py
PYTHONPATH=model python model/scripts/audit_project_status.py
```

Use the [Testing protocol](docs/TESTING_PROTOCOL.md) for evaluation rules, required split discipline, promotion gates, and interpretation requirements. Do not overwrite historical evaluation evidence with an in-sample fit or compare metrics from different populations, grains, or time windows as though they were matched experiments.

## Data and scientific limitations

1. **No public inventory ground truth.** The available public data do not directly reveal pharmacy shelf counts, fill denials, backorders, or wholesaler allocations.
2. **Targets are proxies.** Claims, prescriptions, drug spending, disease activity, shortage listings, recalls, and ARCOS distribution answer different questions and operate at different grains.
3. **Source time is not always event time.** Publication lag and historical-vintage availability can make a retrospective feature invalid for a simulated historical forecast. The source-specific date contract matters.
4. **Mappings are incomplete.** Drug identity, supplier hierarchy, county assignment, API/factory links, and geography coverage may be partial. Unknowns must stay visible.
5. **A signal is not automatically useful.** Incremental value must be measured with matched chronological ablations against a suitable baseline; integration alone is not uplift.
6. **Research architecture is not model validation.** A large neural model can compile, train, and pass shape tests without beating a persistence or regression baseline.
7. **Publication is separately gated.** A generated row or metric file is not necessarily qualified for external use. Check the audit status, target semantics, evaluation coverage, calibration, and provenance.

See [Target availability](docs/TARGET_AVAILABILITY.md), [Data sources](docs/DATA_SOURCES.md), and [Research basis](docs/RESEARCH_BASIS.md) for detailed evidence and source-specific limitations.

## Documentation map

| Document | Use it for |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System layers, forecast grid, model architecture, and output concepts. |
| [Codebase guide](docs/CODEBASE_GUIDE.md) | Execution graph, package roles, operational commands, and maintenance principles. |
| [Data sources](docs/DATA_SOURCES.md) | Local source inventory, roles, source citations, and caveats. |
| [Target availability](docs/TARGET_AVAILABILITY.md) | What the sources can and cannot directly measure. |
| [Project goal](docs/PROJECT_GOAL.md) | The acceptance contract and forecast-quality gates. |
| [Implementation contract](docs/IMPLEMENTATION_CONTRACT.md) | Input/output expectations and minimum implementation requirements. |
| [Testing protocol](docs/TESTING_PROTOCOL.md) | Chronology, evaluation, and model-promotion requirements. |
| [Research basis](docs/RESEARCH_BASIS.md) | Research rationale and methodological references. |
| [Metric research matrix](docs/METRIC_RESEARCH_MATRIX.md) | Candidate metrics, evidence, and measured utility. |
| [Input variable audit](docs/INPUT_VARIABLE_AUDIT.md) | Feature availability and input-contract auditing. |
| [News-only integration](docs/NEWS_ONLY_INTEGRATION.md) | Separate FLAN-T5 feature source and adapter boundary. |
| [Review log](docs/REVIEW_LOG.md) | Historical reviews and recorded decisions; not a substitute for current artifacts. |
| [Project status](docs/PROJECT_GOAL.md), [expansion status](docs/EXPANSION_STATUS.md), and [completion notes](docs/GOAL_COMPLETION.md) | Current/recorded completion criteria and expansion tracking. Check file dates and generated audits for the newest state. |

## Safe extension rules

When adding a source, feature, target, or model:

1. Document the source, retrieval method/date, licensing/access conditions, coverage, cadence, keys, suppression behavior, and publication delay.
2. Define the target grain and semantics precisely. Never relabel a public proxy as pharmacy inventory.
3. Preserve raw or source-native values and record transformations; avoid turning unknown or suppressed observations into zero.
4. Add identity/geography crosswalks only with an explicit evidence basis. Keep unresolved joins and missingness explicit.
5. Build features using only information available at the forecast origin; test time barriers and target windows.
6. Compare against persistence or another appropriate baseline on identical chronological folds. Evaluate the feature family and model change with a matched ablation.
7. Add unit/contract tests for schema, chronology, missingness, grain, and serialization, not merely a smoke test that code executes.
8. Update the relevant source, target, architecture, and implementation documents; regenerate provenance metadata and audit artifacts as appropriate.
9. Keep the neural/research path distinct from qualified operational paths until it passes the same evidence gates.

For changes that affect output semantics or promotion, update the contracts before treating old artifacts as current. Generated artifacts may become stale when source data, feature builders, or model contracts change; their metadata and hashes should be inspected before reuse.
