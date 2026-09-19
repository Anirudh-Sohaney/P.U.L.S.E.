# Architecture

## Objective

Build an Arkansas-first, geographically expandable external-signal forecasting model that converts
readily available inputs into filterable pharmaceutical decision metrics. The
model is intended to be paired with downstream pharmacy inventory-prediction
systems that currently rely on simple regression. The revised metric contract
and accuracy gates are authoritative in `model/docs/PROJECT_GOAL.md`.

The output is not direct shelf inventory. Public data does not observe Arkansas pharmacy inventory or wholesaler allocation. The output is a broad signal layer:

- drug demand forecast,
- demand shock forecast,
- national supply disruption forecast,
- Arkansas shortage impact forecast,
- driver attribution by disease, weather, news, recall, shortage, price, provider, and supplier/labeler signals.

Arkansas remains the primary evaluation and highest-resolution geography. The
same interfaces may ingest U.S. national, bordering-state, and global context
when a local series is unavailable. Such context must retain its original
geography and may only be projected into an Arkansas feature through an
audited, point-in-time-safe exposure mapping.

## Current implementation audit

The code includes separate adapters for CMS, Medicaid, FDA, CDC, ARCOS, news,
weather, provider geography, supplier hierarchy, and universal forecast output.
The neural path is an implemented research architecture, not a validated
replacement for the simpler baselines. The latest focused audit passes 50
expert-gold, input-contract, and ARCOS evaluation tests. Forecast evidence
also remains proxy evidence; no local
inventory, stockout, backorder, or allocation label is available in the public
data. These limits are part of the architecture contract, not optional caveats.

The latest strict operational annual run used 173 features and produced WAPE
`0.145785` versus `0.146271` for the strongest naive baseline. The calibrated
bridge improved the benchmark by only `0.33%`; ARCOS is available in the panel
but is not promoted as a demonstrated accuracy gain.

The current operational surface contains 14 qualified numeric or five-state
external proxies. The 307,798,423-parameter multimodal network is a
research architecture until its own rolling-origin outputs satisfy the same
contract; operational rows use validation-approved proxy adapters and record
that method in provenance.

## Design Principles

1. Use real data only: local files in `data/` and documented public sources.
2. Preserve provenance: every feature and output row must carry source, transformation, and timestamp metadata.
3. Separate extraction from forecasting: language-model/text extraction creates structured intermediate variables, never a direct final forecast.
4. Use layered models: interpretable regression baselines, neural representation learning, graph/distance propagation, and text-derived event features.
5. Keep the final research model within the requested 300-400 million active-parameter budget: the default implementation uses one auditable multimodal model plus compact production baselines. Any optional language model must be used as an offline extractor or small encoder, not an always-active frontier model.
6. Favor Arkansas resolution first, then national/global context: county/city/ZIP3/region features dominate local forecasts; national shortage and trade features are projected into Arkansas by drug demand exposure and supplier mappings.

## Current Code-Level Layer Contract

The trainable path is explicitly separated into the following independently
inspectable modules:

1. `NewsDocumentEncoder` converts timestamped article tokens into a document
   state, with safe handling for empty documents.
2. `TypedGraphReasoner` performs masked directed message passing. Relation ID
   `0` and entity ID `0` are padding/unknown values; they cannot create graph
   messages or affect pooled states. Each graph block aggregates source entity
   embeddings plus typed relation embeddings over valid incoming edges, then
   applies Transformer reasoning with padded nodes masked.
3. `TemporalFusionReasoner` applies causal attention to historical demand and
   external-state vectors. Future timesteps cannot enter the representation.
4. `CrossModalInteractionReasoner` exposes news, graph/geography, and temporal
   states as typed tokens for cross-domain interaction.
5. `DeepConnectionReasoner` applies a second typed interaction stage before
   output fusion; this is the learned political/geographic/economic/supply
   connection seam, not a hard-coded rule table.
6. `ArkansasPredictionHeads` emits point and quantile outputs, target-specific
   five-state target-specific logits, a legacy three-channel aggregate
   state-risk tensor for compatibility,
   and driver contributions. `target_state_risk[batch, horizon, target, state]`
   is the authoritative state-output contract for metric heads; state index
   semantics are supplied by the target registry rather than hard-coded into
   the shared network.
   The state head is trained when the evaluator receives target-specific
   labels and `state_loss_mode="categorical"`; the canonical public demand run
   records both the state target mode and training loss mode. Numeric forecasts
   are also evaluated through a separate five-state projection with thresholds
   fit only on the training split. A state head with no target labels remains
   an exposed forward contract, not a qualified forecast.

`model/tests/test_modular_model.py` verifies intermediate shapes, empty-history
stability, graph edge sensitivity, and graph padding invariance. The model
inventory reports active, trainable, and frozen parameter counts before any
training run, plus a disjoint parameter count and contract for each of the six
code-level stages. These counts are serialized into training and evaluation
metadata so a checkpoint cannot be described only by an approximate total.

## Forecast Grid

The canonical output grid is:

`forecast_date x horizon x geography x drug_ingredient x supplier x disease_driver x target`

Recommended horizons:

- 1 week,
- 4 weeks,
- 8 weeks,
- 13 weeks,
- 26 weeks.

Recommended geography:

- county for public health, population, COVID, weather/disaster exposure;
- Arkansas planning region for stable regional reporting;
- prescriber city when CMS Part D provider-drug data is the target;
- ZIP3 for DEA ARCOS controlled-substance signals;
- state when only Arkansas-level demand is available.

The universal forecast surface also materializes five explicit broad Arkansas
reporting regions (`northwest`, `northeast`, `central`, `southwest`, and
`southeast`) from a versioned county-FIPS crosswalk based on the Arkansas
DHS/TEFRA five-region definition. Empty or unresolved
county mappings stay empty; they are never assigned a region by city name.

It additionally emits `neighbor_state_supply_event_risk` rows at
`geography_level=neighbor_state` for Missouri, Tennessee, Mississippi,
Louisiana, Texas, and Oklahoma when observed FDA/FEMA event geography exists.
These are external-event context rows only; Arkansas demand is never copied
into a neighboring state.

FDA DRLS establishment matches are kept in a separate supplier-context table.
Because the local registration snapshot has no NDC-to-establishment edge, the
product hierarchy leaves factory/API fields empty and records the limitation
explicitly.

Publishability therefore gates the required local supplier categorization on
direct product-labeler coverage plus a nonempty supplier context artifact; it
does not pretend that a non-product-specific FDA establishment name is a
product-specific factory or API relationship.

Recommended targets:

- `demand_claims`: expected claims or fills proxy,
- `demand_cost`: expected drug cost proxy,
- `demand_shock_index`: abnormal demand pressure from disease, weather, disasters, news, and seasonality,
- `supply_disruption_risk`: probability-like score from shortage, recall, enforcement, trade, price, and supplier signals,
- `arkansas_shortage_impact`: combined local demand exposure and national supply-risk score,
- Historical three-state shortage adapters remain compatibility code only;
  promoted shortage outputs are numeric or five-state. The universal grid
  does not emit a binary shortage state.
- `driver_contribution`: signed contribution of a feature family to the above forecasts.

## System Layers

### 0. Research multimodal model (not yet promoted)

`arkansas_pharma_signal.modular_model` contains the first real roughly 300M-class
architecture.  It is intentionally separate from the currently deployable
statistical models until it has been trained and passes the same temporal
gates.  The active path is:

`NewsDocumentEncoder -> TypedGraphReasoner -> TemporalFusionReasoner ->
CrossModalInteractionReasoner -> DeepConnectionReasoner ->
ArkansasPredictionHeads`

`CrossModalInteractionReasoner` receives three typed state tokens after their
independent encoders: news/event state, graph-derived Arkansas exposure, and
causal temporal/economic context. Its learned interaction blocks are followed
by `DeepConnectionReasoner`, a separately parameterized Transformer stage with
role embeddings for those three state types. The deep stage is the explicit
middle representation used by the output heads, rather than an uninspectable
direct concatenation. Neither stage adds time steps, so neither can introduce
a future-period feature; temporal leakage remains controlled by the upstream
history construction and split contracts.

The default configuration uses a 16-layer, 1,024-wide encoder with an 8,192
token context, typed entity/relation message passing, causal temporal
attention, and multi-task Arkansas heads. The measured inventory is
307,798,423 active and trainable parameters before any optional freezing, split
across news `242,698,240`, graph `47,874,400`, temporal `7,519,344`,
cross-modal `4,158,672`, deep-connection `3,749,872`, and heads `1,797,895`.
The model inventory reports total, trainable, and frozen parameters; no
lazy/uninitialized parameters are allowed in the inventory. Intermediate
representations are returned for evidence and component ablations. Outputs
include point forecasts, three quantiles, state risk, and a ten-channel driver
contribution vector.

This module is an architectural and forward-contract milestone, not evidence
of predictive quality.  Promotion requires real time-split training labels,
component-level evaluation artifacts, calibration, and improvement over the
existing transition/ridge baselines.  Random initialization, synthetic rows,
or parameter count alone never qualify a model for production.

Qualified external metric rows follow the same conservative rule: persistence
state forecasts carry point state values but null confidence, risk, and
interval fields until calibration is demonstrated. Their explicit
`uncertainty_status` and `calibration_status` fields prevent a downstream
pharmacy regression from mistaking a placeholder probability for measured
uncertainty.

### Qualified metric context boundary

The modular model exposes an optional metric-context vector at the temporal
boundary. Its ordered fields are the eight context metric values followed by
their eight missingness indicators from `metric_feature_store`; missing values
are never imputed. The vector is projected into the temporal state before
cross-modal and deep-connection reasoning, so external shortage, recall,
disease, acquisition-cost, regional-demand, NSSP pathogen, and national RSV
context can participate in the
same code-level architecture when a caller supplies it.

Historical training can opt into the point-in-time context adapter when its
local dated sources are present. The current adapter admits exact-NDC FDA
shortage archive observations, CMS NADAC prices, FDA enforcement events, and
prior NSSP influenza ED pressure and RESP-NET RSV hospitalization pressure
whose observation/report dates are at or before the feature origin; unmatched
NDCs and incompatible regional/supplier metrics remain explicitly missing. It is a sparse
research feature and does not claim learned
metric-context lift or production pharmacy-inventory improvement. The feature
is disabled in the default research configuration and must be enabled
explicitly for ablation. The live feature-store path remains available for
future pharmacy-level data and vintage-complete retraining.

The first post-NSSP rolling baseline was run with the canonical end-to-end
evaluation command using three chronological folds and one training epoch.
Neural WAPE was 20.89%, 21.19%, and 20.15%, while mean within-five-percent
accuracy was 15.97%. The validation-selected deep blend beat the fold ridge
baseline on every fold, but the result remains research-only because it does
not approach the 75% numeric accuracy requirement for pharmacy-oriented
outputs. These figures are pre-correction historical evidence: the evaluator
previously serialized a validation-selected single policy under the
`stacked_blend` name.

### Component ablation interpretation

The six-fold component report in
`model/artifacts/evaluation/modular_component_rolling.json` now scores the
intact path before ablations and separately preserves direct neural forecasts.
Validation selected a zero neural-blend weight in every fold, so the blended
WAPE comparison is inconclusive: removing a component cannot change a blend
that contains no neural contribution. Direct neural WAPE is therefore the
useful diagnostic. Graph and temporal components improved the direct neural
forecast in four of six folds each, and the deep-connection component improved
it in three of six; news removal did not improve any fold. These are component
diagnostics only and do not satisfy a publishable accuracy gate until the
neural path meets the declared metric requirements.

The current CPU-sized research regeneration (`epochs=1`, frozen news encoder)
contains 32,531 training rows, 3,373 validation rows, and 2,597 test rows. Its
test WAPE is `0.1176`; the six-fold rolling report improves over the strongest
naive baseline by only `0.59%` on average and sets
`publishable_rolling_candidate` to `false`. Both artifacts record
`forecast_mode: research_training_only` and three genuinely non-operational
periodic fields. These are current evidence values, not a production claim.

The historical change-state run is retained for comparison, but it is not the
current contract result. The canonical level-state regeneration projects the
numeric demand head into five absolute demand bands fit only on training rows:
fold exact accuracy was `72.24%`, `72.51%`, and `72.94%`, while high-state
precision was `82.99%`, `83.67%`, and `85.42%`. The numeric raw route still
fails, but every fold passes the declared 65% raw plus 80% event route. This
is a learned public-demand proxy, not direct pharmacy inventory truth.

The state diagnostic also projects the numeric residual head through the same
frozen cut points. On the first canonical fold, that projection reached only
`19.18%` exact accuracy and `19.76%` high-state precision, below the learned
state head's `21.50%` and `21.48%`. The numeric forecast therefore cannot be
treated as a better state classifier by post-processing alone; future work
must improve the shared representation or state target itself.

The publishable NDC9 demand adapter now trains its five-state change head using
cut points fit only on the training rows and reports exact, balanced, and
high-state event precision alongside numeric error. In the first real fold
after this change, state exact accuracy was `21.50%`, balanced accuracy was
`20.57%`, and high-state precision was `21.48%`; numeric within-five-percent
coverage was `14.05%`. The state head is therefore implemented and auditable,
but it does not yet satisfy either the 75% raw or the 65% plus 80% event route.

### End-to-end public-suite evaluation

`model/scripts/evaluate_publishable_model.py` is the canonical NDC9-quarter
adapter for the multimodal demand head. It preserves the public suite's raw
NDC9 identity and strict next-quarter target, maps only prior CDC/news context
into the declared history vector, and represents unavailable external metrics
explicitly as missing/neutral features. It does not create county or supplier
labels. Run it with:

The adapter also exposes a `train_only_transition_prediction` raw feature.
For training rows it is an expanding estimate from strictly earlier training
observations; validation and test rows use the completed training pool only.
It is therefore a derived, fold-local input rather than a statistic computed
over the full panel.

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_model.py \
  --root . --suite-root data/targeted_additions/publishable_test_dataset \
  --epochs 5 --batch-size 2048
```

The first controlled five-epoch run selected its checkpoint on validation WAPE
(epoch 2) and produced `0.8618` blended test WAPE. The independent ridge
benchmark on the same public split produced `0.8465` WAPE, so the neural path
is retained as research-only and is not promoted as an improvement.

The temporal contract subsequently added current FDA shortage state and
supplier-count evidence, producing 36 fields. The matched ablation reached
`0.8852` blended WAPE, so the added fields remain valid inputs but are not
credited with predictive uplift.

The adapter also evaluates the architecture in its intended augmentation mode:
the post-`DeepConnectionReasoner` state is concatenated with the public raw
features and passed to a validation-selected ridge head. This stacked result
reached `0.8639` test WAPE, improving the neural head but remaining worse than
the raw-feature ridge benchmark (`0.8465`). It is therefore an interpretable
research diagnostic, not a promoted model.

The same script supports `--rolling`, which retrains on three chronological
origins and compares stacked versus raw ridge within each fold. The current
one-epoch run fails the all-fold 10% gate: improvement occurs only on the
latest fold, while earlier folds show large historical distribution shift.
History normalization is fitted separately on each selected training fold;
validation and test rows are transformed with those training-only statistics.

Rolling evaluation also supports a fixed training-only 99.5th-percentile
prediction cap to prevent early-fold outliers from setting the forecast range.
That robustness ablation reduces the first-fold raw-ridge WAPE from `89.60` to
`13.47`, but the stacked improvements remain `81.70%`, `-1.51%`, and `0.13%`;
the all-fold gate still fails.

`--mask-drug-identity` tests transfer without learned NDC memorization. With
the same robust cap, masked stacked improvements are `83.36%`, `2.83%`, and
`0.18%` across the three folds. Masking helps the middle origin but does not
meet the all-fold gate, so it remains a cold-start research ablation.

The current whole-system evaluator additionally gates the deep stack against
the raw ridge using validation-only convex blending. This protects against
forcing a learned representation into the forecast when validation favors the
baseline, but it is not a guarantee under temporal drift. The corrected
three-fold artifact now reports the actual convex blend: mean improvement
versus fold ridge is `-18.50%`, mean within-five-percent accuracy is `1.68%`,
and mean five-state exact accuracy is `22.55%`. The event route is false, as
are both 75% learned-model gates. The deep weights were `0.0`, `0.0`, and
`0.0`; this is valid evidence that the current learned stack does not improve
the benchmark, not evidence of a successful deep model. The path remains
research-only and unpromoted.

The latest target-alignment ablation fixes a representation/head mismatch:
the neural path is trained on the log persistence residual, while the former
stacked ridge was trained on absolute log demand. The residual-aligned stacked
head reconstructs predictions from the observed-quarter persistence anchor;
the raw ridge remains unchanged as the benchmark. On the three strict public
folds, the canonical whole-system aligned stack/blend improvements versus raw
ridge were `80.14%`, `69.48%`, and `-6.45%`, mean `47.72%`. This is evidence
that target geometry matters, but the third-fold failure means the all-fold
promotion gate remains false and the path is research-only.

A predeclared two-year validation-window ablation was then run to reduce
single-year selection overfit. It trained through the year before each origin,
selected on the next two complete years, and tested only afterward. The
aligned stack/blend improvements were `76.31%`, `-1.92%`, and `-5.51%`, mean
`22.96%`; only one of three folds passed. Widening validation therefore did
not establish temporal stability or change the research-only status.

The next predeclared regime-robustness test selected a residual scale from
`{0.0, 0.2, ..., 1.0}` using validation WAPE only. Selected scales were `1.0`,
`1.0`, and `0.0`; scaled-stack WAPEs were `0.9464`, `0.9400`, and `0.9345`.
The latest fold therefore correctly shrank to persistence, but its
validation-selected raw/stack blend still lost `6.23%` versus raw ridge. The
scale grid is retained for auditability but does not meet the all-fold gate or
justify promotion.

The training module also supports `--small`, a CPU-testable research model
using the identical 38-feature temporal contract. This mode validates the
data/forward/loss/evaluation path only and writes to a separate
`modular_research_small` checkpoint directory; it is not a substitute for
300M-scale training or a promotion result.

### Research training pipeline

`arkansas_pharma_signal.training` trains against the real next-quarter SDUD
count. It constructs four observed-quarter history windows, real article
windows grouped by publication quarter, and canonical-graph neighbors. The
target is log1p next-quarter count; the point and 0.1/0.5/0.9 quantile heads
use smooth-L1 plus pinball losses. Feature years through 2020 are training,
2021 is validation, and later years are untouched future test data.

The numeric target also has a five-state companion target. By default it is
the next-period log-demand change (`target_log - persistence_log`), which is
more useful for inventory decisions than relabeling an already observed demand
level. The `level` mode remains an explicit research ablation. Five fit-only
quantile thresholds divide the selected state value into five ordered states;
the same thresholds are applied to validation and test rows. A weighted
cross-entropy term trains
`target_state_risk[:, horizon, target, state]`, and checkpoints record both
the thresholds, target mode, and loss weight. A slice with insufficient target variation is
marked unavailable rather than assigned fabricated states, and receives no
state score. State accuracy is reported as exact matches and balanced
accuracy, separately from numeric WAPE. Evaluation also discretizes the
numeric point forecast with the same thresholds, preventing a classifier head
from hiding whether the continuous demand forecast carries the state signal.

The temporal stream now has 38 inputs: 15 demand/history and current FDA-state
fields, 19 bounded annual external context fields, and 4 exact-NDC NADAC price
fields. The two additional demand fields are log change and relative change
from the immediately previous observed quarter.

The matched rolling ablation produces stacked improvements of approximately
`0.00%`, `-3.75%`, and `0.19%`; these scale-invariant fields remain research
inputs but do not qualify as a model improvement.
The annual context covers Arkansas weather/disaster and employment context,
national/state surveillance, wastewater, global supply pressure, tariff
context, and Layer 1 news states. External context is joined
from the previous completed annual panel year, never the current or target
year. This makes the multimodal middle path consume qualified connections
without converting annual summaries into falsely current observations.

Entity ID `0` is reserved for zero-padding/unknown entities. A
research-only `mask_drug_identity` mode tests cold-start behavior by removing
the learned drug hash while retaining Arkansas and observed graph-relation
nodes. The initial single-split ablation did not improve the validation-
selected blend and is not active. The optional shared drug-token graph
representation was also tested and did not improve the matched held-out
blend; it is not active.
An optional canonical-edge priority mode similarly preserves direct
ingredient/manufacturer evidence but has not improved the matched blend and
is not active.

The active small research checkpoint uses a persistence-residual anchor, five
epochs, batch size 256, learning rate 2e-5, a frozen document encoder, and
the NADAC ablation disabled after matched evaluation. With the cross-modal
interaction stage under the strict consecutive-quarter contract, its
validation-selected held-out blend is 11.34% WAPE and its test blend is 9.79%
WAPE; the transition baseline remains stronger, so it is not a production
artifact. The fresh one-epoch six-fold rolling-origin evaluation selects a
blend on each validation fold, beats the strongest naive baseline on 3/6
folds, and has mean improvement 0.04%; the rolling candidate is not
publishable. Subgroup diagnostics show materially worse cold-start
performance and stronger but sparse manufacturer-linked rows; they are not
promotion gates or supplier-level validation.

A matched three-way audit of the residual objective (persistence anchor,
transition anchor, and an `absolute` no-anchor control on the same strict
fixed split) shows the residualization itself is necessary: the absolute mode
collapses to a mean prediction (neural test WAPE `0.9828`, selected blend
weight `0.0`), while persistence-residual and transition-residual neural WAPE
are `0.1320` and `0.1293` with validation-selected blends `0.1019` and
`0.1029` against a transition baseline of `0.09998`. Neither anchored mode
beats the strongest naive baseline, so no objective mode is promoted; the
result is recorded in
`model/artifacts/evaluation/modular_residual_objective_ablation.json`.

The quarterly target builder now requires the observed target row to be the
immediately following calendar quarter. It also withholds a lag when the
previous observed row is separated by a missing quarter. This removed 2,768
invalid multi-quarter transitions from the prior view and invalidated the
older rolling metrics; all current quarterly artifacts were regenerated under
the strict contract.

The NADAC-to-multimodal ablation is retained for research reproducibility but
is disabled by default. In a matched three-epoch fixed split, adding the four
price fields changed validation-selected blend WAPE from `0.1125` to `0.1169`
and test neural WAPE from `0.1388` to `0.2300`; this is a rejection, not a
claim that NADAC is unavailable or intrinsically uninformative.

### Operational training mode

Operational mode is the default train and evaluation mode. It excludes the 17
`PERIODIC_TRAINING_ONLY` columns from training while retaining them in the
panel for research ablations. Periodic data remains available only for
research comparisons via `--include-periodic-training-features`; it is never
part of the default operational artifact. All reported metrics must include
`feature_mode` (`operational` or `periodic_included`) so operational and
research results cannot be conflated.

The modular checkpoint carries a parallel input declaration. Local pharmacy
history is explicitly recognized as a future application-supplied input, while
the name audit distinguishes ten local-history fields from three genuinely
non-operational periodic fields (unemployment, GSCPI, and tariff). Modular
checkpoints and evaluation metadata therefore persist `input_contract` and
set `forecast_mode` to `research_training_only` when those fields are present.
This prevents the 300M-class architecture from being described as operational
merely because its neural layers are deployable.
The same fields are included in full rolling and component-ablation reports,
so a marginal layer result cannot be detached from the input-availability
status of the run.

### 1. Canonical Feature Store

Inputs:

- `data/final_data/features/external_state_features.csv.gz`
- `data/final_data/events/events.csv.gz`
- targeted Arkansas additions under `data/targeted_additions/`
- entity graph and drug dictionaries under `data/final_data/entities/`

Responsibilities:

- load long-format feature rows,
- validate required fields,
- align all time series to a weekly or monthly forecast calendar,
- build lagged, rolling, seasonal, and anomaly features without synthetic values,
- preserve source metadata.

### 2. Entity Graph

Nodes:

- Arkansas counties, cities, ZIP3s, regions, HSA/HRR areas,
- drugs, generic names, active ingredients, NDCs,
- suppliers, labelers, manufacturers,
- diseases and event classes,
- providers/pharmacies where public identifiers exist.

Edges:

- drug contains ingredient,
- NDC maps to ingredient/generic/supplier,
- provider/pharmacy located in city/county/ZIP/region,
- county belongs to planning region and HSA/HRR,
- disease maps to drug classes or likely therapies,
- supplier/labeler maps to NDC/drug products,
- event mentions or affects a disease, geography, drug, or supplier.

The graph is used to project national/global signals into Arkansas exposure scores. Example: a national recall for a labeler should affect Arkansas counties most where Part D demand for mapped ingredients is high.

### 3. Text and Event Extraction

Text inputs:

- Arkansas 3DLNews2 article metadata and monthly keyword counts,
- FDA shortage and enforcement text,
- disease/global event text already extracted in final data,
- optional live news feeds when added later.

Language-model role:

- classify event type,
- extract location, disease, drug, supplier, severity, date, and uncertainty,
- normalize mentions to dictionaries,
- emit structured event features.

The text extractor must not output final shortage forecasts directly. Its outputs feed regression and neural models through time, geography, and entity graph joins.

The explicit Layer 1 contract is the wide 300-variable news-state table produced by
`news_signals.build_news_state_features`. It contains 50 deterministic
boolean/count/state variables (10 per event family for shortage, recall,
disease outbreak, disaster, and policy/trade), plus per-disease presence,
outbreak-risk, recent spread-rate, and uncertainty states for 62 diseases. Every value is
derived from an article event span and keyed by observation date plus county/location. Unknown
publication timestamps are excluded; an `as_of` cutoff excludes future news.
When the weakly supervised relevance artifact is available, two additional
states expose the learned article relevance mean and scored-article count.
They are relevance evidence only, not event or forecast labels.
This gives downstream layers a stable, filterable input surface while keeping
the final market forecast separate from text extraction.

Qualified forecast features have two join modes. Exact mode preserves the
full period/geography/county/drug/labeler/supplier grain. Explicit context mode
uses only the target registry's declared source and destination keys, such as
period+drug for national NDC shortage context or period-only for Arkansas-wide
respiratory context. Unsupported local dimensions remain missing, and the
projection scope is persisted in metadata; no geography or supplier is inferred.

The panel builder also consumes this same artifact path and joins its
year-aggregated states into the strict next-period feature view. Thus these
news states flow through the interaction layer and final demand/shortage heads
without using future publication dates.

The monthly FDA supplier-drug model is also exposed as optional
`supplier_drug`/`supplier_shortage_probability` context rows in the universal
output. These rows are not copied to counties: FDA reports identify supplier and
drug event history, but do not provide Arkansas pharmacy allocation. The rows
therefore retain supplier-level geography, a 30-day horizon, source month, and
the explicit semantics that a zero means no observed FDA event rather than no
true shortage.

For exact normalized drug matches to the Arkansas panel, the universal output
also emits `geography_level=arkansas_supplier_drug` context rows targeting
`arkansas_supplier_drug_shortage_pressure`. These are statewide relevance
signals with low confidence; supplier-to-Arkansas allocation is explicitly
unverified, so no county or regional allocation is implied. The universal
`--supplier` filter applies consistently to both supplier context surfaces;
county/region filters retain statewide supplier context as background.

Every universal output row now carries machine-readable target provenance:
cadence, geography scope, supplier resolution, whether it is a direct pharmacy
observation, observation type, promotion status, and target semantics. The
validator rejects unqualified targets. This prevents a county demand forecast,
an FDA supplier event, and an ARCOS distribution proxy from appearing as the
same kind of medical-market label in downstream clinic filters.

The deterministic event extractor remains a provenance-preserving fallback.
The learned relevance layer (`news_relevance.py`) trains a hashed unigram/bigram
`LogisticRidge` model on the weak PADI-web article-relevance corpus, with a
publication-time split. It is evaluated separately on expert CIRAD sentences;
those expert references are never used for training and relevance is never
treated as an event truth label. A transformer/LLM event extractor remains a
future upgrade requiring a real human-health gold set.

`learned_event_state.py` is a separate research candidate trained on the 1,244
CIRAD expert sentences using an article-grouped holdout. Its CE/RE target means
current or risk event in animal-health surveillance, so its artifact is not
joined to Layer 1 and cannot pass the publishability audit. The command
`train-research-event-state` exists to reproduce the benchmark while a human-
health event gold set is being assembled.

### 4. Regression Baselines

Regression baselines are mandatory because they establish calibration and auditability:

- seasonal naive and moving-average baselines,
- ridge-style linear regression over lagged features,
- Poisson/negative-binomial-style demand proxy where available,
- logistic-style shortage risk from shortage/recall/event labels.

Annual shortage risk is evaluated as a validation-selected leaderboard, not as
a single fixed score. Candidate risk scores include logistic ridge over the
full feature set, current shortage/recall exposure columns, national active
shortage exposure, labeler exposure, and simple exposure blends. Selection is
made on validation top-k recall, then AUPRC, then Brier score; the best
test-period model is reported separately as diagnostic only.

Rolling annual shortage-risk evaluation is exported separately from the main
annual demand metrics. Each fold trains on earlier feature years, validates on
one feature year, and tests the next feature year. The artifact keeps both the
deployable validation-selected model and the diagnostic test-best model so
research can distinguish model-selection failure from absent predictive signal.

These baselines provide backtesting references and feature sanity checks.

### 5. Neural Multi-Task Model

The neural model learns shared representations across drugs, geographies, suppliers, diseases, and external signals:

- categorical embeddings for drug ingredient, supplier, geography, disease, and horizon,
- numeric towers for lagged demand, disease surveillance, weather, economic, trade, and supply features,
- text/event tower for extracted event intensities and hashed text classes,
- multi-task heads for demand, demand shock, supply risk, and shortage impact.

Parameter budget target:

- embeddings and MLP/tower model: less than 50 million parameters by default,
- optional graph neural encoder: less than 100 million active parameters,
- optional small language/text encoder: less than 200 million active parameters and offline feature extraction preferred.

### 6. Graph and Spatial Propagation

Spatial relevance is learned or computed from:

- county/city/ZIP3 distance,
- region membership,
- HSA/HRR crosswalks,
- provider and pharmacy density,
- disease surveillance geography,
- supplier/drug exposure.

News or disease events outside a target county are attenuated by distance, region membership, population, healthcare demand, and known cross-region structure. This creates the requested deep connection path:

`news text -> event extraction -> location/disease/drug entities -> graph/spatial relevance -> disease/drug therapy mapping -> demand/supply/shortage model features -> forecast grid`

### 7. Quarterly Medicaid Demand Evidence Model

The formal high-frequency demand proof path uses real Arkansas Medicaid SDUD
prescription counts aggregated by drug and quarter. It predicts quarter `t+1`
from features available at quarter `t` only.

Current validated layers:

- current-quarter Medicaid prescriptions by drug,
- lagged quarterly demand and two-quarter moving average,
- fixed quarter seasonality,
- historical drug-by-quarter transition ratio learned on the train split only,
- historical drug-by-quarter delta and recent-window transition variants,
- MA2-guarded transition blends for folds where moving-average persistence is
  the stronger deployable base,
- previous-completed-year annual Part D/drug/supply/news layers where a public
  drug-key match exists,
- quarter-level FDA/openFDA event count, recall count, shortage count,
  severity, and confidence.
- CMS NADAC weekly acquisition-cost layer joined by exact normalized NDC:
  quarter mean, observation count, log price, and consecutive-quarter price
  change. This layer covers 4 additional variables in the current real-data
  evaluation and leaves unmatched NDCs missing.
- prior-completed-quarter CDC weekly surveillance summaries: FluView ILI
  (Arkansas and national) and Arkansas site-level wastewater WVAL (overall
  mean, reporting site count, and per-pathogen SARS-CoV-2/influenza-A/RSV
  means) from
  `data/targeted_additions/disease_surveillance_current/data/`.

The weekly surveillance layer is leakage-safe by construction: each weekly
observation is assigned to the quarter of its own week, aggregated to a
quarterly summary, and joined to feature rows one quarter later. A feature row
at quarter `t` therefore sees only completed weeks strictly before `t` begins;
no same/current-quarter observation is used. Missing weeks, sites, or
pathogens stay missing rather than being imputed, and absent or unusable
source files leave the view and feature list unchanged.

Scientific limitation: the prior-quarter constraint deliberately discards
same-quarter early-season surveillance (e.g., current-quarter ILI onset),
which in practice would be partially observable at the forecast date. The
layer also cannot separate reporting-lag artifacts (revised `release_date`
snapshots are collapsed to the latest published issue per epiweek) from true
disease activity, and wastewater coverage only begins in 2022, so the layer is
empty for earlier feature quarters.

In addition to means and coverage, the FluView layer retains within-quarter
Arkansas and national WILI trajectory features: last observed value, maximum,
and first-to-last change. These are deterministic summaries of real weekly
inputs, not forecasts or labels, and follow the same completed-quarter join
boundary.

The separate weekly FluView proxy evaluator uses the source at its native
weekly grain. It predicts next-week Arkansas WILI from current and lagged WILI,
rolling history, and calendar terms, with latest-release deduplication and
seven chronological folds. It is an upstream respiratory healthcare-demand
proxy, not a pharmacy demand or inventory label; its current mean WAPE is
`27.76%` versus `17.78%` for persistence, so it is not promoted.

The NADAC price layer is available by feature-quarter end and is therefore
distinct from the prior-completed-quarter surveillance join. It is useful for
the operational price-pressure contract, but the strict evaluation selected
the same transition policy with or without it: pointwise WAPE remained
`0.0940`, unrestricted rolling improvement remained `9.0%`, and one-year
rolling improvement remained `17.6%`. It is retained as a qualified input and
not claimed as measured predictive uplift.

The publishability gate for this path is strict: the best learned model must
beat the strongest naive model by at least 10% WAPE on the held-out future test
split. Validation may select blending weights; test labels are never used for
selection.

Rolling-origin evidence is stricter. Each rolling fold trains on years before
the validation year, validates on that year, and tests on later years. The
rolling candidate flag is true only when every fold beats its strongest naive
baseline and mean WAPE improvement is at least 10%.

Two rolling surfaces are tracked:

- all-future test folds, which stress long-horizon stability;
- one-year test folds, which better match near-term inventory planning but
  still must beat the strongest naive baseline in every fold to clear the gate.

Rolling publishability uses the model selected by validation WAPE, not the
lowest test WAPE row. Test-best rows are retained only as diagnostic upper
bounds.

The user-requested acceptance criterion is tracked separately from this WAPE
research gate. A quarterly rolling result is promotion-eligible only when it
also reaches `0.75` mean accuracy by either numeric within-5-percent error or
both raw and balanced five-state demand accuracy. The real one-year rolling
artifact has six folds, mean WAPE improvement `17.62%`, mean within-5-percent
accuracy `25.72%`, raw state accuracy `66.74%`, and balanced state accuracy
`41.09%`. Therefore `publishable_rolling_candidate` is true for the WAPE
research gate, but `promotion_candidate` is false and the requested 75%
accuracy goal remains unmet.

An exploratory histogram-gradient-boosting regression candidate was tested on
the same leakage-safe base features. Its best validation WAPE was `13.90%`
versus persistence at `11.33%`, and its held-out one-year WAPE was `15.22%`
versus persistence at `11.29%`; it was rejected. A class-balanced boosted
three-state classifier also reached only `52.6%` held-out balanced accuracy.
These results support retaining the simpler transition ensemble until a
better evidence-backed feature or target source is available.

A train-only correlation-screened residual ridge was also evaluated to test
whether sparse external features were being diluted by the full 800-column
ridge. Validation selected 16 features, alpha `10`, and a `0.20` persistence
blend weight, but held-out WAPE was `11.99%` versus persistence `11.29%` and
within-5-percent accuracy was `27.26%` versus `29.11%`. It was rejected; the
screening procedure remains an experiment, not a production feature path.

A log-ratio median transition with count shrinkage was then evaluated across
all six one-year rolling folds. It beat persistence on only 3/6 folds, with
mean WAPE improvement `4.45%`; mean within-5-percent accuracy was `26.51%`
versus persistence `26.69%`. It was rejected and the existing transition
ensemble remains unchanged.

An ingredient-level hierarchical transition was also assessed using the
source-provided ingredient when present and a deterministic drug fallback.
Ingredient coverage is only `47.9%` in the cached Medicaid panel; the
validation-selected hierarchy beat persistence on 5/6 folds but delivered
only `3.83%` mean WAPE improvement and remained weaker than the existing
transition ensemble. It was rejected until NDC-to-ingredient coverage is
improved with a separately audited catalog join.

The follow-up FDA NDC catalog audit found only `2,666` unique mappings among
`22,983` missing-ingredient rows, with `1,229` rows having multiple candidate
ingredients. This would increase coverage by only about six percentage points
and would use a current catalog for historical observations, so the join is
not promoted into the demand target path.

The deployable quarterly signal is exported separately from the annual
city-drug forecast:

- `model/artifacts/forecasts/quarterly_forecast.csv`;
- `model/artifacts/metadata/quarterly_forecast.json`;
- target `medicaid_prescription_count`;
- geography `state=AR`;
- model family `quarterly_medicaid_selected_transition`.

This artifact is intended as a state-level Medicaid demand-pressure signal for
inventory models. It is not a supplier-level or pharmacy-level inventory
prediction.

The annual city-drug model also receives an audited Medicaid quarterly demand
pressure layer. Real Arkansas Medicaid SDUD quarterly counts are annualized by
drug key into annual total prescriptions, Q4 prescriptions, Q4 share, recent
Q4-vs-Q3 change, log transforms, observed quarter count, bridge-record count,
and bridge-family count. Production bridge keys come only from local real
identifiers: normalized Medicaid names, NDC-to-drug relationships, FDA product
NDC nonproprietary/substance names, and local dictionary NDC/RxCUI links.
Unmatched Part D drugs remain missing; no fuzzy matching or synthetic fill is
used. The annual calibrated production blend does not blindly consume the
widest bridge. It evaluates no-Medicaid, exact-only, bridge-only, and all
Medicaid feature scopes on validation WAPE, then uses the selected scope for
the held-out test forecast.

### 7b. Annual County Demand Evaluation

`model/arkansas_pharma_signal/county_evaluation.py` (CLI
`evaluate-county-demand`) evaluates annual county-drug demand against the real
`county_demand` artifact: 301,361 county-year-drug rows, 73 counties, 1,577
drugs, years 2013-2024.

Strict pointwise result: train 191,059 / validation 22,994 / test 45,372 rows,
1,326 drugs in the evaluated view. Selected `ridge_log1p` test WAPE `0.118587`
versus previous-year naive `0.118902` (`0.27%` improvement);
`publishable_candidate` false. Rolling: 5 folds, mean improvement `0.12%`,
`publishable_rolling_candidate` false.

Scope: this is annual county-level demand evidence only. It is not evidence
for weekly/monthly county forecasting, supplier-specific labels, or shortage
accuracy.

All 73 counties in the rebuilt outcome artifact now carry a deterministic
region bucket derived from their sourced county FIPS. The v3 publishable test
suite preserves those region labels for filtering; it does not infer a county
from ARCOS ZIP3 or create a county-by-supplier inventory target.

The same rolling contract is now evaluated independently for each of the five
regions and written to
`model/artifacts/evaluation/county_region_rolling_metrics.json`. The real
artifact has five folds per region: mean improvement versus the previous-year
naive is `0.00%` central, `0.00%` northeast, `0.36%` northwest, `0.00%`
southeast, and `0.00%` southwest. No region clears the publishability gate.
Small regions with insufficient rows are reported as unevaluable rather than
pooled into another region.

### 7c. FDA Supplier-Drug Shortage Event Evaluation

`model/arkansas_pharma_signal/supplier_shortage.py` (CLI
`evaluate-supplier-shortage`) evaluates FDA-reported supplier-drug shortage
event forecasting against the real normalized FDA shortage records under
`data/S_D/data/by_source/fda_shortages`.

The evaluation builds a complete monthly supplier-drug panel from each pair's
first observed month through the global last month; it does not create leading
negative rows before a pair enters the FDA record. It evaluates a strict
next-month view: features at month `t`
(lagged supplier-drug events, prior-only drug-wide event context, supplier/drug
identity encodings, calendar month) predict the
`shortage_event` observed at `t+1` for the same supplier-drug pair. The
current-month event is retained only for the previous-month persistence
baseline and is not a model feature. Supplier/drug one-hot vocabularies are
fit separately on each training slice; identities appearing only in a future
validation/test slice map to all-zero unknown columns. The logistic model is a
`LogisticRidge`; the binary threshold is selected on validation only. Rolling
folds reserve the final pre-test month for that threshold selection.

Verified local run: 789 eligible `shortage_active` records across 2012-2022,
188 supplier-drug pairs, 66 suppliers, 39 ingredients. Strict chronological
split: train 4,304 / validation 1,652 / test 1,457 rows, with event counts
8 / 1 / 4. Test results: previous-month persistence accuracy `0.988332`,
balanced accuracy `0.994150`, F1 `0.320000`, AUROC `0.994150`, AUPRC
`0.168670`; logistic ridge accuracy `0.997255`, balanced accuracy `0.500000`,
F1 `0.000000`, AUROC `0.627065`, AUPRC `0.004358`. Rolling-origin evaluation
runs 26 folds; the logistic mean AUROC is `0.560525` versus persistence
`0.727945`, and sparse positives make raw accuracy non-acceptance evidence.
The drug-wide context is the maximum FDA event across suppliers for the same
drug, shifted and rolled over prior months only; it is a valid research input,
but it did not improve the rolling gate and is not a promoted supplier model.

An additional Arkansas-exposure supplier view filters the same archive to NDC9s
observed in the Arkansas Medicaid exposure panel, retaining national supplier
identity without claiming county allocation. Its current artifact covers 75,683
panel rows, 865 NDCs, and 161 suppliers across 37 folds; mean AUROC is `0.521`
and balanced accuracy is `0.500` under a roughly 98% active-label rate. It is
therefore an upstream supplier-risk signal, not a promoted shortage classifier.

Scope: this is FDA-reported supplier-drug event forecasting only. A zero
label means no observed FDA event for that supplier-drug pair in that month,
not a confirmed absence of shortage. It is not pharmacy inventory, not county
allocation, not Arkansas-specific shortage allocation, and it is not
publishable evidence.

The censor-safe public archive rerun is serialized by
`model/scripts/evaluate_publishable_supplier_shortage.py`. It covers 254,316
panel rows, 2,551 drugs, and 299 suppliers from 2012-01 through 2026-07. The
protocol excludes 2,698 right-censored next-month targets and scores 131,533
observed targets across 37 rolling folds. Because 99% of scored targets remain
shortage-active, raw accuracy is rejected as a success criterion: the rolling
logistic model has balanced accuracy `0.500` and mean AUROC `0.517`, so this
layer is upstream supplier evidence only and is not promoted.

For the Arkansas-exposed quarterly shortage-onset task, a class-weighted
logistic candidate and validation-selected balanced-accuracy threshold were
added as a rare-event experiment. After adding the observed supplier-count
and archive-row evidence fields, validation selects the weighted candidate,
but the real held-out result still has balanced accuracy `0.500` and AUROC
`0.549`. Class weighting therefore cannot be claimed as a shortage-model
improvement.

The active-state task now also has a six-fold rolling evaluator. It reaches
mean raw accuracy `99.42%` and balanced accuracy `92.38%`, satisfying the
descriptive 75% boolean threshold, but its balanced-accuracy improvement over
previous-quarter persistence is `-0.023` percentage points and AUROC
improvement is `0.065` percentage points. The label is predictable largely
because shortage state persists; `skill_claim_supported` and
`promotion_candidate` therefore remain false.

The active-state model does gain ranking/calibration evidence from the same
lag-safe fields: AUROC rises from `0.9528` to `0.9603`, AUPRC from `0.8521` to
`0.8883`, and Brier score falls from `0.00531` to `0.00497`; balanced accuracy
remains `0.9528`. This is retained as qualified upstream evidence, not as a
promoted inventory forecast.

### 7d. Arkansas ARCOS Regional Distribution Proxy Evaluation

`model/arkansas_pharma_signal/arcos_evaluation.py` (CLI
`evaluate-arcos-regional`) evaluates a strict next-quarter model on the public
DEA ARCOS Report 01 artifact. The source contains quarterly reported grams by
Arkansas ZIP3 and controlled-substance code. It is explicitly a distribution
proxy, not direct pharmacy demand or inventory. The 2011 source gap is retained
as missing; rows are targets only when the next calendar quarter is observed.

The ridge model predicts log1p grams using current and lagged history,
seasonal context, and training-slice-only ZIP3/drug one-hot identities. It is
compared with previous-quarter and seasonal-naive baselines. The verified run
has 9,003 / 1,510 / 3,624 train/validation/test rows, 85 ZIP3s, and 39 drug
codes. The selected pointwise model has test WAPE `0.205728`, a `38.301%`
improvement over the strongest naive. A validation-selected rolling policy
using 4-quarter validation windows has 15 folds and mean WAPE `0.233688`, a
`16.700%` improvement over the strongest naive rolling mean, passing the local
rolling candidate threshold. Early folds select persistence and later folds
select ridge; this is a model-selection result, not evidence that a fixed ridge
model wins every period. This layer is not evidence that the overall
architecture has achieved the requested 75% accuracy or publishability.

The universal forecast emits optional `geography_level=zip3` rows with target
`regional_distribution_pressure` when an ARCOS drug name maps exactly to a
canonical panel drug. These are horizon-zero observed context rows, not future
inventory predictions. It also emits `regional_distribution_pressure_forecast`
rows at a 91-day horizon using the strict next-quarter ARCOS evaluator's
validation-selected ridge or temporal baseline. The forecast is produced only
for ZIP3/drug pairs observed in the global latest ARCOS quarter; stale pairs
are excluded from current scoring rather than forecast from their last
historical observation. Both rows retain explicit proxy semantics: ARCOS
reports controlled-substance distribution grams, not direct pharmacy demand or
inventory.

The separate `evaluate_arcos_state.py` layer encodes next-quarter log grams
using low/mid/high tertiles learned only from each fold's fit period. The
state artifact contains 15 rolling folds and 11,503 held-out ZIP3/drug rows;
the validation-selected policy reaches 93.51% exact and 93.36% balanced
accuracy across 39 controlled-substance codes and 85 ZIP3s. Persistence is
selected in every fold, so this is a stable quarterly distribution-context
metric, not evidence that the learned ARCOS head improves forecasting. The
numeric within-5%-error ARCOS target remains separately rejected by the metric
contract.

## Output Contract

Required output columns:

- `forecast_run_id`
- `forecast_created_at`
- `forecast_date`
- `horizon_days`
- `geography_level`
- `geography_id`
- `geography_name`
- `drug_key`
- `drug_name`
- `ingredient_key`
- `ingredient_name`
- `supplier_key`
- `supplier_name`
- `disease_key`
- `disease_name`
- `target`
- `prediction`
- `prediction_interval_low`
- `prediction_interval_high`
- `risk_score`
- `model_family`
- `driver_summary_json`
- `source_feature_window_start`
- `source_feature_window_end`

The forecast writer should support 100 to 10,000+ rows per run. More rows are preferred when supported by source coverage and compute.

## Backtesting

Backtests must use time-based splits only:

- no random split across time,
- train on earlier dates, validate on later dates,
- evaluate COVID/flu/disaster/shortage stress windows separately,
- compare against simple regression and seasonal baselines.
- for annual shortage risk, run rolling-origin validation/test folds in
  addition to the single strict split.

Metrics:

- MAE, RMSE, WAPE/sMAPE for demand,
- AUROC/AUPRC/Brier score for risk targets,
- calibration by decile,
- top-k shortage-impact recall, precision-at-k, lift-at-k, and positives
  captured,
- Arkansas-region fairness/residual checks.

## Production Constraints

- Target runtime: hundreds to thousands of words of news plus structured features to final variables in under 8 minutes.
- Batch forecast path must avoid expensive language-model calls at scoring time when possible.
- Cache all extracted text/event features with source hashes.
- Fail closed: when a source is missing, report missingness and use available model families; do not fabricate values.

## Operational Input Boundary

Periodic-only features remain in the panel for research ablations but are
excluded by default from training/evaluation. The opt-in flag is
`--include-periodic-training-features`. Every metrics artifact must record
`feature_mode` and `operational_ready`.

## Canonical Public Test Suite

`data/targeted_additions/publishable_test_dataset/` is the canonical whole
system evaluation setup. Build it with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/build_publishable_test_dataset.py --root .
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_test_dataset.py
```

The suite verifies SHA-256 manifests, duplicate keys, consecutive target
periods, source-grain semantics, and target censoring before evaluation. It
reports separate demand, shortage, and distribution-proxy results for the
Arkansas NDC-quarter, county-drug-year, supplier-NDC-month, and ARCOS ZIP3-
drug-quarter tasks. These scores must not be
averaged into a single accuracy because the targets and time grains differ.

The public data do not observe county-by-supplier-week pharmacy inventory; the
suite therefore does not synthesize that target. A model may be promoted only
after it beats the task-specific persistence baseline under the documented
held-out and rolling-origin gates.

The suite metrics now carry a machine-readable `target_adequacy` section. It
records the missing local-inventory target as unavailable and reports that the
current uncensored supplier evidence has only 101 non-shortage feature rows
and 96 observed next-month shortage positives. This prevents the thin onset
slice from being treated as a publishable shortage-onset benchmark.

The versioned suite also includes 14,137 DEA ARCOS ZIP3-by-controlled-
substance next-quarter transitions. Its rolling benchmark improves 16.70%
over the best temporal baseline across 15 folds. This is a geographic
distribution proxy only: ARCOS Report 01 has no supplier field and does not
observe pharmacy inventory. The numeric target is not promoted under the
within-5%-error contract; the promoted representation is the separately
evaluated low/mid/high state proxy.

The next predeclared robustness experiment applied Huber IRLS to the
persistence-residual stack and applied train-only recency weights. The grid
was Huber multiplier `{0.7, 1.0, 1.5}` by recency gamma `{0.5, 0.7, 0.85,
1.0}`, selected by validation WAPE only. The one-year rolling improvements
were `81.10%`, `70.40%`, and `-6.12%`, with mean `48.46%`; the latest fold
selected residual scale `0.2` and still failed the raw-ridge comparison. This
experiment remains research-only and does not establish the 75% target or
promotion readiness.

### Corrected NDC9 rolling-model verification

After the public suite's NDC9 normalization defect was fixed, the layered
model was retrained on the corrected panel. The three one-year rolling folds
contain 64,116 held-out NDC-quarter rows. The validation-gated neural/stacked
blend beats the fold Ridge baseline by a mean 62.37% WAPE improvement and
clears the model-family research gate on every fold. This is not the revised
metric-contract promotion gate: the artifact reports WAPE, not the required
numeric within-5-percent accuracy or three-state accuracy, and the model
still lacks direct pharmacy inventory labels. The result is therefore a
corrected architecture benchmark, not evidence that the 75% project target
has been achieved. The contract-aware rerun reports mean stacked-blend
within-5-percent accuracy of only `1.43%` across the three folds, with fold
values from `1.17%` to `1.60%`; its explicit `contract_75pct_numeric_accuracy`
flag is false.
