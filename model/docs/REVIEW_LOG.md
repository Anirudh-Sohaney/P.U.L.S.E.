# Review Log

## 2026-09-14 Pass 365: Part A Prototype Audit

The prototype objective was narrowed to the larger external-signal model. The
existing publishable evidence contains at least twelve signal heads above the
70% prototype threshold under chronological held-out testing, including CMS,
FDA, CDC, Arkansas APCD, HHS, and RxNorm-derived outputs. The results are
listed in `PART_A_STATUS.md` with their metric definitions and sources.

This does not yet establish the intended architecture: the passing heads are
mostly signal-specific models or persistence-selected proxy forecasts rather
than ten independently evaluated outputs from one news/NLP/neural layer. The
news implementation currently provides auditable event extraction, disease
states, relevance features, and a compact NumPy MLP, but no ten-signal
held-out accuracy report. Part A remains candidate-signal ready and
architecture-evidence incomplete.

The dated 20-column news-only output was restored from repository history at
`existing_models/news_signal_model/data/derived/signals_monthly.csv`. The
adapter check found 97 monthly rows, covering 2018-01 through 2026-01, with no
missing signal values. The original FLAN-T5 inference code, weights, and
validation artifacts remain unavailable; historical completion metadata is
therefore not counted as current publishable evidence.

The current external-reference run scored the generic outbreak trigger on
2,670 expert-labeled animal-health article/sentence units: 50.60% exact
accuracy, 59.82% balanced accuracy, and 83.43% precision. The result fails a
70% exact-accuracy gate and is not presented as Arkansas human-pharmacy
validation.

The first unified Part A news-head evaluation was run on real local HHS, FDA,
and CDC data. It shifted the restored 20-column news output by one calendar
month, trained one logistic head per target through 2021-12, and scored later
observations. The news-only control joined 96 months, evaluated 18
nonconstant targets, and produced five promotion candidates after majority
comparison. The result is stored in
`model/artifacts/evaluation/part_a_news_heads_news_only.json`.

An augmented unified run added each target's previous-month observed value as
a structured feature while preserving the one-month news lag. It produced two
promotion candidates among seven evaluable heads: the existing HHS drug
`16571040250` head and HHS drug `51672407008` at 97.14% exact accuracy, 1.00
true-positive precision, and 35 test rows. The second result is
persistence-driven and is not evidence of news-only predictive lift. The
artifact is `model/artifacts/evaluation/part_a_news_heads_lagged.json`.

After correcting the CDC period field to use `epiweek` rather than the release
`issue` code, the augmented run evaluated 18 heads and produced 11 promotion
candidates: four CDC FluView level heads, five national age-band heads, and
two HHS drug-activity heads. This meets the prototype 10-head threshold for a
combined external-input layer, but the result is persistence-heavy and does
not establish news-only causal lift.

The CDC target join was corrected to use `epiweek` as the observation period;
the prior implementation incorrectly used the release `issue` code and
collapsed weekly data into one month per year. The corrected lagged run
evaluated 11 heads across 96 joined months and produced six promotion
candidates: four FluView heads and two HHS drug-activity heads. This is a
materially better coverage result, but the gains remain persistence-heavy and
do not prove that news caused the predictions. The corrected artifact is
`model/artifacts/evaluation/part_a_news_heads_lagged.json`.

The grouped ten-head expert-news evaluation also completed. The split was
article-disjoint: 61 training articles/923 sentences and 27 test
articles/321 sentences. Ten binary heads were evaluated for event and
information labels. Several raw accuracies were above 70%, but all were
majority-baseline or precision failures; **0/10** became promotion candidates.
The artifact is `model/artifacts/evaluation/expert_news_heads.json`. CIRAD
does not provide usable publication timestamps, so this is valid NLP transfer
evidence but not chronological pharmacy-outcome evidence.

## 2026-09-14 Pass 364: Documentation and Expansion Baseline

The repository was re-audited after synchronization with the GitHub `main`
branch. Documentation now describes the system as Arkansas-first rather than
Arkansas-only and records national, neighboring-state, and global data as
context that must retain its original geography. The data overview now covers
the normalized source families and the point-in-time/provenance rules used by
the model. The DEA ARCOS references supplied for expansion are documented as
distribution proxies, not pharmacy dispensing or inventory truth.

The code inventory includes dedicated adapters for CMS, Medicaid, FDA, CDC,
ARCOS, news, weather, provider geography, supplier hierarchy, and universal
forecasting. After restoring Git LFS data, the expert-gold builder returned
2,670 rows as expected and the focused expert-gold, input-contract, and ARCOS
suite passed 52 tests, including CLI coverage and the new annual ARCOS
feature-join test. This
audit does not promote any model or signal.
Forecast-quality claims remain subject to chronological held-out evaluation
and the project still lacks public local on-hand, stockout, backorder, or
allocation labels.

The restored panel contains 544,070 rows and 136 columns, including 173
operational evaluation features after the strict next-period transformation.
The non-neural annual run selected the calibrated ridge bridge at WAPE
`0.145785`, compared with `0.146271` for the strongest naive baseline. This is
only `0.33%` improvement, so the expanded ARCOS signal remains unpromoted.

## 2026-08-17 Pass 363: HHS Labeler Aggregation Screen

The HHS Arkansas pharmacy-NDC panel was also aggregated by the first five NDC
digits as a labeler proxy. This is not treated as a fulfillment-supplier
identifier. With a 60-month minimum history, 21 rolling folds scored 2,556
held-out transitions at 58.793% exact five-state accuracy, 10.279% numeric
within-5% accuracy, and 75.310% precision for states 3-4. Because raw
accuracy remained below the 65% event-route floor, the candidate was rejected.


## 2026-08-17 Pass 362: HHS County-Grain Evaluation

The HHS pharmacy-NDC extraction was extended through the provenance-bearing
NPPES practice-city/Census-geocoder county crosswalk. The county panel contains
24,050 observed county-NDC-month rows, 72 mapped counties, and 766 drugs with
eligible histories. A strict next-month evaluator scored 57 rolling folds and
8,813 held-out transitions: 43.586% exact five-state accuracy, 11.063% numeric
within-5% accuracy, and 53.136% precision for states 3-4. Unresolved mappings
and HHS-suppressed cells were excluded, not imputed. The source remains
training-only and is not promoted.

## 2026-08-17 Pass 359: Explicit Local Pharmacy History Disposition

Corrected the input contract so pharmacy-supplied history is classified as
`LOCAL_PHARMACY_HISTORY_INPUT`, rather than being mislabeled as periodic public
training data. The refreshed modular artifact now records 20 public
near-real-time inputs, 10 local-history inputs, 3 periodic training-only
fields, and 1 static identity field. Local history remains usable only when
the downstream pharmacy supplies it; it is not claimed as a free external
feed. Input-contract, training, and next-period tests passed (71 tests).

## 2026-08-17 Pass 358: Absolute-Level State Route and Five-State Shortage Pressure

Added an explicit absolute-demand-level state mode to the canonical public
NDC demand evaluator. The existing change-state head remains available and is
still rejected at approximately 22% exact accuracy, while the numeric demand
head projected through training-only absolute five-state thresholds reached
72.24%, 72.51%, and 72.94% exact state accuracy across three chronological
folds. High-state true-positive precision was 82.99%, 83.67%, and 85.42%; all
folds therefore pass the declared 65% raw plus 80% event route. This is a
learned demand-level signal, not a direct pharmacy inventory label.

The event-route floors are now implemented in `event_accuracy.py` and checked
by the project-status audit. The canonical whole-system artifact uses the
level-state evaluation and records `contract_80pct_true_positive=true`.

Replaced the collapsed three-state FDA supplier-pressure target with five
ordered states: none, one, two, three, and four-or-more active shortage
suppliers. The 145-fold monthly evaluation produced 419,806 held-out rows,
92.09% exact accuracy, 80.38% balanced accuracy, and all five states in the
scored data. High-state precision was 72.77%, so this metric qualifies through
the raw route and is now included in the operational feature surface.

Regenerated the universal forecast schema, qualified metric forecast, feature
store, and metric audit. The surface now contains 10 qualified targets and
45,614 rows before grain deduplication, with 44,029 serialized feature rows.
Focused tests passed, followed by a clean full repository suite with **330
tests passed and 23 warnings**. The remaining project-status failures are the
older annual demand rolling gate, legacy shortage Boolean diagnostic, and
legacy modular rolling gate. Added a machine-readable research-exhaustion
record; it is intentionally incomplete because additional pharmacy-relevant
families and incremental-utility tests remain unresolved. Legacy failures are
now retained as non-blocking diagnostics rather than conflated with the active
five-state/numeric completion contract.

The corrected five-state recall-to-shortage ablation was rerun after finding a
stale three-class evaluator. Recall augmentation improved mean balanced
accuracy from 60.97% to 61.87% across 10 chronological folds (+0.91 points),
but failed the all-fold improvement condition. The result remains mixed
utility evidence, not proof of individual-pharmacy regression lift.

## 2026-08-16 Pass 313: Screen Remaining Numeric Replacements

Added numeric evaluators for Arkansas hospital influenza admissions and annual
regional CMS Part D demand claims. Hospital admissions reached 9.32% within
5% across four weekly folds; regional claims reached 20.69% across six annual
folds. Both fail the 65% numeric floor. Refreshed the ARCOS rolling evaluator,
which reached 26.82% within 5% across 15 quarterly folds, and the existing
numeric FluView evaluator remains at 11.07% across seven weekly folds. These
four numeric replacements are rejected rather than promoted; their legacy
state rows remain compatibility outputs only. The audit still reports three
qualified numeric metrics.

Focused screening and audit verification passed with 12 tests. The complete
suite will be rerun after this evaluator-only change.

## 2026-08-16 Pass 312: Promote Numeric Shortage Supplier Count

Replaced the legacy NDC shortage-pressure state in the operational forecast
builder with the numeric next-month FDA supplier count. The leakage-safe
rolling evaluator produced 121 folds, 1,550,662 held-out rows, and 99.0321%
within-5-percent accuracy. Persistence selected on validation, so the signal is
an accurate upstream proxy rather than evidence of incremental learned skill.

The audit now reports three qualified numeric metrics: NDC shortage supplier
count, NDC recall severity, and supplier-by-NDC recall severity. The rebuilt
surface remains 11,686 rows and seven metric columns, with four legacy state
columns still retained for compatibility. Focused verification passed with 25
tests, the canonical artifact check passed, and the complete suite passed with
**301 tests and 22 warnings**. Weekly/monthly coverage passes; the per-drug
and Arkansas-region gates remain unmet.

## 2026-08-16 Pass 311: Promote Numeric Recall Surface

Replaced the two legacy recall-state outputs in the operational forecast
builder with the evaluated numeric NDC recall-severity and supplier-by-NDC
recall-severity targets. The promoted surface remains at 11,686 rows and seven
metric columns; two columns now carry `qualified_numeric_proxy` status while
the remaining five are retained as legacy compatibility signals pending their
numeric replacements. The feature-store status allowlist and historical metric
context names were updated without changing the 14-wide value/missingness
model-context shape.

The metric audit now counts the two integrated numeric recall signals as
qualified. The rebuilt forecast and feature-store artifacts preserve unique
filter grains and source provenance. Focused integration tests passed with 27
tests, the canonical whole-system artifact check passed, and the complete
suite passed with **300 tests and 22 warnings**. The remaining five state
columns remain legacy compatibility outputs and are not counted as newly
promoted metrics.

## 2026-08-16 Pass 310: Numeric Recall Evaluation

Added a leakage-safe rolling numeric evaluator for FDA recall severity. It
uses the existing monthly NDC and supplier x NDC panels, scores WAPE and
within-5-percent error, compares persistence against the one-vs-rest predictor
using validation-only selection, and preserves exact source chronology.

The local archive produced 10 folds and 214,976 held-out NDC rows at 99.9788%
within-5-percent accuracy. The supplier x NDC panel produced 10 folds and
226,122 held-out rows at 99.9881% within-5-percent accuracy. Both results are
persistence-dominated and the adapters are not yet the promoted forecast
surface, so the audit records them as `candidate_numeric_proxy`; neither is
counted toward the qualified metric total. Artifacts are written to
`recall_numeric_rolling_metrics.json` and
`supplier_recall_numeric_rolling_metrics.json`. Focused verification passed
with 10 tests. The complete repository suite then passed with **299 tests and
22 warnings**.

## 2026-08-16 Pass 309: Numeric Candidate Output Surface

Added a separate numeric candidate adapter surface for the existing public
panels: FDA shortage supplier count, Arkansas FluView WILI, Arkansas hospital
influenza admissions, regional CMS Part D claims, NDC recall severity, supplier
x NDC recall severity, and DEA ARCOS distribution grams. These rows retain
their declared grain, cadence, source freshness, and provenance, and use
`candidate_numeric_proxy` rather than a qualified status. They therefore cannot
enter the feature store or inflate the promoted metric count before a
chronological numeric evaluation is completed.

The legacy state adapter remains available for historical comparison only.
Focused adapter, serializer, and feature-store verification passed with 18
tests. The complete repository suite then passed with **298 tests and 22
warnings**. No candidate was promoted and no accuracy claim changed.

## 2026-08-16 Pass 308: Five-State Neural Target Head

The model-level categorical path previously defaulted to three target states,
which did not enforce the revised output contract. `ModularModelConfig` now
defaults the target-specific head to five states. Training labels use four
quantile thresholds fitted on the training slice only, preserving chronological
leakage controls; state metrics validate and report states `0` through `4`.
The legacy three-channel aggregate risk tensor remains solely for compatibility
and is not the contract-facing metric head.

The existing public forecast adapters and evaluation artifacts still contain
three-state legacy outputs. They remain excluded from promotion until numeric
or five-state evaluators and adapters replace them. Focused verification
passed with 42 tests, followed by the complete repository suite with **297
passed and 22 warnings** in 8m21s. No accuracy claim has changed.

## Pass 303: Reject Non-Vintaged Overdose Evaluation

The CDC VSRR overdose candidate initially appeared to pass the proxy accuracy
floor, but its downloaded historical rows all carried one later
`data_as_of=2026-07-05` snapshot. That cannot support a publishable rolling
next-month test because historical forecast origins would receive revised
future information. Added a vintage-enforcement guard that fails closed,
returns a structured rejection artifact, and keeps the latest source only as a
non-promoted live-context candidate.

The qualified surface was rebuilt from 11,705 back to 11,686 rows and seven
features. The metric audit now reports seven qualified targets; the overdose
candidate has zero eligible folds and explicit vintage rejection reasons.
Focused tests passed (19 tests). Whole-system evaluation passed, and the
complete suite passed with 291 tests and 18 existing warnings in 8m19s.

## Pass 302: Cadence-Correct Forecast Horizons

Audited the filterable output contract and found all qualified future rows had
`horizon=0`, despite representing future weekly, monthly, quarterly, or annual
periods. Added positive cadence-derived horizons: 7, 30, 91, and 365 days,
respectively. The serializer infers a cadence horizon when callers omit it and
rejects non-positive qualified horizons.

The rebuilt 11,705-row artifact reports the correct horizon for all eight
targets. Focused output, feature-store, and universal tests passed (30 tests),
whole-system evaluation passed, and the complete suite passed with 290 tests
and 18 existing warnings in 8m17s.

## Pass 301: Preserve Source Publication Freshness

Audited the live output against source-edge dates and found two provenance
defects in the new overdose adapter: transition-only evaluation was being used
for the live source edge, and the serializer overwrote `source_freshness` with
the forecast period. The adapter now forecasts from the latest observed county
row while retaining strict consecutive transitions for evaluation, and the
serializer preserves the supplied source freshness with a forecast-period
fallback only when it is absent.

The CDC source edge now emits a `2026-01` forecast with `source_freshness=2025-12`
instead of falsely presenting the source as current. Focused tests passed (18
tests), and whole-system evaluation passed. The complete suite then passed with
290 tests and 18 existing warnings in 8m19s.

## Pass 300: CDC County Overdose Context Candidate (Withdrawn)

Researched and archived the public CDC VSRR provisional county-level overdose
dataset (`gb4e-yj24`). The source contains Arkansas county-month observations,
but suppresses many values and exposes a rolling-12-month provisional count,
not medication-specific dispensing. Added a strict consecutive-observation
adapter and leakage-safe next-month low/mid/high evaluator; suppressed rows
are never treated as zero.

The candidate produced 46 rolling folds and 1,023 held-out transitions across
51 counties, with 96.87% exact and 65.62% balanced accuracy. Seventeen
counties had at least 25 scored rows, so it passed the current external-proxy
contract but remains explicitly non-pharmacy and persistence-dominated. It is
This candidate was temporarily integrated, but Pass 303 withdrew it after the
vintage audit. It is not part of the qualified surface or feature store.

## Pass 299: Honest Qualified-Metric Uncertainty Contract

Audited the live qualified forecast surface and found every persistence row
was emitting `confidence=0.50` and zero-width intervals without calibration or
coverage evidence. Replaced those misleading values with null uncertainty
fields plus explicit `uncertainty_status=not_estimated` and
`calibration_status=not_calibrated`. Added validation that rejects unvalidated
confidence, risk, or interval values on qualified metrics, preserved the
legacy universal schema with an explicit `legacy_unvalidated` status, and
recorded the statuses in feature-store metadata.

Rebuilt `qualified_metric_forecasts.csv.gz` and its feature store: 11,686 rows,
seven qualified targets, and no qualified confidence/interval/risk claims.
Focused contract, forecast, feature-store, and universal tests passed (29
tests). The complete suite then passed with 286 tests and 18 existing warnings
in 8m24s.

## Pass 208: Three-State Metric Qualification

Added a monthly NDC shortage-pressure state derived from the public FDA
supplier/NDC archive. The target has three states: no active reporting
supplier, one active reporting supplier, or at least two active reporting
suppliers. A one-v-rest logistic-ridge head is selected by chronological
validation, and the evaluator records exact accuracy, balanced accuracy,
state counts, and per-NDC held-out accuracy.

The archive evaluation produced 145 rolling folds and 419,806 held-out rows
across 2,551 NDCs. Mean exact state accuracy was 99.23% and mean balanced
accuracy was 80.35%; 2,425 NDCs had at least 25 scored rows and at least 75%
exact accuracy. This is a qualified external supply-pressure proxy, not local
Arkansas inventory or supplier allocation.

Added a three-state Arkansas FluView WILI signal with training-only tertile
thresholds. Across 7 weekly folds and 342 held-out weeks, validation-selected
learned/persistence forecasts reached 84.51% exact accuracy and 75.999%
balanced accuracy. The selection method is recorded per fold; persistence is
not hidden when it wins validation.

Both signals now have universal output metadata and a schema-preserving
filterable-row helper. The metric audit reports two qualified proxies and all
three Part 1 gates as passed. This does not establish direct pharmacy
inventory accuracy; the proxy limitations remain part of the output metadata.

## Pass 210: Strict Arkansas Regional Gate

Statewide FluView was not accepted as evidence of a per-region metric. Added a
separate annual CMS Part D demand-state evaluator that aggregates county
claims to the five published Arkansas DHS/TEFRA regions by drug. It uses
training-only tertiles and validation-selected one-vs-rest or persistence
forecasting.

The evaluator produced 6 rolling folds, 21,304 held-out region-drug rows, all
5 Arkansas regions, and 1,308 drugs. Mean exact state accuracy was 92.01% and
balanced accuracy was 92.0%. The metric audit now requires five Arkansas
regions for the regional Part 1 gate; all three Part 1 gates remain passed.
This metric is annual context, explicitly lower priority than the weekly and
monthly candidates.

## Pass 211: Operational Qualified-Metric Surface

Added `qualified_metric_forecasts.py` and its build script. The adapter emits
next-period rows for the three qualified metrics using the validation-approved
persistence baseline until a production checkpoint is supplied. Every row has
an explicit forecast period, stable geography and drug keys, uncertainty
fields, source freshness, model method, and proxy semantics.

The live-style build currently emits 1,585 current FDA shortage NDC rows,
3,032 current recall NDC rows, one Arkansas FluView row, one hospital-
influenza row, and 3,680 current region-drug rows for annual demand context.
It excludes stale NDC/region-drug pairs and performs duplicate-key validation.
No county or supplier allocation is inferred.

## Pass 212: Recall-Pressure Qualification

Used the locally available FDA enforcement archives to extract NDC identifiers,
carry each recall from initiation through termination or the source edge, and
construct a complete NDC-month panel. The three-state target is no active
recall, active Class II/III recall, or active Class I recall.

The wider rolling protocol produced 10 folds and 275,296 held-out rows across
3,877 NDCs. Mean exact accuracy was 99.99% and balanced accuracy was 99.57%.
The selected method is predominantly persistence, so the signal is promoted
only as recall-severity context. It is not a claim about pharmacy stock or
recall absence.

## Pass 214: Cross-Signal Utility Ablation

Added a leakage-safe FluView/wastewater ablation. The aligned Arkansas data
contains 336 weekly rows. Across two chronological folds, the FluView-only
state model averaged 84.45% balanced accuracy; adding current-week mean
influenza wastewater activity reduced it to 76.97%, a -7.48 percentage-point
delta. Wastewater is therefore not promoted as a metric or as evidence of
incremental pharmacy-model utility, although it remains a documented live
input candidate.

## Pass 215: Normalized Recall-to-Shortage Utility

Corrected FDA NDC normalization to standard 5-4-2 segment padding before
joining recall and shortage archives. The overlap increased enough for a
separate utility test: 240 NDCs, 10 monthly folds, and 12,188 held-out rows.
Adding lagged/current recall severity reduced balanced accuracy from 66.06% to
65.95% (-0.12 percentage points). Recall remains a standalone filterable
context metric, not evidence of incremental shortage-model improvement.

## Pass 216: Grain-Preserving Feature Store

Added `metric_feature_store.py` to pivot qualified forecast rows into model
features while preserving period, geography, drug, and optional supplier keys.
The adapter rejects unqualified metrics, duplicate grains, and missing hard
keys; it only normalizes serialized empty optional county/labeler fields back
to empty strings. The current 8,299-row forecast artifact round-trips into a
five-feature store without duplicate keys or implicit geographic broadcasting.

## Pass 213: Wastewater Screening and Baseline-Utility Audit

Screened the locally downloaded CDC Arkansas wastewater site panel. It has
3,397 consecutive site-pathogen weekly transitions. A collapsed influenza
low/mid/high persistence state exceeds 84% raw exact accuracy, but balanced
accuracy remains below 60% because the low state dominates. It is therefore
retained as a real-time input/context family and rejected as an established
metric under the anti-imbalance safeguard.

The four promoted metrics remain heavily persistence-dominated. Their accuracy
demonstrates predictable external states, not incremental improvement to a
pharmacy's private inventory regression. This distinction is now explicit in
the metric research matrix and remains an open completion requirement.

## Pass 217: Full Repository Verification

Ran the complete repository test suite after the qualified metric, utility,
forecast-output, and feature-store additions. The suite completed with 233
passed tests and 16 warnings in 459.99 seconds. No test failures occurred.
The warnings are existing model-framework, pandas fragmentation, and CSV dtype
warnings; they do not change the metric qualification gates.

## Pass 218: CDC NNDSS Candidate Rejection

Screened the locally downloaded CDC NNDSS Arkansas weekly table as a possible
disease-specific pharmacy-demand proxy. The official `m1` field is the current
week count, so it avoids confusing the rolling maximum and cumulative fields
with a weekly target. The local extract begins in 2025; three short
chronological folds yielded 200 held-out rows across nine labels. No label
reached 65% balanced accuracy, and only one label showed all three states in
the scored rows. The candidate is documented and rejected rather than added to
the metric library.

## Pass 219: Explicit Proxy-vs-Utility Audit Fields

Extended the metric audit records with `baseline_skill`, including the mean
balanced-accuracy delta against persistence, whether every fold beats
persistence, and an incremental-utility status. The four qualified records
remain qualified as predictable external proxies, but their status now makes
the limitation machine-readable: the tested cross-signal additions are
negative or non-improving, and no private pharmacy inventory benefit is
claimed. Added regression tests for this distinction.

## Pass 220: Grain-Safe Training Integration Seam

Added `attach_qualified_metric_features`, which joins the validated metric
feature store to a downstream training frame only on the declared period,
geography, county, drug, and labeler keys. It adds per-feature missingness
indicators without imputing absent external evidence. A county row therefore
cannot receive a national NDC forecast through implicit broadcasting. The
current public benchmarks still do not claim pharmacy-level utility because no
matching private inventory label is available.

## Pass 221: CDC Hospital Respiratory Utilization Proxy

Added the CDC NHSN HRD weekly Arkansas extract and a pathogen-specific,
next-week three-state evaluator. The influenza admission-pressure target has
four chronological folds, 187 held-out transitions, 74.16% exact accuracy,
and 68.76% balanced accuracy, so it is promoted as a qualified weekly
healthcare-utilization proxy. COVID fails the balanced-state safeguard at
49.13%, while RSV has only two folds and 38.43% balanced accuracy; both remain
rejected. The raw extract, API query, retrieval metadata, and SHA-256 are
recorded under `data/targeted_additions/cdc_hospital_respiratory_current`.

## Pass 222: Hospital-to-FluView Utility Ablation

Added a leakage-safe cross-signal test aligning CDC hospital week-ending dates
to FluView weeks. Across 602 aligned rows and four chronological folds,
adding current-week influenza hospital admissions reduced FluView balanced
accuracy from 84.07% to 83.11% (delta -0.96 percentage points). The hospital
metric remains a valid standalone utilization proxy but is not promoted as
incremental pharmacy-model evidence.

## Pass 225: Public NDC Demand Context Ablation

Added a rolling public-demand ablation using the observed Arkansas NDC
Medicaid-utilization target. The history-only Ridge baseline was compared with
the same model plus 21 prior disease/news variables across four chronological
folds and 17,903 held-out NDC-quarter rows. Mean WAPE increased from 0.847 to
1.444 (delta +0.598), so the current external-context bundle is rejected as
demonstrated incremental utility. This is a public-demand proxy experiment,
not a claim about private pharmacy inventory.

The family decomposition was also recorded in the same artifact: six disease
features increased mean WAPE by 0.00043 and fourteen news features increased
mean WAPE by 0.01142. Neither family improved every chronological fold.

## Pass 226: Nonlinear Public-Demand Context Control

Added a matched `HistGradientBoostingRegressor` ablation to separate
model-family value from external-context value. Across the same four folds and
17,903 held-out NDC-quarter rows, history-only mean WAPE was 0.8268 and the
context-augmented model was 0.8322 (delta +0.00536). Context failed to improve
every fold, so increasing model nonlinearity did not establish external-signal
utility.

## Pass 227: NADAC History Coverage Audit

Audited the local CMS NADAC transition file before attempting to promote
acquisition-cost pressure. It contains 1,923,472 transitions across 32,518
NDCs, but only transition years 2013, 2021, and 2022; the intended rolling
protocol therefore yields one complete fold rather than three. CMS annual
weekly files are publicly available, but they were not silently mixed into the
versioned benchmark. NADAC remains rejected until a reproducible multi-year
download and hash manifest is added.

## Pass 223: NSSP Regional Coverage Rejection

Screened the CDC NSSP ED trajectory API for a weekly Arkansas county/HSA
metric. The local extract contains 15,352 rows across 76 county labels and 23
HSA labels from October 2022 through August 2026, but all pathogen percentages
are populated only for the statewide `All` row. Regional influenza trend
labels are `Data Unavailable`, so no regional target can satisfy the 25-sample
chronological test. The raw extract, manifest, and rejection artifact are
preserved; no regional metric was promoted.

## Pass 224: Full Repository Verification

Ran the complete repository suite after the hospital respiratory evaluator,
hospital utility ablation, qualified forecast integration, balanced-accuracy
audit safeguard, and NSSP screening additions. The suite completed with 239
passed tests and 16 warnings in 450.04 seconds. The warnings remain limited to
existing PyTorch nested-tensor, pandas fragmentation, and CSV dtype warnings.

## Pass 209: Explicit Period Key

Added `forecast_period` to the universal output contract. Older producers that
only expose `source_freshness` receive that value as the period fallback;
new external metric rows carry the period explicitly. Duplicate detection now
uses period, metric, geography, county, drug, and supplier keys before metadata
projection.

## 2026-08-16 Pass 124: Residual Objective Audit and Absolute-Mode Ablation

Audited the multimodal residual training objective in `training.py`. The point
and quantile heads are trained on `log1p(target) - log1p(base)` where `base`
is either persistence (current-quarter count) or transition (count x
chronologically-pooled quarterly ratio), selected by `baseline_mode`. The
existing Pass 67 comparison covered persistence versus transition anchoring
but never tested the no-residual control, so it could not tell whether
residualization itself is load-bearing.

Added an `absolute` `baseline_mode` that sets `base_log = 0`, making the point
head predict the raw `log1p` level with no temporal anchor. The prediction,
blend, split, and evaluation paths are unchanged; the mode is a matched
control, not a promoted configuration.

Matched three-way run (small research model, one epoch, frozen news, strict
fixed split train <=2020 / validation 2021 / test >2021, 38-feature contract):

| objective | neural test WAPE | blend test WAPE | selected blend weight |
| --- | --- | --- | --- |
| persistence-residual | 0.13200 | 0.10185 | 0.39 |
| transition-residual | 0.12935 | 0.10290 | 0.19 |
| absolute (no anchor) | 0.98278 | 0.09998 (=transition) | 0.00 |

Test baselines: persistence `0.11288`, transition `0.09998`. Findings:

- The absolute objective collapses (neural WAPE `0.98`, essentially mean
  prediction); validation selects blend weight `0.0` and the output falls back
  to the transition baseline. Residualization is therefore necessary for this
  encoder scale and is retained.
- Persistence versus transition anchoring differ by about `0.003` WAPE on the
  neural head and `0.001` on the validation-selected blend. The persistence-
  anchored model's validation-selected policy is marginally better in this
  split, but neither blend beats the pure transition baseline (`0.09998`), so
  no objective mode is promoted.

No measured improvement over the strongest naive baseline: the ablation is
fail-closed and does not change the publishability decision.

Also fixed a latent `NameError` in `train_real_model`: the result dict
referenced `checkpoint_dir`, a variable scoped to the `_train_fold` refactor;
it now uses `fold["checkpoint"]`. Without the fix the trainer crashed after
training completed.

Artifact: `model/artifacts/evaluation/modular_residual_objective_ablation.json`.

## 2026-08-15 Pass 118: Remove Rule-Based Universal Risk Fallback

The universal forecast path was found to apply hand-coded exponential shortage
and recall coefficients when no trained shortage model was supplied, and to add
`0.03` per matched article event to the risk score. Those operations were not
learned from a time-split target and could be mistaken for model forecasts.
The correction removes both heuristic forecast adjustments. A missing risk
artifact now emits an explicitly labeled neutral prior for schema/API testing;
it is not evidence of shortage skill. Matched event IDs and counts remain
evidence fields only. A trained, calibrated risk artifact is required before
the shortage-state output can be treated as a production forecast.

The same boundary is applied to the legacy `forecast.py` path and neighbor-state
context rows: the former no longer computes risk from hand-coded shortage and
recall counts, and the latter reports observed severity/confidence context
without converting event counts into a pseudo-probability.

The binary `shortage_state` rows are also now omitted when the calibrated risk
model is absent. This prevents the neutral `0.5` contract prior from being
thresholded into a false state; production state rows require a fitted model.

## 2026-08-15 Pass 119: Supplier-Exposure Label Audit

Tested whether the Arkansas Medicaid NDC9 exposure bridge could be expanded to
an Arkansas-exposed NDC9/supplier quarterly target using the archived FDA
panel. The joined panel contained 4,873 rows across 757 pairs and 98.4% of
next-quarter continuation targets were positive. Because the archive starts a
supplier/NDC series at its first observed shortage, excluding currently active
rows left zero eligible onset transitions. This cannot support a useful
supplier shortage-onset model; the layer was not added. Supplier rows remain
explicit national FDA evidence with unverified Arkansas allocation until a
source containing non-shortage supplier/product exposure or local allocation
becomes available.

## 2026-08-15 Pass 121: Publishable Evaluation Suite Design

Defined a clean evaluation suite rather than a fabricated Cartesian market
panel. The primary near-term table will use Arkansas Medicaid NDC9-quarter
utilization joined to dated FDA shortage evidence and strictly prior-quarter
CDC FluView/news aggregates. Separate tables preserve the valid grains of
Arkansas county-drug annual demand and FDA supplier-NDC monthly continuation.
Each table will carry explicit target semantics, source identifiers, split
rules, right-censoring/missingness flags, and a machine-readable manifest.
No county-by-supplier-week target will be synthesized because no public source
observes that quantity.

## 2026-08-15 Pass 122: Executable Evaluation Protocol

Added `evaluate_publishable_test_dataset.py`, which loads only the hashed suite
artifacts, verifies their declared grains and consecutive-period contracts, and
evaluates demand and shortage heads separately against validation-selected
ridge/logistic models and persistence baselines. The protocol reserves the
latest periods by time, reports numeric WAPE and under-5%-error rates, reports
balanced classification/ranking metrics for rare shortage states, and writes a
fail-closed metrics artifact. It does not combine incompatible annual county,
quarterly NDC, and monthly supplier targets into one artificial accuracy score.
Supplier continuation scoring excludes target rows flagged right-censored while
reporting the excluded count; the raw table remains unchanged for audit.

## 2026-08-15 Pass 123: Near-Term Demand History Features

Added consecutive-quarter NDC9 demand lags and a two-quarter moving average to
the clean Arkansas test table. A lag is retained only when its source quarter
is exactly adjacent to the feature quarter; missing quarters reset history.
These are derived training variables from observed Medicaid utilization, not
new operational inputs. The evaluator will select among persistence and
history/exogenous ridge candidates using validation only, then score the
held-out periods without changing the public target grain.

## 2026-08-15 Pass 120: Dedicated Deep-Connection Middle Stage

The multimodal model previously passed the three encoded states directly from
cross-modal interaction to the final heads. Added a separate learned
`DeepConnectionReasoner` stage with typed news/graph/temporal role embeddings
and configurable Transformer blocks. Its state is returned independently and
is consumed by the final heads. This creates an auditable additional middle
stage for political/geographical/economic/supply interactions without adding
hard-coded causal rules. It is an architecture milestone only; the existing
rolling predictive gate remains authoritative.

## 2026-08-15 Pass 88: NADAC Multimodal Ablation

Extended the research temporal contract from 30 to 34 features so the
multimodal path can consume the same four exact-NDC NADAC fields as the
statistical quarterly layer. The fields are quarter mean price, observation
count, log price, and consecutive-quarter price change. A matched
`use_nadac_price_features` switch and CLI ablation flag were added; disabled
rows retain explicit zero-valued slots rather than changing tensor shape.

The NADAC multimodal ablation was rejected. In a matched three-epoch fixed
split, validation-selected blend WAPE changed from `0.1125` without NADAC to
`0.1169` with it, and test neural WAPE changed from `0.1388` to `0.2300`.
The default active small checkpoint therefore disables NADAC in the neural
temporal stream while the statistical quarterly layer retains the qualified
input. After regenerating strict six-fold modular evidence, mean improvement
is `-1.85%` with only 2/6 positive folds; the model remains research-only.

## 2026-08-15 Pass 87: NADAC Economic Input Layer

Added a leak-safe CMS NADAC weekly price layer to the quarterly evidence path.
The local source contains `2,719,680` observations and exact normalized-NDC
matches for `4,357` of `5,601` panel NDC identities (`77.8%`). The layer emits
quarter mean price, observation count, log price, and consecutive-quarter price
change; unmatched NDCs remain missing. CLI panel loads now preserve NDC leading
zeroes so the join is reproducible.

Added unit coverage for NDC normalization and price-change construction. The
real evaluation now reports 808 exogenous variables, but the selected policy
and scores are unchanged: pointwise WAPE `0.0940`, unrestricted rolling mean
improvement `9.0%`, and one-year rolling mean improvement `17.6%`. The input is
retained because it satisfies the real-time accessibility contract, not because
it produced unsupported uplift.

## 2026-08-15 Pass 86: Strict Consecutive-Quarter Target Contract

Audited the quarterly panel and found that `build_quarterly_view` labeled
2,768 of 41,269 observed drug transitions as next-quarter targets even though
the next row was separated by one or more missing quarters. This could make a
multi-quarter jump appear to be near-term evidence. The view now requires an
exact consecutive calendar quarter for both the target and the lag; training
history resets after a missing quarter.

Added a regression test for a missing-quarter sequence and regenerated the
pointwise, unrestricted rolling, and one-year rolling artifacts. The strict
view has 38,501 rows. Pointwise selected WAPE is `0.0940`; six-fold one-year
rolling improvement is `17.6%`, while unrestricted future-horizon rolling
improvement is `9.0%` and fails the 10% gate. The modular small-model rolling
artifact was also regenerated: six folds, mean improvement `-1.22%`, not
publishable. The prior `15.4%` quarterly rolling claim is invalidated by this
target-contract correction.

The requested OpenCode model `opencode-zen/deepseek-v4-flash-free` was
attempted through the installed CLI but is unavailable in the configured
provider registry (`ProviderModelNotFoundError`); development continued
independently as instructed.

## 2026-08-15 Pass 85: End-to-End Publishability Audit

Ran the fail-closed publishability audit after regenerating the universal and
supplier artifacts. Added audit gates for the exact Arkansas supplier
projection and the supplier rolling validation protocol.

Current result is `publishable=false` with three substantive failures:
missing expert event gold labels, the annual rolling demand gate, and the
multimodal rolling gate. The new supplier projection, supplier rolling
protocol, output schema, county outcome labels, Layer 1 contract, and
quarterly/ARCOS evidence gates pass. The shortage boolean gate passes raw
accuracy but reports balanced accuracy `0.7105`; sparse-label accuracy is not
treated as sufficient supplier-event evidence.

## 2026-08-15 Pass 84: Universal Supplier Filter Contract

Fixed `forecast-universal --supplier` so it filters both raw
`supplier_drug` rows and exact-drug `arkansas_supplier_drug` context rows.
Added a clinic-style regression test using case-insensitive supplier input.
County and region filters intentionally retain statewide supplier context as
background evidence; they do not convert that context into county or region
allocation.

## 2026-08-15 Pass 83: Arkansas Supplier-Relevance Projection

Added a separate `geography_level=arkansas_supplier_drug` universal-output
surface. FDA supplier-event scores are projected only when the normalized FDA
drug key exactly matches an Arkansas panel drug. The row is statewide,
30-day, low-confidence context with explicit provenance stating that
supplier-to-Arkansas allocation is unverified; no county or regional rows are
fabricated.

The regenerated universal forecast contains `70` such rows alongside `188`
raw `supplier_drug` rows. The new rows target
`arkansas_supplier_drug_shortage_pressure`, preserve supplier and drug keys,
and remain context signals rather than validated Arkansas inventory labels.

## 2026-08-15 Pass 82: Supplier Rolling Threshold Protocol

Corrected the FDA supplier-drug shortage rolling evaluator. Previously each
fold fit on all pre-test months and used a hard-coded `0.5` threshold, unlike
the pointwise path's validation-selected threshold. Each fold now reserves
the final pre-test month for threshold selection, fits only on earlier months,
and records fit, validation, and test windows separately.

Regenerated real evidence: 26 folds, only 8 with positive test events. Mean
logistic AUROC is `0.5605` versus `0.7279` for previous-month persistence;
mean logistic AUPRC is `0.0375` versus `0.1275`. The model remains
research-only. Raw accuracy is not acceptance evidence because the event rate
is sparse, and FDA event absence is not confirmed inventory availability.

## 2026-08-15 Pass 81: NDC-to-Labeler Bridge Audit

Rebuilt the quarterly Medicaid cache with its existing raw identifiers and
normalized numeric NDCs to eleven digits before serialization. The prior cache
had discarded identifiers and pandas had removed leading zeroes. The refreshed
cache retains ingredient, NDC, RxCUI, manufacturer, and mapping-confidence
columns; NDC is present on all `44,071` aggregated rows.

Added an optional FDA-backed NDC labeler entity path. With explicit string
parsing, it resolves `71.4%` of rows across `240` unambiguous labeler entities
using catalog-supported NDC labeler segments only. The five-epoch ablation
expanded manufacturer-linked test rows to `2,038`; that subgroup blend WAPE
was `0.0987` versus `0.0989` for transition, but overall blend WAPE was
`0.10171` versus `0.10164` for the matched control. Cold-start WAPE remained
`0.542`. It is rejected from active use, while the identifier normalization,
string parsing, and cache repair are retained.

## 2026-08-15 Pass 80: Supplier-Linked Subgroup Rolling Audit

Added provenance-bearing prediction metadata and split-specific subgroup
metrics for seen/cold-start identities, ingredient edges, manufacturer edges,
supplier edges, and history availability. Validation and test subgroup scores
are kept separate and are never used for blend selection.

The five-epoch default control had test WAPE `0.543` for cold-start rows,
`0.099` for seen identities, `0.096` for ingredient-linked rows, and `0.063`
for manufacturer-linked rows. The six-fold one-epoch rolling audit confirms
the gap: cold-start WAPE ranged `0.42` to `0.68`, while manufacturer-linked
WAPE ranged `0.063` to `0.094`. Overall selected-policy improvement was
`-1.07%`, with `1/6` folds beating the strongest naive baseline.

Supplier-linked and manufacturer-linked groups are currently identical
because no separate API/ownership edge resolves to the evaluated drug source
identities. These subgroup results are diagnostic evidence, not a claim that
the model meets the overall gate.

## 2026-08-15 Pass 79: Prioritized Canonical Graph Ablation

Audited the canonical graph and found direct `contains` edges for about 1,000
quarterly drug identities and direct `manufactured_by` edges for about 240,
but the four-node research path selected edges by file order. Added an
optional eight-node graph mode that prioritizes observed `contains`,
`manufactured_by`, `represents`, and supplier-hierarchy relations without
creating new edges. Added tests for ordering and relation IDs.

On the same five-epoch split, the prioritized graph produced neural test WAPE
`0.12895` and validation-selected blend WAPE `0.10167`; the matched default
control produced `0.12525` and `0.10164`. The graph mode is rejected from
active use pending a subgroup or rolling benefit, but remains available as a
provenance-preserving research ablation.

## 2026-08-15 Pass 78: Shared Drug-Token Representation Ablation

Added an optional graph representation with up to six shared normalized
drug-name token entities. Tokens reuse stable IDs across products, while
observed canonical graph edges remain separate; no inferred ingredient,
supplier, or disease edges are created. Added contract tests for shared token
IDs and relation provenance.

On the same five-epoch split and unmasked control, the content-enabled model
had neural test WAPE `0.12716` and validation-selected blend WAPE `0.10166`.
The matched unmasked control had neural WAPE `0.12525` and blend WAPE
`0.10164`. The content representation is therefore rejected from active use
for now; it remains an explicit research flag rather than evidence of
cold-start improvement.

## 2026-08-15 Pass 77: Cold-Start Identity Ablation

Audited rolling data coverage before changing the multimodal representation.
Early test folds contain `19.4%` to `30.0%` drug identities absent from their
training slice; only `1.6%` to `1.8%` of test rows have no observed history,
and graph-edge coverage is approximately `56%`. This indicates a learned
hashed identity cold-start risk rather than a general missing-history issue.

Reserved entity ID `0` as a true zero-padding/unknown embedding and added a
research-only `mask_drug_identity` ablation that preserves Arkansas and graph
relation nodes. On the same five-epoch single split, the masked model's
validation-selected test blend WAPE was `0.10174`, versus `0.10164` for the
same-code unmasked control. The ablation is rejected from active use. The
identity-masking flag remains available for future rolling experiments, and
the default remains unmasked so this result is not overinterpreted from one
split.

## 2026-08-15 Pass 76: Rolling Policy Evaluation Correction

Audited `evaluate_modular_rolling` and found that it reported the raw neural
head even though the training path selects a transition/neural blend using
validation WAPE. The evaluator now scores that validation-selected policy,
retains raw neural WAPE as a diagnostic, and records the selected fold weight.
Added a regression test proving that rolling improvement is calculated from
the selected policy rather than the unblended neural output.

After regeneration, the corrected six-fold result beats the strongest naive
baseline on `1/6` folds, with mean improvement `-0.92%`; it remains below the
10% all-fold publishability gate. The active single-split artifact was
restored separately after the rolling run.

## 2026-08-15 Pass 75: Corrected Modular Rolling Evidence

Regenerated modular next-quarter rolling-origin evidence after the
chronological transition-pool correction. The run used six folds, one epoch
per fold, the frozen-news research model, the 30-feature contract, and the
default `log_huber` objective.

The corrected result is research-only: `0/6` folds beat the strongest
validation-selected naive baseline, with mean relative improvement
`-156.4%`. The prior two-fold artifact was stale and has been replaced. The
active single-split artifact was restored separately after the rolling run;
the rolling model is not promoted.

## 2026-08-15 Pass 74: WAPE-Weighted Objective Ablation

Added an optional `wape_weighted` objective that weights the log-residual
Huber and quantile terms by the observed training volume, with a cap and
batch normalization. The default remains `log_huber`; the new mode uses only
the current training label and is exposed in `TrainConfig` and the CLI for
reproducibility.

The five-epoch frozen-news ablation improved the neural test WAPE to `0.1239`,
but the validation-selected blend achieved `0.1015`, slightly worse than the
chronology-corrected default-objective blend at `0.1014`. It is therefore
retained as a separate research checkpoint and rejected from the active
objective. Added finite-value, missing-label, and invalid-mode tests.

## 2026-08-15 Pass 73: Transition-ratio leakage audit and rejected ablation

Audited the multimodal training target construction and found that the
transition-ratio fallback pools were being populated in drug-major order.
Later years from earlier drugs could therefore influence earlier rows for
other drugs. Changed sample construction to global chronological order and
added a regression test demonstrating that later-year ratios cannot alter an
earlier transition base.

Tested an explicit temporal transition-base and ratio feature extension after
that correction. The five-epoch trainable-news ablation produced neural WAPE
`0.1514` and validation-selected blend WAPE `0.1022`, both worse than the
transition baseline (`0.1019`). The extension was rejected from the active
32-feature contract; the active research contract remains 30 features. A
one-epoch smoke run regenerated the active artifact after the rejection.

This is negative evidence against simply exposing more baseline internals to
the residual network. The next multimodal iteration should change the
representation or objective, not add the rejected fields back without a new
ablation.

## 2026-08-15 Pass 72: Validation baseline fallback

Corrected quarterly selection so naive baselines are eligible deployment
candidates when their validation WAPE is lower than every learned transition
candidate. Model-only test winners remain diagnostic; a learned model is no
longer forced into the selected output. Forecast selection uses the same
fallback candidates.

After regeneration, the historical one-year rolling artifact remains a
publishable candidate: 6/6 folds beat the validation-selected strongest naive
baseline with mean improvement `15.4%`. The live 2025 extension also reports
6/6 and `15.4%` under the one-year window, but that window does not include
the isolated 2025 target because local target history ends in 2022 and the
source has an intervening gap. The unrestricted live rolling artifact includes
the 2025 target and is the conservative result: 6/7 folds beat naive, only 1
clears 10%, and mean improvement is `4.7%`.

This distinction prevents a favorable protocol from being misrepresented as
evidence for the newest target year.

## 2026-08-15 Pass 71: Live quarterly forecast path

Extended `forecast-quarterly` with the same explicit `--live-sdud-url` target
refresh used by evaluation. The current CMS SDUD file is merged in memory,
the historical quarterly cache is not modified, and forecast metadata records
the source URL, retrieval date, and whether the target was refreshed. Live
forecast evidence points to the matching live rolling artifact rather than the
historical rolling artifact.

The end-to-end command generated 20 current state-level forecast rows across
7 Medicaid products. This closes the operational refresh path but does not
change the scientific scope: the output remains state/NDC Medicaid demand
pressure, not pharmacy inventory or county/supplier ground truth.

## 2026-08-15 Pass 70: Live CMS SDUD extension and rolling audit

Extended the opt-in CMS SDUD adapter to accept the actual downloadable CSV
headers (`State`, `Product Name`, and `Number of Prescriptions`) as well as the
descriptive catalog labels. The first live run caught this mismatch before any
target data was used, and the adapter now streams the national file, filters
Arkansas, excludes suppressed rows, and aggregates product-quarter
prescriptions.

The live 2025 CMS file produced 5,270 strict next-quarter test rows. The
validation-selected quarterly model achieved WAPE `0.0891` and a `13.5%`
improvement over the validation-selected strongest naive baseline. This is
pointwise evidence only: the unrestricted rolling run produced 7 folds, 6
beating the naive baseline, but only 1 clearing the 10% threshold, with mean
improvement `4.7%`. The live target therefore does not establish a publishable
rolling model.

Fixed the CLI so live evaluations write `*_live_target` artifacts rather than
overwriting historical metrics. The historical quarterly artifacts remain
unchanged. The result is still Medicaid state/NDC utilization evidence, not
pharmacy inventory, county demand, or supplier allocation ground truth.

## 2026-08-15 Pass 68: Validation-Selected Hybrid Output

Formalized a convex blend between the transition baseline and the trainable
multimodal residual output. The blend weight is selected only on validation
WAPE and then frozen for the held-out test. On the five-epoch trainable-news
experiment, validation selected weight `0.10`; test WAPE was `0.1009` versus
`0.1019` for transition alone, a `0.94%` improvement.

This is below the 10% promotion threshold and has not been promoted. The
trainer now writes the blended prediction column and exposes the selected
weight in its metrics artifact; the helper has a bounded-weight regression
test.

## 2026-08-15 Pass 67: Residual and Text-Branch Ablations

Compared identical 30-feature small research models under strict fixed
validation/test splits. Persistence-residual with frozen news produced test
WAPE `0.1656`; transition-residual with frozen news improved to `0.1613`;
transition-residual with a trainable news encoder for two epochs improved to
`0.1335`. The transition baseline remained stronger at `0.1019`, so no model
was promoted. Ablation metrics are persisted in
`model/artifacts/evaluation/modular_baseline_ablation.json` and
`modular_trainable_news_metrics.json`.

This supports training the document branch rather than freezing random text
states, but it does not establish rolling or production validity.

## 2026-08-15 Pass 66: Arkansas Weather Context

Added three weather aggregates to the annual panel builder:
`ar_temperature_mean`, `ar_precipitation_mean`, and `ar_wind_mean`. Only NOAA
station rows whose public station name contains `, AR US` are included, so
national station data cannot be mistaken for Arkansas exposure. These fields
join the multimodal temporal stream through the previous-completed-year
barrier, expanding it from 27 to 30 inputs.

The rebuilt panel contains all three weather fields across 544,070 rows. The
regenerated fixed research run produced neural test WAPE `0.1656` versus
transition `0.1019`; the two-fold rolling mean improvement was `-65.56%`.
Weather availability improves input completeness but does not imply forecast
improvement or promotion.

## 2026-08-15 Pass 65: Multimodal Rolling Evaluation

Added leakage-safe rolling fold construction and `evaluate_modular_rolling` to
the real-label multimodal trainer. Each fold trains through year `Y`, uses
`Y+1` for model selection, and evaluates later years against both persistence
and transition baselines. The publishability audit now includes a fail-closed
`modular_rolling_gate` requiring at least three folds and >=10% improvement in
every fold.

Real small-model run with the available history produced two valid folds
(the source window does not support three under the selected minimum training
history). Mean improvement versus the strongest naive was `-39.75%`; fold
improvements were `-43.82%` and `-35.68%`. The candidate is therefore not
publishable. This is useful negative evidence: the multimodal architecture is
not promoted merely because it has many parameters or a connected input path.

## 2026-08-15 Pass 64: Multimodal Training Smoke Run and Mask Safety

Added an explicit small-model research mode to `training.py` so the real-label
multimodal pipeline can be exercised on CPU without overwriting the
production-scale checkpoint path. It trained on the real quarterly SDUD split:
35,148 train rows, 3,482 validation rows, and 2,639 future test rows. The
540,145-parameter smoke model used 27 temporal features and produced finite
row-level predictions.

Test WAPE was `0.1382`; persistence was `0.1129` and transition was `0.1019`,
so the smoke model did not beat the statistical baselines and is not
publishable. This is an honest pipeline validation, not a success claim.

Also fixed causal temporal attention for entities with fully or partially
padded history. The model now clears undefined padded-query attention states;
the new regression test prevents NaNs from entering saved evaluation rows.

## 2026-08-12 Pass 1: Architecture and Constraints

Status: documentation initiated before implementation.

Findings:

- `model/` was empty.
- Local `data/.venv` contains pandas and numpy.
- Local `data/.venv` does not contain scikit-learn, PyTorch, transformers, sentence-transformers, LightGBM, or XGBoost.
- `opencode` is installed at `/home/ubuntu/.opencode/bin/opencode`.
- Available matching model is `opencode/deepseek-v4-flash-free`.
- `data/final_data` validates successfully with 6,479,152 feature rows and 20,153 event rows.
- Targeted Arkansas additions include CMS Part D provider-drug-year data, FDA NDC products/packages, FDA shortages/enforcement, disease surveillance, Arkansas news, ARCOS, provider locations, and pharmacy rosters.

Decision:

- The default model code must run with pandas and numpy only.
- Optional ML backends can be detected and used when installed.
- The first implementation target is a robust end-to-end panel/training/forecast/evaluation pipeline rather than a notebook.

## 2026-08-12 Pass 2: Implementation

Status: end-to-end pipeline implemented and run.

What shipped:

- Package `model/arkansas_pharma_signal/` with config, io, entities, features,
  text_signals, regression, neural, forecast, evaluate, and cli modules.
- `build-panel` builds the annual city x generic-drug demand panel from CMS
  Part D (2013-2024), maps generic names to FDA NDC active ingredients and
  labelers by normalized nonproprietary name, and joins annual aggregates of
  the external-state feature store plus deterministic news/event/shortage
  features. Result: 544,070 panel rows, 1,592 drugs, 310 cities.
- `text_signals` builds deterministic annual event/news/shortage/recall
  features from local files only (no LLM calls).
- `train` fits numpy ridge demand models, a logistic shortage-risk model, and
  a compact `NeuralSignalBlender` MLP (833 params, well under budget).
- `forecast` writes the full 25-column output schema with 10,000 rows in the
  test run; demand_claims wape ~0.32 on training data.
- `evaluate` uses a time-based split (train <= 2021): model wape 0.384 vs
  moving-average baseline 1.288.
- Tests pass as plain assert scripts (pytest not installed in the venv).

Data notes:

- `consumer_price_index_medical_care` covers 2000-2009 only and is empty in
  the 2013-2024 panel window; it was dropped from default features rather than
  imputed.
- Shortage labels are sparse (~0.6% of rows); risk precision/recall metrics
  are reported but degenerate, as expected for the sparse label distribution.

## 2026-08-12 Pass 3: Strict Publishability Audit and Improvements

Status: improved architecture and evidence, but **not publishable yet** by the
driving criterion.

What changed:

- Added strict next-period evaluation: feature year `t` predicts demand at
  `t+1` for the same Arkansas city-drug pair. This removes same-year target
  leakage from the first evaluation.
- Expanded the panel from 49 to 100 columns with real-data layers:
  - drug identity: active ingredient, labeler, dosage form, route, marketing
    category, DEA schedule, pharma class, product counts;
  - provider/demand: provider-type count, beneficiary proxy, cost per fill,
    additional lag/moving-average/log history features;
  - disease: national FluView, Arkansas/national wastewater WVAL, NNDSS burden
    features where public data exists;
  - supply: shortage reasons, recall class counts, recall reason keywords,
    labeler shortage/recall exposure;
  - news: existing Arkansas 3DLNews keyword groups.
- Added `datasets.py` for strict next-period views, train-mask-fitted category
  encoders, and layer feature groups.
- Reworked `evaluate.py` to write:
  - `leaderboard.csv`,
  - `layer_uplift.csv`,
  - strict metrics with WAPE, MAE, RMSE, sMAPE, R2, directional accuracy,
    shortage AUROC/AUPRC/Brier/top-k recall,
  - publishability fields in `metrics.json`.
- Added residual/shock models against the strongest naive baseline:
  `target_residual_log = log1p(demand_t1) - log1p(demand_t)`.
- Added a conservative calibrated demand blend: validation-selected convex
  combination of `city_drug_last` and full ridge. The forecast path now uses
  this calibrated blend for demand.

Strict computed result:

- Best naive baseline: `city_drug_last`, WAPE `0.146271`.
- Best learned/calibrated model: `calibrated_ridge_blend`, WAPE `0.144925`.
- Relative WAPE improvement over strongest naive baseline: `0.9203%`.
- Publishability threshold set for this audit: at least `10%` WAPE improvement
  over `city_drug_last` on strict test.
- `publishable_candidate`: `false`.

Shortage-risk result:

- Label: `shortage_events_t1`
- AUROC: `0.8465`
- AUPRC: `0.0524`
- Brier: `0.1245`
- Top-k recall: `0.0024`
- Test label rate: `0.0103`

Decision:

- The model now shows a small, real, strict-test demand uplift over the
  strongest annual city-drug baseline, but not enough to claim a publishable,
  marketable model with extreme confidence.
- Annual CMS Part D city-drug demand is highly persistent; public annual
  external signals do not yet add the magnitude of uplift required for a
  strong pharmacy-inventory impact claim.
- The next credible improvement path is higher-frequency local demand labels
  (pharmacy weekly/monthly inventory/dispense data, claims feeds, or a drug-level
  Arkansas Medicaid quarterly panel) and shortage-impact labels closer to
  Arkansas pharmacies. Without those, further complexity risks overfitting.

Additional high-frequency check:

- Local `data/S_D/data/combined/*.json.gz` Medicaid SDUD records were inspected
  as a real Arkansas drug-level quarterly target.

## 2026-08-12 Pass 4: Formal Quarterly Evidence Model

Status: quarterly Medicaid demand model now clears the predefined strict gate;
annual Part D demand and shortage-risk paths remain weaker and are not covered
by this claim.

What shipped:

- `model/arkansas_pharma_signal/quarterly.py`: loads real Arkansas Medicaid
  SDUD records from `data/S_D/data/combined/*.json.gz` where
  `source.source_id == medicaid_sdud`, `geography.admin1 == AR`,
  `observation.metric == prescription_count`, and `observation.value` is
  present, then aggregates by year, quarter, canonical drug.
- Strict next-quarter view: features at quarter `t` (previous-quarter count,
  lag-2 count, two-quarter moving average, log1p transforms, fixed quarter
  seasonality) predict the real prescription_count at quarter `t+1` for the
  same drug. No same-quarter target leakage; an earlier target-shift bug that
  made the previous-quarter baseline perfect was fixed and is covered by
  `model/tests/test_quarterly.py`.
- Real exogenous/transition layers:
  - modal Medicaid identifiers retained from raw records: ingredient, NDC,
    RxCUI, manufacturer, mapping confidence;
  - previous-completed-year annual Part D drug/supply/news layers from
    `model/artifacts/panel/panel.csv` where public drug-key matching exists;
  - quarter-level FDA/openFDA event count, shortage count, recall count,
    severity, and confidence from `data/final_data/events/events.csv.gz`;
  - historical drug-by-quarter transition ratio learned from train rows only.
- Baselines `previous_quarter`, `ma2`, `drug_mean`, `global_mean`; models
  `drug_quarter_transition`, `drug_quarter_transition_blend`,
  `ridge_history`, `calibrated_ridge_blend`, `ridge_history_exogenous`,
  `calibrated_exogenous_blend`, and `residual_shock_exogenous`.
- Metrics WAPE/MAE/RMSE/sMAPE/R2 with train (feature years <= 2020),
  validation (2021), test (>= 2022).
- Artifacts `model/artifacts/evaluation/quarterly_leaderboard.csv` and
  `quarterly_metrics.json` with publishability fields and improvement vs
  `previous_quarter`.
- CLI command `evaluate-quarterly --root .`; existing commands unchanged.
- Tests `model/tests/test_quarterly.py`: toy-frame no-leakage,
  next-quarter-shift logic, previous-completed-year annual-layer join, event
  feature-quarter join, and artifact schema.

Strict computed result (formal):

- Rows: train `35,148` / validation `3,482` / test `2,639` drug-quarter
  observations.
- Best naive baseline: `previous_quarter`, WAPE `0.112931`.
- Best learned/calibrated model: `drug_quarter_transition_blend`, WAPE
  `0.098647`.
- Relative WAPE improvement over `previous_quarter`: `12.6485%`.
- Publishability threshold: `>=10%` WAPE improvement over strongest naive
  baseline on strict next-quarter test.
- `publishable_candidate`: `true` for the quarterly Medicaid demand path.
- Strongest supporting non-blended transition row:
  `drug_quarter_transition`, WAPE `0.102775`, `8.9932%` improvement.
- Exogenous ridge models did not improve the leaderboard; validation selected
  zero learned weight for the ridge blends, so the evidence comes from the
  historical drug-by-quarter transition layer rather than the annual/event
  ridge layer.

Decision:

- The quarterly Medicaid demand model now has computed evidence of genuine
  improvement over a strong persistence baseline. This is a credible
  publishable/marketable component for feeding pharmacy inventory models with
  high-frequency Arkansas drug-demand pressure.
- The claim is scoped: annual CMS Part D demand improvement remains small
  (`0.9203%` WAPE improvement), and shortage-risk evidence remains sparse.
  Further work should extend the successful quarterly transition layer into
  pharmacy-level weekly/monthly inventory or dispense data when available.

## 2026-08-12 Pass 5: Rolling-Origin Robustness

Status: robustness improved, but the broader rolling-origin publishability
gate is **not** cleared.

What changed:

- Added rolling-origin quarterly evaluation via `evaluate-quarterly --rolling`.
- New artifacts:
  - `model/artifacts/evaluation/quarterly_rolling.csv`;
  - `model/artifacts/evaluation/quarterly_rolling_metrics.json`.
- Added a validation-selected `multi_transition_blend` using only real
  historical transitions:
  - previous-quarter persistence;
  - historical drug-by-quarter ratio;
  - historical drug-by-quarter delta;
  - recent-window drug-by-quarter ratio.
- Added toy-data tests proving rolling folds are chronological and have
  non-overlapping train/validation/test years.

Computed rolling result:

- Eligible rolling folds: `6`.
- Folds beating the strongest naive baseline: `6/6`.
- Folds clearing the 10% WAPE-improvement gate: `1/6`.
- Mean WAPE improvement vs strongest naive: `7.4654%`.
- Median WAPE improvement vs strongest naive: `6.7893%`.

## 2026-08-13 Pass 6: Universal Intelligence Layer

Status: upstream contracts and real-data artifacts shipped; county/supplier
hierarchy coverage remains explicitly incomplete.

What shipped:

- `canonical_graph.py`: versioned nodes and edges with source, timestamps,
  confidence, transformation method, and direct/unresolved evidence fields.
- `news_corpus.py`: restored 10,228 historical article records from local
  JSONL/metadata sources, including 8,000 full-text records and 2,228
  metadata-only records; publication-time filtering is enforced.
- `event_schema.py`: structured event contract for extraction and provenance.
- `universal_forecast.py`: county × drug × supplier × target output contract.
- CLI commands `build-graph`, `build-news-corpus`, and `forecast-universal`.

Computed artifacts:

- canonical graph: 29,923 nodes and 170,289 deduplicated edges;
- universal forecast: 10,000 rows across five targets and five horizons;
- current panel has no defensible city-to-county crosswalk, so all universal
  rows are marked `evidence_type=unresolved`, confidence `0.35`, and have zero
  parent-company, factory, or API-source coverage.

Decision:

The system now fails transparently on missing geography and supplier hierarchy
instead of silently presenting city or labeler proxies as county or parent
relationships. The next data milestone is a sourced ZIP/city-to-county
crosswalk plus FDA establishment/ownership/factory/API records.
- Minimum WAPE improvement vs strongest naive: `5.2115%`.
- `publishable_rolling_candidate`: `false`.

Fold best models:

- validation `2016`, test `>2016`: `multi_transition_blend`, WAPE `0.112081`
  vs previous-quarter `0.120366`, improvement `6.8830%`.
- validation `2017`, test `>2017`: `multi_transition_blend`, WAPE `0.110594`
  vs previous-quarter `0.118837`, improvement `6.9364%`.
- validation `2018`, test `>2018`: `multi_transition_blend`, WAPE `0.113125`
  vs previous-quarter `0.121243`, improvement `6.6956%`.
- validation `2019`, test `>2019`: `drug_quarter_transition_blend`, WAPE
  `0.119346` vs previous-quarter `0.125907`, improvement `5.2115%`.
- validation `2020`, test `>2020`: `drug_quarter_transition_blend`, WAPE
  `0.106043` vs previous-quarter `0.113314`, improvement `6.4172%`.
- validation `2021`, test `>2021`: `drug_quarter_transition_blend`, WAPE
  `0.098647` vs previous-quarter `0.112931`, improvement `12.6485%`.

Decision:

- The model has a reliable positive high-frequency demand signal: every
  chronological fold beats the strongest naive baseline.
- The evidence is still not strong enough for the broader claim of
  publishable, extreme-confidence inventory impact across rolling origins.
  The correct claim is narrower: the quarterly model is a useful Arkansas
  demand-signal component, but more local pharmacy-level demand/inventory data
  or stronger shortage-impact labels are still needed for the full objective.

## 2026-08-12 Pass 6: Near-Term Rolling Window

Status: one-year rolling evidence is stronger and more inventory-relevant, but
initially did not satisfy the strict all-fold gate.

What changed:

- Added optional bounded rolling test windows:
  `evaluate-quarterly --rolling --rolling-test-window-years 1`.
- New artifacts:
  - `model/artifacts/evaluation/quarterly_rolling_1y.csv`;
  - `model/artifacts/evaluation/quarterly_rolling_1y_metrics.json`.
- The existing all-future rolling artifacts are unchanged in meaning.

Computed one-year rolling result:

- Eligible one-year rolling folds: `6`.
- Folds beating the strongest naive baseline: `5/6`.
- Folds clearing the 10% WAPE-improvement gate: `4/6`.
- Mean WAPE improvement vs strongest naive: `15.0517%`.
- Median WAPE improvement vs strongest naive: `13.0947%`.
- Minimum WAPE improvement vs strongest naive: `-1.1261%`.
- `publishable_rolling_candidate`: `false`.

Fold note:

- The failed fold is validation `2020`, test `2021`: strongest naive is `ma2`
  with WAPE `0.112669`; best model is `drug_quarter_transition_blend` with
  WAPE `0.113937`, a `-1.1261%` gap.

Decision:

- Near-term demand forecasting evidence is now commercially interesting:
  average one-year rolling improvement is above 10%, and four of six folds clear
  the 10% threshold.
- The objective is still not complete because the model does not beat the
  strongest naive baseline in every one-year fold and does not clear the
  all-future rolling gate. The next improvement should target the 2021 fold and
  compare against `ma2` explicitly during deployable model selection.

## 2026-08-12 Pass 7: MA2-Guarded Transition Layer

Status: diagnostic one-year test-best rows clear the formal near-term gate, but
this was later tightened in Pass 8 to validation-selected deployable models.

What changed:

- Added `ma2_transition_blend`: validation-selected blend of the two-quarter
  moving-average baseline and the drug-quarter transition forecast.
- Added `ma2_multi_transition_blend`: validation-selected blend of `ma2`,
  drug-quarter transition ratio, drug-quarter delta, and recent-window
  drug-quarter transition.
- The added rows are model rows, not naive baselines. Weights are selected on
  validation data only; test labels are used only for scoring.

Computed one-year rolling result after MA2 guardrail:

- Eligible one-year rolling folds: `6`.
- Folds beating the strongest naive baseline: `6/6`.
- Folds clearing the 10% WAPE-improvement gate: `5/6`.
- Mean WAPE improvement vs strongest naive: `15.9223%`.
- Median WAPE improvement vs strongest naive: `13.0947%`.
- Minimum WAPE improvement vs strongest naive: `1.5872%`.
- `publishable_rolling_candidate`: `true`.

One-year fold summary:

- validation `2016`, test `2017`: `multi_transition_blend`, WAPE `0.110179`
  vs previous-quarter `0.127435`, improvement `13.5410%`.
- validation `2017`, test `2018`: `multi_transition_blend`, WAPE `0.082034`
  vs previous-quarter `0.109706`, improvement `25.2234%`.
- validation `2018`, test `2019`: `ma2_multi_transition_blend`, WAPE
  `0.075498` vs previous-quarter `0.108481`, improvement `30.4041%`.
- validation `2019`, test `2020`: `ma2_transition_blend`, WAPE `0.133818`
  vs previous-quarter `0.152291`, improvement `12.1297%`.
- validation `2020`, test `2021`: `ma2_multi_transition_blend`, WAPE
  `0.110880` vs strongest naive `ma2` WAPE `0.112669`, improvement `1.5872%`.
- validation `2021`, test `2022`: `drug_quarter_transition_blend`, WAPE
  `0.098647` vs previous-quarter `0.112931`, improvement `12.6485%`.

All-future rolling result after MA2 guardrail:

- Folds beating strongest naive: `6/6`.
- Folds clearing 10%: `1/6`.
- Mean WAPE improvement: `7.8124%`.
- Minimum WAPE improvement: `6.4172%`.
- `publishable_rolling_candidate`: `false`.

Decision:

- The test-best diagnostic result is promising, but not sufficient as a
  deployable publishability claim because the winning row in each fold was
  selected after test scoring. Pass 8 corrects this by using validation-selected
  model rows for publishability.

## 2026-08-12 Pass 8: Deployable Model Selection Audit

Status: evidence is now stricter and honest; one-year rolling remains a
near-term publishable candidate under validation-selected deployment, but the
single split and all-future rolling gates do not clear.

What changed:

- Added `validation_wape` and `deployable_candidate` fields to model rows where
  validation predictions are available.
- Added `selected_model` to `quarterly_metrics.json`; it is chosen by
  validation WAPE, not test WAPE.
- Rolling artifacts now separate:
  - `best_model_by_test`: diagnostic upper-bound row selected after test
    scoring;
  - `selected_model`: deployable row selected by validation WAPE;
  - `selected_improvement_vs_best_naive`: the only improvement used for
    rolling publishability.
- `publishable_candidate` for the single split now uses `selected_model`.

Corrected single-split result:

- Test-best diagnostic row: `drug_quarter_transition_blend`, WAPE `0.098647`,
  `12.6485%` better than previous-quarter.
- Validation-selected row: `ma2_multi_transition_blend`, WAPE `0.106649`,
  `5.5629%` better than previous-quarter.
- `publishable_candidate`: `false` because the deployable selected row does not
  clear 10%.

Corrected all-future rolling result:

- Folds beating strongest naive: `6/6`.
- Folds clearing 10% using selected models: `0/6`.
- Mean selected improvement: `5.9711%`.
- Minimum selected improvement: `4.0116%`.
- `publishable_rolling_candidate`: `false`.

Corrected one-year rolling result:

- Folds beating strongest naive: `6/6`.
- Folds clearing 10% using selected models: `3/6`.
- Mean selected improvement: `13.7492%`.
- Median selected improvement: `10.8447%`.
- Minimum selected improvement: `1.1857%`.
- `publishable_rolling_candidate`: `true`.

Decision:

- The near-term one-year Arkansas Medicaid quarterly demand model still has
  credible deployable evidence for inventory-relevant forecasting: every fold
  beats the strongest naive baseline and mean improvement is above 10%.
- The full objective remains active because the single split, all-future
  rolling, annual Part D demand, and shortage-risk/impact targets still do not
  meet the same deployable evidence standard.

## 2026-08-12 Pass 9: Quarterly Forecast Artifact

Status: deployable near-term quarterly Medicaid demand signal is now exported
as a forecast artifact using the same output schema as the annual grid.

What changed:

- Added `forecast-quarterly` CLI command.
- Added `build_quarterly_forecast_grid`, which:
  - fits quarterly transition candidates on historical Arkansas Medicaid SDUD
    data;
  - selects the deployable model by validation WAPE on the latest
    target-observed year;
  - scores the latest observed real quarter for each Medicaid drug;
  - writes the existing forecast output contract.
- New artifacts:
  - `model/artifacts/forecasts/quarterly_forecast.csv`;
  - `model/artifacts/metadata/quarterly_forecast.json`.

Computed/generated output:

- Rows: `8,406`.
- Drugs: `2,802`.
- Geography: `state`, `AR`.
- Horizons: `91`, `182`, `365` days.
- Target: `medicaid_prescription_count`.
- Model family: `quarterly_medicaid_selected_transition`.
- Selected scoring model in the artifact: `multi_transition_blend`.
- Selection validation year: `2022`.
- Selection validation WAPE: `0.096188`.
- Evidence pointer: `model/artifacts/evaluation/quarterly_rolling_1y_metrics.json`.

Decision:

- The proven near-term quarterly demand signal is now available to downstream
  inventory models as a real forecast table rather than only an evaluation
  report.
- Scope remains important: this is a state-level Arkansas Medicaid
  prescription-count pressure signal by drug. It improves inventory-model input
  coverage but is not pharmacy-level inventory, supplier allocation, or
  shortage-impact proof by itself.

## 2026-08-12 Pass 10: Annual Shortage-Risk Leaderboard

Status: annual shortage-risk evaluation is more honest and more decision
oriented, but still not publishable as a pharmacy inventory-impact model.

What changed:

- Replaced the single annual logistic shortage-risk report with a
  validation-selected risk leaderboard.
- Candidate risk scores now include:
  - full-feature logistic ridge;
  - current `shortage_events` and `shortage_active`;
  - national active shortage/recall exposure;
  - labeler shortage/recall exposure;
  - mean/max exposure blends;
  - logistic plus max-exposure blend.
- Selection uses validation top-k recall, then validation AUPRC, then
  validation Brier score.
- The best model by test metrics is reported as diagnostic only so hindsight
  selection cannot be confused with deployable evidence.
- Added artifact-schema tests for selected risk model, diagnostic test-best,
  top-k capture, precision-at-k, lift-at-k, and publishability fields.

Computed evidence from `model/artifacts/evaluation/metrics.json`:

- Risk label: `shortage_events_t1`.
- Selected model: `shortage_events`.
- Selected test AUROC: `0.503888`.
- Selected test AUPRC: `0.032982`.
- Selected test top-k recall: `0.020384`.
- Selected test precision-at-k: `0.020384`.
- Selected test lift-at-k: `1.982058`.
- Selected positives captured: `17 / 834`.
- Diagnostic test-best model: `na_recall_active_mean`.
- Diagnostic test-best top-k recall: `0.118705`.
- Diagnostic test-best positives captured: `99 / 834`.
- Diagnostic test-best lift-at-k: `11.542570`.
- `publishable_candidate`: `false`.

Decision:

- The risk path now exposes the operationally relevant failure mode: broad
  ranking metrics can look reasonable while the top inventory-triage list is
  weak.
- National active shortage/recall exposure contains useful signal in the test
  window, but the strict validation year did not support selecting it for
  deployment.
- The annual shortage-risk model is not marketable yet. Further work needs
  rolling risk selection, more pharmacy-adjacent shortage impact labels, and
  higher-resolution supplier allocation data before a strong inventory-impact
  claim can be made.

## 2026-08-12 Pass 11: Rolling Annual Shortage-Risk Evidence

Status: rolling annual shortage-risk evaluation is now implemented, and it
shows the annual shortage-risk path is not publishable.

What changed:

- Added `evaluate_risk_rolling` for annual shortage-risk folds.
- Added `evaluate --rolling-risk` CLI support.
- New artifacts:
  - `model/artifacts/evaluation/risk_rolling.csv`;
  - `model/artifacts/evaluation/risk_rolling_metrics.json`.
- Each fold uses:
  - training feature years `<= cutoff - 1`;
  - validation feature year `cutoff`;
  - test feature year `cutoff + 1`;
  - test shortage target year `cutoff + 2`.
- Fold rows include both deployable selected-model metrics and diagnostic
  test-best metrics.
- Added artifact-schema tests for chronological fold ordering, selected model,
  diagnostic test-best model, top-k recall, precision-at-k, lift-at-k, and
  captured positives.

Computed rolling result:

- Eligible folds: `6`.
- Mean selected top-k recall: `0.000000`.
- Median selected top-k recall: `0.000000`.
- Mean selected precision-at-k: `0.000000`.
- Median selected lift-at-k: `0.000000`.
- Selected positives captured: `0 / 1,535`.
- Folds beating label-rate baseline: `0 / 6`.
- Diagnostic test-best positives captured: `20 / 1,535`.
- Best diagnostic fold: cutoff `2022`, `shortage_events`, captured `17 / 43`.
- `publishable_rolling_candidate`: `false`.

Decision:

- The rolling evidence rules out a strong annual shortage-risk claim with the
  current label and feature set.
- The failure is not only model selection. Even diagnostic test-best selection
  captures only `20 / 1,535` positives across folds, with most of the signal
  concentrated in one fold.
- Next shortage work should stop treating annual FDA shortage presence as the
  final proof target. It should instead build a pharmacy-adjacent shortage
  impact label from real data, such as backorder/substitution events, local
  fill disruptions, supplier allocation records, or higher-frequency shortage
  status histories.

## 2026-08-12 Pass 12: Medicaid Quarterly Layer in Annual Demand Model

Status: a real quarterly Medicaid demand-pressure layer is now integrated into
the annual Part D city-drug model, but the computed annual demand uplift is
still not marketable.

What changed:

- Added annualized Arkansas Medicaid SDUD quarterly features:
  - `medicaid_rx_annual`;
  - `medicaid_rx_q4`;
  - `medicaid_rx_recent_qoq`;
  - `medicaid_rx_q4_share`;
  - `medicaid_rx_log`;
  - `medicaid_rx_q4_log`;
  - `medicaid_rx_quarters_observed`.
- Features are deterministic transforms of real Medicaid quarterly
  prescription counts only.
- Join is by exact normalized drug or ingredient key for the same annual
  feature year. No fuzzy matching or synthetic fill is used.
- Added a `medicaid` ablation group to strict annual evaluation.
- Rebuilt `model/artifacts/panel/panel.csv`.
- Regenerated annual evaluation and rolling shortage-risk artifacts.
- Added tests for Medicaid feature generation, panel artifact columns, and
  `ridge_medicaid` leaderboard presence.

Coverage:

- Medicaid feature rows: `13,085`.
- Medicaid normalized drugs: `2,719`.
- Annual Part D row coverage: `124,926 / 544,070` (`22.9614%`).
- Claim-weighted coverage: `30.0926%`.
- Matched annual drug keys: `182 / 1,582`.

Computed annual demand result:

- Best naive: `city_drug_last`, WAPE `0.146271`.
- Best model: `calibrated_ridge_blend`, WAPE `0.144631`.
- Relative improvement vs strongest naive: `1.1209%`.
- Previous annual best before this layer: WAPE `0.144925`,
  improvement `0.9203%`.
- Standalone Medicaid ablation: `ridge_medicaid`, WAPE `0.157877`,
  worse than `ridge_history_only` WAPE `0.158797` only by a small amount but
  still worse than the strongest naive.
- `publishable_candidate`: `false`.

Decision:

- The quarterly Medicaid layer adds a real, auditable demand-pressure input and
  slightly improves the calibrated full annual blend.
- The improvement is too small for a marketable annual inventory-impact claim.
- The limited exact-match coverage means the next demand improvement should
  focus on better public drug identity alignment, especially brand/generic and
  RxCUI/NDC-to-ingredient mapping, before adding more model complexity.

## 2026-08-12 Pass 13: Audited Medicaid Identifier Bridge

Status: real identifier bridging substantially improves Medicaid-to-Part-D
coverage, but the wider bridge does not improve strict annual WAPE.

What changed:

- Medicaid annualization now prefers raw local Medicaid SDUD records when
  available instead of only the cached four-column quarterly panel.
- Added production bridge keys from real local identifiers:
  - normalized Medicaid drug/canonical/ingredient/original names;
  - final-data `ndc:* represents drug:*` relationship edges;
  - final-data `drug:* contains ingredient:*` relationship edges;
  - FDA NDC product `NONPROPRIETARYNAME` and `SUBSTANCENAME`;
  - local drug-dictionary NDC and RxCUI links.
- Added bridge diagnostics:
  - `medicaid_rx_bridge_records`;
  - `medicaid_rx_bridge_families`.
- Medicaid record-to-final-key pairs are deduplicated before aggregation to
  avoid double-counting the same Medicaid source record through multiple
  bridge families.
- No fuzzy matching or synthetic fill was added.

Coverage after bridge:

- Medicaid feature rows: `15,942` vs `13,085` before bridge.
- Medicaid normalized keys: `3,129` vs `2,719` before bridge.
- Annual Part D row coverage: `228,107 / 544,070` (`41.9260%`) vs
  `124,926 / 544,070` (`22.9614%`) before bridge.
- Claim-weighted coverage: `55.5359%` vs `30.0926%` before bridge.
- Matched annual drug keys: `401 / 1,582` vs `182 / 1,582` before bridge.
- Maximum bridge-family count on matched rows: `3`.

Computed annual demand result after bridge:

- Best naive: `city_drug_last`, WAPE `0.146271`.
- Best model: `calibrated_ridge_blend`, WAPE `0.145009`.
- Relative improvement vs strongest naive: `0.8631%`.
- Previous exact-match Medicaid-layer best: WAPE `0.144631`,
  improvement `1.1209%`.
- Standalone Medicaid ablation: `ridge_medicaid`, WAPE `0.157203`, slightly
  better than the exact-match `ridge_medicaid` WAPE `0.157877`, but still far
  worse than the strongest naive.
- `publishable_candidate`: `false`.

Decision:

- The bridge is useful infrastructure because it more than doubles claim-weighted
  Medicaid feature coverage using auditable local identifiers.
- More identity coverage did not translate into better annual WAPE. The broader
  bridge appears to add noisy brand/generic alignment for the annual Part D
  target.
- The production annual demand selector should remain conservative. Next work
  should test guarded Medicaid features, such as validation-selected
  exact-match-only vs identifier-bridge variants, before allowing wider bridge
  features into the full annual blend.

## 2026-08-12 Pass 14: Guarded Medicaid Feature Selector

Status: guarded Medicaid selection improves strict annual demand evidence, but
the annual model is still not publishable.

What changed:

- The panel now keeps both Medicaid scopes:
  - exact-only columns `medicaid_exact_rx_*`;
  - audited bridge columns `medicaid_rx_*`.
- Added annual feature groups:
  - `medicaid_exact`;
  - `medicaid_bridge`;
  - combined `medicaid`;
  - `full_no_medicaid`;
  - `full_medicaid_exact`;
  - `full_medicaid_bridge`.
- The calibrated production blend now evaluates four validation candidates:
  - `none`;
  - `exact`;
  - `bridge`;
  - `all`.
- The selected candidate is recorded in `metrics.json` under
  `calibrated_blend.selected_medicaid_variant`.
- Diagnostic leaderboard rows are retained for every calibrated candidate.

Coverage:

- Exact Medicaid row coverage: `124,926 / 544,070` (`22.9614%`).
- Exact claim-weighted coverage: `30.0926%`.
- Exact matched annual drug keys: `182`.
- Bridge Medicaid row coverage: `228,107 / 544,070` (`41.9260%`).
- Bridge claim-weighted coverage: `55.5359%`.
- Bridge matched annual drug keys: `401`.

Validation candidate result:

- Selected Medicaid variant: `all`.
- Validation WAPE:
  - `all`: `0.120137`;
  - `exact`: `0.120315`;
  - `bridge`: `0.120598`;
  - `none`: `0.120870`.

Strict annual test result:

- Best naive: `city_drug_last`, WAPE `0.146271`.
- Selected calibrated model: `calibrated_ridge_blend`, WAPE `0.144370`.
- Relative improvement vs strongest naive: `1.2994%`.
- Diagnostic exact-only calibrated WAPE: `0.144472`.
- Diagnostic no-Medicaid calibrated WAPE: `0.144925`.
- Diagnostic bridge-only calibrated WAPE: `0.145009`.
- `publishable_candidate`: `false`.

Decision:

- Guarded validation fixed the previous bridge regression and produced the best
  annual Part D WAPE so far.
- The improvement remains far below the `10%` marketability gate.
- The next annual-demand work should add rolling-origin demand evaluation for
  the guarded selector. A single strict split is not enough to claim inventory
  impact, even though this pass improved the point estimate.

## 2026-08-12 Pass 15: Annual Demand Rolling Audit

Status: rolling annual demand audit added; the guarded annual lift remains
unproven until the new artifacts are regenerated and inspected.

What changed:

- Corrected annual encoder fitting so identity categories are fit on feature
  years `<= cutoff - 1`, excluding the validation year in both strict demand
  and rolling shortage-risk evaluation.
- Added `evaluate_demand_rolling` and `annual_fold_cutoffs` for chronological
  annual demand folds.
- Each fold selects the Medicaid scope and calibrated blend on validation,
  selects the naive comparator on validation, and reports test-best models as
  diagnostic only.
- Added optional bounded one-year test windows and CLI flags:
  `evaluate --rolling-demand` and
  `--rolling-demand-test-window-years 1`.
- Added toy-data tests for chronological fold construction and
  validation-selected rolling rows.

Controlled strict-split audit before artifact regeneration:

- Guarded Medicaid ridge `alpha=0.01` versus `alpha=10.0` produced the same
  selected `all` variant and effectively unchanged validation WAPE; no alpha
  change was justified.
- Excluding validation-year encoder categories changed the calibrated test
  WAPE from `0.144370` to `0.144258`, still only about a `1.4%` gain over the
  city-drug persistence baseline and far below the `10%` gate.

Decision:

- The annual path now has the missing rolling audit needed to distinguish a
  stable improvement from a single-split result.
- No additional model complexity or broader identity bridge is justified until
  the rolling artifacts show consistent out-of-sample lift.

## 2026-08-13 Pass 16: Therapy-Context Interaction Layer

Status: interaction architecture improved and validated, but the model is not
publishable for pharmacy inventory deployment.

What changed:

- Added a bounded interaction layer combining real disease, surveillance,
  news, recall, shortage, and event variables with train-fitted ingredient,
  pharmaceutical-class, therapeutic-category, dosage-form, and route flags.
- Interactions are deterministic products of forecast-time variables. Target
  columns ending in `_t1` are explicitly excluded, and the existing strict
  no-leakage tests pass.
- Added `full_no_interactions` as an ablation so interaction uplift is measured
  against the prior architecture rather than assumed.
- Added a raw-scale prediction bound to keep unstable diagnostic log fits from
  overflowing during rolling evaluation.

Computed evidence from real Arkansas CMS Part D provider/drug claims:

- Strict 2021-cutoff test: calibrated WAPE `0.143604` versus city-drug-last
  WAPE `0.146271`, a `1.8235%` improvement; R2 `0.987768`.
- Six one-year rolling folds: all six beat the city-drug-last baseline, but
  mean improvement is only `1.2147%`, median `1.0804%`, and zero folds clear
  the configured `10%` publishability gate.
- All-future rolling folds remain weaker: mean improvement `-2.4%` and only
  two of six folds beat persistence.

Decision:

- The added variables improve short-horizon consistency but do not establish
  marketable pharmacy-inventory impact. The evidence supports a modest demand
  signal, not extreme-confidence deployment. Higher-frequency pharmacy or
  dispense/inventory labels remain the limiting requirement.

## 2026-08-13 Pass 17: Quarterly Selection Audit and Seasonal Comparator

Status: the quarterly Medicaid demand signal is promising for one-quarter
planning, but the overall pharmacy-inventory model is not yet publishable with
extreme confidence.

What changed:

- Corrected quarterly baseline selection to use validation labels only. The
  prior evaluator selected the best naive comparator from test labels, which
  overstated uplift. Test labels are now used only for final scoring.
- Added a standard same-drug, same-target-quarter-last-year seasonal-naive
  comparator, fitted only from observed training targets.
- Added tests asserting validation-only baseline selection and comparator
  presence.
- OpenCode was requested for this audit but returned a server error before
  producing a usable change; the correction was implemented independently.

Leakage-free computed evidence from real Arkansas Medicaid SDUD prescription
counts:

- Strict next-quarter test: validation-selected multi-transition model WAPE
  `0.106649` versus validation-selected `ma2` WAPE `0.127241`, a `16.18%`
  improvement; R2 `0.939735`.
- One-year rolling: six of six folds beat their validation-selected naive
  comparator; mean improvement `15.7%`, all folds clear the 10% gate.
- Unrestricted rolling horizon: six of six folds beat naive, but mean
  improvement is only `7.7%` and one of six folds clears 10%.
- The seasonal-naive comparator did not beat persistence on the current panel,
  but is retained as an auditable benchmark rather than removed.

Decision:

- The model has credible near-term quarterly demand value, but the long-horizon
  rolling gate fails and the label is state-level Medicaid prescription count,
  not individual pharmacy inventory or dispense demand. The publishability
  claim for real pharmacy inventory impact remains unproven.

## 2026-08-13 Pass 18: Pharmacy Access and ARCOS Layers

Status: real supply/access layers added and audited; no publishability claim.

What changed:

- Added official Arkansas pharmacy-directory facility, hospital, and
  retail/community counts by normalized prescriber city/year to the annual
  panel. Coverage is `50.9%` of panel rows because directory years and city
  matches are incomplete; unmatched rows remain missing.
- Added DEA ARCOS quarterly retail-distribution grams by normalized drug and
  quarter, including log level and observed quarter-over-quarter change. Only
  four Medicaid drugs overlap the ARCOS catalog; unmatched drugs are not
  imputed.
- Passed ARCOS through strict quarterly and rolling evaluation.
- Added tests for pharmacy-access aggregation and ARCOS quarter alignment.

Computed evidence:

- Annual strict test remained unchanged at WAPE `0.143604`, or `1.8235%`
  improvement over city-drug persistence. Pharmacy-access features did not
  produce measurable additional uplift.
- Quarterly strict and one-year rolling selected transition scores remained
  unchanged: `16.18%` strict improvement over validation-selected `ma2`, and
  `15.7%` mean one-year rolling improvement.
- ARCOS worsened the diagnostic exogenous ridge (`0.193968` WAPE versus
  `0.185464` without ARCOS), so it is retained as an auditable feature source
  but is not selected for deployment.

Decision:

- Added layers are real and leakage-safe, but current public data still does
  not demonstrate individual-pharmacy inventory impact. The model remains
  below the extreme-confidence publishability standard, especially on
  unrestricted rolling horizons.

## 2026-08-13 Pass 19: Universal Geography, Events, and Supplier Coverage

What changed:

- Added a Census geocoder city-to-county resolver with weak-inference
  provenance. It resolved 54/310 unique panel cities; ambiguous and failed
  cities remain unresolved.
- Regenerated the universal forecast with 9,725/10,000 rows carrying county
  evidence and 275 unresolved rows.
- Added article-span event extraction over the 8,000 available full-text
  articles. It produced 1,293 events with evidence spans, article IDs, source
  timestamps, impact direction, uncertainty, negation, and confidence.
- Event and article IDs now flow into universal forecast attribution and event
  pressure contributes to demand-shock and supply-risk signals.
- Added a supplier hierarchy coverage artifact with 115,223 direct FDA
  labeler/product rows. Parent-company, factory, and API-source tiers remain
  empty because those source records are not present locally.

Validation:

- Geography, event extraction, supplier, universal-contract, config, and model
  tests pass; Python compilation and CLI artifact generation pass.

The final universal model is not yet publishable: expert event labels,
county-level demand outcomes, and complete supplier/factory/API mappings remain
required acceptance gates.

## 2026-08-13 Pass 20: FDA Establishments and Publishability Audit

What changed:

- Added the official FDA DECRS annual registration download as a real local
  source (`drls_reg.zip`).
- Parsed exact normalized labeler matches only. The resulting hierarchy has
  56,526 establishment-linked product rows, 5,642 parent-company links, and
  4,626 API-manufacture links. Unmatched labelers remain labeler-only.
- Universal forecast now enriches rows from the supplier hierarchy and carries
  factory, parent, and API fields where directly supported.
- Added `audit-publishability`, which fails closed against the strict gates and
  writes `artifacts/evaluation/publishability_audit.json`.

Current audit result: `publishable=false`, with five failed gates: expert gold
set, county outcome coverage, county crosswalk threshold, supplier factory/API
coverage threshold, and unrestricted rolling-origin forecast performance.
The quarterly rolling gate and universal output/event schemas pass.

## 2026-08-13 Pass 21: County Outcomes and Annotation Manifest

- Improved Census geography resolution using real Arkansas NPPES practice
  addresses: 260/310 panel cities resolve to a county.
- Built `artifacts/outcomes/county_demand.csv.gz` from real CMS Part D
  city/provider demand. It contains 301,361 county-year-drug rows and covers
  504,493/544,070 panel rows (92.73%); unmapped rows are excluded, not filled.
- Built a 2,000-row real-text expert annotation manifest with balanced
  candidate event classes. All 2,000 rows remain explicitly unlabeled; this
  artifact cannot satisfy the gold-set gate until an expert annotates it.

The strict audit is now down to three failed gates: expert labels, supplier
factory/API coverage, and unrestricted rolling-origin performance.

## 2026-08-13 Pass 22: Supplier Provenance Correction

Test:

- Targeted plain-Python suites for annotation, county outcomes, event
  extraction, models, next-period evaluation, publishability, quarterly
  evaluation, and universal contracts passed.
- The schema suite was started against real raw Medicaid data but interrupted
  after prolonged execution while recomputing the large feature layer; it did
  not produce a result and is not counted as a pass.
- Before the change, the supplier artifact reported factory coverage 49.06%
  and API coverage 4.01% based on exact name matching against the FDA
  establishment snapshot.

Review and research:

- FDA eDRLS documentation states that establishment registration and drug
  listing are separate submissions and that establishment registration data is
  used as a current list of establishments. The local `drls_reg.txt` contains
  FEI, firm, registrant, and operations fields but no NDC-to-establishment
  product edge.
- FDA DECRS: https://www.fda.gov/drugs/drug-approvals-and-databases/drug-establishments-current-registration-site-decrs
- FDA eDRLS instructions: https://www.fda.gov/drugs/electronic-drug-registration-and-listing-system-edrls/electronic-drug-registration-and-listing-instructions
- Therefore an exact labeler/firm name match cannot support a product-specific
  factory edge, parent-company edge, or API-source edge. This was a false
  provenance claim even though the code and coverage numbers looked plausible.

Change implemented:

- `supplier_hierarchy.py` now emits only direct FDA labeler/product rows from
  the NDC product file. Parent, factory, API, and country remain empty until a
  source provides an explicit product-to-establishment relationship.
- Validation rejects any row marked `direct_labeler_only` that claims one of
  those unsupported tiers.
- Added a regression test for that invariant.

Actual effect:

- Regenerated supplier hierarchy: 115,223 rows; factory coverage 0%, API
  coverage 0%. Universal forecast regenerated at 10,000 rows.
- Publishability remains false, now for the correct reason: supplier factory/API
  coverage is genuinely unavailable rather than overstated. Other failed gates
  are expert gold labels and unrestricted rolling-origin performance.

Decision: KEEP correction; INVESTIGATE acquisition of explicit FDA drug-listing
establishment edges before attempting hierarchy enrichment again.

Next step: run an independent event-layer ablation with the same validation-
selected quarterly protocol, then add only features that improve against the
selected transition baseline under rolling-origin evaluation.

## 2026-08-13 Pass 23: Event-Layer Ablation

Test:

- Real Arkansas Medicaid SDUD quarterly panel, train years <=2020,
  validation 2021, test >=2022.
- Compared the existing validation-selected transition candidates with and
  without the real `final_data/events/events.csv.gz` quarterly event layer.
- The same annual panel and ARCOS layers were retained in both runs.

Result:

- No-event exogenous feature count: 156; event-enabled count: 161.
- Validation-selected model: `ma2_multi_transition_blend` in both runs.
- Selected test WAPE: 0.1066490684 in both runs.
- Drug-quarter transition diagnostic test WAPE: 0.0986472295 in both runs.
- Persistence baseline test WAPE: 0.1129313176 in both runs.
- Event features produced 0.0 measurable incremental WAPE value in this
  evaluation. The exogenous ridge blend weight remained zero.

Review:

- The event layer is schema-valid and provenance-bearing, but this test does
  not demonstrate predictive value. Keeping it in the feature store is useful
  for attribution, but it must not be described as improving demand forecasts.
- The result is consistent with the existing review finding that the current
  deterministic event aggregate is too coarse and the source labels are not a
  held-out expert NLP gold set.

Decision: INVESTIGATE, not PASS. Do not increase model complexity based on
these events. The next high-value experiment is a source-family/time-held-out
evaluation of event extraction and a real annotated subset before claiming
news-to-demand lift.

## 2026-08-13 Pass 24: Pointwise Accuracy Audit

Test: real Arkansas Medicaid SDUD quarterly prescription counts; train years
<=2020, validation 2021, test >=2022; 2,639 held-out drug-quarter rows.

Results:

- Validation-selected `ma2_multi_transition_blend`: WAPE 0.106649, MAE
  178.78, RMSE 1,137.20, within-5% 26.34%, within-10% 44.75%, within-20%
  66.54%.
- Persistence baseline: WAPE 0.112931, within-5% 28.95%, within-10% 47.25%,
  within-20% 68.40%.
- Quarter-level selected within-5%: Q1 17.49%, Q2 30.07%, Q3 31.56%.

Failure: the requested >=75% within-5% numerical prediction gate fails badly.
WAPE improvement is not evidence of accurate individual predictions.

Research: Croston-family research reports that ordinary pointwise metrics can
be inappropriate for sparse demand and evaluates specialized methods and
inventory outcomes:

- Croston comparison, *International Journal of Forecasting*:
  https://doi.org/10.1016/0169-2070(94)90021-3
- Teunter and Duncan intermittent-demand comparison:
  https://doi.org/10.1057/palgrave.jors.2602569

Decision: FAIL. Keep the 75% gate; test accuracy-optimized blending and a
Croston/SBA comparator.

## 2026-08-13 Pass 25: Accuracy-Optimized Blend and Croston/SBA

Added validation-only `within5_transition_blend` and a per-drug Croston/SBA
baseline. Both were fitted without test labels.

Results:

- Accuracy-selected blend: WAPE 0.102476, within-5% 28.69%, versus
  persistence 28.95%; no primary-gate improvement.
- Croston/SBA: WAPE 0.338529, within-5% 6.67%, within-10% 13.87%,
  within-20% 26.98%; rejected.

Decision: FAIL. The next action is expanding-origin forecasting so each
one-step test forecast can use earlier observed quarters without using its own
future target.

## 2026-08-13 Pass 26: Expanding-Origin Transition

Added `expanding_transition_blend`. Each row is forecast before its target is
appended; test forecasts use training, validation, and earlier test
observations only. The initial dataframe-concatenation implementation was too
slow and was replaced with incremental ratio statistics.

Results:

- Expanding transition: WAPE 0.099073, MAE 166.08, RMSE 1,179.43,
  within-5% 30.20%, within-10% 48.01%, within-20% 67.68%.
- Static transition: WAPE 0.098647, within-5% 28.72%.
- Persistence: WAPE 0.112931, within-5% 28.95%.

Decision: FAIL. Expanding history improves pointwise accuracy by 1.48 points
over static transition but remains far below 75% and is not selected by
validation WAPE. Run rolling-origin validation next.

## 2026-08-13 Pass 27: Rolling Pointwise Accuracy

Test: six eligible one-year rolling-origin folds on the real 2012–2022
quarterly Medicaid panel. Each fold used validation-selected deployment and
the same baselines.

Results:

- All 6/6 folds beat their strongest naive baseline by WAPE.
- Mean WAPE improvement: 15.66%; the rolling WAPE gate remains true.
- Mean selected within-5%: 24.86%; within-10%: 42.31%; within-20%: 63.33%.
- Fold within-5% ranged from 21.09% to 27.48%.

Failure: rolling robustness confirms the pointwise >=75% gate is not close.
Only six folds are possible because the real panel covers 2012–2022; an
eight-fold test cannot be honestly manufactured from this time span.

Decision: FAIL for pointwise accuracy, PASS only for the narrower WAPE
improvement claim. Keep the model scoped to a demand-pressure signal.

## 2026-08-13 Pass 28: Accuracy Simplex and Bias Correction

Added a validation-within-5%-optimized simplex and validation-only
multiplicative bias correction, informed by forecast-combination research:

- Bias-corrected transition: scale 1.02, WAPE 0.095080, within-5% 29.06%,
  within-10% 48.09%, within-20% 67.45%.
- Accuracy simplex: WAPE 0.105247, within-5% 27.97%.

The bias correction improves WAPE over the static transition but not the
pointwise gate; the simplex is rejected. No change reaches >=75% within-5%.

Research:

- Bias-corrected panel forecasts, Journal of Econometrics:
  https://doi.org/10.1016/j.jeconom.2009.01.002
- Forecast combination and bias/variance effects, International Journal of
  Forecasting: https://doi.org/10.1016/j.ijforecast.2019.03.010

Decision: FAIL. Do not claim numerical accuracy success. The remaining
highest-value limitation is target resolution and coverage: CMS SDUD is
state/NDC quarterly prescription utilization, not daily/weekly county,
pharmacy, or supplier demand. CMS documentation confirms the quarterly,
state/NDC structure and suppression/late-reporting behavior:
https://www.cms.gov/data-research/cms-data/data-available-researchers/limited-data-set-lds-files/medicaid-state-drug-utilization-lds
and https://www.medicaid.gov/faq/2020-04-09/91831

Next iteration: quantify target volatility and suppression/coverage by drug,
then determine whether a coarser operational target or a new real data source
is required before further model changes.

## 2026-08-13 Pass 29: Target Volatility and Data-Resolution Audit

Tested the real quarterly panel directly: 41,269 next-quarter rows after
constructing the strict view, 2,518 drugs, years 2012–2022.

Results:

- Median absolute relative next-quarter change: 12.74%.
- Persistence is within 5% on only 23.63% of all eligible historical
  transitions, before any model error is added.
- Target counts are all above one in the current panel (minimum 11; median
 128), so the one-count near-zero tolerance does not explain the failed gate.
- Drug-series length is sparse: median 10 quarters, first quartile 5, and
 25% of drugs have <=5 observed quarters.
- 2022 has 2,639 rows versus 4,996 in 2012, reflecting changing coverage.

Review: a 75% within-5% gate is empirically incompatible with this target’s
observed quarter-to-quarter volatility for the current state/drug aggregate.
The model’s 29.06% held-out result is only modestly above the historical
persistence ceiling and cannot be fixed credibly by adding more ensemble
complexity.

CMS confirms that SDUD is state/NDC quarterly utilization and is updated with
late reports and revisions; it is not a daily/weekly pharmacy or county
inventory target:

- https://www.cms.gov/data-research/cms-data/data-available-researchers/limited-data-set-lds-files/medicaid-state-drug-utilization-lds
- https://www.medicaid.gov/faq/2020-04-09/92101
- https://www.medicaid.gov/faq/2020-04-09/91831

Decision: BLOCKED for the requested universal >=75% within-5% pointwise gate
on this target, pending an external real-data change. The required next data
are pharmacy-level weekly/monthly dispense or inventory outcomes, or a
defensible county-level high-frequency utilization source. Synthetic targets
and labels are prohibited, and the repository contains neither those outcome
labels nor an expert NLP gold set.

## 2026-08-13 Pass 30: 100M-Class Modular Architecture Contract

Implemented `arkansas_pharma_signal.modular_model` as a separate research
architecture rather than replacing the validated statistical baseline.  The
default inventory is computed from initialized parameters and reports:

- active parameters: 148,766,885
- trainable parameters: 148,766,885
- frozen parameters: 0
- architecture: document encoder, typed graph reasoner, causal temporal
  reasoner, and Arkansas multi-task heads

Forward-contract tests pass for intermediate news/graph/temporal states,
point forecasts, three quantiles, state-risk outputs, and driver
contributions.  The architecture has not yet been trained on real labels and
therefore has zero evidence of predictive value; it is not promoted and does
not alter the current forecast artifacts.

The strongest measured quarterly baseline remains WAPE 0.095080 and
within-5% 29.06% on the held-out future split; rolling one-year folds average
24.86% within-5%.  The new architecture must beat those numbers under the
same leakage audit before it can replace the baseline.

## 2026-08-14 Pass 31: First Real-Label Modular Training

Trained the modular model on the real Arkansas Medicaid SDUD next-quarter
count target with strict feature-year partitions:

- train: 35,148 rows through 2020
- validation: 3,482 rows in 2021
- future test: 2,639 rows after 2021
- real article windows: 51 publication quarters
- active parameters: 148,653,073
- trainable parameters: 32,729,617; frozen document encoder: 115,923,456

The best persistence-residual checkpoint was selected on validation WAPE:

- validation WAPE: 0.112013; within-5%: 27.54%
- future-test WAPE: 0.119551; within-5%: 25.21%

An absolute-log-demand run was worse (test WAPE 0.303730). A past-only
transition-residual run was also worse (test WAPE 0.302484). The persistence
residual is therefore retained as the neural candidate, but it does not beat
the statistical transition baseline (WAPE 0.095080). The checkpoint is not
promoted, and this result is evidence against the current multimodal training
recipe rather than evidence of publishability.

## 2026-08-14 Pass 32: Neural Residual Blend Audit

Using the trained persistence-residual checkpoint and a separate online
past-only transition column, the validation-selected blend used 56% neural
and 44% transition prediction:

- validation WAPE: 0.102665; within-5%: 26.84%
- future-test WAPE: 0.099922; within-5%: 30.03%

This improves the neural model substantially and improves the held-out
within-5% rate over the existing bias-corrected transition row (29.06%), but
does not beat its WAPE (0.095080). It is not a publishable replacement. The
blend artifact is retained for subgroup and rolling-origin follow-up; its
weight was selected only on validation.
## 2026-08-14 Pass 31: Align Default Research Model With Requested Parameter Budget

The default `ModularModelConfig` was re-sized from 148,766,885 active
parameters to 301,745,749 active parameters. The new configuration uses a
1,024-wide, 16-layer text encoder, 400-wide typed graph stream, four graph
layers, and a 1,536-wide temporal feed-forward block. `model_inventory` counts
initialized parameters before any optional freezing, and the modular tests
assert the requested 300M--400M range. This changes only the research
multimodal architecture; the lightweight statistical production forecast and
its generated artifacts are unchanged.

Evidence: `pytest -q model/tests/test_modular_model.py` -> 2 passed; measured
active/trainable parameters = 301,745,749.
## 2026-08-14 Pass 32: Add Per-Disease Outbreak Risk and Spread States

Layer 1 remains within the requested 50--300 variable budget at 266
variables, but the prior disease fields were mostly mention counts. They now
emit, for each of 54 disease lexicon entries, `present`, `outbreak_risk`,
`spread_rate`, and `uncertain_count`. `outbreak_risk` is the maximum observed
article probability for matching disease-outbreak events. `spread_rate` is a
strictly trailing, clipped change between the current seven-day and preceding
seven-day article-mention windows within the same geography; it does not use
future articles. The generated artifact contains 979 observed date/geography
rows and 266 variables.

Evidence: `test_news_signals.py` -> 2 passed; regenerated
`artifacts/news/layer1_news_state_features.csv.gz` -> 979 rows, 266 variables.
## 2026-08-14 Pass 33: Connect Layer 1 Into the Predictive Panel

Previously the 266-variable Layer-1 artifact was generated and published but
the annual panel only consumed the older event-count feature path. The panel
builder now reads `artifacts/events/article_events.csv.gz`, builds the same
Layer-1 state table, aggregates it by observed calendar year (sum for boolean
and count states; mean for outbreak risk and spread rate), and joins those
states before strict next-period target construction. Year `t` news therefore
flows into the year `t+1` demand/shortage features without future leakage.
## 2026-08-14 Pass 34: Real-Label Run of the 300M-Class Architecture

The resized multimodal architecture was trained for one epoch on the real
quarterly Arkansas Medicaid SDUD target with the document encoder frozen. The
strict partition was 35,148 train rows through 2020, 3,482 validation rows in
2021, and 2,639 future-test rows. The initialized run reports 301,568,001
active parameters, 58,869,761 trainable parameters, and 242,698,240 frozen
document-encoder parameters. Test WAPE was 0.23764 and within-5% accuracy was
16.22%; this is below the statistical production path and the requested
numeric target, so the model remains research-only. The separately validated
shortage boolean path remains the promoted final output.
## 2026-08-14 Pass 35: Expand Layer-1 Disease Coverage From External Reference

The free EventEpi repository was added under
`data/targeted_additions/eventepi`. Its processed incident database contains
169 expert public-health surveillance records. The Layer-1 lexicon was
expanded from 54 to 62 disease states, adding cholera, Ebola, anthrax, Lassa
fever, MERS, yellow fever, plague, and leptospirosis. The resulting contract is
298 variables (`50 + 62*4`), still within the hard 300-variable limit. Exact
disease-name coverage against the external reference increased from 47/169 to
93/169; this is a lexicon coverage diagnostic, not an article-extraction
accuracy claim, because the repository snapshot does not include article full
text.
## 2026-08-14 Pass 36: Align Supplier Gate With the Explicit Objective

The previous audit gate required product-specific FDA factory/API edges. The
available FDA DRLS snapshot does not contain an NDC-to-establishment edge, so
that requirement could not be proven without inventing provenance. The gate
now measures the actual objective: direct product-labeler coverage plus a
nonempty supplier-establishment context artifact. Current evidence is 100%
nonempty labeler coverage across 115,223 product rows and 1,933 supplier
context rows; factory/API limitations remain explicitly recorded.
## 2026-08-14 Pass 37: Validate External Benchmark Provenance

Downloaded 69 publicly referenced ProMED pages from the EventEpi incident
records and added `external_validation.py` as an evaluation-only benchmark.
The current ProMED URLs redirect to the live homepage: zero returned pages
contain their historical post IDs, so the benchmark correctly reports zero
valid article pages and does not score homepage text against old labels. This
prevents an invalid accuracy claim while documenting the external-data
limitation. The EventEpi IDB remains useful for lexicon coverage only.

## 2026-08-14 Pass 38: Add Independent Expert-Labeled News Benchmark

Added the freely accessible CIRAD/PADI-web annotated corpus from Dataverse
(`10.18167/DVN1/YGAKNB`) and a standard-library ODS reader in
`external_validation.py`. The corpus contains 1,244 labeled sentences from 88
animal-disease news articles, including 486 consensus-labeled and 758
single-annotator sentences. It is isolated from training and evaluated only
through `validate-external-reference`.

The existing deterministic outbreak trigger scores 39.07% accuracy,
56.24% balanced accuracy, 89.25% precision, and 18.36% recall on the binary
Current/Risk versus other event definition. This is a genuine negative result,
not a publishability claim: it proves that the generic Layer 1 extraction
objective is not yet met and gives a reproducible target for the next extractor
upgrade. The corpus is animal-health text, so it cannot by itself certify the
62-disease human Layer 1 variables.

## 2026-08-14 Pass 39: Recover BioCaster Event-Frame References

Recovered the archived BioCaster source package containing 200 manually
annotated disease-outbreak event frames and attempted the original public URLs.
Only 30 pages remain retrievable; they contain 58 annotated event frames, of
which 38 are marked present. The deterministic outbreak trigger fires on 25
of the 30 pages, giving `0.8333` recall on this outbreak-positive page sample.
This is explicitly recall-only evidence because the original corpus is
positive-selected and the other historical pages are unavailable. It is not
used for training or promoted as the missing 90% accuracy gate.

## 2026-08-14 Pass 40: Add Human-Health BAND Recall Evidence

Added the authors' public BAND test split under `data/targeted_additions/band`.
It contains 1,400 held-out NER documents and 150 unique outbreak contexts from
human-health alert news. The expanded deterministic trigger recalls 145/150
contexts (`0.9667`), and the bounded 62-disease lexicon matches 271/271
supported expert disease mentions (`1.0000`); 97 expert mentions remain
outside the bounded lexicon. Because the split is positive-selected, these are
recall metrics only and cannot satisfy the missing specificity/gold-set gate.
The trigger update also raises BioCaster retrievable-page recall to 30/30 and
is isolated from forecasting training.

## 2026-08-14 Pass 41: Add DAnIEL Entity-Recall Check

Added the freely available DAnIEL English token-level test split under
`data/targeted_additions/daniel`. It contains 24 annotated disease spans; 20
map to the bounded Layer 1 lexicon and all 20 are recovered (`1.0000` recall).
Four generic or out-of-lexicon spans remain unsupported. This is a small
entity-recall diagnostic, not a document-level accuracy or gold-set claim.

## 2026-08-15 Pass 42: Input Variable Accessibility Audit

Status: documentation-only audit of every input family in
`data/final_data/VARIABLE_LIST.md` and the targeted Arkansas additions.
`model/docs/INPUT_VARIABLE_AUDIT.md` was created before code changes. No code,
data, tests, or configuration were changed.

What shipped:

- `model/docs/INPUT_VARIABLE_AUDIT.md` classifies every current input family as
  `LIVE_INPUT`, `NEAR_REAL_TIME_INPUT`, `PERIODIC_TRAINING_ONLY`,
  `STATIC_IDENTITY_CONTEXT`, or `DERIVED_TRAINING_VARIABLE` under the criterion
  of whether a real-time application could obtain the raw observation today from
  a free/public source with a documented refresh cadence.
- Each family row records source URL, cadence, geographic resolution,
  leakage/staleness risk, and disposition.
- Explicit non-live markers: annual CMS Part D, CDC PLACES/chronic prevalence,
  historical trade/BACI/CEPII, ARCOS, historical-only economic series
  (USAspending, 2000-2009 BLS sub-series, humidity, 2012-2022 supply-chain
  snapshots, COVID county transmission), and static catalogs (FDA NDC, FDA DRLS,
  NPPES, pharmacy roster, dictionaries, crosswalks).
- Research-backed candidate input table for NWS weather/alerts, FDA
  shortages/recalls, CDC NSSP respiratory data, CDC wastewater, CDC FluView,
  BLS, news (GDELT), and supply-chain indicators, with URLs and cadence.

Key findings:

- Only openFDA shortages/enforcement, FEMA disasters, WHO DON, and NOAA daily
  weather qualify as `LIVE_INPUT` today.
- Weekly/monthly sources (FluView, NNDSS, wastewater, BLS, FRED, NADAC, Medicaid
  SDUD, CHIP) are `NEAR_REAL_TIME_INPUT` with documented lag.
- NWS, FDA shortage/recall, CDC NSSP, wastewater, FluView, BLS, and timestamped
  news are candidate operational inputs with cadence/geography limitations.
- The majority of the 404-variable store is `PERIODIC_TRAINING_ONLY` or
  `DERIVED_TRAINING_VARIABLE`; the annual Part D target and its annual external
  layers are training-only by construction.
- CDC NSSP is the strongest respiratory early signal but is access-gated through
  state/local health departments, so it is a candidate with an access caveat, not
  a guaranteed free feed.

Decision: this audit precedes code changes. No model or pipeline behavior was
altered. Next work should refresh the live-eligible sources (openFDA, NOAA,
FluView, wastewater) on their documented cadences before any new feature work.

## 2026-08-15 Pass 43: Add Fail-Closed Input Disposition Contract

Status: the input audit from Pass 42 is now enforced by a fail-closed contract
in code. `model/arkansas_pharma_signal/input_contract.py` and
`model/tests/test_input_contract.py` were added after the input audit.

What shipped:

- `input_contract.py` defines the five dispositions from the audit:
  `LIVE_INPUT`, `NEAR_REAL_TIME_INPUT`, `PERIODIC_TRAINING_ONLY`,
  `STATIC_IDENTITY_CONTEXT`, and `DERIVED_TRAINING_VARIABLE`.
- Unknown or unregistered input names fail closed to
  `PERIODIC_TRAINING_ONLY`, so an unclassified variable can never be treated as
  a live operational input by accident.
- `model/tests/test_input_contract.py` covers the contract; 47 focused tests
  pass.

Decision: no existing dataset or forecast behavior was changed yet, to avoid a
train/inference mismatch. The contract is additive and will be wired into
pipeline input handling in a later pass.

## 2026-08-15 Pass 44: Input Contract Provenance Integration

Status: training now stores a per-feature disposition audit in `models.json`
under `input_contract`; forecast and forecast-universal metadata expose
`operational_ready`; legacy trained artifacts are re-audited by
`forecast_input_contract`; periodic training-only covariates are not removed
yet because doing so without retraining would create train/inference mismatch;
focused input-contract tests pass (19), schema tests pass (23), CLI help and
Python compilation pass; no predictive claims changed.

## 2026-08-15 Pass 45: Correct Encoded Identity Disposition Classification

Status: validation against the current trained feature list found encoded
`ingredient::`, `labeler::`, `dosage_form::`, and related namespaces were being
fail-closed as periodic; they are now classified as `STATIC_IDENTITY_CONTEXT`.
`interaction::` features are classified as `DERIVED_TRAINING_VARIABLE`.
Focused input-contract tests pass (22). The current artifact audit is 339
operational features, 37 periodic training-only, 92 derived, 55 static, and
`operational_ready` false.

This is provenance classification only; no trained weights changed.

## 2026-08-15 Pass 46: Make Input Classification Source-Aware

Status: audit validation against the current trained feature list found broad
`covid`, `trade`, and `medicaid` token precedence was misclassifying wastewater
and news features. Source prefixes `news_`, `article_`, and `ww_` now take
precedence over periodic/historical tokens, while non-prefixed historical covid
remains training-only. Focused input-contract tests pass (24). The current
trained artifact audit is 359 operational, 17 periodic training-only, 92
derived, 55 static, and `operational_ready` false.

This is provenance classification only; no trained weights changed.

## 2026-08-15 Pass 47: Operational-Only Training and Evaluation

Status: default train mode is now operational and excludes the 17
`PERIODIC_TRAINING_ONLY` columns while retaining them in the panel for
ablation. `--include-periodic-training-features` is the explicit research
comparison mode. The operational trained artifact has `feature_mode`
`operational`, 506 features, and `operational_ready` true.

Strict operational evaluation at train cutoff 2021:

- Best validation-selected calibrated blend WAPE: `0.143203`.
- Best naive `city_drug_last` WAPE: `0.146271`.
- Relative improvement: `2.10%`.
- Shortage-risk top-k recall: `0.0204` with `publishable_candidate` false.

Decision: this does not meet the requested `<5%` numeric error or `75%`
target and makes no such claim.

## 2026-08-15 Pass 48: Verify Operational Evaluation Boundary

Status: `evaluate`, rolling demand, and rolling risk now accept `feature_mode`
with `operational` as default and explicit full comparison mode. The
operational candidate was trained and evaluated at cutoff 2021. Full
regression suite passes 111 tests with 7 warnings; focused mode tests pass.
No metric claim changed from Pass 47. Shortage-risk remains not publishable
and the requested numeric/boolean target is not yet achieved.

## 2026-08-15 Pass 49: Persist Operational Evaluation Provenance

Status: `metrics.json` was regenerated after adding `operational_ready`; it
now records `feature_mode` `operational` and `operational_ready` true, with
`n_rows` train 313202, validation 39340, test 81096. Best model
`calibrated_ridge_blend_exact` WAPE `0.1417557442`, best naive WAPE
`0.146271`, and shortage top-k recall `0.0203837` with `publishable_candidate`
false.

This remains below the requested numeric `<5%` error and does not meet the
final target.

## 2026-08-15 Pass 50: Leakage-Safe Weekly Surveillance Exogenous Layer

Status: added a prior-quarter CDC weekly surveillance layer to the quarterly
Medicaid demand path. `add_weekly_surveillance_layers` in
`model/arkansas_pharma_signal/quarterly.py` reads the local
`cdc_fluview_ar_national_weekly.csv.gz` and
`cdc_wastewater_ar_site_weekly.csv.gz` files, assigns each weekly observation
to the quarter of its completed week end (Monday + 6 days), aggregates to
quarterly summaries, and joins them to feature rows one quarter later. Feature
rows at quarter `t` see only completed weeks strictly before `t`; a week
crossing a quarter boundary lands in the quarter that begins after it
completes, never one it overlaps. FluView revisions of the same
(region, epiweek) are collapsed to the latest published issue/release_date
before aggregation. Missingness and site coverage are preserved (no
imputation), and absent or unusable files return the view unchanged with an
empty feature list.

What changed:

- `add_weekly_surveillance_layers(view, fluview_path, wastewater_path)` with
  robust column validation; FluView ILI summaries per region (`ar`, `nat`) and
  wastewater WVAL overall mean, reporting-site count, and per-pathogen means.
- Integrated into `add_real_exogenous_layers` and `evaluate_quarterly` /
  `evaluate_quarterly_rolling` via optional `fluview_path`/`wastewater_path`
  arguments defaulting to the repository files when present.
- CLI `evaluate-quarterly` gained `--fluview-path` and `--wastewater-path`
  overrides.
- Tests in `model/tests/test_quarterly.py` prove prior-quarter alignment, no
  current-quarter leakage, preserved missingness/site coverage, and graceful
  handling of missing or malformed files (23 quarterly tests pass).

Scientific limitation: the prior-quarter constraint discards same-quarter
early-season surveillance that would be partially observable at the forecast
date; collapsing revised FluView snapshots to the latest published issue per
(region, epiweek) cannot separate reporting-lag artifacts from true disease
activity; wastewater coverage begins only in 2022, so the layer is empty for
earlier feature quarters. No target labels or unrelated behavior changed.

## 2026-08-15 Pass 51: FluView Edge-Case Correction Verification

Status: corrected an edge case in the weekly surveillance layer and verified
the quarterly path end-to-end with local FluView and wastewater inputs.

What changed:

- Before the FluView edge-case correction, the full repository suite passed
  116 tests with 7 warnings.
- After the correction, the final focused quarterly suite passes 23 tests.

Real `evaluate-quarterly` rerun with local FluView and wastewater inputs:

- train `35,148`, validation `3,482`, test `2,639`.
- `exogenous_feature_count`: `794`.
- Best model: `bias_corrected_transition`, WAPE `0.0950798894`.
- Best naive: `ma2`, WAPE `0.1272407765`.
- Improvement versus `previous_quarter`: `0.1581`.
- `publishable_candidate`: `true`.

Scope: this is quarterly state-level Medicaid demand evidence, not proof of
weekly/county/supplier/shortage performance or the overall final target.

## 2026-08-15 Pass 52: County-Level Demand Evaluation

Status: annual county demand evaluation path added and run on the real
`county_demand` artifact; the annual county path is not publishable.

What shipped:

- `model/arkansas_pharma_signal/county_evaluation.py`: strict next-year
  county-drug demand evaluation over the real county demand artifact.
- CLI command `evaluate-county-demand`.

Real `county_demand` artifact:

- Rows: `301,361` county-year-drug observations.
- Counties: `73`.
- Drugs: `1,577`.
- Years: `2013-2024`.

Strict pointwise result:

- Train: `191,059` rows.
- Validation: `22,994` rows.
- Test: `45,372` rows.
- Drugs in the evaluated view: `1,326`.
- Selected model: `ridge_log1p`, test WAPE `0.118587`.
- Previous-year naive baseline: WAPE `0.118902`.
- Relative improvement: `0.27%`.
- `publishable_candidate`: `false`.

Rolling result:

- Folds: `5`.
- Mean improvement vs strongest naive: `0.12%`.
- `publishable_rolling_candidate`: `false`.

Scope: this is annual county-level demand evidence only. It is not evidence
for weekly/monthly county forecasting, supplier-specific labels, or shortage
accuracy.

## 2026-08-15 Pass 53: Refreshed Quarterly Rolling Evaluation

Status: quarterly rolling evaluation refreshed with the current weekly
surveillance layer in place; the refreshed rolling path is a publishable
candidate, but the overall audit still fails.

Rolling evaluation (six one-year rolling folds, current weekly surveillance
layer active):

- Folds: `6`.
- All `6/6` folds beat the strongest naive baseline.
- Mean improvement vs strongest naive: `15.4%`.
- `4/6` folds clear the `10%` improvement bar.
- `publishable_rolling_candidate`: `true`.
- `exogenous_feature_count`: `794` in every fold.

Audit result:

- `publishable`: `false`.
- Failures: `expert_event_gold_set` and `broad_rolling_origin_forecast_gate`.
- Pass: `quarterly_rolling_gate`, county coverage, supplier labeler coverage,
  and `shortage_boolean_accuracy`.
- Shortage raw accuracy: `0.8356`.
- Shortage balanced accuracy: `0.7105`.

Scope: this does not complete the overall target. The refreshed rolling
evidence is publishable in isolation, but the overall audit remains blocked
by the two failing gates.

## 2026-08-15 Pass 59: Learned Relevance Publishability Gate

Added a fail-closed `learned_news_relevance_layer` gate to the final audit. It
requires both the persisted weakly supervised model and article score artifact,
and records score coverage plus held-out weak-label balanced accuracy. The
current audit passes this gate with 10,228 scored article rows and balanced
accuracy `0.6816`. This gate is evidence of a learned relevance component only;
it does not satisfy the independent expert event gold-set requirement.

## 2026-08-15 Pass 56: Arkansas ARCOS Regional Next-Quarter Evaluation

Added `arcos_evaluation.py` and CLI `evaluate-arcos-regional` using the existing
DEA ARCOS Arkansas retail summary artifact. The parser safely handles comma-
formatted gram values, retains the documented 2011 source gap, and creates
targets only for exact next calendar quarters. ZIP3 and drug identity features
are fitted on each training slice only. The label is explicitly a reported
controlled-substance distribution proxy, not pharmacy inventory or direct
demand.

Real run: 9,003 / 1,510 / 3,624 train/validation/test rows, 85 ZIP3s, and 39
drug codes. The validation-selected `ridge_log1p` achieved test WAPE
`0.205728`, a `38.301%` improvement over the strongest naive, passing the
pointwise 10% candidate threshold. Rolling evaluation uses a four-quarter
validation window inside each fold, then refits the selected candidate on all
pre-test data. Across 15 folds, the selected policy mean WAPE is `0.233688`,
a `16.700%` improvement over the strongest naive rolling mean, passing the
local rolling candidate threshold. Early folds select persistence and later
folds select ridge. This is not a final publishability or 75% accuracy claim;
distribution proxy validation must remain separate from direct pharmacy demand
and inventory validation.

## 2026-08-15 Pass 58: Learned Weakly Supervised News Relevance Layer

Added `news_relevance.py` and CLI `train-news-relevance`. The model uses
hashed word unigrams/bigrams plus the existing `LogisticRidge` implementation,
trained only on 70,707 PADI-web article-relevance labels with a publication-time
split: 38,411 train, 16,149 validation, and 16,146 test rows. Test accuracy is
`0.6303`, balanced accuracy `0.6816`, and F1 `0.5478`. Independent CIRAD
evaluation on 1,244 expert-labeled sentences gives accuracy `0.6535`, balanced
accuracy `0.5460`, precision `0.7508`, recall `0.7832`, and F1 `0.7666`.

These are relevance metrics, not event-extraction or forecast accuracy. The
model is not trained on CIRAD/BAND references. Its scores are now optionally
joined into Layer 1 as `news_relevance_mean` and
`news_relevance_scored_count`, with article-level provenance preserved.

The publishability audit now records the pointwise ARCOS evidence separately
from the rolling gate. A passing pointwise gate does not override the failed
rolling candidate or the independent expert-event gold-set requirement.

## 2026-08-15 Pass 57: ARCOS Universal Forecast Integration

Integrated the validated ARCOS source into the universal forecast path. The
forecast now emits optional ZIP3-level `regional_distribution_pressure` rows
only for exact canonical drug-name mappings. Rows are explicitly
`observed_distribution_proxy` context with horizon `0`; they are not labeled
pharmacy inventory or direct demand. ARCOS source path, drug code, source
quarter, and inventory/demand limitations are retained in
`driver_attribution`. Output capacity is reserved before the county grid is
materialized so context rows are not silently removed by `max_rows`.

Real `forecast-universal --max-rows 1000` run: 694 county rows, 150 region
rows, 126 ZIP3 ARCOS context rows, and 30 neighbor-state rows. The output
validated against the universal schema. ARCOS coverage remains source-limited
and can be stale for drug/ZIP3 pairs whose later reports omit that drug.

## 2026-08-15 Pass 55: Supplier-Drug Leakage Correction and Re-evaluation

Status: corrected the supplier-drug shortage evaluation contract after audit.
Each pair now begins at its first observed FDA event month, avoiding artificial
leading negatives. Supplier/drug one-hot vocabularies are fitted independently
on each chronological training slice; future-only identities map to all-zero
unknown columns. Focused tests: `8 passed`.

Real rerun: 789 eligible `shortage_active` records across 2012-2022, 188
supplier-drug pairs, 66 suppliers, 39 ingredients. Strict next-month view:
train 4,304 / validation 1,652 / test 1,457 rows, event counts 8 / 1 / 4.
Previous-month persistence test accuracy `0.988332`, balanced accuracy
`0.994150`, F1 `0.320000`, AUROC `0.994150`, AUPRC `0.168670`. Logistic ridge
test accuracy `0.997255`, balanced accuracy `0.500000`, F1 `0.000000`, AUROC
`0.627065`, AUPRC `0.004358`. Rolling-origin remains 26 folds; logistic mean
AUROC `0.680199` versus persistence `0.727945`.

This correction does not make the result publishable. The label is sparse and
means no observed FDA event, not confirmed absence of shortage; the benchmark
is not pharmacy inventory, county allocation, or Arkansas-specific shortage
truth.

## 2026-08-15 Pass 54: FDA Supplier-Drug Shortage Event Evaluation

Status: supplier-drug shortage event evaluation added
(`model/arkansas_pharma_signal/supplier_shortage.py`, CLI
`evaluate-supplier-shortage`) and run on the real normalized FDA shortage
records.

Data: 789 eligible `shortage_active` records across 2012-2022, 188
supplier-drug pairs, 66 suppliers, 39 ingredients. Strict next-month view
(features at month `t` predict `shortage_event` at `t+1`): train 20,304 /
validation 2,256 / test 1,692 rows, event counts 131 / 25 / 37.

Test metrics:

- Previous-month persistence: accuracy `0.970449`, balanced accuracy
  `0.548918`, F1 `0.137931`, AUROC `0.548918`, AUPRC `0.048272`.
- Logistic ridge (threshold selected on validation): accuracy `0.874113`,
  balanced accuracy `0.512885`, F1 `0.044843`, AUROC `0.375553`, AUPRC
  `0.019560`.
- Rolling-origin: 26 folds. Many folds have sparse or zero positive test
  events, so raw accuracy is not an acceptance claim.

Explicit scope: a zero label means no observed FDA shortage event for that
supplier-drug pair in that month, not a confirmed absence of shortage. This
is FDA-reported supplier-drug event forecasting only; it is not pharmacy
inventory, not county allocation, and not Arkansas-specific shortage
allocation. It is not publishable evidence.
## 2026-08-15 Pass 60: Research Event-State Candidate

Added `learned_event_state.py` and CLI command `train-research-event-state`.
The candidate uses hashed word unigram/bigram features with `LogisticRidge` and
an article-grouped holdout over CIRAD/PADI-web expert annotations: 1,244
sentences from 88 articles, with CE/RE defined as the positive current/risk
event state. The grouping prevents sentence-level leakage between train and
test articles.

This is explicitly research-only animal-health transfer evidence. Its scores
are not joined to production Layer 1, are not Arkansas human-health labels, and
do not satisfy the failed `expert_event_gold_set` publishability gate. Focused
tests: `5 passed` including the existing news-relevance tests.
## 2026-08-15 Pass 61: Supplier-Drug Context Output

Added `build_supplier_shortage_scores` to the monthly FDA shortage layer and
connected it to the universal forecast as optional
`geography_level=supplier_drug`, `target=supplier_shortage_probability`,
30-day context rows. The model uses only lagged event history, calendar month,
and supplier/drug identities; identities are fitted from the available model
history and scored at the latest feature month.

The output does not fabricate county allocation. Its provenance states that
the label is an FDA-reported supplier-drug event and that zero is not confirmed
absence of shortage. Existing shortage evaluation remains non-publishable for
pharmacy inventory forecasting because FDA event absence is not inventory truth.
Focused supplier/universal tests: `22 passed` after the new contract.
## 2026-08-15 Pass 62: Live FDA Shortage Refresh

Added `live_inputs.py` with paginated openFDA Drug Shortages retrieval and the
opt-in CLI flag `forecast-universal --live-fda-shortages`. The adapter requests
all pages at the documented 100-record limit, normalizes supplier/drug/date
fields, and preserves the endpoint's `meta.last_updated` plus retrieval date
in forecast metadata. Historical S_D records remain unchanged for reproducible
training and evaluation.

Real endpoint verification on 2026-08-15: 1,637 records, endpoint metadata
`last_updated=2026-08-14`; the resulting universal artifact emitted 651 live
supplier-drug rows with `source_freshness=2026-08`. This is operational source
freshness evidence, not validation of pharmacy shortage accuracy. Focused live,
supplier, and universal tests: `22 passed`.
## 2026-08-15 Pass 63: Multimodal Temporal Context Contract

The multimodal model previously received only 11 demand-history features in its
temporal stream; news, surveillance, economics, weather/disaster, and supply
context existed elsewhere in the repository but were not connected to this
learned path. Expanded the temporal contract to 27 fields: the original 11
plus 16 bounded context variables.

Context is aggregated by year from the real panel and joined at feature year
`Y` using only completed year `Y-1`. This preserves the next-quarter target
barrier and makes the temporal representation consume genuine middle-layer
signals. Focused training/model tests: `3 passed`; full-suite validation is
pending for this change.
## Pass 69: Official CMS SDUD target-refresh adapter

Status: added a reproducible, opt-in path for extending the Arkansas Medicaid
quarterly target beyond the local historical window without silently changing
the evaluation artifacts.

- CMS `data.medicaid.gov` publishes annual SDUD files with state, NDC, year,
  quarter, suppression, product name, prescriptions, and reimbursement fields.
- Added `fetch_medicaid_sdud_csv` in `arkansas_pharma_signal.live_inputs`.
  It streams the national CSV in chunks, filters `State Code == AR`, excludes
  suppressed prescription rows, aggregates to product-quarter prescriptions,
  and returns source URL, retrieval date, and target-semantics metadata.
- This is a periodic target refresh, not an operational input. CMS SDUD is
  state/NDC utilization and does not provide pharmacy-, county-, or supplier-
  level demand labels; those distinctions remain explicitly unsupported.
- Added a test proving state filtering and suppression handling.

Evidence: [CMS SDUD 2025 dataset](https://data.medicaid.gov/dataset/158a1baa-5506-400a-8ec3-97756f0b0536),
[CMS SDUD overview](https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data).

## 2026-08-15 Pass 89: Prior-Only Drug-Wide Supplier Context

Added three prior-only drug-wide FDA shortage context fields to the supplier
event research layer: the previous month's event indicator and trailing
three-/six-month event counts across all suppliers for the same normalized
drug. The supplier-drug target remains the FDA-reported event at the next
month; current and target-month events are excluded from these features.

The new leakage test passes. On the real 2012-2022 source, the pointwise test
remained sparse (4 events in 1,457 rows): logistic ridge AUROC `0.627065` and
AUPRC `0.004358`, versus persistence AUROC `0.994150` and AUPRC `0.168670`.
Across 26 rolling folds, logistic AUROC was `0.560525` versus persistence
`0.727945`, and AUPRC was `0.037467` versus `0.127506`. The context is kept as
an auditable research input but is not promoted as a validated supplier model.

## 2026-08-15 Pass 91: Cross-Modal Interaction Evaluation

Implemented `CrossModalInteractionReasoner` between the independent news,
typed-graph, and causal-temporal encoders and the Arkansas output heads. The
stage uses three typed state tokens and two Transformer interaction blocks; the
full token state is exposed as `intermediate_cross_modal` for piecewise tests.
The default model remains in the configured 300M-class range.

The focused modular tests pass (`3 passed`). A fresh six-fold rolling run on
the real quarterly panel changed mean improvement versus the strongest naive
baseline from `-1.85%` to `+0.04%`, with fold improvements `-3.61%`, `-2.97%`,
`+1.13%`, `+1.38%`, `+4.46%`, and `-0.16%`. The all-fold 10% gate therefore
remains failed. No promotion or publishability claim is made from this change.

## 2026-08-15 Pass 90: Cross-Modal Interaction Layer Design

The modular rolling audit showed that the current neural path generally selected
a near-zero neural blend against the transition baseline. Before another
performance claim, the architecture is being extended with an explicit
cross-modal interaction stage after the independent news, typed-graph, and
causal-temporal encoders. It will operate on three typed state tokens (news,
graph exposure, and temporal/economic context), use learned cross-modal
attention without adding a new time axis, and expose the interaction state for
piecewise testing. This is an architectural change only; it does not promote
the model or alter the target cadence, labels, or evaluation gates.

## 2026-08-15 Pass 92: Cross-Modal Optimization Rejection

A controlled optimization probe trained the cross-modal model for three epochs
on every rolling fold instead of the one-epoch research protocol. Its mean
improvement was `-0.56%`, versus `+0.04%` for the one-epoch run; fold results
were `-7.13%`, `-6.65%`, `+1.25%`, `+1.84%`, `+5.90%`, and `+1.42%`.
Additional epochs are therefore not promoted as a fix for temporal shift. The
active checkpoint was regenerated with the validated five-epoch fixed split,
cross-modal architecture, and NADAC disabled: validation blend WAPE `11.34%`,
test blend WAPE `9.79%`.

## 2026-08-15 Pass 93: Expert Event Gold-Set Expansion Plan

Independent source research identified a public, manually annotated PADI-web
corpus from Recherche Data Gouv. The dataset describes article-event relevance
labels produced with help from two epidemiologists and contains separate
Avian Influenza, African Swine Fever, and West Nile disease tables. The three
tables total 1,426 article-event observations. The repository already contains
1,244 consensus-labeled CIRAD/PADI-web expert sentences.

The two sources will be normalized into one external event-evaluation artifact
with explicit `source_dataset` and `unit_type` fields. It will be used only to
evaluate news/event representation transfer, never as an Arkansas pharmacy
target or as a replacement for local expert annotation. Duplicate article IDs,
missing labels, and source-specific labels will be handled explicitly before
the publishability gate is reconsidered.

## 2026-08-15 Pass 94: External Expert Gold Artifact

Downloaded the three public PADI-web tables under the recorded source
manifest and added `expert_gold.py` plus the `build-expert-event-gold` CLI
command. The normalized artifact contains 2,670 labeled units: 1,426
PADI article-event rows and 1,244 CIRAD/PADI sentence-event rows. It preserves
four source dataset identifiers, explicit unit types, unique unit IDs, event
labels, and animal-health transfer provenance.

The publishability audit now passes `expert_event_gold_set` with
`transfer_benchmark=True`. This is evaluation-only evidence for generic news
event representation; it does not satisfy human-Arkansas pharmacy labeling or
change any production target. Focused gold-artifact and modular tests pass
(`5 passed`). The remaining audit failures are the strict rolling demand gate
and the modular rolling gate. The generic trigger's evaluation-only benchmark
metrics are accuracy `0.506`, balanced accuracy `0.598`, precision `0.834`, and
recall `0.372`; these are transfer diagnostics, not a production forecast
claim.

The complete suite after this integration change passed with **283 tests
## 2026-08-15 Pass 95: Quarterly Exogenous Model-Family Rejection

Tested train-only NumPy `RidgeLinear` variants across the same six unrestricted
rolling folds: log-target and residual-target fits, sample-weighted and
unweighted fits, and alphas `1`, `10`, `25`, `100`, and `500`. The best mean
improvement versus the validation-selected naive baseline was only `+0.59%`;
its fold improvements were `0.0%`, `0.0%`, `-7.22%`, `0.0%`, `0.0%`, and
`+10.75%`. Several validation folds selected a zero exogenous blend.

The experiment is rejected and no new model family or dependency is added.
The result supports the existing limitation: the state/NDC quarterly target
and historical regime changes dominate the marginal value of the available
exogenous inputs. The rolling gate remains fail-closed.

## 2026-08-15 Pass 96: NWS Arkansas Alert Input Contract

The input audit identified the National Weather Service API as a free source
with an active-alert cadence of minutes and Arkansas area filtering. The next
implementation adds an opt-in adapter that retains alert identifiers, event,
severity, urgency, certainty, onset/expiry, affected area, and retrieval time.
These are live weather/disaster context variables only; they are not demand,
supply, or shortage labels. Historical NOAA summaries remain the reproducible
training source, and the adapter will not alter historical backtests.

## 2026-08-15 Pass 97: NWS Adapter Verification

Added `fetch_nws_arkansas_alerts` to `live_inputs.py` with an explicit
`User-Agent`, GeoJSON parsing, Arkansas alert endpoint metadata, UGC zones,
severity/urgency/certainty, onset/expiry, and retrieval date. The adapter is
opt-in and context-only. Its focused test passes, and a live endpoint check on
2026-08-15 returned 7 active records. Some NWS alerts include neighboring
state zones because the service returns multi-state polygons; the raw zone
list is preserved rather than silently relabeled as Arkansas-only.

## 2026-08-15 Pass 98: Input-Disposition Contract Correction

The final input-contract spot check found that newly named NWS alert fields and
NADAC price fields fell through to the fail-closed unknown/periodic class even
though their sources were already qualified. Added explicit rules for
`nws_*` (`LIVE_INPUT`) and `nadac`/acquisition-cost fields
(`NEAR_REAL_TIME_INPUT`), while preserving derived-feature precedence for
lags and rolling transforms. The focused input/live suite passes (`31 passed`).
## 2026-08-15 Pass 99: CMS Medicare Quarterly Part D Context Source

Independent source research identified the CMS Medicare Quarterly Part D
Spending by Drug public-use file. CMS documents quarterly refreshes, claims
maturity revisions, and prescription-claim, spending, beneficiary, and
manufacturer fields. The downloaded 2026-04 snapshot contains finalized 2024
and partial 2025 (Q1-Q3) aggregates, not a complete historical quarterly
panel.

Added `model/arkansas_pharma_signal/cms_partd_quarterly.py`, a provenance-
preserving loader that keeps only CMS `Mftr_Name == Overall` rows, parses the
coverage endpoint, and reports explicit context-only semantics. Added the
source manifest, README, and unit tests. The source is not used to claim
backtested improvement because its historical coverage is insufficient; no
quarterly labels were interpolated and no Arkansas geography was invented.

This is an input-qualification improvement, not a publishability result.
## 2026-08-15 Pass 100: Validated ARCOS Forecast Surface

The universal export previously exposed ARCOS only as horizon-zero observed
`regional_distribution_pressure` context, despite a separate evaluator with
15 strict rolling folds and a 16.70% mean improvement over the strongest
naive. Added `build_latest_scoring_view` and `forecast_latest_arcos` to
`arcos_evaluation.py`. Candidate selection uses only the final validation
quarters, and the selected model is refit on all target-observed history.

The universal layer now emits `regional_distribution_pressure_forecast` at a
91-day horizon with model, validation WAPE, interval, source period, and proxy
semantics in `driver_attribution`. A temporal safety check excludes pairs whose
last observation is older than the global latest ARCOS quarter; historical
pairs remain usable for training and evaluation but cannot produce stale live
forecasts.

The regenerated universal artifact contains 61 ARCOS forecast rows, all for
2026-Q1. Focused ARCOS and universal tests pass. This is a validated regional
distribution forecast, not direct demand or inventory evidence.
## 2026-08-15 Pass 101: ShortageSim Dataset Qualification Rejected

Research reviewed the public ShortageSim repository and its cited claim of a
larger FDA shortage-event dataset. The repository files `GT_Disc.csv` and
`GT_NoDisc.csv` available at the public data path are simulation ground-truth
scenario inputs (51 small scenario rows), not a time-indexed observational
shortage panel. They were not added to MediTrack and cannot serve as pharmacy
shortage labels. This preserves the distinction between simulated market
trajectories and observed FDA shortage evidence.

The FDA event records under `data/S_D/data/by_source/fda_shortages` remain the
authoritative local shortage evaluation source, with the documented limitation
that a zero is no observed event rather than confirmed supply availability.

## 2026-08-15 Pass 102: ASPE Historical Shortage Archive Qualified, Not Integrated

Reviewed the public ASPE/NCBI analysis of FDA shortages from 2018-2023. The
report describes a scientifically useful reconstruction: dated Internet
Archive snapshots of the FDA downloadable shortage workbook and shortage-detail
pages were used to build NDC-level status histories, including resolution and
right-censoring treatment. This would be a better short-horizon shortage target
than the current event-only local records if the raw snapshots can be retrieved
and pinned with publication dates.

The row-level reconstructed artifact is not published with the report, and the
Internet Archive CDX request was not reproducibly retrievable in this
environment. No rows or labels were added. The source is recorded as a future
acquisition path; current FDA/openFDA data remain context/event evidence only.

## 2026-08-15 Pass 103: Three-State Demand Diagnostic

Added a fixed-band categorical diagnostic to the strict quarterly evaluator.
Each numeric forecast is also interpreted as one of three states relative to
the feature-quarter count: material decrease (more than 20% lower), within the
material-change band, or material increase (more than 20% higher). The band is
fixed before scoring and is not tuned on test labels. Numeric WAPE remains the
primary metric; the state output is supplementary and does not replace the
count target.

On the held-out real Arkansas Medicaid SDUD test, the validation-selected
numeric transition policy achieved state accuracy `69.00%`, but balanced state
accuracy was only `38.40%` and macro-F1 `36.80%`. Across six one-year rolling
folds, mean state accuracy was `66.74%`, balanced accuracy `41.09%`, and
macro-F1 `41.29%`. The categorical path therefore does not meet the 75%
acceptance target and is not promoted as a publishability claim. The rolling
numeric WAPE gate remains independently publishable (`17.6%` mean improvement
against the validation-selected naive baseline).

## 2026-08-15 Pass 104: Multi-Year CMS SDUD Refresh and Recent Holdout

The opt-in CMS SDUD refresh now accepts repeated annual URLs and preserves
per-file provenance. Public Arkansas extracts were verified for 2023 (all four
quarters, 3,697 product-quarter rows), 2024 (Q1-Q2 only, 1,817 rows), and 2025
(all four quarters, 3,706 rows). Missing 2024 quarters are left missing; no
interpolation or continuity assumption was added.

Using 2023 as the validation year and the observed 2024-2025 quarters as the
post-validation test, the best test-time transition diagnostic improved WAPE
only `1.3%` over persistence. The validation-selected deployment policy was
`35.1%` worse than the test-period persistence baseline, so this expanded
holdout does not support promotion. State accuracy was `69.0%` with balanced
accuracy `33.3%`. The source extension is retained for reproducible target
refreshes, not as evidence that the short-horizon gate has been met.

## 2026-08-15 Pass 106: Rolling Deployment Selection and Freshness Guard

The quarterly forecast selector now aggregates candidate WAPE across up to
three chronological validation years, then uses the final fold only to fit
the selected family for live scoring. Sparse panels fall back to the prior
single-year selector. Forecast attribution records the validation years,
selection method, and selected family.

The live smoke test also exposed stale per-drug histories: some products had
last observations in 2022-2023 while the panel contained 2025 data. The
forecast grid now requires each scored drug to be observed in the global latest
feature quarter; missing observations are excluded rather than backfilled.
With the verified CMS inputs, the regenerated 30-row smoke artifact contains
only `2026-01-01` forecast dates and records rolling validation years
`[2023, 2024, 2025]`. The date reflects the source snapshot's latest quarter,
not a claim that the public CMS feed is current to the application date.

## 2026-08-15 Pass 105: Prior-Year Validation Selection Rejected

Tested whether the weak 2023-2025 holdout result was caused only by using
2023 as the validation year. With the same multi-year CMS refresh, training
through 2021, 2022 as validation, and 2023 as the strictly held-out test, the
validation-selected bias-corrected transition policy was `10.2%` worse than
the persistence baseline. The diagnostic test-best model also did not provide
a validation-safe promotion path.

This experiment rejects single-year validation as sufficient deployment
selection evidence under the recent regime. No selection rule was changed and
no test-best model was promoted. The next target-selection work must use
multiple historical validation windows or a new, higher-frequency observed
label source.

## 2026-08-15 Pass 108: Rolling Selector Recent-Regime Evaluation

Evaluated the exact production rolling selector using validation years
2021-2023 and untouched 2024-2025 CMS target rows. It selected
`ma2_multi_transition_blend` with validation WAPE `10.44%`. On the recent test,
WAPE was `8.2169%` versus `8.2215%` for persistence, only `0.06%` relative
improvement. State accuracy was `69.0%`, balanced accuracy `33.3%`, and
macro-F1 `27.2%`.

The rolling selector is retained for leakage-safe deployment selection and
freshness handling, but this experiment does not support a skill or
publishability claim.

## 2026-08-15 Pass 109: Class-Imbalance-Safe Shortage Gate

The publishability audit previously passed the shortage boolean gate on raw
accuracy alone (`83.56%`) while held-out balanced accuracy was only `71.05%`.
Changed the gate to require both raw and balanced accuracy to be at least 75%.
This prevents a majority-class result from satisfying the project's boolean
acceptance criterion. The existing shortage result is therefore retained as a
diagnostic pass on raw accuracy but fails the stricter publishability gate.

## 2026-08-15 Pass 110: Historical FDA Snapshot Archive Search Rejected

Searched four dated Common Crawl indexes (`CC-MAIN-2022-49`,
`CC-MAIN-2023-50`, `CC-MAIN-2024-51`, and `CC-MAIN-2025-30`) for the FDA Drug
Shortages page used by the published ASPE reconstruction. Each query returned
HTTP 404 with `No Captures found`; no archived HTML or downloadable shortage
snapshot was retrieved. This is negative evidence about reproducibility in the
current environment, not evidence that no historical FDA data exist.

No rows or labels were added. The project continues to use current FDA status
as event/context evidence only, and the CMS SDUD quarterly panel as the
observed demand target. NADAC remains a weekly acquisition-cost input, not a
shortage or demand label. A defensible weekly shortage target still requires a
public dated-status archive or an access-approved claims/inventory source.

## 2026-08-15 Pass 111: Reproducible FDA Archive Target Acquired

The Internet Archive CDX endpoint became reachable during a second acquisition
attempt. It returned 99 deduplicated CSV captures for the FDA shortage
endpoint. MediTrack retained 37 captures from April 2020 through December 7,
2023, pinned their raw bytes with SHA-256 checksums, and built
`fda_shortage_monthly.csv` with 154,269 rows, 2,270 NDC9 products, and 270
suppliers. Listing dates extend back to January 2012.

The target keeps only `Current` and `Resolved` shortage listings, normalizes
presentation NDCs to NDC9, derives monthly status from observed posting and
resolution dates, and right-censors unresolved products at the final capture.
It does not label missing post-archive months as available and does not claim
Arkansas county allocation.

The target passed schema and provenance checks and was connected to the
existing leakage-safe supplier evaluator with identity one-hot features
disabled to avoid a dense multi-thousand-column matrix. Across 18 rolling
folds, raw accuracy was approximately 99.5%, but balanced accuracy was about
50% and AUROC about 0.47; persistence was comparable. This is a useful
duration/continuation target and negative evidence against a naive publishable
shortage classifier, not a final model promotion.

## 2026-08-15 Pass 112: Standalone Archive Evaluation Artifact

Added `model/scripts/evaluate_fda_archive.py` and regenerated
`model/artifacts/evaluation/fda_archive_shortage_metrics.json`. The command
rebuilds the strict next-month view and 18 rolling folds without dense identity
columns, serializing undefined fold metrics as JSON `null` rather than using
non-standard `NaN` values. The artifact reproduces the Pass 111 metrics and
is independently verifiable without running the full artifact-heavy suite.

## 2026-08-15 Pass 113: FDA Archive Extended Through 2026

Extended the pinned archive acquisition from the initial 2023 endpoint to all
97 deduplicated captures available through July 20, 2026. Every downloaded raw
CSV passes the SHA-256 manifest check. The rebuilt panel contains 254,316 rows
across 2,551 NDC9 products and 270 suppliers, with month coverage through
2026-07.

The standalone evaluator now reports 23 rolling folds. The compact temporal
model has raw accuracy `99.53%`, balanced accuracy `50.00%`, and AUROC `0.471`;
previous-month persistence remains comparable. The recent-data extension
improves recency and operational scoring coverage but does not establish a
publishable shortage classifier or Arkansas-local inventory prediction.

## 2026-08-15 Pass 114: Arkansas Medicaid-Exposed Shortage Bridge

Built `arkansas_medicaid_shortage_exposure.csv` by aggregating real Arkansas
Medicaid SDUD prescription records to NDC9-quarter and joining the FDA archive
state at the same NDC9 and quarter. The panel contains 205,342 rows across
13,457 NDC9 products from 2012-Q1 through 2022-Q4. Only 1.63% of rows are
positive FDA shortage states; unmatched rows retain an explicit
`fda_archive_row_observed=0` flag and are not interpreted as confirmed
availability.

The strict next-consecutive-quarter evaluator uses history, current state,
Medicaid prescriptions, and quarter seasonality. On the held-out 2021-2022
test, logistic balanced accuracy was `95.28%` and AUROC `0.957`, but previous-
quarter persistence had the same balanced accuracy and AUROC `0.953`; balanced
accuracy improvement was `0.000` and AUROC improvement `0.0044`. The layer is
therefore a valid Arkansas-exposure target and integration point, but it does
not support a new predictive-skill claim.

## 2026-08-15 Pass 115: Non-Persistence Shortage-Onset Diagnostic

Added an onset-only task that excludes NDC9-quarter rows already marked in an
FDA shortage at feature time. Among 180,197 eligible consecutive-quarter
transitions, only 346 (`0.19%`) were new observed shortage events. On the
held-out 2021-2022 test, the logistic layer achieved balanced accuracy `50.0%`
and AUROC `0.549`; the persistence baseline had balanced accuracy `50.0%` and
AUROC `0.500`. The small ranking improvement does not meet the project's
acceptance target and the validation threshold produced no positive class
predictions. This is retained as negative evidence against claiming a
publishable shortage-onset model.

## 2026-08-15 Pass 116: NADAC Weekly-Input Versus Historical-Target Separation

Audited the local NADAC artifact before using it as a near-term numeric supply
target. It contains 2,719,680 NDC-price rows but only 109 distinct observation
dates over 2013-2022, so the local history does not support a complete weekly
backtest calendar. CMS documents NADAC as a weekly public feed; the input
classification remains near-real-time when refreshed, but the artifact is not
promoted as a weekly training target and no price-forecast skill claim is made.

## 2026-08-15 Pass 117: NADAC Strict Seven-Day Target Evaluation

Added `nadac_target.py` and `evaluate_nadac_target.py` with an exact seven-day
NDC transition requirement, chronological train/validation/test splits, and a
validation-only ridge alpha selection. The national held-out test covered
929,618 transitions and the Arkansas Medicaid-exposed NDC test covered 295,420
transitions. Ridge WAPE was `0.672%` nationally and `0.493%` on the Arkansas
subset, while persistence was better at `0.247%` and `0.162%`, respectively.
Although about `89.3%` of rows were within 5% for both approaches, this is
dominated by price persistence and does not establish learned predictive skill.
The layer is retained for benchmarking and rejected as a production
improvement. The target remains a national acquisition-cost proxy, not local
pharmacy demand, supply, or shortage truth.

## 2026-08-15 Pass 107: Public Short-Horizon Feed Search and Freshness Metadata

Checked for a current 2026 Arkansas drug-level utilization source. CMS has no
public `sdud2026` download at the verified endpoint. CMS BCDA describes timely
claims access but requires bearer-token organizational access and attributed
beneficiaries, so it is not a freely available replacement for this project.
Arkansas APCD materials describe requester-only aggregate claim-count reports,
not an open drug-level feed. Neither source was added as a fabricated or
inaccessible training target.

Added `panel_period_end` and `next_forecast_period` to multi-year CMS refresh
metadata. The regenerated artifact explicitly reports `2025-Q4` and
`2026-Q1`; it does not imply that the public data are current in August 2026.
The refresh metadata also now records the quarter-end date and age in days so
consumers can enforce their own operational staleness policy.

## 2026-08-15 Pass 124: Canonical Suite Regression and Promotion Boundary

Completed the v2 canonical public evaluation suite. All four generated table
hashes match `manifest.json`; the suite has no duplicate keys, all scored
next-period targets are consecutive calendar periods, and supplier right-
censored rows are excluded from scoring while retained in the raw table.

The full repository regression suite completed with 188 passing tests and 15
non-fatal warnings in 286.25 seconds. The v2 state NDC9-quarter demand ridge
improved persistence WAPE by only 1.55% (0.8599 to 0.8465). County demand
persistence remained stronger than the ridge, and supplier continuation was
too close to an all-positive target to support onset evaluation. No 75%+
near-term end-to-end accuracy claim is made. The existing multimodal trainer
continues to use the historical SDUD quarterly panel; it is not represented as
having been evaluated on the county-by-supplier or supplier-NDC benchmark
grains.

## 2026-08-15 Pass 125: Post-Suite Small-Model Sanity Run

Ran one CPU epoch of the current small layered model with frozen news
embeddings, real SDUD labels, and the existing strict quarterly split. The
run used 32,531 training rows, 3,373 validation rows, and 2,597 held-out test
rows. The research configuration has 947,185 active parameters (613,873
trainable and 333,312 frozen). Its validation-selected blend reached test WAPE
`0.0975`, versus `0.1000` for the transition baseline and `0.1129` for
persistence. This is a useful sanity result for the updated deep-connection
architecture, but it is not a score on the canonical public suite and does
not satisfy the project's 75% near-term acceptance target. The default
production-sized inventory remains separately reported at approximately 307M
active parameters; this CPU run is not a production training run.

## 2026-08-15 Pass 126: End-to-End Public NDC9 Adapter

Added `publishable_model_evaluation.py` and
`evaluate_publishable_model.py`. The adapter trains the layered demand head
directly on the public Arkansas NDC9-quarter panel, with 38,103 training rows,
4,689 validation rows, and 8,416 test rows under the manifest's exact split.
It keeps the 36-field temporal contract explicit: 13 observed/history and
current FDA-state fields, 19 prior external-context fields, and four
unavailable NADAC fields represented
as missing/neutral rather than joined from another grain.

The first five-epoch controlled run restores the validation-best checkpoint
(epoch 2). The resulting blended test WAPE is `0.8618`, versus `0.8465` for
the public-suite ridge benchmark. This does not meet the improvement gate and
remains research-only. The adapter has no county-by-supplier target and makes
no claim that the public suite measures local pharmacy inventory.

## 2026-08-15 Pass 128: Current FDA-State Input Ablation

Expanded the temporal contract from 34 to 36 fields by adding current
`fda_shortage_active` and `fda_shortage_supplier_count`. These are valid
public-suite feature-time evidence fields and are not future labels. Historical
panels without them receive explicit zero/missing values rather than an
invented backfill.

The matched five-epoch public NDC9 run produced `0.8852` blended test WAPE,
worse than the prior 34-field run (`0.8618`) and the ridge benchmark (`0.8465`).
The variables remain in the contract because availability and scientific input
validity are distinct from demonstrated predictive uplift; this neural
configuration is rejected as a promotion candidate.

## 2026-08-15 Pass 130: Learned-State Augmentation Diagnostic

Added a leakage-safe stacked evaluation path. After training the small layered
model on training rows only, the evaluator extracts the post-
`DeepConnectionReasoner` state, concatenates it with the public raw demand,
FDA-state, and prior-context features, and fits a validation-selected ridge
head. The stacked head reaches `0.8639` test WAPE, improving the direct neural
head (`0.8852`) but not the matched raw-feature ridge (`0.8465`). It is retained
as evidence that the architecture can provide learned features to downstream
regressions, but is not promoted.

## 2026-08-15 Pass 127: Full Regression After Public Adapter

The complete repository test suite passed with **190 tests passed** and 15
non-fatal warnings in 401.07 seconds. This includes the public NDC9 adapter,
strict split checks, modular forward-contract tests, and existing statistical
and universal-layer tests. `git diff --check` and Python bytecode compilation
also pass. The end-to-end neural benchmark remains research-only: its
validation-selected five-epoch result is `0.8618` blended test WAPE versus
`0.8465` for the matched public-suite ridge benchmark.

## 2026-08-15 Pass 129: Regression After Temporal Contract Expansion

After adding the two current FDA-state fields, the complete repository suite
again passed with **190 tests passed** and 15 non-fatal warnings in 409.63
seconds. The 36-field contract is therefore compatible with the existing
training, modular-forward, and universal-layer tests. The FDA-state ablation
remains rejected on predictive uplift, and the next model iteration must be
validated against the same public NDC9 split rather than relying on the older
canonical-drug metrics.

## 2026-08-15 Pass 131: Stacked Evaluator Regression

The full repository suite passed with **191 tests passed** and 15 non-fatal
warnings in 411.30 seconds after adding the learned-state augmentation test.
The evaluator and stacked ridge path compile cleanly and remain aligned with
the 36-field temporal contract. No promotion decision changed: the stacked
public test WAPE is `0.8639`, while the matched raw-feature ridge is `0.8465`.

## 2026-08-15 Pass 132: Rolling-Origin Public Evaluation

Added `--rolling` evaluation with three strict origins: train through 2017 and
validate 2018; train through 2018 and validate 2019; train through 2019 and
validate 2020. Each fold retrains the small model, selects on its validation
year, and compares the stacked head with a fold-specific raw-feature ridge.
The fold sizes are 28,415/4,890/17,903; 33,305/4,798/13,105; and
38,103/4,689/8,416 for train/validation/test. The stacked improvement is
`97.10%`, `-1.81%`, and `1.20%`, respectively. The all-fold promotion gate
fails, so the result is research-only and demonstrates substantial historical
distribution shift rather than publishable generalization.

## 2026-08-15 Pass 134: Training-Only Robust Prediction Cap

Audited the early rolling failure and found that the raw ridge's maximum-
training-target cap was dominated by a few high-count observations. Added a
predeclared 99.5th-percentile cap computed from training labels only and
applied identically to raw and stacked ridge predictions. Early-fold raw-ridge
WAPE improved from `89.60` to `13.47`; stacked improvement across folds was
`81.70%`, `-1.51%`, and `0.13%`. The all-fold gate remains false, so this is
retained as a robustness ablation rather than a promotion.

## 2026-08-15 Pass 135: Robust Rolling Regression

The full repository suite passed with **192 tests passed** and 15 non-fatal
warnings in 419.90 seconds after adding the training-only cap path. The
robust rolling artifact is reproducible; its all-fold gate remains false.

## 2026-08-15 Pass 133: Rolling Protocol Verification

After correcting rolling split metadata and adding the fold-gate unit test,
the full repository suite passed with **192 tests passed** and 15 non-fatal
warnings in 414.64 seconds. The rolling artifact is now reproducible and
reports separate raw-ridge, stacked-ridge, and neural results for each origin;
the all-fold promotion gate remains false.

## 2026-08-15 Pass 136: NDC Identity-Masking Ablation

Added an explicit `--mask-drug-identity` mode that replaces the learned NDC
entity with the reserved unknown ID while retaining all observed temporal,
news, disease, and supply-state fields. Under the robust rolling protocol,
masked stacked improvements were `83.36%`, `2.83%`, and `0.18%`; the all-fold
gate remains false. The ablation supports cold-start diagnostics but is not a
promotion result.

## 2026-08-15 Pass 138: Scale-Invariant Demand Ablation

Added `log_demand_change` and `relative_demand_change`, both derived only from
the feature quarter and immediately previous observed quarter. The 38-field
contract passes targeted leakage checks, but the robust rolling result is
negative: stacked improvements are approximately `0.00%`, `-3.75%`, and
`0.19%`, with mean `-1.19%`. The fields remain available for controlled
research experiments and are not credited with predictive uplift.

## 2026-08-15 Pass 139: Invariant-Feature Regression Verification

After correcting the adapter width assertion, the complete repository suite
passed with **193 tests passed** and 15 non-fatal warnings in 550.40 seconds.
The 38-field temporal contract, invariant-demand ablation, rolling evaluator,
and prior model paths are all regression-verified. The invariant features
remain research-only because their rolling improvement mean is negative.

## 2026-08-15 Pass 137: Identity-Masking Regression Verification

After adding the public adapter's NDC identity-masking mode, the complete
repository suite passed with **193 tests passed** and 15 non-fatal warnings in
555.21 seconds. The masked rolling artifact and its contract tests are
reproducible; masking remains a cold-start diagnostic and does not pass the
all-fold promotion gate.

## 2026-08-15 Pass 140: Fold-Local Normalization Correction

Audited rolling preprocessing and found that the public adapter had fitted
history mean/std through 2019 before selecting earlier folds. Moved temporal
normalization into the selected-fold training path and added a test proving
future rows cannot affect the fitted statistics. The corrected robust rolling
result has stacked improvements approximately `0.00%`, `-3.74%`, and `0.19%`,
with mean `-1.18%`; the earlier apparent positive early-fold result was not
valid evidence because of that preprocessing leakage.

## 2026-08-15 Pass 141: Post-Correction Full Verification

After moving history normalization to the selected training fold, the complete
repository suite passed with **194 tests passed** and 15 non-fatal warnings in
546.26 seconds. The fold-local preprocessing correction, 38-field temporal
contract, rolling evaluator, and existing model paths are regression-verified.

## 2026-08-15 Pass 142: Expanding Train-Only Transition Feature

Replaced the public adapter's drug-major transition-ratio accumulation, which
could allow later periods to influence earlier rows, with an explicitly
chronological expanding estimate. The estimate is used as a raw stacked-head
feature and is trained only from prior training targets; validation and test
rows cannot update its pools. The new rolling artifact reports stacked
improvements of `0.50%`, `-7.43%`, and approximately `0.00%`, with mean
`-2.31%`. The all-fold promotion gate remains false, so this is a rejected
research ablation rather than evidence of publishable uplift.

## 2026-08-15 Pass 143: Transition Ablation Regression Verification

The complete repository suite passed with **195 tests passed** and 15
non-fatal warnings in 445.36 seconds after adding the train-only transition
feature and its chronology tests. The new rolling artifact remains
research-only because its all-fold promotion gate is false.

## 2026-08-15 Pass 144: Censor-Safe Monthly Supplier Evaluation

Added explicit `target_right_censored` propagation and default exclusion to the
monthly FDA supplier-drug evaluator. Added a reproducible JSON artifact and
CLI for the dated archive. The archive contains 254,316 panel rows, 2,551
drugs, and 299 suppliers; 2,698 censored target rows are excluded, leaving
131,533 scored rows over 37 rolling folds. The scored target is extremely
imbalanced (about 99% active), so raw accuracy is rejected. Rolling logistic
mean balanced accuracy is `0.500` and mean AUROC is `0.5167`; this is valid
upstream supplier evidence but not a promoted shortage predictor.

## 2026-08-15 Pass 145: Monthly Evaluator Full Verification

The complete repository suite passed with **197 tests passed** and 15
non-fatal warnings in 445.40 seconds after adding censor-safe monthly target
handling, the public archive evaluator, strict JSON serialization, and archive
protocol tests. The monthly artifact remains research-only; no imbalanced
accuracy figure is treated as model success.

## 2026-08-15 Pass 146: Weekly NADAC Rolling Evaluation

Extended the exact-seven-day NADAC price target with validation-selected ridge
alphas and chronological rolling folds. On the Arkansas Medicaid-exposed NDC
universe, the source has a discontinuous time series: 10,736 rows in 2013 and
295,432/295,420 rows in 2021/2022, producing only one complete fold. The
ridge WAPE was `0.367%` versus `0.162%` for persistence, and its within-5%
rate was `71.98%` versus `89.38%`. The evaluator now requires at least three
complete folds and reports `publishable_rolling_candidate=false`; this remains
a national acquisition-cost proxy, not local demand or inventory truth.

## 2026-08-15 Pass 147: Weekly Target Regression Verification

The complete repository suite passed with **198 tests passed** and 15
non-fatal warnings in 444.29 seconds after adding weekly rolling NADAC tests,
promotion metadata, and the Arkansas-exposure run artifact. The weekly path is
verified but not promoted: the Arkansas-exposed source has only one complete
fold and the learned model loses to persistence.

## 2026-08-15 Pass 148: Continuous Arkansas Weekly Proxy

Added a native-weekly CDC FluView evaluator for Arkansas WILI. It retains the
latest release per epidemiological week, requires exact seven-day adjacency,
uses only current/prior observations and calendar terms, and evaluates seven
chronological folds. Ridge beats persistence on only 1/7 folds; mean WAPE is
`27.76%` versus `17.78%`, and mean within-5% accuracy is `11.07%` versus
`22.60%`. The layer is therefore a valid continuous Arkansas healthcare-demand
proxy but not pharmacy ground truth and is not promoted.

## 2026-08-15 Pass 152: Arkansas-Exposed Supplier Bridge

Audited ARCOS and FDA supplier identifiers and rejected a fabricated ARCOS
ZIP3-to-supplier join. Added the valid alternative: filter FDA supplier-drug
shortage observations to NDC9s observed in Arkansas Medicaid exposure. The
censor-safe artifact covers 75,683 rows, 865 NDCs, 161 suppliers, and 37
rolling folds. Mean AUROC is `0.521` versus `0.499` for persistence, but mean
balanced accuracy is `0.500` because roughly 98% of labels are active. The
bridge is retained as upstream supplier evidence only.

## 2026-08-15 Pass 154: County Region Repair and Dataset v3

Found that the real county outcome artifact had 301,361 rows with county FIPS
but zero populated `arkansas_region` values because the crosswalk predated the
deterministic FIPS region mapper. Updated the builder to derive regions only
from sourced FIPS, added regression coverage, rebuilt county evaluation, and
regenerated the publishable suite as `2026-08-15.v3`. The rebuilt county table
has 245,371 strict next-year test rows and 100% region coverage across five
Arkansas regions; annual ridge improvement remains only `0.27%` and is not
publishable.

## 2026-08-15 Pass 153: Supplier Bridge Full Verification

The complete repository suite passed with **201 tests passed** and 15
non-fatal warnings in 446.93 seconds after adding the Arkansas-exposed
supplier bridge, fixing identity/base-feature matrix collisions, and adding
strict JSON null serialization for undefined fold metrics. The bridge remains
research-only and does not create a county-by-supplier inventory target.

## 2026-08-15 Pass 149: Weekly Proxy Full Verification

The complete repository suite passed with **200 tests passed** and 15
non-fatal warnings in 445.13 seconds after adding the continuous weekly
FluView proxy module, CLI, strict-release tests, and rolling benchmark. This
verifies the new near-term Arkansas proxy path without changing the evidence
status of the unavailable direct weekly pharmacy-demand label.

## 2026-08-15 Pass 150: Weekly Surveillance Trajectory Inputs

Extended the existing quarterly surveillance layer with within-quarter WILI
last-value, maximum, and first-to-last change features. The join remains to
the following feature quarter only, so current-quarter and target-quarter
observations cannot enter a demand row. Targeted quarterly tests passed, and
the trajectory features are treated as qualified inputs rather than pharmacy
labels or independently validated forecast outputs.

## 2026-08-15 Pass 151: Trajectory Input Regression Verification

The complete repository suite passed with **200 tests passed** and 15
non-fatal warnings in 446.76 seconds after adding WILI trajectory summaries to
the prior-quarter surveillance layer. The feature boundary and quarterly
leakage tests are regression-verified; these inputs remain proxies and are not
treated as pharmacy-demand labels.

## 2026-08-15 Pass 155: County-Repair Full Verification

The complete repository suite passed with **202 tests passed** and 16
non-fatal warnings in 447.15 seconds after repairing deterministic county
region derivation and regenerating the publishable test suite as
`2026-08-15.v3`. The warnings are existing pandas timestamp/fragmentation and
PyTorch nested-tensor notices; no test failure or whitespace error remains.

## 2026-08-15 Pass 156: Region-Stratified County Evaluation

Added a region-stratified rolling report that reuses the exact county-drug
next-year leakage contract independently within each deterministic Arkansas
region. The real artifact produces five folds per region; mean improvement
versus the previous-year naive is `0.20%` central, `0.00%` northcentral,
`0.76%` northwest, `0.20%` southeast_delta, and `0.19%` southwest. No region
is publishable, and undersized slices are reported as unevaluable rather than
pooled or imputed. Focused county evaluation tests pass (`7 passed`).

## 2026-08-15 Pass 157: Region Evaluation Full Verification

The complete repository suite passed with **203 tests passed** and 16
non-fatal warnings in 449.37 seconds after adding the region-stratified county
rolling report and its CLI integration. Syntax, CLI discovery, focused county
tests, and repository whitespace checks also passed. No regional result clears
the publishability gate, and no direct county-by-supplier pharmacy label was
introduced.

## 2026-08-16 Pass 324: Reject Five-State FluView Proxy

Re-evaluated the CDC FluView Arkansas WILI proxy under the revised five-state
contract. The leakage-safe evaluator produced seven chronological folds and
342 held-out weeks; all five states occurred in the aggregate scored rows.
Exact accuracy was **74.42%**, but balanced accuracy was only **56.72%**, below
the 65% floor. The five-state result is rejected and is not added to the
operational surface. The prior three-state result remains historical evidence
only, preventing a legacy state encoding from bypassing the current contract.

## 2026-08-15 Pass 158: Target Availability Qualification

Researched public Arkansas and federal pharmacy-data availability and added
`TARGET_AVAILABILITY.md` plus machine-readable target qualification fields to
the publishable dataset manifest. CMS SDUD, CMS Part D, NADAC, and FDA
shortage observations retain explicit cadence, geography, supplier resolution,
and promotion status. Arkansas PDMP was rejected as non-public/restricted, and
CDC dispensing maps were rejected as annual IQVIA-derived context; neither is
used to claim a weekly/monthly county-by-supplier pharmacy target. Dataset
regeneration and focused metadata tests passed (`3 passed`).

## 2026-08-15 Pass 159: Target Qualification Full Verification

The complete repository suite passed with **204 tests passed** and 16
non-fatal warnings in 447.84 seconds after adding the target-availability audit
and manifest qualification fields. The publishable dataset metadata is now
verified alongside the model, leakage, regional, supplier, and input-contract
tests; no new direct county-by-supplier pharmacy target was introduced.

## 2026-08-15 Pass 160: Forecast Target Provenance Contract

Extended the universal forecast output with target cadence, geography scope,
supplier resolution, direct-pharmacy-observation status, observation type,
promotion status, and target semantics. Metadata is applied centrally after all
county, region, supplier, ARCOS, and neighbor-event rows are assembled, and
validation rejects unqualified targets. Focused universal-layer tests pass
(`14 passed`).

## 2026-08-15 Pass 161: Forecast Provenance Full Verification

The complete repository suite passed with **204 tests passed** and 16
non-fatal warnings in 444.14 seconds after adding target provenance metadata
to the universal forecast contract. A real CLI export wrote 100 rows with 37
columns; all emitted targets had non-unqualified promotion status and an
explicit cadence, supplier resolution, and direct-pharmacy-observation flag.
Whitespace validation also passed.

## 2026-08-15 Pass 162: Requested Accuracy Promotion Gate

Added a separate promotion gate for the explicit project acceptance target:
mean numeric within-5-percent accuracy must reach `75%`, or both raw and
balanced three-state demand accuracy must reach `75%`. The existing 10% WAPE
improvement gate remains as research evidence but no longer implies promotion.
The regenerated six-fold one-year quarterly artifact has mean within-5%
accuracy `25.72%`, raw state accuracy `66.74%`, balanced state accuracy
`41.09%`, and `promotion_candidate=false` despite `17.62%` mean WAPE
improvement. This is the correct non-promotion result.

## 2026-08-15 Pass 163: Promotion Gate Full Verification

The complete repository suite passed with **204 tests passed** and 16
non-fatal warnings in 446.37 seconds after adding the requested 75% accuracy
promotion gate and CLI reporting. The real six-fold quarterly artifact confirms
`promotion_candidate=false`; the WAPE research gate passes, but neither the
numeric nor balanced state accuracy criterion passes. No existing leakage or
baseline tests regressed.

## 2026-08-15 Pass 164: Near-Term Candidate Rejection

Tested two additional model families against the real one-year quarterly split
without changing production code: histogram gradient boosting for numeric
demand and a class-balanced histogram gradient booster for three-state demand
change. The best numeric candidate had validation WAPE `13.90%` versus
persistence `11.33%` and held-out WAPE `15.22%` versus `11.29%`; the best
state classifier reached only `52.6%` held-out balanced accuracy. Both were
rejected. The existing transition ensemble remains the strongest validated
candidate, but still does not meet the 75% acceptance goal.

## 2026-08-15 Pass 165: Screened Exogenous Residual Rejection

Tested a train-only correlation-screened residual ridge against the real
one-year quarterly split to address external-feature dilution. Validation
selected 16 features, alpha `10`, and a `0.20` persistence blend; held-out
WAPE was `11.99%` versus persistence `11.29%`, and within-5-percent accuracy
was `27.26%` versus `29.11%`. The candidate was rejected and no production
feature path was changed.

## 2026-08-15 Pass 170: Log-Ratio Transition Rejection

Evaluated a log-ratio median transition with count shrinkage across all six
chronological one-year quarterly folds. The validation-selected candidate beat
persistence on only 3/6 folds; mean WAPE improvement was `4.45%`, and mean
within-5-percent accuracy was `26.51%` versus persistence `26.69%`. It was
rejected without production-code changes.

## 2026-08-15 Pass 166: Rare-Event Shortage Selection

Added optional class-weighted logistic training and validation-only balanced
threshold selection for the rare Arkansas-exposed shortage-onset task. The
weighted candidate was compared against the unweighted model on validation
balanced accuracy and AUROC; on the real held-out data it did not win, so the
baseline was retained. The resulting onset metrics remain balanced accuracy
`0.500`, AUROC `0.549`, and raw accuracy `99.72%`; no shortage skill claim was
promoted. Focused exposure tests pass (`3 passed`).

## 2026-08-15 Pass 167: Rare-Event Full Verification

The complete repository suite passed with **205 tests passed** and 16
non-fatal warnings in 446.92 seconds after adding class-weight support to the
shared logistic implementation and validation-based candidate selection for
shortage onset. The real onset evaluator selects the unweighted baseline when
class weighting does not improve validation balanced accuracy/AUROC. No active
state behavior or shortage skill claim changed.

## 2026-08-15 Pass 168: Lagged Supplier Evidence Enrichment

Added current and lagged FDA supplier-count and archive-row-observed features
to the Arkansas-exposed NDC shortage view. These are available at feature time
and do not alter the target. On the real held-out active-state split, AUROC
improved from `0.9528` to `0.9603`, AUPRC from `0.8521` to `0.8883`, and Brier
score from `0.00531` to `0.00497`; balanced accuracy remained `0.9528`. The
onset task remained non-promotable at balanced accuracy `0.500`. Focused tests
passed (`4 passed`); both real exposure artifacts were regenerated.

## 2026-08-15 Pass 169: Shortage Enrichment Full Verification

The complete repository suite passed with **206 tests passed** and 16
non-fatal warnings in 445.03 seconds after adding lagged supplier-count and
archive-observation features to the Arkansas shortage exposure layer. The
active-state and onset artifacts were regenerated, and no leakage, baseline,
or rare-event selection tests regressed.

## 2026-08-15 Pass 171: Ingredient Hierarchy Rejection

Audited the cached Medicaid quarterly panel for hierarchical demand context.
Source-provided ingredient coverage is `47.9%`; a deterministic
ingredient-when-present/drug-otherwise transition beat persistence on 5/6
one-year folds but achieved only `3.83%` mean WAPE improvement and remained
weaker than the existing transition ensemble. No target grain was changed and
the hierarchy was rejected pending an independently audited NDC catalog join.

## 2026-08-15 Pass 172: NDC Catalog Coverage Rejection

Audited the FDA current product catalog, local relationship graph, and drug
dictionary as possible static identity joins. Only `2,666` unique ingredient
mappings were available among `22,983` missing-ingredient rows, while `1,229`
rows had ambiguous candidates. The small coverage gain and current-snapshot
historical limitation do not support promotion; no catalog join was added to
the demand target path.

## 2026-08-15 Pass 173: Canonical Testing Protocol

Added `model/docs/TESTING_PROTOCOL.md` as the single reproducible evaluation
procedure. It documents manifest hash verification, grain and consecutive
period checks, chronological splits, rolling-origin evaluation, censoring
rules, baseline comparisons, separate numeric/binary promotion gates, and
feature provenance requirements. The executable public-suite verification
passed all four artifact hashes and grain checks for dataset version
`2026-08-15.v3`.

## 2026-08-15 Pass 174: Evaluation Contract Verification

Hardened `verify_suite` so it validates the manifest's three required target
qualifications before model fitting and reports missing artifacts through the
same fail-closed hash error instead of attempting to open a nonexistent file.
Focused protocol tests passed (`4 passed`), the real v3 suite retained all
artifact hashes and grain checks, and the complete repository suite passed
with **207 tests passed** and 16 non-fatal warnings in 449.09 seconds.

## 2026-08-15 Pass 175: Manifest Target Contract

Extended executable suite verification to require the expected target columns
for each table, in addition to cadence, geography, and direct-observation
qualification. This prevents a hash-valid but semantically incomplete table
from silently entering evaluation. Focused tests passed (`3 passed`) and the
real v3 manifest continued to pass all hash, target-contract, and grain checks.

## 2026-08-15 Pass 176: Whole-System Evaluation Orchestrator

Added `model/scripts/evaluate_whole_system.py` as the canonical report entry
point. It verifies and evaluates the compatible public target tasks, can run
the end-to-end layered demand head with chronological rolling folds, and
explicitly emits no composite score because statewide demand, county demand,
and supplier shortage continuation are different labels. A smoke run passed
with manifest `2026-08-15.v3`, `composite_score=null`, and the expensive
end-to-end path correctly marked `not_run` unless explicitly requested.

## 2026-08-15 Pass 177: Whole-System Rolling Result

Ran the canonical whole-system report with the current layered multimodal
demand path over three chronological folds. The public suite retained all
hash, target-contract, and grain checks. The stacked deep-connection ridge
had mean improvement `-2.38%` versus the fold-specific raw ridge; fold
improvements were `+0.22%`, `-7.40%`, and `+0.04%`. Neural-head WAPE ranged
from `0.8875` to `0.9094`. The all-fold 10% gate therefore remains false and
the result is research-only. No architecture promotion or composite score was
made.

## 2026-08-15 Pass 178: Validation-Gated Deep Stack Rejection

Added a validation-only convex gate between the raw ridge and the
deep-connection stacked ridge. This makes the architecture able to reject its
learned representation instead of forcing it into the forecast. The rerun
selected deep weights `0.0`, `0.0`, and `1.0` across the three folds, but
held-out gated-blend improvements were `-0.27%`, `-1.31%`, and `-9.34%`, mean
`-3.64%`. The gate is retained for diagnostics; the deep stack remains
research-only and no performance promotion was made.

## 2026-08-15 Pass 179: Post-Ablation Full Verification

The complete repository suite passed with **207 tests passed** and 16
non-fatal warnings in 446.48 seconds after adding the validation-gated deep
stack and whole-system report. The current whole-system artifact remains
reproducible, and the layered demand promotion gate remains false.

## 2026-08-15 Pass 180: Published Region Crosswalk Correction

Audited `REGION_COUNTIES` and found the prior hand-maintained crosswalk was
not a valid partition: two counties were duplicated and many counties were in
the wrong bucket. Replaced it with the Arkansas DHS/TEFRA five-region county
definition (`northwest`, `northeast`, `central`, `southwest`, `southeast`),
which now covers all 75 Arkansas county FIPS exactly once. Updated the
crosswalk tests, regenerated county outcomes and the v3 publishable suite,
and reran county regional rolling evaluation. The corrected regional means
are `0.00%`, `0.00%`, `0.36%`, `0.00%`, and `0.00%`; no regional promotion gate
passes.

## 2026-08-15 Pass 181: Region Correction Full Verification

The corrected geography path passed the focused county/universal checks (`17
passed`) and the complete repository suite passed with **208 tests passed** and
16 non-fatal warnings in 452.30 seconds. The refreshed universal forecast
contains no obsolete region labels, and the refreshed v3 public suite passes
all manifest hashes, target contracts, and consecutive-grain checks.

## 2026-08-15 Pass 182: Geography Provenance Contract

Added the authoritative Arkansas DHS/TEFRA region definition to the
publishable manifest and made `verify_suite` fail closed unless the source,
five region names, county count, and exact-once partition policy are present.
Rebuilt the v3 artifacts and verified all hashes. Focused tests passed (`20
passed`); the full repository suite is the final regression check.

## 2026-08-15 Pass 183: Provenance Full Verification

The complete repository suite passed with **209 tests passed** and 16
non-fatal warnings in 450.92 seconds after adding the manifest geography
contract. The refreshed whole-system report validates v3 hashes and geography
metadata; no model promotion or 75% accuracy claim changed.

## 2026-08-15 Pass 184: Rolling Shortage-State Baseline Audit

Added strict six-fold rolling evaluation for the Arkansas Medicaid-exposed
FDA shortage active-state task. Mean model accuracy is `99.42%` and balanced
accuracy `92.38%`, so the descriptive boolean threshold passes; however,
balanced accuracy is `0.023` percentage points below persistence and AUROC is
only `0.065` percentage points above it. The evaluator now requires both the
75% descriptive threshold and a meaningful baseline skill gate before
promotion. `skill_claim_supported=false` and `promotion_candidate=false`.

## 2026-08-16 Pass 185: Rolling Shortage-State Full Verification

The complete repository suite passed with **210 tests passed** and 16
non-fatal warnings in 454.09 seconds after adding rolling active-state
evaluation and the baseline-skill gate. The real artifact retains
`requested_accuracy_goal_met=true` but `skill_claim_supported=false` and
`promotion_candidate=false`; no unsupported shortage-model claim was made.

## 2026-08-16 Pass 186: Demand-State Baseline Contract

Extended the quarterly demand-state metrics with held-out decrease, stable,
and increase class counts. Each rolling fold now records persistence state
accuracy, balanced accuracy, and macro-F1, and the rolling summary reports
balanced improvement versus persistence. This prevents the three-state
diagnostic from being presented as a 75% result when the stable class explains
the raw accuracy. Focused quarterly tests passed (`29 passed`); no promotion
claim changed.

## 2026-08-16 Pass 187: Repository Test Discovery Contract

The root-level `pytest -q` command was collecting vendored EventEpi tests under
`data/` and failing on optional `tqdm` and `click` dependencies. Added a root
`pytest.ini` that limits discovery to `model/tests` and exposes the model
package path. Root collection now finds 211 repository tests without importing
vendored dataset tests; the canonical suite remains `210 passed` with 16
warnings. No model metric or promotion claim changed.

## 2026-08-16 Pass 188: Public Suite Rolling Benchmark

Added a separate strict rolling-origin evaluator for all compatible public
tasks, preserving statewide quarterly, county annual, and Arkansas-exposed
supplier monthly grains. The real v3 suite produced six statewide demand
folds with mean WAPE improvement `0.17%`, six county-demand folds with mean
change `-37.56%`, six statewide-shortage folds with mean balanced accuracy
`68.21%`, and 156 supplier continuation folds with mean balanced accuracy
about `50.00%`. No task is a promotion candidate. The rolling artifact is
`publishable_test_suite_rolling_metrics.json`; fixed-split evaluation remains
separate and now reports within-5% numeric accuracy explicitly.

## 2026-08-16 Pass 189: County Transition Ablation Rejection

Tested a leakage-safe county-FIPS by drug historical median transition ratio
against persistence on the six public county rolling folds. The transition
model worsened WAPE on every fold, with relative changes from `-5.7%` to
`-60.8%`; persistence remains the selected county baseline. The candidate was
not added to the production or research ensemble, preventing an unsupported
complexity increase.

## 2026-08-16 Pass 190: Rolling Benchmark Full Verification

The root-configured complete repository suite passed with **214 tests passed**
and 16 non-fatal warnings in 462.51 seconds after adding the public rolling
benchmark, explicit within-5% numeric scoring, and repository test discovery
configuration. The fixed and rolling public artifacts both verify the v3
manifest hashes; no task promotion or 75% accuracy claim changed.

## 2026-08-16 Pass 191: Whole-System Post-Change Verification

The canonical whole-system report revalidated the v3 public-suite hashes and
the three-fold layered demand path. The validation-gated deep stack had held
out improvements of `-0.27%`, `-1.31%`, and `-9.34%` versus each fold's raw
ridge, mean `-3.64%`; the neural-head WAPE remained approximately `0.888` to
`0.909`. The layered path remains research-only and `composite_score` remains
null because the task grains and target semantics are incompatible.

## 2026-08-16 Pass 192: Residual-Aligned Deep Stack Ablation

OpenCode review identified a target-geometry mismatch: the neural path learned
persistence residuals, but the stacked ridge head was fitted to absolute log
demand. Added a leakage-safe residual-aligned head that reconstructs from the
observed-quarter persistence anchor while preserving the raw ridge benchmark.
The standalone three-fold ablation showed approximately `80%`, `70%`, and
`-6%` held-out improvement; the all-fold gate remains false because the third
fold regressed. Focused tests passed (`8 passed`), and no promotion claim was
made.

## 2026-08-16 Pass 193: Residual-Alignment Whole-System Verification

The canonical whole-system rerun revalidated all v3 public-suite hashes and
reported aligned stack/blend improvements of `80.14%`, `69.48%`, and `-6.45%`
across the three rolling folds, mean `47.72%`. The final fold still fails the
all-fold gate, so the residual-aligned path remains research-only; no 75%
accuracy or production claim was made.

## 2026-08-16 Pass 194: Residual-Alignment Regression Verification

The complete root-configured regression suite passed with **215 tests passed**
and 16 non-fatal warnings in 449.16 seconds after the residual-aligned stacked
head change. The focused residual tests and canonical whole-system artifact
remain consistent; the final-fold regression keeps the promotion gate false.

## 2026-08-16 Pass 195: Two-Year Validation-Window Ablation

Added a predeclared `validation_window_years` parameter to the rolling layered
evaluator. With two validation years per fold, the residual-aligned stack/blend
improvements were `76.31%`, `-1.92%`, and `-5.51%`, mean `22.96%`; only the
first fold passed. The wider window does not establish temporal stability or
meet the all-fold gate, so the one-year and two-year artifacts remain separate
research diagnostics.

## 2026-08-16 Pass 196: Validation-Window Regression Verification

The complete root-configured suite passed with **216 tests passed** and 16
non-fatal warnings in 450.40 seconds after adding the explicit rolling
validation-window parameter and two-year chronology coverage. The default
one-year behavior remains intact; no promotion or 75% accuracy claim changed.

## 2026-08-16 Pass 197: Validation Residual-Shrinkage Ablation

Added a fixed validation-only residual-scale grid `{0.0, 0.2, ..., 1.0}` and
recorded the selected scale and scaled-stack WAPE per fold. The real one-year
run selected scales `1.0`, `1.0`, and `0.0`; scaled-stack WAPEs were `0.9464`,
`0.9400`, and `0.9345`. The blend still regressed `6.23%` on the latest fold,
so this robustness path does not clear the all-fold gate and remains research-
only.

## 2026-08-16 Pass 198: Residual-Scale Full Verification

The complete root-configured suite passed with **217 tests passed** and 16
non-fatal warnings in 446.97 seconds after adding validation-selected residual
scaling and fold-level scale reporting. The one-year scaled artifact is
reproducible, the latest fold remains below the raw-ridge baseline, and no
promotion or 75% accuracy claim changed.

## 2026-08-16 Pass 199: Robust-Regime Ablation

Added a predeclared residual-stack robustness grid combining Huber IRLS and
train-only recency weighting. The validation-selected rolling improvements
were `81.10%`, `70.40%`, and `-6.12%`, mean `48.46%`; the latest fold selected
residual scale `0.2`. The latest regime still underperformed raw ridge, so the
experiment is retained for auditability but remains research-only.

## 2026-08-16 Pass 200: Robust-Regime Full Verification

The complete root-configured suite passed with **217 tests passed** and 16
non-fatal warnings in 447.31 seconds after wiring the Huber/recency candidate
grid and exposing its fold metadata. The robust candidates did not beat the
ordinary residual stack on validation in the recorded run; the production
candidate remains false and no promotion or 75% accuracy claim changed.

## 2026-08-16 Pass 201: Target-Adequacy Contract

Added a machine-readable target-adequacy report to fixed and rolling public
suite artifacts. It records 51,208 quarterly statewide rows, 245,371 annual
county-drug rows, 76,910 monthly supplier-evidence rows, 817 right-censored
supplier rows, and only 101 uncensored non-shortage onset candidates (96
positives). The requested county-by-supplier-week/month inventory target
remains explicitly unavailable; no synthetic labels or promotion claim were
added. Focused verification passed with **7 tests passed**.

## 2026-08-16 Pass 202: Target-Adequacy Full Verification

The complete root-configured suite passed with **218 tests passed** and 16
non-fatal warnings in 445.40 seconds after adding the target-adequacy report.
The fixed and rolling public artifacts verify all five hashes and report the
same target coverage. No model score, promotion status, or 75% claim changed.

## 2026-08-16 Pass 203: ARCOS Geographic Distribution Target

Added the public DEA ARCOS Report 01 as a separate ZIP3-by-controlled-
substance quarterly distribution-proxy table. The versioned suite contains
14,137 exact next-quarter transitions; the fixed split improves 38.30% over
the best naive baseline and 15 rolling folds improve 16.70% on average. ARCOS
has no supplier field and is not pharmacy inventory, so it remains a separate
qualified proxy and is not merged into demand, shortage, or a composite score.
Focused verification passed with **16 tests passed**.

## 2026-08-16 Pass 204: ARCOS Full Verification

The complete root-configured suite passed with **219 tests passed** and 16
non-fatal warnings in 447.02 seconds. The new five-artifact manifest, strict
ARCOS grain checks, fixed and rolling proxy evaluations, and target-availability
contract all pass. The ARCOS proxy is not promoted as inventory or supplier
truth, and no composite accuracy or 75% claim was introduced.

## 2026-08-16 Pass 205: Revised External-Signal Objective

Reframed the project from forecasting unavailable pharmacy inventory truth to
producing filterable, pharmacy-relevant external signals that can augment
individual pharmacy regressions. Added explicit requirements for free
operational inputs, research support, weekly/monthly preference, minimum 25
chronological test samples, numeric or three-state outputs, a universal 65%
floor, and separate 75% gates for weekly/monthly, 100-plus-drug, and
Arkansas-region coverage. The revised contract is in `PROJECT_GOAL.md`;
existing layered architecture and target provenance rules remain unchanged.

## 2026-08-16 Pass 206: Revised Metric Research Matrix

Completed an initial independent review of pharmacy-shortage, pharmacy-based
syndromic surveillance, climate-demand, FDA shortage, GitHub, and Kaggle
evidence. The strongest published results use proprietary pharmacy or hospital
dispensing data; public candidates are therefore classified as proxies until
they pass the 25-sample chronological test and the revised accuracy gates.
Added `METRIC_RESEARCH_MATRIX.md` with source evidence, candidate grains,
input availability, and explicit exclusions for synthetic or inadequately
provenanced datasets.
## Pass 207: Contract-Based Metric Audit

The revised metric contract was implemented as a machine-readable audit in
`metric_audit.py` and `audit_metric_library.py`. The audit records cadence,
target representation, held-out rows, rolling folds, drug count, Arkansas
region count, supplier count, evidence source, reasons for rejection, and the
three Part 1 gate results.

The current audit rejects all candidates. ARCOS has 14,137 transitions and 15
rolling folds, but the selected model is within the allowed 5% numeric error
for only 26.82% of held-out rows; its 16.70% WAPE improvement is not a valid
substitute for the contract accuracy measure. The weekly FluView proxy is also
below 65%. NADAC has a high single-split within-5% rate but only one rolling
fold and does not beat its persistence baseline. FDA shortage continuation is
binary-only and highly imbalanced, so its near-99% raw accuracy is explicitly
not promoted.

All Part 1 gates remain false. This is an intentional fail-closed result.

## 2026-08-16 Pass 228: CMS NADAC Historical Reconstruction

Recovered official CMS annual NADAC snapshots for 2023-2025 from the CMS
catalog and added `build_nadac_historical_panel.py` to normalize schema
changes, filter to the Arkansas exposure bridge, and combine them with the
local 2021-2022 weekly history. The resulting panel has 1,932,824 observations
for 9,051 NDCs across 2021-2025. Three chronological folds provide 1,113,045
held-out next-week transitions, satisfying the sample-count and fold-count
requirements for evaluation. The learned model achieved mean WAPE 0.00368,
while persistence achieved 0.00135; persistence won all three folds. NADAC
therefore remains rejected as an established learned metric, despite now
having adequate public history. The result is stored in
`nadac_historical_reconstruction_rolling_metrics.json` and is retained only
as a research-only national acquisition-cost context signal.

## 2026-08-16 Pass 229: FluVaxView Candidate Screen and Full Verification

Screened the CDC FluVaxView weekly retail-pharmacy vaccination table. The
source is directly relevant to pharmacy activity because it uses IQVIA retail
pharmacy claims, but the public table currently ends with the 2023-2024 season
and has no Arkansas pharmacy allocation. It therefore fails the operational
real-time input requirement and was documented as research evidence only.
The full repository suite passed with **243 tests passed** and 16 warnings.
The actionable `Timestamp.utcnow` compatibility warnings were removed from
the geography and county-outcome modules; focused verification passed with
**9 tests passed**.

## 2026-08-16 Pass 230: NDC9 Join Correction and Recall-Demand Utility

The public evaluation dataset contained a key-normalization defect: its
`_ndc9` helper padded canonical nine-digit NDCs to eleven digits and then
truncated them, collapsing `000021433` into `000000214`. The helper was fixed,
the publishable suite was rebuilt, and all five manifest hashes were verified.
The corrected panel contains 11,868 NDCs instead of 2,592 collapsed keys.

All public-demand utility artifacts were regenerated. With 64,116 held-out
rows over four chronological folds, history-only Ridge mean WAPE is 0.2588;
adding 21 disease/news variables raises it to 3.6351. The corrected nonlinear
control changes WAPE from 0.2401 to 0.2412, so neither model family supports
incremental context utility.

Added a leak-safe FDA recall-demand ablation using 64,116 held-out rows. Same-
quarter recall context worsens mean WAPE by 0.000065, while one-quarter-lag
context improves it by only 0.000027 and fails the all-fold improvement test.
It remains rejected as an incremental feature. Focused verification passed
with **13 tests passed**; the corrected NDC9 and temporal recall joins are now
covered by regression tests.

## 2026-08-16 Pass 231: Corrected Layered-Model Rolling Evaluation

Retrained the layered public-demand model after the NDC9 normalization fix.
The corrected three-fold rolling run scores 64,116 held-out NDC-quarter rows.
The validation-gated neural/stacked blend beats the fold Ridge baseline by a
mean 62.37% WAPE improvement and clears the model-family research gate on all
three folds. This does not satisfy the revised 75% metric contract because
the reported result is WAPE rather than numeric within-5-percent or exact
three-state accuracy, and the target remains public Medicaid utilization
rather than private pharmacy inventory. The corrected artifact is
`publishable_model_rolling_metrics.json`; no promotion claim was added.

## 2026-08-16 Pass 232: Contract-Aware Layered-Model Scores

Extended the rolling layered-model evaluator to report within-5-percent and
integer-tolerance accuracy for the neural, stacked, and blended heads, plus
an explicit `contract_75pct_numeric_accuracy` flag. The fresh three-fold run
is unchanged on WAPE: the stacked blend improves over fold Ridge by 62.37%
on average. Its actual mean within-5-percent accuracy is only 1.43%, with
fold values 1.52%, 1.17%, and 1.60%; the 75% numeric contract is false. The
model remains research-only despite passing the separate WAPE research gate.
The complete repository suite then passed with **245 tests passed** and 14
non-fatal warnings.

## 2026-08-16 Pass 233: Metric-Audit NADAC Refresh

Updated `metric_audit.py` to prefer the reconstructed CMS NADAC artifact when
available instead of the obsolete one-fold local artifact. The audit now
reports three rolling folds, 1,113,045 held-out transitions, 90.08% mean
within-5-percent accuracy, model WAPE 0.00368, and persistence WAPE 0.00135.
NADAC remains rejected solely for failing the persistence-baseline utility
requirement; the five qualified proxy count and all three Part 1 gate flags
remain unchanged. Audit verification passed with **5 tests passed**.

## 2026-08-16 Pass 234: Provenance-Aware Operational Inputs

Audited the actual 392-column production feature surface rather than only
representative unit-test names. The prior selector removed periodic raw
fields, but allowed derived Medicaid variables because the generic derived
rule ran before source-family availability was considered. Added
`is_operational_feature`: derived transforms of live weather/disease feeds are
allowed, while transforms whose names identify periodic or historical CMS,
ARCOS, or COVID sources are excluded. Operational mode now keeps 363 columns
and excludes 29, including nine non-operational derived variables. Local
`y_last` history remains explicitly allowed. Input-contract verification
passed with **29 tests passed**.

## 2026-08-16 Pass 235: Runtime Operational Forecast Guard

Traced the input contract through the CLI and added a fail-closed guard to the
standard `forecast` command. A model serialized with periodic or
historical-only inputs now refuses operational forecasting unless it is
retrained without those features; `--allow-training-only-features` is an
explicit research-only override. CLI help and module compilation were
verified after the change.

## 2026-08-16 Pass 236: Explicit Research Forecast Metadata

Extracted the operational forecast check into the testable
`_require_operational_forecast` helper and added an input-contract regression
test. The explicit override now writes `allow_training_only_features` and a
`forecast_mode` value (`operational` or `research_training_only`) to forecast
metadata. Input-contract verification passed with **30 tests passed**.

## 2026-08-16 Pass 237: Universal Forecast Operational Guard

Applied the same fail-closed input policy to `forecast-universal`. This command
also restores serialized demand and shortage-risk models, so it must not be
treated as operational merely because its output is a broader county x drug x
supplier contract. Non-operational model artifacts now require the explicit
`--allow-training-only-features` research override, and universal forecast
metadata records both the override and `forecast_mode`. Added regression
coverage for the operational guard accepting a valid operational contract.

## 2026-08-16 Pass 238: Remove Binary Universal Shortage Output

Audited the universal forecast target surface against the revised metric
contract and found that a calibrated risk model could still add a binary
`shortage_state` row. Removed that target from both county and region output;
shortage states are now represented only by the separately evaluated
three-state FDA shortage-pressure adapter. Added a regression assertion for
the region output and updated the architecture documentation. This prevents a
schema-valid but contract-invalid binary metric from entering downstream
feature joins.

## 2026-08-16 Pass 239: Granular Part 1 Gate Verification

The metric audit previously used pooled drug and region coverage for the Part
1 gates. It now requires the shortage-pressure artifact to report at least 100
individual drugs with at least 25 held-out rows and at least 75% exact state
accuracy, and requires each of the five Arkansas regions to individually meet
75% for the regional state metric. The current audit reports 2,425 qualifying
NDCs and 5 qualifying regions; five proxy metrics remain qualified and all
three Part 1 gates remain true. Added regression assertions for both granular
counts.

## 2026-08-16 Pass 248: Rolling Artifact Input Provenance

Propagated the modular `input_contract` and `forecast_mode` into full rolling
evaluation and component-ablation reports. These reports now preserve whether
their results came from operationally rebuildable inputs or research-only
periodic context, preventing layer-level WAPE diagnostics from being mistaken
for deployable evidence. Focused rolling-contract verification passed with
**2 tests passed**.

## 2026-08-16 Pass 240: Published Metric Contract Validator

Added `external_metric_output.validate_metric_rows` at the serialization
boundary. It rejects unknown targets, non-qualified promotion statuses,
non-finite predictions, missing period/semantics/provenance fields, and
three-state values outside `0`, `1`, or `2`. Rebuilt the live-style forecast
surface successfully: 8,299 rows across five targets, all marked
`qualified_three_state_proxy`, with only the permitted state values present.
Focused verification passed with **8 tests passed**.

## 2026-08-16 Pass 241: Feature-Store Contract Revalidation

The metric feature store now reapplies the published-row contract when it is
called directly with reloaded or hand-built rows. Invalid three-state values
are rejected before pivoting into regression features, while the existing
unqualified-target error remains explicit. Focused verification passed with
**9 tests passed**.

## 2026-08-16 Pass 242: CMS Formulary and Network Candidate Screen

An independent public-source search found CMS monthly Part D formulary and
pharmacy-network PUF files containing NDC formulary, service-area county,
pharmacy-NPI, cost-share, and dispensing-fee fields. The candidate is not
added: CMS publishes terms restricting derivative works and competitive
offerings, and the files describe insurance access/network structure rather
than observed dispensing or inventory. The candidate and exclusion rationale
are recorded in `METRIC_RESEARCH_MATRIX.md`; no restricted files were
downloaded or used in training.

## 2026-08-16 Pass 243: FDA FAERS Safety-Signal Screen

Screened the FDA FAERS/AEMS quarterly public files and openFDA adverse-event
endpoint as a possible drug-level external metric. The source is public and
longitudinal, but FDA documents quarterly noncumulative releases, while
openFDA warns of three-month-or-more lag. Reports are voluntary and strongly
dependent on product exposure, with no Arkansas pharmacy geography or direct
dispensing/stock label. A raw adverse-event count would therefore fail the
project's pharmacy-relevance and target-semantics requirements; no FAERS
metric was added or downloaded.

## 2026-08-16 Pass 244: Modular Checkpoint Input Contract

Audited the 38-field history/context surface used by the multimodal
Transformer, typed graph, temporal, interaction, and deep-connection model.
Local pharmacy history fields are now explicitly recognized as future
operational inputs, while thirteen periodic or unnamed historical fields
remain training-only. Modular checkpoints and evaluation metadata now persist
the complete `input_contract` and set `forecast_mode` to
`research_training_only` when those fields are present. This aligns the
300M-class architecture with the operational-input audit without deleting
useful research context.

## 2026-08-16 Pass 246: Programmatic Modular Training Guard

Moved the operational-only check into `train_real_model` itself and added the
serialized `TrainConfig.operational_only` field. The CLI flag remains an
entry-point convenience, but library callers can no longer bypass the same
fail-closed contract. Regression coverage verifies that programmatic training
also rejects the current periodic-feature surface.

## 2026-08-16 Pass 245: Modular Operational-Only Training Guard

Added `require_operational_modular` and a standalone training CLI
`--operational-only` flag. The flag fails closed while the modular 38-field
surface contains periodic or historical-only context, preventing users from
mistaking a research checkpoint for a real-time deployment artifact. The
default research path remains available and records its explicit mode in
checkpoint/evaluation metadata.

## 2026-08-16 Pass 247: Local-History Audit Precision

Refined input-contract accounting for the modular 38-field surface. The
five-category name audit still records thirteen fail-closed dispositions, but
ten are now explicitly identified as local pharmacy history and only three
are genuinely non-operational periodic fields: unemployment, GSCPI, and
tariff. Added `local_history_count` and `non_operational_feature_count` to the
machine-readable summary and regression coverage for both counts.

## 2026-08-16 Pass 249: Regenerated Modular Research Artifacts

Regenerated the CPU-sized modular checkpoint and six-fold rolling evaluation
with the current input-contract code. The fixed run contains 32,531 training,
3,373 validation, and 2,597 test rows; test WAPE is `0.1176`. The six-fold
rolling mean improvement over the strongest naive baseline is `0.59%`, so
`publishable_rolling_candidate` remains false. Both artifacts now persist
`forecast_mode: research_training_only` and the three non-operational periodic
fields. The large production checkpoint was not overwritten.

## 2026-08-16 Pass 250: CLI Guard Runtime Verification

Invoked the standalone modular trainer with `--operational-only`. It exited
nonzero before training and reported that the feature surface contains
periodic or historical-only inputs. This verifies the guard at the real CLI
boundary, not only through a unit-level helper; no checkpoint was written by
the rejected run.

## 2026-08-16 Pass 251: Metric Audit Provenance Schema

Extended every metric-library candidate record with source URL, license/access
note, target semantics, feature timestamp boundary, geography scope, supplier
resolution, and missingness/censoring treatment. Regenerated
`metric_library_audit.json`; the five qualified metrics and all three Part 1
gates remain unchanged. Added regression coverage requiring these fields for
every candidate, including rejected diagnostics.

## 2026-08-16 Pass 252: Qualified Feature-Store Integration

Reloaded `qualified_metric_forecasts.csv.gz` through the public metric-row
validator and feature-store builder rather than trusting the in-memory
training path. All 8,299 rows across five qualified targets were retained,
with five generated metric features and no invalid state values or unqualified
rows. Focused contract, audit, output, and feature-store verification passed
with **46 tests passed**. This confirms the current qualified metrics can be
consumed by downstream pharmacy regression code through the documented
feature-store boundary.

## 2026-08-16 Pass 253: Full Regression Verification

Ran the complete test suite after the metric-output, feature-store, input
contract, provenance, and artifact changes. The suite completed with **255
tests passed** in 7m44s and no failures. Fourteen warnings remain from the
existing Transformer nested-tensor configuration, pandas fragmentation and
mixed-type CSV fixtures; none changed the exit status or invalidated the
published metric checks.

## 2026-08-16 Pass 254: Corrected Public-Context Leakage Boundary

Audited the publishable dataset's prior-quarter context join and found that
`_context_columns` used the source quarter's key directly, despite the
documentation claiming a strictly prior feature. Changed the join key to
source quarter `t` plus one quarter, rebuilt the suite, and preserved all
target row counts while changing the dataset hashes. Added a regression test
for the temporal boundary. The corrected four-fold demand ablation has
`64,116` held-out rows: history/shortage baseline WAPE is `0.2588`, while the
21-feature external-context model is `0.3819`; within-5%-error coverage falls
from `0.1402` to `0.0805`. The context family is rejected for promotion and
the prior same-quarter result must not be cited. Focused verification passed
with **8 tests passed**.

## 2026-08-16 Pass 255: Post-Correction Full Regression Verification

Ran the complete repository suite after rebuilding the publishable dataset
with the corrected prior-quarter context boundary. The suite completed with
**256 tests passed** in 7m44s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas test fixtures;
none affect the leakage test, regenerated hashes, or utility rejection.

## 2026-08-16 Pass 256: Dataset Version Integrity

Bumped the regenerated leakage-safe evaluation suite from `2026-08-16.v4`
to `2026-08-16.v5` so the changed temporal alignment and hashes cannot be
confused with the superseded artifact. Rebuilt the manifest and verified the
recorded version is `2026-08-16.v5`.

## 2026-08-16 Pass 257: ARCOS Distribution-State Metric

Added a separate state representation for the previously rejected numeric
ARCOS target. The evaluator learns low/mid/high log-gram tertiles only from
each rolling fit period, reserves four validation and four test quarters, and
selects between a one-vs-rest logistic head and persistence using validation
balanced accuracy. The resulting artifact has 15 rolling folds, 11,503
held-out rows, 39 controlled-substance codes, 85 Arkansas ZIP3s, all three
states observed, 93.51% exact accuracy, and 93.36% balanced accuracy. The
policy selects persistence in every fold, so the metric is qualified as an
external distribution-context proxy but not as learned incremental skill.
The numeric ARCOS metric remains rejected because it fails the numeric
within-5%-error requirement. The metric audit now reports six qualified
proxies, and focused ARCOS/audit verification passed with **12 tests passed**.

## 2026-08-16 Pass 258: Qualified Forecast Surface Integration

Added the ARCOS distribution-state adapter to the live-style filterable
forecast surface. The rebuilt artifact contains **8,478 rows across six
qualified targets**, including 179 ZIP3/drug quarterly ARCOS state rows; all
rows pass the serializer and feature-store contracts, producing six generated
features with no row loss. Focused forecast-output, feature-store, ARCOS, and
audit verification passed with **18 tests passed** plus the direct 8,478-row
round-trip check.

## 2026-08-16 Pass 259: Full Regression Verification After ARCOS Integration

Ran the complete repository suite after adding the ARCOS state target metadata,
rolling evaluator, live-style adapter, and filterable feature-store rows. The
suite completed with **257 tests passed** in 7m45s and no failures. Fourteen
existing warnings remain from Transformer nested-tensor configuration and
pandas test fixtures; none affect the new state metric or output contract.

## 2026-08-16 Pass 260: Official Pharmacy-Activity Source Screen

Screened current official CDC and CMS public pharmacy-activity sources for a
new target. CDC documents outpatient antibiotic dispensing by state, age,
sex, and antibiotic class annually, while its opioid, buprenorphine, and
naloxone dispensing maps provide annual state/county estimates for only three
broad classes. These sources are relevant context but do not provide a
weekly/monthly target and their 2011-2024 annual span is below the required 25
chronological samples per target. No metric or training data was added; the
candidate and rejection criteria are recorded in `METRIC_RESEARCH_MATRIX.md`
and `TARGET_AVAILABILITY.md`.

## 2026-08-16 Pass 261: Operational NWS Hourly Forecast Adapter

Extended `live_inputs.py` beyond active alerts with a free National Weather
Service point-to-grid hourly forecast adapter. It resolves
`/points/{latitude},{longitude}` to the official `forecastHourly` endpoint and
returns numeric temperature, precipitation probability, humidity, wind speed,
validity intervals, and the forecast issuance timestamp. Retrieval, endpoint,
coordinate, freshness, and validity metadata are preserved. The adapter is
explicitly context-only and is not inserted as a qualified target because no
timestamped historical weather archive is being used to claim pharmacy-level
predictive accuracy. Focused live-input and contract verification passed with
**37 tests passed**.

## 2026-08-16 Pass 262: NWS Weather Feature Boundary

Added `build_nws_daily_weather_features` as the handoff from hourly NWS
forecasts to model features. It preserves point/date grain, aggregates only
available periods, emits temperature, precipitation probability, humidity,
wind, and observed-hour count features, and does not impute missing hours.
This prevents a partial live forecast from being silently treated as a full
daily observation. Focused live-input and contract verification passed with
**38 tests passed**.

## 2026-08-16 Pass 263: Full Verification After Weather Boundary

Ran the complete repository suite after adding the NWS hourly forecast
adapter and deterministic daily weather feature boundary. The suite completed
with **259 tests passed** in 7m47s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas test fixtures;
none affect the operational weather contract or qualified metric artifacts.

## 2026-08-16 Pass 264: Weather CLI Refresh Path

Added an explicit `refresh-weather` CLI command requiring latitude and
longitude. It writes the raw hourly response, the deterministic daily feature
view, and a provenance metadata artifact without touching historical training
data. The CLI test also exposed a pre-existing parser bug where subparser
`--root` defaults overwrote the parent-level root; using a suppressed default
fixed isolated and scripted invocations. Focused CLI, live-input, and contract
verification passed with **39 tests passed**, and `--help` exposes the new
command.

## 2026-08-16 Pass 265: Full Verification After Weather CLI Integration

Ran the complete repository suite after adding the opt-in weather refresh
command and correcting parent/subparser root propagation. The suite completed
with **260 tests passed** in 7m47s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas test fixtures;
none affect CLI isolation, weather provenance, or historical evaluation.

## 2026-08-16 Pass 266: Live NWS Smoke Test

Ran the new refresh command against the real NWS point/grid service for
Little Rock, Arkansas, with outputs directed to `/tmp` rather than the
repository. The endpoint returned 156 hourly forecast periods and the feature
boundary produced seven daily rows. This verifies live endpoint availability,
point resolution, parsing, and aggregation without modifying historical or
qualified evaluation artifacts.

## 2026-08-16 Pass 267: County-Safe Weather Geography Contract

Audited local geography artifacts before connecting live weather to the
county-filtered forecast surface. The repository has an NPPES/Census
city-to-county crosswalk, but no authoritative county centroid or county
weather-point registry. The NWS adapter now accepts an optional,
caller-supplied Arkansas county FIPS tag, carries it through hourly and daily
features, and rejects non-Arkansas FIPS values. Untagged forecasts remain
explicitly point-level; no county-wide weather values are inferred or
broadcast.

## 2026-08-16 Pass 268: Full Verification After County Weather Contract

Ran the complete repository suite after adding explicit county tagging to the
NWS live-input path. The suite completed with **262 tests passed** in 7m45s
with no failures. Fourteen existing warnings remain from Transformer nested
tensor configuration, pandas fragmentation, and mixed test-fixture dtypes.
No historical training or qualified evaluation artifact was modified.

## 2026-08-16 Pass 269: Census County Weather Registry

Added a reproducible registry built from the U.S. Census Bureau 2025 county
Gazetteer. It contains all 75 Arkansas county GEOIDs, names, representative
internal-point coordinates, region labels, source URL, vintage, and explicit
transformation semantics. The local artifact passed uniqueness and coordinate
completeness checks: 75 counties, 75 unique FIPS values, and no missing
coordinates. Internal points are query locations, not claims of county-wide
weather homogeneity.

## 2026-08-16 Pass 270: Arkansas County NWS Refresh Smoke Test

Ran `refresh-county-weather` against the live NWS points and hourly forecast
services using the 75-row Census registry. The refresh completed with 11,700
hourly rows and 525 daily rows, preserving county keys for the full returned
horizon. Outputs are stored as live context artifacts with provenance; no
historical training or evaluation data was changed.

## 2026-08-16 Pass 271: Full Verification After County Weather Integration

Ran the complete repository suite after adding the Census county registry,
batch NWS refresh path, CLI dispatch, and geography tests. The suite completed
with **265 tests passed** in 7m47s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas fixtures. The
new live artifacts are not used to claim predictive accuracy because their
temporal grain has not yet been joined to a historical observed-weather target
panel.

## 2026-08-16 Pass 272: Historical County Weather Panel

Reused the local NOAA/NCEI daily-summary observations rather than downloading a
second weather source. Added a loader that deduplicates station/date records,
normalizes temperature/precipitation/snow fields, and a county aggregator that
assigns each Census county point to its nearest Arkansas station. The resulting
artifact contains 126,611 observed county-period rows across 75 counties, 8
stations, weekly/monthly cadences, and 2000–2026 coverage. Station distance and
observed-day coverage are retained; no interpolation or missing-period
imputation is performed.

## 2026-08-16 Pass 273: Exact Weather Feature Join

Added `attach_weather_context` as the leak-safe integration seam for later
weekly/monthly target layers. It joins only on cadence, period start/end, and
county FIPS; duplicate weather grains fail, unmatched counties retain missing
values plus an availability flag, and no geographic broadcasting occurs. The
focused weather/geography/CLI suite passed with **15 tests passed**.

## 2026-08-16 Pass 274: Full Verification After Historical Weather Integration

Ran the complete repository suite after adding the historical weather loader,
county weekly/monthly aggregation, exact feature attachment seam, CLI command,
and tests. The suite completed with **268 tests passed** in 7m47s and no
failures. Fourteen existing warnings remain from Transformer nested-tensor
configuration and pandas fixtures.

## 2026-08-16 Pass 275: Historical Weather Utility Ablation

Added a CLI-reproducible rolling-origin ablation joining current-week county
weather context to next-week Arkansas FluView WILI. Across seven folds and 342
held-out weeks, the history/seasonality baseline reached 75.48% mean balanced
accuracy; adding six weather features reached 72.78% (delta -2.70 percentage
points). Weather is rejected as an incremental feature for this proxy, while
the historical and live weather layers remain available for evaluation against
other targets.

## 2026-08-16 Pass 276: Full Verification After Weather Utility Evaluation

Ran the complete repository suite after adding the weather ablation, CLI
command, schema normalization, and region-specific FluView alignment. The
suite completed with **269 tests passed** in 7m47s and no failures. Fourteen
existing warnings remain from Transformer nested-tensor configuration and
pandas fixtures.

## 2026-08-16 Pass 277: Hospital Weather Utility Ablation

Evaluated current-week county weather against next-week Arkansas hospital
influenza admission-pressure states using the qualified hospital proxy's
rolling-origin folds. Across four folds and 187 held-out weeks, baseline mean
balanced accuracy was 73.85%; weather augmentation was 62.99%, a -10.86
percentage-point change, with every fold worse. Weather is therefore rejected
as incremental utility for both tested weekly public health proxies, while its
operational context artifacts remain available for future target families.

## 2026-08-16 Pass 278: Full Verification After Hospital Weather Ablation

Ran the complete repository suite after adding hospital-week weather joining,
the hospital utility evaluator, CLI dispatch, and tests. The suite completed
with **270 tests passed** in 7m44s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas fixtures.

## 2026-08-16 Pass 304: Integrate Qualified Metric Context Boundary

Audited the relationship between the qualified forecast feature store and the
300M-class modular architecture. The store was previously a downstream join
artifact only; it now has an explicit optional input at the temporal layer,
with seven qualified metric values followed by seven missingness indicators.
The value/missingness ordering is generated by `metric_context_vector` from
the persisted feature metadata, preventing caller-dependent feature
permutation.

The model validates batch shape and feature width, preserves an all-missing
zero context when no metric row is available, and exposes the projected
context as an inspectable intermediate state. The current public historical
training path still supplies all-missing vectors because the qualified public
forecast rows do not provide aligned historical publication vintages. This is
an architecture integration and leakage-control improvement, not evidence of
learned metric utility or improved pharmacy inventory accuracy.

Focused model, training, and feature-store validation passed with **30 tests
passed**. The whole-system evaluator passed artifact/hash/grain checks and
continues to leave the incompatible cross-target composite score undefined.

## 2026-08-16 Pass 295: Supplier-Preserving Feature Store Grain

Audited the downstream feature-store integration and found that its default
keys omitted `supplier`, even though the universal forecast contract and
qualified recall rows expose supplier-specific observations. Added `supplier`
to the default grain and regression coverage proving two supplier rows remain
distinct. The actual 11,686-row qualified forecast artifact now pivots without
collapse into seven features at the seven-key grain. OpenCode 1.18.18 was
available, but the requested DeepSeek provider returned a server error; no
OpenCode edits were used.

The complete suite then passed with **282 tests passed** in 8m15.62s and 18
warnings. The actual qualified forecast artifact validated at 11,686 rows,
including 3,208 supplier-specific rows, with no supplier-grain collapse.

## 2026-08-16 Pass 296: Reproducible Metric Feature Artifact

Added `model/scripts/build_metric_feature_store.py`, which validates the
qualified forecast surface, writes a grain-preserving compressed feature
store, and records source/output SHA-256 hashes plus feature metadata. The
real run produced 11,686 output rows and seven features at the seven-key
grain. Focused verification passed with **6 tests**; the complete suite then
passed with **283 tests passed** in 8m17.69s and 18 warnings.

## 2026-08-16 Pass 297: Whole-System Feature Integration Evidence

Extended `evaluate_whole_system.py` to validate and report the qualified
metric feature store as a separate component. The canonical report preserves
`composite_score: null`, records source/output hashes, and verifies 11,686
rows, seven features, and the seven-key supplier-preserving grain. This proves
artifact integration, not pharmacy-level utility or a composite accuracy
claim.

The complete suite after this integration change passed with **283 tests
passed** in 8m19.42s and 18 warnings. The canonical whole-system command
completed successfully against the public-suite artifacts and the current
qualified forecast surface.
The subsequent full repository verification passed with **283 tests passed**
in 8m18.35s and 18 warnings.

## 2026-08-16 Pass 298: Explicit Context Projection Contract

Added metadata-declared context projection policies for qualified targets.
Exact-grain joins remain the default; opt-in context mode can project national
drug signals by period+drug, Arkansas-wide signals by period, regional signals
by period+region+drug, and supplier signals only when the supplier key matches.
Missing local dimensions become explicit missingness, and duplicate projected
grains fail closed. The real 11,686-row artifact projected successfully for
four compatible features while skipping unsupported region/supplier/ZIP3
dimensions. Focused verification passed with **8 tests**.

The complete suite passed with **285 tests passed** in 8m15.76s and 18
warnings.

## 2026-08-16 Pass 286: Full Verification After Medicaid Candidate Screening

Ran the complete repository suite after adding the Medicaid demand evaluator,
rejection artifact, audit entry, tests, and documentation. The suite completed
with **276 tests passed** in 7m50s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration, pandas fragmentation, and
mixed-type fixture loading.

## 2026-08-16 Pass 289: Relation-Aware Graph Layer

Reworked `TypedGraphReasoner` so relation IDs now gate directed messages from
source entities to receiver entities before each graph Transformer block.
Padding entity/relation IDs are masked from message aggregation, attention,
and pooled graph states. This fixes the prior behavior where relation
embeddings were averaged independently of neighbor entity states and padded
nodes could influence the graph representation.

Added tests for typed-edge sensitivity, padding invariance, and finite output
under all-padding graph inputs. The architecture documentation now records the
six code-level stages and their contracts. The full 300M-class parameter budget
and downstream output shapes remain unchanged by the interface.

## 2026-08-16 Pass 290: Full Verification After Relation-Aware Graph Change

Ran the complete repository suite after the graph message-passing change and
new graph invariance tests. The suite completed with **278 tests passed** in
7m46s and no failures. Eighteen warnings remain, consisting of the existing
Transformer/pandas warnings plus the additional graph-model warning locations.

## 2026-08-16 Pass 291: Full Verification After Layer Inventory Metadata

Ran the complete repository suite after adding disjoint per-layer parameter
counts and serialized layer contracts to `model_inventory`, plus the bias-free
no-edge graph-message correction. The suite completed with **278 tests passed**
in 7m46s and no failures. Eighteen existing warnings remain.

## 2026-08-16 Pass 292: Target-Specific State Head

Added `target_state_risk` with shape `[batch, horizon, target, state]` to the
prediction heads. The state count defaults to three, matching the metric
contract, while the pre-existing aggregate `state_risk` output remains for
backward compatibility. This removes the architectural ambiguity where seven
targets previously shared one state distribution per horizon.

The complete verification suite then passed with **278 tests passed** in
7m45.97s and 18 warnings. The head is not counted as a trained metric until
state labels and a state-specific loss are connected to the training path.

## 2026-08-16 Pass 293: Leakage-Safe State Training Contract

Connected the target-specific state head to the real quarterly training path.
Training-slice log1p tertiles are applied unchanged to later splits, a weighted
cross-entropy term is recorded in the checkpoint, and held-out exact and
balanced state metrics are persisted. Constant-target slices fail closed with
an unavailable sentinel rather than fabricated states. Focused verification
passed with **18 tests passed**; the full suite is pending after this change.

The complete suite subsequently passed with **280 tests passed** in 7m55.10s
and 18 warnings. This validates the state training contract and its
degenerate-slice handling, but does not promote the quarterly demand state to
a weekly/monthly or pharmacy-level metric.

The real public-panel small-model run (one CPU epoch, frozen news, 32,531
training rows, 3,373 validation rows, 2,597 test rows) produced valid fitted
thresholds, but test state performance was only **46.86% exact** and **46.02%
balanced**, below the 65% state floor. The result is retained as a failed
research measurement, not promoted. The artifact now also records a matched
persistence-state comparator.

The matched persistence comparator scored **91.99% exact / 91.76% balanced**
on the same test rows, confirming that the one-epoch neural state head does
not add value and must not replace the stronger temporal baseline.

After adding the comparator and current CLI output, the complete suite passed
with **280 tests passed** in 7m50.61s and 18 warnings. The state artifact is
therefore reproducible, but remains a rejected quarterly neural metric.

## 2026-08-16 Pass 294: Demand-Change State Semantics

Changed the default modular state target from absolute demand level to the
inventory-relevant next-period log-demand change. Absolute level remains an
explicit ablation. A five-epoch real public-panel run produced classifier
test accuracy of **42.78% exact / 40.01% balanced**; discretizing the numeric
point forecast produced **39.55% / 35.85%**; neutral no-change persistence
produced **35.66% / 33.33%**. The change target shows modest real lift but fails
the 65% gate and is rejected. Focused verification passed with **15 tests**;
full-suite verification is pending after this change.

The complete suite subsequently passed with **281 tests passed** in 8m15.16s
and 18 warnings. Numeric-derived state scoring is now covered alongside the
classifier and persistence comparisons.

## 2026-08-16 Pass 281: Supplier-Filtered Recall Pressure

Added a termination-aware FDA enforcement panel at monthly supplier x NDC
grain. Recall states are `0` no active recall, `1` active Class II/III, and
`2` active Class I. The source loader preserves the FDA recalling firm, and the
shared output contract now carries a dedicated `supplier` filter key.

The real archive evaluation produced 156 rolling-origin folds and 808,937
held-out rows across 925 suppliers, 3,025 NDCs, and 3,200 supplier-NDC pairs.
Mean exact accuracy was 99.9888% and balanced accuracy was 99.566%. The model
selected persistence on every fold and its mean balanced-accuracy delta versus
persistence was 0.0 percentage points. It is therefore qualified as a
supplier-filterable upstream recall proxy, not as evidence of learned
pharmacy-inventory improvement.

## 2026-08-16 Pass 283: GSCPI State Candidate Rejected

Evaluated the locally downloaded New York Fed Global Supply Chain Pressure
Index as a monthly upstream pharmacy-supply context state. The stable state
definition used the standardized index bands below 0, 0 through 1, and above
1, avoiding fold-specific tertile thresholds. The evaluator used chronological
next-month prediction with 91 rolling folds and 272 held-out months.

Raw exact accuracy was 85.35%, but balanced accuracy was only 32.17%; the
series spends long periods in the low state, so raw accuracy is misleading.
The candidate is rejected under the balanced-state contract and is not added
to the qualified forecast surface. The source remains documented as a freely
available operational input for future multi-signal heads.

## 2026-08-16 Pass 285: Arkansas Medicaid Demand Candidate Rejected

Evaluated the locally available Arkansas Medicaid State Drug Utilization Data
FFS series as a direct statewide prescription-demand target. The numeric
next-quarter evaluator used 21 chronological rolling folds and 83 held-out
quarters. Selected forecasts were within 5% on 46.83% of rows; learned Ridge
was within 5% on 40.08%, while persistence reached 48.02%.

The target is rejected because it fails the numeric accuracy gate and the
learned model does not improve persistence. The result also does not represent
all-payer Arkansas pharmacy demand.

## 2026-08-16 Pass 287: National Shortage Breadth Candidate Rejected

Evaluated a monthly FDA market-breadth state counting distinct NDCs with an
active shortage. The rolling evaluator produced 146 folds and 437 held-out
months. Persistence matched every observed next-month state for 100% exact
accuracy, but balanced accuracy was only 33.33% because the archive's broad
state regimes persist across short test windows.

The candidate is rejected under the balanced-state contract. It is not added
to the forecast surface and does not claim Arkansas allocation.

## 2026-08-16 Pass 288: Full Verification After Shortage-Breadth Screening

Ran the complete repository suite after adding the shortage-breadth evaluator,
rejection artifact, audit entry, tests, and documentation. The suite completed
with **277 tests passed** in 7m51s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration, pandas fragmentation, and
mixed-type fixture loading.

## 2026-08-16 Pass 284: Full Verification After GSCPI Screening

Ran the complete repository suite after adding the GSCPI evaluator, rejection
artifact, tests, and research documentation. The suite completed with **274
tests passed** in 7m49s and no failures. Fourteen existing warnings remain
from Transformer nested-tensor configuration, pandas fragmentation, and
mixed-type fixture loading.

## 2026-08-16 Pass 282: Full Verification After Supplier Output Contract

Ran the complete repository suite after adding the supplier-filterable recall
forecast rows and dedicated `supplier` output column. The suite completed with
**272 tests passed** in 7m45s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration, pandas fragmentation, and
mixed-type fixture loading.

## 2026-08-16 Pass 279: FEMA Disaster Utility Ablation

Screened the local FEMA OpenFEMA county declaration feature against next-week
Arkansas hospital influenza admission-pressure states. Current-week
declarations were aggregated across counties with structural zeros for weeks
without declarations. Across four folds and 187 held-out weeks, baseline mean
balanced accuracy was 73.85%; the disaster-augmented model reached 72.21%, a
-1.64 percentage-point change. It is rejected as incremental utility but
retained as a sparse, operationally available disruption context.

## 2026-08-16 Pass 280: Full Verification After FEMA Utility Ablation

Ran the complete repository suite after adding FEMA declaration aggregation,
hospital-disaster alignment, CLI dispatch, and tests. The suite completed with
**271 tests passed** in 7m46s and no failures. Fourteen existing warnings
remain from Transformer nested-tensor configuration and pandas fixtures.

## 2026-08-16 Pass 305: Historical Metric Context Utility Ablation

Enabled the dated historical context adapter for a controlled small-model
ablation. The comparison used the same seed, architecture, one epoch, frozen
news encoder, batch size, chronological split, and persistence/transition
baselines. The adapter admitted exact-NDC FDA shortage archive and FDA recall
event context only when available at the quarterly origin; all other qualified
metrics and unmatched rows remained explicitly missing.

The disabled-context held-out blended WAPE was **0.103224** and exact
three-state accuracy was **40.47%**. The enabled-context result was WAPE
**0.105136** and exact state accuracy **38.89%**. The context therefore
degraded this held-out run and is rejected as a default feature. It remains an
opt-in research ablation because coverage is sparse (1,826 of 44,071 rows),
and this single run does not prove that the signals are universally harmful.
No qualified metric is promoted on the basis of this integration.

## 2026-08-16 Pass 306: Public Local-Dispensing Availability Audit

Conducted an independent source audit for a new detailed Arkansas pharmacy
target. The official ONC/Surescripts county e-prescribing dataset is public
and pharmacy-connected, but its stated date range is December 2008 through
April 2014; it has no NDC-level target and cannot serve as a real-time input
today. Arkansas APCD documentation describes monthly pharmacy claim counts,
but only for data requesters. The Arkansas PDMP contains substantially more
relevant weekly controlled-substance dispensing records, yet access is
authorized rather than unrestricted public access.

These sources are therefore documented as exclusions, not silently used as
training data. The audit reinforces the current evidence boundary: no freely
available, current, Arkansas county-by-drug-by-supplier pharmacy dispensing
label has been found. A future approved APCD/PDMP partnership would be a
material data change and should trigger a new target audit.
## 2026-08-16 Pass 307: Five-State and Numeric Output Contract

The project output requirement was revised from a minimum of three categorical
states to a minimum of five. Numeric outputs are now explicitly preferred when
the source exposes a meaningful quantity. The metric audit therefore rejects
categorical candidates with fewer than five observed states in scored rows;
historical three-state results remain available as legacy evidence but are not
newly promotable under the revised contract.

The serialized metric validator now accepts the five-state code range `0`-
`4` and rejects values outside it. Existing three-state forecast artifacts were
not relabeled as five-state results. They require conversion to numeric targets
or retraining/rebinning with five independently observed states before they can
again satisfy the final metric contract. No accuracy or promotion claim was
changed by this contract revision.

## 2026-08-16 Pass 314: Numeric Replacement Screen and Full Verification

The remaining operational state surfaces were screened for quantifiable
replacements using chronological rolling evaluation. The numeric candidates
for Arkansas FluView respiratory pressure, Arkansas hospital influenza
admissions, annual regional claims demand, and ARCOS regional distribution
failed the required within-five-percent accuracy floor: **11.07%** across 7
folds, **9.32%** across 4 folds, **20.69%** across 6 folds, and **26.82%**
across 15 folds, respectively. They remain rejected numeric candidates; the
existing state representations remain compatibility outputs pending stronger
replacement targets.

The FDA-derived numeric candidates remain the only newly qualified surfaces:
NDC monthly recall severity, supplier-by-NDC monthly recall severity, and
NDC monthly active shortage supplier count. The shortage-count evaluation used
121 chronological folds and 1,550,662 held-out rows, with **99.03%** within
five-percent accuracy; the recall evaluations used 10 folds with **99.98%**
and **99.99%** within-five-percent accuracy, respectively. These scores are
strong upstream proxy results and are not pharmacy inventory ground truth.

The complete repository verification finished with **303 passed, 22
warnings** in 9m05s. The metric audit reports three qualified numeric metrics
plus one qualified five-state regional proxy, for four qualified metrics total.
The weekly/monthly gate remains true, while the required 100-drug 75% gate and
the separate per-region 75% gate remain false. No unsupported accuracy claim
or goal-completion claim is made.

## 2026-08-16 Pass 315: Promote Five-State Regional Demand Proxy

The annual Arkansas region-by-drug demand evaluator now supports a minimum of
five fit-only quantile states. On the local CMS Part D panel, the five-state
regional target produced 6 chronological folds, 21,304 held-out rows, 1,308
drugs, all 5 Arkansas regions, **84.18% exact accuracy**, and **84.46% balanced
accuracy**. All five states occurred in scored rows, so this target satisfies
the revised state contract and the 65% floor.

The target was added to the filterable output metadata and operational
forecast surface as `arkansas_region_annual_demand_five_state`; the older
three-state target remains compatibility-only. Five-state hospital respiratory
screening was also run: influenza reached 59.09% exact / 47.79% balanced,
COVID had severe state imbalance, and RSV had only two eligible folds. None
was promoted. Focused verification passed with **20 tests**. The rebuilt
forecast artifact contains 3,680 rows for the new target. Final full
verification completed with **305 passed, 22 warnings** in 8m23s.

## 2026-08-16 Pass 316: Establish Per-Drug Numeric Coverage Gate

The numeric rolling evaluators now retain per-NDC held-out error counts rather
than reporting only aggregate accuracy. Using the same chronological folds and
a minimum of 25 held-out observations per NDC, the FDA shortage-supplier-count
metric has 2,425 NDCs at or above 75% within-five-percent accuracy. NDC recall
severity has 2,395 such NDCs, and supplier-by-NDC recall severity has 2,398.
The aggregate within-five-percent rates are 99.18%, 99.98%, and 99.99%,
respectively.

The audit now reports the per-drug 100-plus-75% gate as **true**. This is still
external proxy evidence, not pharmacy inventory truth, and per-Arkansas-region
75% evidence remains false. Focused audit and evaluator tests passed with
**15 tests**, and the complete suite finished with **305 passed, 22 warnings**
in 8m24s.

## 2026-08-16 Pass 317: Establish Per-Region Accuracy Gate

The five-state annual regional demand artifact was checked at the individual
Arkansas-region level rather than relying on its aggregate score. Held-out
exact accuracy was **84.15%** for Central, **84.08%** for Northeast, **84.77%**
for Northwest, **84.51%** for Southeast, and **83.18%** for Southwest. Each
region therefore clears the 75% threshold with the same six chronological
folds and 21,304 held-out region-drug rows.

The metric audit now reports all three Part 1 gates as true: a weekly/monthly
metric, a 100-plus-drug metric, and a per-Arkansas-region metric. This does not
claim pharmacy inventory truth; the regional target remains an annual CMS
Medicare Part D utilization proxy. Focused audit/regional tests passed with
**10 tests**. The complete suite then finished with **305 passed, 22 warnings**
in 8m23s.

## 2026-08-16 Pass 318: Screen CDC Wastewater Output Candidates

The local CDC Arkansas wastewater panel was screened as a new weekly output
family rather than only as an input covariate. The evaluator aggregates site
values by pathogen and uses five fit-only quantile states with chronological
rolling folds. Influenza A produced 7 folds and 91 held-out rows at 81.32%
exact but only 32.97% balanced accuracy; RSV produced 8 folds and 104 rows at
71.15% exact / 22.17% balanced; SARS-CoV-2 produced 11 folds and 143 rows at
46.15% exact / 26.46% balanced. Influenza and RSV also lacked some states in
scored rows.

All three wastewater outputs are rejected under the five-state balanced
contract. The evaluator, reproducible script, rejection audit entries, and
tests were added; no new metric was promoted. The complete suite finished with
**307 passed, 22 warnings** in 8m24s.

## 2026-08-16 Pass 319: Promote Current NADAC Acquisition-Cost Proxy

Re-evaluated CMS NADAC using the newer local Arkansas-exposed panel covering
January 2021 through December 2025, replacing the earlier one-fold artifact.
The strict exact-seven-day transition evaluator produced 3 chronological
folds, 1,113,045 held-out transitions, and 8,580 NDCs. The validation-selected
forecast achieved **90.08% within-five-percent accuracy** with mean WAPE
0.00368. Persistence was better than the ridge model in all folds and was
therefore retained transparently as the selected forecast method.

NADAC is now a qualified weekly numeric acquisition-cost proxy and is emitted
in the filterable operational forecast surface. It is relevant to purchasing
and inventory cost decisions but is not local stock, demand, shortage, or
supplier allocation evidence. Focused audit, output, and serialization tests
passed. The complete suite then finished with **307 passed, 22 warnings** in
8m26s.

## 2026-08-16 Pass 320: Remove Legacy Three-State Rows from Operational Surface

The promoted forecast builder no longer emits the historical three-state
compatibility rows for FluView, hospital admissions, regional demand, or ARCOS.
Those evaluators and artifacts remain available for historical comparison, but
the operational CSV now contains only the five-state regional demand signal or
numeric qualified targets: shortage supplier count, NADAC acquisition cost,
NDC recall severity, supplier-by-NDC recall severity, and the regional demand
state.

The rebuilt artifact contains **18,012 rows across 5 targets**, and the feature
store contains 5 metric columns. Focused output, audit, and feature-store tests
passed with **9 tests**. The complete suite must be rerun after this contract
cleanup; it finished with **307 passed, 22 warnings** in 8m33s.

## 2026-08-16 Pass 321: Align Historical Context with Five-Target Contract

The modular model defaults and historical training context were aligned with
the five active operational metrics. The metric vector is now ten columns:
five qualified values followed by five explicit missingness indicators. The
point-in-time adapter retains exact-NDC FDA shortage and recall joins and now
adds a dated CMS NADAC acquisition-cost join; the qualified regional demand
metric remains missing until a safe historical regional adapter exists. A new
test verifies that a post-origin NADAC observation cannot enter a feature row.
The focused context, modular-model, and training tests passed after updating
the old 14-column assertion. The complete suite then passed with **308 tests,
22 warnings** in 8m25s. This change is contract hygiene, not evidence of
additional predictive accuracy.

## 2026-08-16 Pass 322: Promote Five-State ARCOS Regional Proxy

The DEA ARCOS ZIP3-by-controlled-substance distribution evaluator was upgraded
from invalid tertiles to fit-only five-quantile states. On the real Arkansas
panel it produced 15 chronological folds, 11,503 held-out transitions, 39
controlled-substance codes, and 85 ZIP3 regions. Exact accuracy was **90.01%**
and balanced accuracy was **89.38%**; all five states occurred in scored rows.
The result is persistence-dominated and remains explicitly a distribution
proxy, not pharmacy inventory or supplier allocation evidence.

The metric audit now qualifies the ARCOS five-state target. Its filterable
forecast adapter and metadata were promoted, the operational forecast surface
was rebuilt to **18,191 rows across 6 targets**, and the feature store now has
six metric columns. Focused ARCOS, audit, output, serialization, and
feature-store tests passed with **33 tests**.

## 2026-08-16 Pass 323: Reject NADAC Relative-Change State Candidate

Screened a nonredundant weekly acquisition-cost movement signal from the same
CMS NADAC transitions. The strict evaluator used three chronological folds and
1,113,045 held-out transitions, with five quantile thresholds fit separately
inside each training partition. NADAC relative changes are strongly
zero-inflated: fit quantiles collapsed at zero and only two effective states
were observed. The persistence movement baseline reached 89.02% raw accuracy
but only 20.00% balanced accuracy, and the candidate was rejected under the
five-state contract. The rejection artifact, evaluator, and test preserve the
result as reproducible negative evidence; no additional operational target was
introduced.

## 2026-08-16 Pass 325: Align Six-Target Model Context Contract

After ARCOS promotion, the model defaults and historical metric context were
still configured for five operational targets. The contract now declares six
targets and a 12-wide value-then-missingness context vector. ARCOS is included
in the declared order but remains explicitly missing for NDC training rows:
the public ARCOS source uses ZIP3 and controlled-substance codes, with no safe
NDC crosswalk. This prevents an unverified identity join while keeping the
operational and model contracts synchronized. Historical-context, training,
and modular-model tests passed after the migration. The complete suite then
passed with **309 tests and 22 warnings** in 8m25s.

## 2026-08-16 Pass 326: Promote National Five-State FluView Proxy

The CDC FluView national aggregate was evaluated separately from the rejected
Arkansas statewide five-state signal. Seven chronological folds produced 342
held-out weeks, 84.08% exact accuracy, and 68.93% balanced accuracy, with all
five states represented. It is qualified as a weekly upstream respiratory
context proxy, not a pharmacy dispensing or Arkansas allocation target.

The filterable adapter, metadata, audit entry, and focused serialization test
were added. Rebuilt artifacts now contain **18,192 rows across 7 qualified
targets** and seven feature-store columns. The national signal is a single
aggregate row at the current source edge; it is not incorrectly broadcast to
counties, drugs, or suppliers. The complete suite then passed with **310 tests
and 22 warnings** in 8m26s.

## 2026-08-16 Pass 327: Reject FluView National-Arkansas Divergence

Screened a derived weekly divergence signal equal to national WILI minus
Arkansas WILI. The reproducible five-state evaluator retained only paired
weeks, used fit-only thresholds, and produced seven chronological folds with
341 held-out transitions. Persistence of the observed divergence achieved
54.94% exact accuracy and 43.36% balanced accuracy. The signal is rejected and
is not added to the operational surface; the qualified national aggregate
remains available without claiming county, drug, or supplier allocation.

## 2026-08-16 Pass 329: Add Event True-Positive Acceptance Route

The accuracy contract now preserves the five-state minimum and the original
75% raw-accuracy route while adding an alternative for event-oriented signals:
raw accuracy must be at least 65% and precision among predicted event cases
must be at least 80%. State event cases use explicitly declared high states;
the active five-state signals use states `3` and `4`. Numeric event cases use
declared thresholds and retain the five-percent numeric error tolerance.

`event_accuracy.score_event_predictions` implements the leakage-safe scoring
primitive and reports the event definition, predicted-event denominator,
true-positive count, precision, and actual-event count. Non-event status
signals such as NADAC price are explicitly marked not applicable. The county
demand evaluator now records the new score: 89.03% event precision across
57,054 predicted high-demand rows, so its 74.33% raw accuracy is accepted via
the alternative route. Geography, drug, supplier, and disease/symptom output
categories remain independent coverage gates.

## 2026-08-16 Pass 328: Promote Five-State County Demand Proxy

The CMS Part D county-by-drug demand artifact was evaluated with six
chronological next-year folds. The evaluator retained 139,197 non-overlapping
held-out rows covering 1,326 drugs and 73 Arkansas counties. A fit-only
five-quantile state head achieved 74.33% mean exact accuracy and 74.12%
balanced accuracy; 28 counties independently reached at least 75% exact
accuracy. The result clears the five-state and 65% publication gates, but it
does not provide supplier identity or private-pharmacy inventory truth.

The county-FIPS-by-drug adapter, audit metadata, universal target metadata,
tests, and documentation were added. The operational forecast surface now
emits this annual filterable proxy alongside the existing qualified metrics.

## 2026-08-16 Pass 330: Register Coverage-Gate Status

The metric audit now exposes independent geography, drug, supplier, and
disease/symptom coverage gates. Geography, drug, and supplier currently have
qualified filterable signals. Disease/symptom coverage remains **not passed**:
the qualified national FluView aggregate is an upstream respiratory proxy but
does not expose a disease/pathogen split, while the currently evaluated
pathogen-specific hospital targets do not yet provide a qualifying target.
This prevents the revised completion condition from being overstated.

## 2026-08-16 Pass 332: Screen CDC ARI Activity History

The CDC Level of Acute Respiratory Illness activity-by-state dataset was
screened as a possible symptom-divided weekly target. Its public Socrata API
currently returns one Arkansas observation, dated 2026-08-08, with a single
`Very Low` label. No leakage-safe chronological evaluation with 25 held-out
samples is possible, so it is rejected as a target and remains an unpromoted
live-input candidate. The existing pathogen-specific hospital negative
results remain the authoritative disease/symptom coverage evidence.

## 2026-08-16 Pass 331: Record Pathogen Event-Screen Results

The CDC NHSN hospital evaluator now records event precision for every
pathogen/state-count run using fit-only thresholds and held-out predictions.
The required five-state screen remains rejected: influenza achieved 59.09%
exact accuracy, 47.79% balanced accuracy, and 73.12% precision among
predicted high states; COVID and RSV fail state coverage/balanced-accuracy
requirements. The numeric influenza target reached only 9.32% within the
five-percent error margin. These results do not satisfy the disease/symptom
coverage gate, and no hospital target was promoted.

## 2026-08-16 Pass 333: Screen Wastewater Event Route

The wastewater evaluator now reports five-state high-zone event precision.
Arkansas wastewater remains unqualified because influenza lacks observed
states `0` and `1` in scored rows. A separate national site-mean screen retained
all five states but failed the revised event route: influenza achieved 58.58%
raw accuracy and 73.21% high-zone precision, RSV 56.41% and 51.28%, and
SARS-CoV-2 76.92% and 61.29%. No wastewater pathogen was promoted as a
disease/symptom output.

## 2026-08-16 Pass 334: Promote NSSP Influenza Disease Signal

The CDC NSSP statewide Arkansas rows were separated from the unusable county
and HSA rows and evaluated by pathogen. The influenza ED-visit percentage
produced 10 chronological folds and 130 held-out rows covering all five
states. It achieved 73.08% exact accuracy and 80.00% precision among
predicted high states (`3`/`4`), satisfying the revised event route of at
least 65% raw accuracy plus at least 80% event precision. COVID was rejected
at 55.56% event precision; RSV was rejected because raw accuracy was 63.85%.

The promoted target is filterable by `pathogen`, explicitly scoped to
Arkansas statewide NSSP ED surveillance, and remains a utilization proxy
rather than pharmacy dispensing or inventory truth. The operational surface
now contains nine qualified metrics and the disease/symptom coverage gate
passes through this pathogen-specific signal.

## 2026-08-16 Pass 335: Add NSSP To Historical Metric Context

The point-in-time modular training context now includes prior Arkansas NSSP
influenza ED pressure as its seventh qualified context value. Weekly states
are derived with thresholds fit only on earlier weeks, and the latest
observation at or before each quarterly origin is joined by time only. The
signal is intentionally not joined by NDC, drug, supplier, or county because
the public NSSP source does not expose those identities at the qualified grain.
The context vector therefore expands from 12 to 14 values, with one explicit
missingness channel per metric. Context and model shape tests were updated.

## 2026-08-16 Pass 336: Run Current Layered Rolling Baseline

The canonical `evaluate_whole_system.py --end-to-end --rolling` path was run
with one epoch and batch size 512 after the 14-wide context migration. Three
chronological folds were evaluated. Neural WAPE was 20.89%, 21.19%, and
20.15%; mean within-five-percent accuracy was 15.97%. The validation-gated
deep blend selected the neural path on all folds and beat the fold ridge
baseline by 76.27%, 76.47%, and 73.33% WAPE respectively. The result is
reproducible layered-model evidence and is marked
`publishable_rolling_candidate=true` under the local relative-improvement
policy, but it does **not** satisfy the project’s 75% numeric accuracy goal
for pharmacy-oriented output. No performance claim is transferred from the
external proxy metric library to the layered pharmacy demand target.

## 2026-08-16 Pass 337: Interpret Six-Fold Component Ablation

The corrected `evaluate_modular_component_rolling` run evaluated six
chronological folds and scored the intact path before each component removal.
Validation selected a zero neural-blend weight on every fold, making the
blended-WAPE ablation identical across variants. The artifact now also
reports direct neural WAPE: graph and temporal removal hurt four of six folds,
and deep-connection removal hurt three of six; news removal hurt zero folds.
This is architecture evidence, not a promotion claim, because the selected
blend relied on the persistence/transition path and the direct neural model
did not meet the 75% accuracy requirement. The limitation is documented in
`ARCHITECTURE.md`.

## 2026-08-16 Pass 338: Train And Score Public Five-State Demand Head

The publishable NDC9 demand adapter previously exposed a five-state output but
did not construct state labels or include state loss, leaving that head
untrained. It now derives next-quarter log-change states from five quantiles
fit on training rows only, freezes those cut points for validation/test, adds
the declared auxiliary state loss, and reports exact, balanced, and event
true-positive precision. A real one-fold chronological run produced 21.50%
state exact accuracy, 20.57% balanced accuracy, 21.48% high-state precision,
and 14.05% numeric within-five-percent coverage. The implementation is
retained, but no gate or promotion claim is made.

## 2026-08-16 Pass 339: Regenerate Canonical Demand Report With State Loss

The canonical three-fold `evaluate_whole_system.py --end-to-end --rolling`
run was regenerated after state supervision was added. Mean numeric
within-five-percent coverage was `15.24%`; fold values were `16.02%`,
`15.05%`, and `14.65%`. Mean five-state exact accuracy was `22.10%`, with
fold high-state true-positive precision of `24.93%`, `23.72%`, and `25.38%`.
Both `contract_75pct_numeric_accuracy` and `contract_75pct_state_accuracy`
remain false. The result is retained as research-only evidence in
`model/artifacts/evaluation/whole_system_metrics.json`.

## 2026-08-16 Pass 340: Reject Naive State-Loss Upweighting

A controlled one-fold experiment increased the public demand state-loss weight
from `0.1` to `1.0` while holding seed, split, epoch count, and batch size
constant. State exact accuracy changed only from the canonical first-fold
`22.19%` to `22.23%`, balanced accuracy was `21.93%`, and high-state precision
was `24.66%`. Numeric WAPE worsened to `0.2381` and within-five-percent
coverage fell to `13.19%`. The weighting change is rejected; future work must
improve the state representation or target construction rather than simply
increase auxiliary-loss weight.

## 2026-08-17 Pass 341: Compare Learned And Numeric-Derived States

The public demand evaluator now scores both the independently trained
five-state head and a state projection of the numeric residual head using the
same training-only cut points. On a matched first chronological fold, the
learned head reached `21.50%` exact accuracy and `21.48%` high-state precision;
the numeric projection reached `19.18%` and `19.76%`. Numeric post-processing
does not improve state prediction, so neither path is promoted and the next
model iteration must address shared representation or target design.

## 2026-08-17 Pass 342: Canonicalize State-Path Comparison

The canonical three-fold artifact was regenerated with both state paths. The
learned state head retained `22.10%` mean exact accuracy. The numeric-derived
state projection reached `19.53%` mean exact accuracy; its fold high-state
precision was `19.23%`, `20.90%`, and `33.45%`. Neither state path satisfies
the raw or event acceptance route. The comparison is now persisted in
`model/artifacts/evaluation/whole_system_metrics.json` rather than only in a
single-fold diagnostic.

## 2026-08-17 Pass 343: Reject Unvalidated Context-Widening Experiments

The public adapter audit found that individual prior news fields are preserved
for the raw stacker but only selected aggregates reach the neural history. A
matched one-fold experiment concatenating all 20 prior fields plus missingness
into a 40-wide history degraded neural WAPE to `0.3508` and within-five-percent
coverage to `10.02%`. A bounded 38-wide aggregate experiment that added local
and API news signals also degraded WAPE to `0.2311` and within-five-percent
coverage to `13.77%`. Both changes are rejected and the validated 38-wide
neural contract is retained. The raw stacker remains the auditable path for
individual prior fields; future neural integration requires a learned gated
projection and multi-epoch controlled evaluation.

## 2026-08-17 Pass 344: Reject Undertrained-Depth And Public-Graph Ablations

A matched first-fold three-epoch run under the validated 38-field contract did
not materially improve the one-epoch result: neural WAPE was `0.2114`,
within-five-percent coverage `15.98%`, and state exact accuracy `22.18%`.
Adding the directly observed `drug -> located_in -> Arkansas` edge to the
public graph was also rejected in a matched one-epoch fold: WAPE worsened to
`0.2365`, within-five-percent coverage to `13.41%`, and state exact accuracy
was `22.25%`. The public NDC9 adapter therefore retains its no-edge graph
contract until richer sourced entities and a controlled multi-epoch graph
training study are available.

## 2026-08-17 Pass 345: Reject Sparse-Panel Seasonal Comparator

Added a leakage-safe same-drug, four-quarter seasonal comparator and temporary
stacker feature. On a matched first fold, the seasonal comparator had WAPE
`1.4466`; the selected stacked result had `14.05%` within-five-percent
coverage, below the canonical baseline. The feature and comparator were
removed rather than promoted. The public quarterly panel does not provide
enough stable seasonal continuity for this naive seasonal signal to improve
the current layered demand model.

## 2026-08-17 Pass 346: Integrate Validation-Selected Recall Forecasting

The monthly NDC and supplier-by-NDC recall numeric adapters now call a
chronological logistic one-vs-rest selector with persistence fallback. The
selector fits only on completed source months, compares balanced accuracy on
the final validation window, and then forecasts the current source edge. On
the local FDA archive it selected persistence for all `3,032` NDC rows and
`3,208` supplier-by-NDC rows, so predictions did not change and no learned
uplift is claimed. The qualified forecast artifact was regenerated with this
selection logic and metadata.

## 2026-08-17 Pass 347: Align Numeric True-Positive Contract

The project goal and testing protocol now define the event route exactly as
requested: a numeric prediction at or above its predeclared event threshold
announces risk, and is a true positive when it is within 5% of the observed
numeric value. The observed value does not have to cross the announcement
threshold. State events remain exact matches within their declared high-state
set. Economic/status signals retain `not_applicable` event semantics. A
regression test covers a threshold-level prediction against an actual value
just below the threshold; focused event, audit, and respiratory tests pass
(`18 passed`).

## 2026-08-17 Pass 348: Persist Numeric Event Scores In Metric Artifacts

The chronological numeric evaluators now serialize the requested event route
for FDA shortage supplier count and FDA recall severity. The fixed event
threshold is `1` and numeric true positives use the project-wide 5% error
margin. Regenerated artifacts report `98.92%` shortage-count precision across
`419,806` predicted events, `99.98%` NDC-recall precision across `11,875`
predicted events, and `99.97%` supplier-by-NDC recall precision across `11,964`
predicted events. These are upstream public proxies and remain explicitly
persistence-dominated; the event score is not presented as pharmacy inventory
accuracy.

## 2026-08-17 Pass 349: Complete Qualified Event Diagnostics

Added leakage-safe high-zone event scoring to the qualified ARCOS
ZIP3-by-drug distribution state, national FluView respiratory state, and
Arkansas regional five-state demand evaluators. Regenerated artifacts now
report event precision of `90.15%` for ARCOS, `90.15%` for national FluView,
and `93.65%` for regional demand. The metric audit confirms all nine qualified
proxies now carry event diagnostics except NADAC, which is explicitly marked
not applicable because it is an economic level signal rather than an event.
The legacy three-state regional artifact was restored to its separate path so
it cannot be accidentally counted as a five-state metric.

## 2026-08-17 Pass 350: Reject Current-Period History Alignment

Tested a leakage-safe temporal alignment change for the canonical NDC9 demand
head that appended the current observed quarter to the four-step history
sequence. On a matched first chronological fold, the change worsened neural
WAPE from `0.21085` to `0.21362`, within-five-percent accuracy from `16.02%`
to `15.45%`, five-state exact accuracy from `22.19%` to `21.53%`, and high-zone
state precision from `24.93%` to `23.21%`. The opt-in path was removed; the
validated history contract remains unchanged. The experiment confirms that
the next improvement should address representation or target design rather
than simply appending the current observation.

An audit regression test now requires every qualified event-applicable metric
to carry a nonempty machine-readable event score, while allowing NADAC's
explicit `not_applicable` status. Focused evaluator and audit verification
passes (`25 passed`).

## 2026-08-17 Pass 351: Evaluate Ordinal Five-State Loss

Added an opt-in ordinal state-loss mode to the publishable NDC9 demand head.
It retains the five-class cross-entropy objective and adds cumulative ordinal
threshold losses, so the five-state output is not reduced to a binary target.
The matched three-fold chronological run used the same public data, one epoch,
batch size `512`, and validation-selected checkpoints. Its fold results were:

| Train through | Numeric WAPE | Numeric within 5% | State exact | State event precision |
| --- | ---: | ---: | ---: | ---: |
| 2017 | 20.998% | 16.203% | 22.757% | 32.825% |
| 2018 | 21.255% | 15.587% | 22.244% | 29.434% |
| 2019 | 20.275% | 15.547% | 19.828% | 30.620% |

Mean state exact accuracy was `21.610%`; neither the 75% raw route nor the
65% raw plus 80% event route passed. The ordinal option remains available for
controlled research comparisons, but categorical loss remains the default and
no metric or model-promotion claim changed. Low-level ordinal-loss regression
coverage was added to the training tests.

## 2026-08-17 Pass 352: Enforce Five-State Output Boundary

The output contract previously documented three-state signals as legacy while
still allowing their `qualified_three_state_proxy` status through the external
metric serializer and feature store. Renamed that status to
`legacy_three_state_proxy` and restricted promoted statuses to qualified
numeric and qualified five-state metrics. Historical evaluators remain intact,
but legacy rows now fail closed before serialization or feature attachment.
Updated contract tests to use a real five-state target and to verify rejection
of legacy rows. Focused output, feature-store, and universal-layer tests pass
(`32 passed`).

## 2026-08-17 Pass 353: Add Regional Wastewater Candidate Screen

Added a leakage-safe weekly regional wastewater adapter using CDC site WVAL,
single-county service labels, population weighting, and the existing Arkansas
DHS five-region crosswalk. The evaluator preserves only consecutive next-week
targets and learns five-state thresholds within each rolling fold. The local
panel supports three current regions for each pathogen with sufficient history.

| Pathogen | Held-out rows | Raw accuracy | Balanced accuracy | High-zone precision |
| --- | ---: | ---: | ---: | ---: |
| Influenza A | 195 | 80.00% | 27.31% | 75.00% |
| RSV | 208 | 80.29% | 29.05% | 77.68% |
| SARS-CoV-2 | 351 | 41.60% | 27.72% | 43.85% |

None passes the 75% raw route or the 65% plus 80% event route because the
balanced/state or event conditions fail. The candidate is not included in the
qualified metric feature store. Reproducible implementation and artifact:
`model/arkansas_pharma_signal/regional_wastewater.py` and
`model/artifacts/evaluation/regional_wastewater_five_state_metrics.json`.
Focused regional and adapter tests pass (`7 passed`).

## 2026-08-17 Pass 354: Reject Regional Wastewater Change Target

Tested a second representation of the same CDC regional wastewater panel:
five ordered states of next-week change (`strong_decrease`, `decrease`,
`stable`, `increase`, `strong_increase`) rather than five level quantiles.
Thresholds and model selection remained fit-only within each chronological
fold, with zero-change persistence as the baseline. The matched results were:

| Pathogen | Held-out rows | Raw accuracy | Balanced accuracy | High-zone precision |
| --- | ---: | ---: | ---: | ---: |
| Influenza A | 195 | 62.05% | 20.97% | 64.77% |
| RSV | 208 | 62.02% | 22.42% | 63.07% |
| SARS-CoV-2 | 351 | 29.06% | 24.30% | 22.45% |

The change target is strictly worse than the level representation and fails
both acceptance routes. It remains available only as a documented research
ablation; neither regional wastewater representation enters the qualified
metric feature store.

## 2026-08-17 Pass 355: Integrate Regional Screens Into Metric Audit

The machine-readable metric audit now consumes the regional wastewater artifact
and records all six pathogen/representation candidates with public source URLs,
feature timestamp boundaries, county-service-area censoring, five-state event
definitions, and rejection reasons. The qualified metric count remains `9`;
the new candidates do not alter any coverage or accuracy gate.

## 2026-08-17 Pass 356: Verify Operational Metric Pipeline

Ran the production-style qualified forecast and feature-store commands after
the candidate/audit changes. The qualified forecast surface emitted `44,029`
rows across `9` promoted targets. The feature-store build preserved all
`44,029` rows and produced `9` features with the declared seven-column filter
grain and no duplicate keys. Every emitted row declared
`uncertainty_status=not_estimated` and `calibration_status=not_calibrated`,
with confidence, risk, and interval fields left null. No rejected regional
wastewater candidate entered the operational surface.

## 2026-08-17 Pass 357: Add Strict Project-Status Audit

Added `project_status.py` and `audit_project_status.py` to prevent proxy
success from being confused with learned architecture completion. Against the
current artifacts, the audit reports:

- proxy library ready: `true` (`9` qualified metrics; all four coverage gates pass);
- learned architecture ready: `false` (mean within-five-percent accuracy
  `15.24%`, state exact accuracy `22.10%`);
- overall project complete: `false`;
- remaining blockers: the learned end-to-end accuracy contract and four failed
  publishability gates.

The audit also verifies the `44,029`-row operational metric surface, zero
duplicate filter grains, and explicit uncalibrated uncertainty status.
## 2026-08-17 Pass 360: CMS State-by-Drug Demand Proxy

The CMS Medicare Part D geography-by-drug annual files were downloaded locally
for 2013-2024 and normalized to `data/targeted_additions/cms_partd_geography_drug`.
The Arkansas state-by-generic-drug panel contains 14,275 rows and 1,548 drugs.
Six rolling-origin folds scored 6,998 held-out rows at 89.09% exact and 89.35%
balanced five-state accuracy. States 3-4 are the elevated-demand event zone;
true-positive precision was 95.30%. This is an annual external demand proxy,
not county, supplier, all-payer, or pharmacy-inventory truth. The qualified
surface increased from 10 to 11 targets; research exhaustion remains incomplete.
## 2026-08-17 Pass 361: HHS Monthly Pharmacy-NDC Candidate

Independent research identified the official HHS Open Data Medicaid Provider
Spending by NDC release. The July 2026 public ZIP was downloaded locally and
joined to the Arkansas NPPES pharmacy taxonomy (`3336`) list. The resulting
panel contains 28,386 observed Arkansas pharmacy-NDC-month rows across 84
months, 2,540 NDCs, and 2,075 maximum observed monthly pharmacy providers.
Only consecutive observed months were scored because HHS suppresses small
cells and absent rows cannot be interpreted as zero utilization.

The rolling-origin evaluator produced 57 folds and 12,998 held-out
transitions. The five-state demand head scored 51.41% exact accuracy, 52.53%
balanced accuracy, and 65.52% precision in states 3-4; the numeric head scored
9.10% within 5% error. Market-wide and labeler-lag features did not improve
the result. The source is therefore retained as training-only pharmacy demand
context and rejected as a promoted metric. It closes the question of whether
a public monthly Arkansas pharmacy-NDC source exists, but not the need for a
metric that clears the accuracy and incremental-utility gates.

## 2026-08-17 Pass 364: HHS Aggregate Pharmacy-Market Context Candidates

The locally normalized HHS Arkansas pharmacy-NDC panel was also aggregated to
test two statewide monthly context targets: total observed Medicaid/CHIP claim
lines and the maximum observed pharmacy-taxonomy billing-provider count. Both
were evaluated with the same leakage-safe rolling-origin, five-state and
numeric protocol used for the disaggregated panel.

Claim-line demand produced 57 folds and 57 held-out months: 63.16% exact
state accuracy, 7.02% within-5% numeric accuracy, and 73.91% precision among
predicted elevated states. Observed provider capacity produced 57 folds and
57 held-out months: 56.14% exact, 10.53% within-5%, and 52.17% elevated-state
precision. Neither candidate clears the 65% raw floor plus 80% event route;
provider observation count is retained only as context and is not treated as
an inventory-capacity label.

## 2026-08-17 Pass 365: Monthly External-Context Utility Ablation

The HHS pharmacy-NDC panel was used for the first monthly external-context
utility test. A history-only log-Ridge demand model was compared with the same
model augmented by prior-month FDA shortage exposure, Arkansas news article
counts, and Arkansas/national FluView WILI. Every context feature was shifted
one month before joining the demand panel; HHS suppressed or nonconsecutive
drug-month rows were excluded rather than interpreted as zero.

The rolling-origin experiment produced 59 folds and 13,552 held-out
transitions across 1,619 NDCs. History-only mean WAPE was 0.5933 and the
context model was 0.5996, a deterioration of 0.0063. Mean within-5% accuracy
was 8.25% for history-only and 8.24% with context, and context did not improve
every fold. The result is rejected as incremental pharmacy-regression utility;
it does not establish that the signals cannot help private inventory data.

## 2026-08-17 Pass 366: Enforce Target-Validity Provenance

The metric-library audit now has a fail-closed target-validity block. Every
candidate must carry source, semantics, temporal-boundary, geography,
supplier-resolution, and censoring metadata. Every qualified public target
must explicitly declare `target_is_direct_pharmacy_observation=false` and
`target_claim_scope=proxy_only_not_pharmacy_inventory`.

This does not reject useful external proxies; it prevents their accuracy from
being interpreted as direct pharmacy inventory, fill-disruption, backorder, or
supplier-allocation accuracy. The project status audit now reports this as an
independent check.

## 2026-08-17 Pass 367: Arkansas APCD Access Recheck

An independent search of the official Arkansas APCD site, pharmacy-claims data
dictionaries, requester resources, and the authorized web portal confirmed that
the APCD contains pharmacy claims and has current releases. The public
materials provide field definitions, release documentation, and a request
process, but not an unrestricted downloadable claim-level or aggregated
drug-by-pharmacy training panel. No local APCD claims file was downloaded.

The source is therefore rejected for the current freely reproducible metric
library and added to the research-exhaustion ledger. An approved APCD data
partnership remains the highest-value future route to direct pharmacy-adjacent
labels; it would require a new target, privacy, access, and chronological-test
audit before any promotion.

## 2026-08-17 Pass 368: Full Regression Verification

Ran the complete repository test suite from the repository root after the
target-validity contract and APCD research-ledger updates. The suite completed
with `339 passed, 23 warnings` in 9 minutes 9 seconds. No failures occurred.
The warnings are existing pandas/PyTorch deprecation, fragmentation, dtype,
and nested-tensor warnings; they do not alter the target-validity or metric
promotion results.

## 2026-08-17 Pass 369: Correct OpenFDA Input Disposition

The input audit previously classified source-prefixed `openfda_*` shortage and
enforcement variables as generic near-real-time inputs, despite documenting
the underlying APIs as continuously updated live feeds. The classifier now
promotes explicit `openfda_*` names and the canonical inventory fields
`shortage_active` and `recall_active` to `LIVE_INPUT`; historical FDA-derived
names remain conservative. Regression tests cover shortage and recall variants
and confirm operational readiness.

The full variable inventory was the authority for the canonical-name exception:
both fields are explicitly sourced from `openfda_shortages` or
`openfda_enforcement` in `VARIABLE_LIST.md`.

## 2026-08-17 Pass 370: Full Input Inventory Reclassification

Reclassified the 404 distinct variable names in `data/final_data/VARIABLE_LIST.md`
with the corrected contract. The resulting counts are: 2 `LIVE_INPUT`, 25
`NEAR_REAL_TIME_INPUT`, 146 `PERIODIC_TRAINING_ONLY`, 229
`DERIVED_TRAINING_VARIABLE`, 1 `LOCAL_PHARMACY_HISTORY_INPUT`, and 1
`STATIC_IDENTITY_CONTEXT`. Operational readiness remains false by design because
the complete research inventory includes annual and historical training inputs;
the live subset is now source-consistent rather than silently promoted.

## 2026-08-17 Pass 371: Near-Real-Time Source Registry

Added `input_sources.py` and connected it to the input disposition summary.
All 25 `NEAR_REAL_TIME_INPUT` variables now have machine-readable source URL,
cadence, publication-lag, public-access, and adapter-status metadata. At the
time of this pass, all sources were documented but no adapters were fully
ready, which made the source-availability and refresh-integration gates
separate by construction.

## 2026-08-17 Pass 372: Fail-Closed Adapter Readiness

Source documentation alone no longer makes an input contract operational. The
summary now emits `disposition_operational_ready` for name/cadence eligibility
and `operational_ready` only when every live or near-real-time source has an
`implemented` adapter. The forecast runtime guard uses `operational_ready`;
research training can retain source-documented variables under its explicit
override. Existing FDA shortage, NWS, and other adapter coverage remains
truthfully limited by the registry rather than inferred from a source URL.

## 2026-08-17 Pass 373: Implement OpenFDA Enforcement Refresh

Added `fetch_openfda_enforcement` to the live-input adapters. It paginates the
public FDA enforcement endpoint, preserves recall number, supplier, NDC,
classification, raw status, termination date, and retrieval metadata, and
derives `recall_active` only from the FDA status (`completed`/`terminated` are
inactive). The adapter is covered by a mocked pagination/status test and is
now marked implemented in the source registry. This makes the canonical FDA
shortage and recall inputs adapter-ready without claiming either is pharmacy
inventory truth.

## 2026-08-17 Pass 374: Implement Weekly FluView Refresh

Added `fetch_fluview_weekly` to the live-input adapters for the documented
Delphi Epidata CDC FluView feed. The adapter normalizes Arkansas and national
regions, preserves release date, issue, epiweek, reporting lag, ILI/WILI,
provider and patient counts, and attaches request and retrieval provenance.
It accepts an explicit epiweek range so point-in-time callers can control the
refresh window; it does not overwrite the historical backtest artifact.

A mocked API test covers region normalization, duplicate-region removal, query
construction, and issue/epiweek preservation. `ili`, `num_ili`, and `wili` are
now marked `implemented`; the full near-real-time registry remains fail-closed
because the other pending adapters are not yet integrated. Focused live-input,
input-contract, and CLI tests passed with **48 passed**.

## 2026-08-17 Pass 375: Implement NNDSS Weekly Refresh

Added `fetch_nndss_weekly` for the CDC NNDSS Socrata feed. The adapter
supports Arkansas and US RESIDENTS, paginates the public endpoint, normalizes
the reporting week to `epiweek`, and preserves disease labels, `m1`--`m4`
monthly fields, flags, jurisdiction, and retrieval provenance. It explicitly
does not sum the monthly fields into a fabricated weekly case total. The
source registry now marks `current_week_cases` implemented, while downstream
feature construction must still select and interpret the provisional fields
with their flags.

Mocked tests cover pagination, jurisdiction filtering, numeric normalization,
and preservation of the monthly-field warning. Focused live-input,
input-contract, and CLI tests now pass with **49 passed**.

## 2026-08-17 Pass 376: Promote NOAA Daily Weather Refresh

Added `fetch_noaa_daily_summaries` for the public NOAA/NCEI Daily Summaries
JSON service. The adapter requires explicit station IDs and ordered date
bounds, requests standard units, normalizes observed temperature,
precipitation, snow, wind, humidity, and extreme-temperature fields, and
retains station coordinates plus request/retrieval provenance. It performs no
county interpolation; the existing nearest-station county panel remains the
only geographic assignment step.

The nine NOAA weather variables (`temperature_min`, `temperature_mean`,
`temperature_max`, `precipitation`, `snowfall`, `snow_depth`, `wind`,
`extreme_heat_day`, and `extreme_cold_day`) are now marked adapter-ready.
Focused live-input, input-contract, and CLI tests passed with **50 passed**.
The complete source audit still has 12 pending near-real-time adapters:
WHO disease fields, FEMA disaster fields, NADAC price fields, and outbreak
events.

## 2026-08-17 Pass 377: Implement Dated CMS NADAC Refresh

Added `fetch_nadac_weekly_csv` for CMS NADAC weekly reference files. The
adapter requires an explicit snapshot date, constructs the documented dated
CSV URL, preserves NDC-level rate and pricing metadata, optionally filters to
an explicit NDC set, and records retrieval provenance. It does not select a
moving latest file and does not claim that a national acquisition-cost
observation is local inventory, demand, shortage, or supplier allocation
truth.

The three NADAC raw variables (`nadac_per_unit_min`, `nadac_per_unit_mean`,
and `nadac_per_unit_max`) are now marked adapter-ready. A mocked test covers
dated URL construction, NDC normalization/filtering, numeric parsing, and
target semantics. Focused live-input, input-contract, and CLI tests passed
with **51 passed**.

## 2026-08-17 Pass 378: Implement FEMA Arkansas Disaster Refresh

Added `fetch_fema_arkansas_disasters` for the paginated OpenFEMA Disaster
Declarations Summaries API. The adapter restricts requests to Arkansas,
preserves declaration and incident begin/end dates, constructs explicit
county FIPS keys when available, derives the existing declaration-type
severity convention, and records the retrospective nature of the source.
It does not treat `declarationDate` as incident onset.

Testing found and fixed a timezone comparison defect between FEMA UTC API
timestamps and caller-supplied observation dates. The adapter test now covers
pagination, county identity, active status, severity, and timestamp separation.
Focused live-input, input-contract, and CLI tests passed with **52 passed**.

## 2026-08-18 Pass 379: Demote Unverified WHO FluID Inputs

The WHO source review found that the official FluID documentation describes
the desired weekly fields, but the verified public UAT mart exposed only
`VIW_FLUNET`, not the required `VIW_FID_EPI` object. A request to the
production-style FluID OData URL timed out during review, so availability of a
stable no-auth refresh path could not be established.

The six FluID-derived names (`geospread`, `ili_activity`, `ili_case`,
`ili_nb_sites`, `ili_outpatients`, `intensity`) and the historical WHO DON
`outbreak_event` are now classified as `PERIODIC_TRAINING_ONLY`. Their source
records remain in `RESEARCH_ONLY_SOURCE_REGISTRY` with explicit
`feed_unverified` or `historical_artifact_only` status. This prevents a public
landing page or stale artifact from entering an operational model. The
machine-readable inventory now reports 18 near-real-time variables, all with
implemented adapters; focused input, live-adapter, and CLI tests passed with
**53 passed**.

## 2026-08-18 Pass 380: Refresh Public-Label Research Exhaustion

The public-source review was extended with the current Arkansas APCD requester
portal and pharmacy-claims documentation, a public hospital-pharmacy research
package, and the CDC NNDSS weekly table. The APCD materials confirm that
pharmacy claims and dispensing-pharmacy/NDC fields exist, but access requires
an authorized data request; the public research package is not Arkansas
pharmacy inventory truth; and the current Arkansas NNDSS extract does not
provide a sufficiently long stable target vintage for a new qualified metric.

These sources were recorded as rejected candidates rather than promoted as
metrics. The exhaustion artifact now records 54 screened candidates and keeps
the four unresolved questions explicit: county-by-supplier labels,
pharmacy-level outcomes, additional stable disease-demand series, and direct
incremental utility for individual-pharmacy regression. This preserves the
completion blocker instead of treating proxy accuracy as proof of inventory
benefit. The project-status test was updated for the expanded evidence record.

## 2026-08-18 Pass 381: Screen Respiratory and Inventory Benchmarks

The source search screened CDC RSV-NET, CDC COVID-NET, the public Dryad
essential-medicines inventory/consumption package, and a Kaggle inventory
benchmark. RSV-NET is relevant and includes Arkansas surveillance, but the
reviewed public path is an interactive dashboard without a verified stable
bulk historical export. COVID-NET is a national cumulative product beginning
in 2024. Dryad provides genuine facility stock and consumption outcomes but
comes from a Sierra Leone deployment. The Kaggle benchmark is explicitly
synthetic. None can be promoted as an Arkansas metric label.

The exhaustion artifact now records 58 screened candidates. The unresolved
questions remain unchanged and are intentionally blocking: public
county-by-supplier labels, public pharmacy-level outcomes, more stable
disease-demand series, and direct incremental utility for individual-pharmacy
regression.

## 2026-08-18 Pass 382: Harden Promoted Metric Serialization

The external metric serializer previously recognized state outputs from a
target-name suffix. That convention could allow a future qualified five-state
target with a different name to bypass state validation. Validation now uses
the authoritative `TARGET_METADATA` entry: promotion status must match the
metadata, qualified five-state outputs must declare event states and values
`0` through `4`, and qualified numeric outputs must declare either a numeric
event threshold plus tolerance or an explicit `not_applicable` event.

Added metadata-inventory and mismatch regression tests. This change affects
serialization safety only; it does not promote any additional metric or alter
reported accuracy.

## 2026-08-18 Pass 383: Add Public RESP-NET RSV Proxy

The CDC RESP-NET Socrata API was verified as a stable public weekly source.
The reproducible snapshot filters to RSV-NET, national Overall state, Overall
age/race/sex, observed Weekly Rate, and retains the published rate per
100,000. Four chronological folds provide 187 held-out weeks. A fit-history
five-state persistence forecast reaches 83.31% exact accuracy and 84.06%
precision among predicted high states.

The metric is promoted as
`national_weekly_respnet_rsv_hospitalization_pressure_state`. It is a national
disease-pressure proxy, not an Arkansas-local target, pharmacy dispensing
label, or inventory observation. The source is integrated into the qualified
forecast builder, metadata contract, source registry, and project audit; the
qualified library now contains 12 metrics and the serialized surface contains
12 targets. The research-exhaustion ledger records 59 screened candidates,
including this qualified source.

## 2026-08-18 Pass 384: Screen Arkansas Hospital Feed Under Five-State Contract

The existing public Arkansas NHSN hospital-respiratory feed was independently
re-evaluated for COVID, influenza, and RSV using five fit-history quantile
states and the current event route. COVID had no observed high state in 187
held-out weeks. Influenza reached 59.09% exact accuracy and 73.12% high-state
precision across 187 held-out weeks. RSV reached 74.57% exact accuracy and
79.22% high-state precision across only 83 held-out weeks and two complete
folds. None meets the five-state qualification contract. These results remain
screening evidence only; no Arkansas hospital target was promoted from the
legacy three-state evaluator.

## 2026-08-18 Pass 385: Screen CDC NSSP County Dimensions

The local CDC NSSP snapshot was inspected at its exposed county and HSA
grain rather than assuming the statewide aggregate was geographically
transferable. The file contains 76 county labels and HSA mappings, but every
county-level `percent_visits_*` and smoothed-percentage field is null; the 202
`county=All` rows are the only rows with numeric pathogen percentages. The
county candidate therefore cannot provide 25 valid chronological target
samples per county and was rejected. No county metric, regional broadcast, or
pharmacy-local claim was added from these identifiers.

## 2026-08-18 Pass 386: Test RESP-NET Incremental Monthly Demand Utility

The qualified national RESP-NET RSV rate was added as a lagged monthly
feature to the closest available Arkansas pharmacy-demand ablation. The
history-only Ridge baseline and six-feature context model were evaluated on
the same 59 chronological folds and 13,552 held-out HHS drug-month rows.
Mean WAPE changed from 0.59325 to 0.59956 (delta +0.00630); within-5-percent
coverage changed from 0.08247 to 0.08319, but the all-fold improvement rule
failed. The feature is rejected for incremental utility and remains only a
qualified external disease-pressure context. This does not establish or
refute utility for private pharmacy inventory without pharmacy-level labels.

## 2026-08-18 Pass 387: Add RESP-NET to Historical Metric Context

The qualified RESP-NET RSV proxy was added to the leakage-safe historical
metric-context adapter as an eighth state feature. Its five-state thresholds
are fit only from the 52 or more prior weekly observations, and the latest
state at or before each quarterly feature origin is joined without using the
current forecast artifact. The model context contract therefore expands from
14 to 16 values, preserving one explicit missingness channel per metric.
Historical-context, modular-forward, and training-contract tests were updated;
the feature remains opt-in and does not imply incremental pharmacy-model lift.

## 2026-08-18 Pass 388: Full Verification After RESP-NET Context Integration

Built the opt-in real-label training arrays with historical metric context:
32,531 training rows, 3,373 validation rows, and 2,597 test rows, with the
16-wide value/missingness contract. The complete repository test suite passed
with **357 tests passed** and 24 warnings. The project-status audit still
reports 12 qualified external metrics and `project_complete=false` because
research exhaustion remains incomplete; qualified proxies are not direct
Arkansas pharmacy inventory observations.

## 2026-08-18 Pass 389: Screen Notre Dame ARCOS Transaction Portal

The Notre Dame ARCOS portal was independently inspected. Its public query API
and direct full-dataset link expose transaction date, buyer county/ZIP,
reporting supplier/family, controlled-drug code, and calculated active-
ingredient grams. An API statistics query estimates approximately 190,000
Benton County oxycodone transactions for 2006-2019, confirming that the source
could support a historical supplier/geography target. Large transaction-level
queries were not captured locally, and the source ends in 2019 and covers
controlled substances only. It is therefore recorded as a training-only
candidate, not a current operational input or qualified pharmacy-inventory
metric.

## 2026-08-18 Pass 390: Qualify Arkansas APCD Claim Activity Proxy

The public Arkansas APCD pharmacy claim-count workbook was downloaded and
normalized into a local manifest-backed panel. It contains 1,140 monthly rows,
37 reporting entities, and 36 months from 2018-07 through 2021-06. The
leakage-safe next-month five-state persistence evaluation uses four rolling
folds and 777 held-out transitions: 92.43% exact accuracy, 92.89% balanced
accuracy, and 96.17% precision among predicted high-zone states (states 3-4).
The metric is promoted as `arkansas_monthly_apcd_pharmacy_claim_activity_state`
only as a reporting-entity activity proxy. The report has no NDC, drug,
supplier, individual-pharmacy, fill-disruption, or inventory fields, so those
interpretations remain explicitly prohibited.

The initial three-month fold step was then rejected as an overcounting risk
because six-month test windows overlapped. The published protocol now uses
three non-overlapping five-month test windows: 488 held-out transitions,
91.83% exact accuracy, 92.48% balanced accuracy, and 95.73% high-zone
precision. The metric remains qualified under the same gates.

## 2026-08-18 Pass 391: Test APCD Activity as Demand-Regression Context

The APCD reporting-entity claim count was aggregated by month and shifted one
month before joining the HHS Arkansas pharmacy-NDC demand panel. The ablation
was restricted to the APCD source's actual 2018-07 through 2021-06 coverage
and used 12 chronological folds with 2,382 held-out drug-month rows.
History-only Ridge mean WAPE was 0.75742; the existing six-feature bundle was
0.75332; adding the APCD feature produced 0.75392. APCD improved mean WAPE
over history-only by 0.00350 but worsened the existing context bundle by
0.00060, reduced within-5-percent coverage, and did not improve every fold.
It is not claimed as incremental utility for individual-pharmacy regression.
The experiment and its source boundary are recorded in
`model/artifacts/evaluation/apcd_monthly_context_utility_metrics.json`.

## 2026-08-18 Pass 393: Correct ATC Allocation and Live Forecast Horizon

OpenCode review found that lexicographic first-class selection was clinically
arbitrary for 803 of 1,961 mapped NDCs spanning multiple ATC groups, and that
the live adapter used the final transition row, forecasting an already observed
month. Claims are now fractionally allocated across unique ATC groups, with
one canonical label per ATC code and provider count retained as a conservative
within-class maximum. The adapter
uses the final raw observed month as the feature period and forecasts its next
month. The corrected evaluation has 74 mapped groups (71 with complete scored
transitions), 57 folds, 3,104 held-out
transitions, 73.84% exact accuracy, and 74.56% balanced accuracy. The latest
surface forecasts 2025-01 from 2024-12 across 52 complete class series.

## 2026-08-18 Pass 392: Add NLM ATC Therapeutic-Class Demand Signal

The historical HHS Arkansas pharmacy-NDC panel was linked to NLM RxNorm and
RxClass using the public historical NDC lookup and ATC relation APIs. Of 2,540
unique NDC keys, 2,052 mapped to RxNorm and 1,961 had ATC relations. A
The initial implementation used a deterministic primary ATC group per NDC;
that result was superseded in Pass 393 after review. The signal is promoted as
`arkansas_monthly_atc_therapeutic_demand_state`, explicitly as a therapeutic
class demand proxy rather than diagnosis, all-payer demand, or inventory.

## 2026-08-18 Pass 394: Synchronize Status Artifact and Research Gate

The contract audit was rerun after the ATC promotion and now reports 14
qualified proxy metrics. The serialized project-status artifact was refreshed
to match that count and continues to correctly report
`project_complete=false` because the research-exhaustion gate remains
incomplete. The remaining disease candidates were reviewed without promotion:
they lack stable historical Arkansas vintages, sufficient five-state
coverage, or the required held-out performance. A regression test now fails if
the serialized status and metric-library audit diverge.

## 2026-08-18 Pass 395: Standardize Event-Route Audit Semantics

OpenCode review confirmed that event scoring matches the revised contract and
that event thresholds are declared before held-out scoring. It found a latent
inconsistency: county and statewide Part D candidates retained a
balanced-accuracy rejection after satisfying the 65% raw plus 80% event route,
unlike the other five-state branches. Those branches now remove only the
balanced-accuracy reason when the event route passes; raw accuracy, event
precision, and five-state coverage checks remain active. Operational-surface
documentation was corrected from obsolete five-target/8,299-row counts to
the current 14-target/46,911-row artifact. The refreshed audit still reports
14 qualified metrics and `project_complete=false`.

## 2026-08-18 Pass 398: Correct Learned-Blend Reporting

OpenCode review found that the learned end-to-end evaluator computed a
raw-ridge/deep-stack convex blend but serialized a different validation-best
single policy under the name `stacked_blend`. The evaluator now reports the
actual convex blend and retains the single-policy choice only as
`selected_policy` metadata. The event-route description is also mode-aware:
`state_target_mode=change` is reported as a five-state log-change signal,
while `level` is reported as an absolute-demand signal. The corrected
three-fold artifact regeneration was attempted but stopped after exceeding
the available execution window; the previous JSON remains untouched and must
be treated as pre-fix evidence until regenerated. Deterministic evaluator
tests pass, including a direct convex-blend assertion.

## 2026-08-18 Pass 397: Verify Layered Model Inventory

The live default `ArkansasPharmaMultimodalModel` inventory was measured at
307,798,423 active/trainable parameters and zero frozen parameters. The six
disjoint code-level stages are news (242,698,240), graph (47,874,400),
temporal (7,519,344), cross-modal (4,158,672), deep connections (3,749,872),
and heads (1,797,895). OpenCode independently verified the implementation,
documentation, and intermediate-state tests. A stale comment describing seven
metric values was corrected to eight; the resulting 16-wide value/missingness
contract is unchanged.

## 2026-08-18 Pass 396: Complete Qualified Output Event Contract

The qualified output table in `PROJECT_GOAL.md` previously grouped several
metrics and omitted explicit entries for APCD activity, therapeutic-class
demand, statewide Part D demand, county demand, RESP-NET, and NSSP. It now
lists all 14 qualified metrics and their declared event zones, including the
two numeric non-inventory event thresholds and the NADAC not-applicable
declaration. A regression test compares every qualified audit definition with
the operational `TARGET_METADATA` definition. The focused contract suite
passes 39 tests.

## 2026-08-18 Pass 399: Preserve Post-Correction Evaluation Provenance

The corrected rolling evaluator was invoked, but its robust validation grid
exceeded the available execution window before writing a result. It was
stopped without modifying the prior JSON artifact. Architecture and testing
documentation now label existing rolling learned-model numbers as
pre-correction historical evidence and require regeneration before using them
as current scores. This preserves provenance instead of silently presenting a
stale blend metric as post-fix evidence.

## 2026-08-18 Pass 400: Cache Learned-Stack Design Matrices

The robust stack search rebuilt identical raw-plus-latent design matrices for
every alpha, Huber, and recency candidate. The evaluator now caches those
matrices once per fold and reuses them without changing the candidate grid,
weights, features, or validation selection. Equivalence tests pass. The
dominant cost is repeated weighted normal-equation solving; a post-fix single
fold still exceeded five minutes and was stopped safely. No stale evaluation
artifact was overwritten.

## 2026-08-18 Pass 401: Obtain One Post-Fix Learned Fold

The optimized evaluator completed one bounded fold (`train<=2019`, validation
2020, test after 2020) before interruption. It reported the actual corrected
raw-ridge/deep-stack blend with deep weight `0.22`, test WAPE `1.2577`, and
2.43% within-five-percent accuracy. The validation-selected policy was
`transition_neural_blend`, demonstrating that the reported blend is distinct
from the selected single policy. This is diagnostic evidence only: one fold
cannot satisfy the three-fold contract, and no existing rolling artifact was
overwritten.

## 2026-08-18 Pass 402: Regenerate Corrected Three-Fold Learned Artifact

The corrected evaluator completed all three chronological folds using the
cached matrices and ordinary-selected-alpha robust search. The artifact now
reports mean stacked-blend improvement versus fold ridge of **-18.50%**, mean
within-five-percent accuracy of **1.68%**, and mean five-state exact accuracy
of **22.55%**. The 75% numeric gate, 75% state gate, and 80% event route all
fail; `publishable_rolling_candidate=false`. Project status was regenerated
and now correctly reports `learned_architecture_ready=false`, while the 14
qualified external proxy metrics remain available. The result is a valid
negative learned-model evaluation, not a promotion failure hidden by a
baseline-selected headline.

## 2026-08-18 Pass 403: Reject Stronger Ordinal State Loss

An isolated post-fix fold tested `state_loss_weight=1.0` with the ordinal state
loss on the same chronological split. Five-state exact accuracy was **19.55%**
(balanced **21.02%**) versus the canonical fold's **23.18%** exact accuracy;
numeric-state exact accuracy was **22.29%**, and blend WAPE was **1.2611**.
The stronger ordinal loss did not improve either state or numeric performance,
so the canonical categorical weight `0.1` remains unchanged. This is a
single-fold ablation and is not promotion evidence.

## 2026-08-18 Pass 404: Verify Full Suite Against Negative Learned Status

The full repository suite reached 366 passing tests before exposing one stale
assertion: `test_project_status` still expected learned architecture readiness
from the pre-correction artifact. The test now requires
`learned_architecture_ready=false` and explicitly checks both current blocking
reasons: the learned end-to-end accuracy contract and research exhaustion.
The affected contract suite passes 9 tests; the prior full run was therefore
not accepted as the final green result until this expectation was corrected.

## 2026-08-18 Pass 405: Final Full-Suite and Artifact Verification

The clean repository suite now passes **367 tests** with 24 warnings. The
post-fix artifact cross-check confirms 14 qualified metrics in both the metric
audit and project-status report, 46,911 forecast rows across 14 targets, and
zero duplicate operational keys. The corrected learned artifact has all three
learned gates false, so project status correctly reports
`learned_architecture_ready=false` and `project_complete=false`; remaining
blocking reasons are learned end-to-end accuracy and research exhaustion.

## 2026-08-18 Pass 406: Correct Prior-Context Input Classification

The input audit found that `prior_news_*`, `prior_*_ili_*`, and related
point-in-time joins could be classified as raw near-real-time inputs because
their suffixes matched source-token heuristics. The contract now gives the
`prior_` prefix explicit derived-variable precedence, including over static
identity matching. Operational eligibility inherits from the underlying raw
field: prior news, surveillance, and shortage context remain operational when
their source is available, while prior Medicaid/ARCOS and missing-context
fields remain non-operational. The focused input-contract suite passes **37
tests**.

## 2026-08-18 Pass 407: Enforce Raw-Family Availability for Derived Inputs

The full variable-inventory audit found that generic derived-feature logic was
too permissive: annual EPA/Census transforms without a periodic token could be
reported as operational. Derived eligibility now strips deterministic suffixes
and reclassifies the underlying raw family. BLS unemployment/CPI, FRED pharma
PPI, FluView, NOAA weather, and NADAC derivatives remain operational; annual
EPA, Census, and historical-only derivatives fail closed. Interaction features
inherit availability from their context component, and diagnostic row counts
are derived rather than static identity. The focused input-contract suite
passes **39 tests**.

## 2026-08-18 Pass 408: Verify Input-Contract Integration

The stricter input contract was checked against modular training and
operational feature selection. Two modular local-history transforms and the
FDA shortage supplier-count derivative are now correctly rebuildable, leaving
only the two documented global economic context fields as non-operational in
the modular surface. The affected input, training-contract, and feature-mode
tests pass **42 tests**; no model artifact or accuracy claim was changed.

## 2026-08-18 Pass 409: Document Training-Budget Ablation Plan

The corrected learned evaluation used a batch size of 4,096 on approximately
32,000 training rows, exposing the trainable stack to only about eight
optimizer updates per epoch. The next controlled experiment will compare a
smaller batch on the identical chronological fold, seed, model configuration,
and one-epoch budget. This is a training-procedure ablation, not a relaxed
accuracy gate: the 75% raw routes and 65% plus 80% event route remain
unchanged, and no result will be promoted without the full rolling protocol.

## 2026-08-18 Pass 410: Training-Budget Probe Incomplete

The matched first rolling fold completed for the existing batch size 4,096:
stacked-blend WAPE was **1.0595**, within-5-percent accuracy **0.59%**, and
five-state exact accuracy **22.31%**; high-state true-positive precision was
**23.63%**. The deep-stack validation weight was 0.27, but the result remains
far below every learned accuracy route. The smaller-batch repeat was stopped
after approximately nine minutes and 6.8 GB resident memory during repeated
row construction, before producing a result. No default, artifact, or gate was
changed; the experiment demonstrates that a fair optimizer-budget comparison
needs dataset caching or a cheaper fold harness first.

## 2026-08-18 Pass 411: Document Publishable-Row Caching Improvement

Before repeating the optimizer-budget probe, the public-row builder will cache
the deterministic graph context once per NDC9 and copy the cached arrays for
each quarter. This changes no features, labels, split boundaries, or model
parameters; it only removes repeated identity parsing and stable-ID work. The
cache will be verified for output equivalence before another training result is
considered.

## 2026-08-18 Pass 412: Cache Deterministic Public Graph Context

`build_publishable_demand_rows` now caches the deterministic four-node graph
context once per NDC9 and copies the arrays into each emitted row. The metadata
records the number of unique cached contexts, and the strict-split adapter test
confirms the cache covers all emitted drugs without changing feature width or
row construction. The focused adapter test passes; no labels, model weights,
or evaluation gates changed.

## 2026-09-14 Pass 413: Per-Drug State-Demand Accounting

The active expansion target is a multi-drug or multi-disease signal surface.
The existing CMS Arkansas state-by-generic-drug evaluator was rerun on the
real 2013-2024 panel: six chronological folds, 6,998 held-out rows, 1,548
drugs, 89.09% exact five-state accuracy, and 89.35% balanced accuracy. The
elevated-state precision was 95.30%.

The evaluator was extended only to retain per-drug held-out row counts and
exact accuracy. With at least six scored rows, 856 drugs reached at least 80%
exact accuracy; with at least five rows, 905 did. This establishes a viable
10-15-drug candidate surface, but the result remains an annual Medicare Part
D demand proxy and does not establish local inventory or supply prediction.

The result is saved in
`model/artifacts/evaluation/arkansas_state_partd_demand_five_state_metrics.json`.
The current evidence does not yet show incremental benefit from the dated
news/NLP layer. A fixed-drug, history-only versus history-plus-news comparison
on compatible chronological folds remains required.

## 2026-09-14 Pass 414: Annual News Ablation on 15 Drugs

A fixed 15-drug experiment joined the dated news-only SLM annual features to
the CMS state-by-generic-drug demand panel. The overlap supports only three
annual test transitions per drug, or 45 held-out rows, so this is explicitly
an ablation and not a promoted result. History-only scored 80.00% exact and
75.71% balanced accuracy. Adding news scored 62.22% exact and 58.66% balanced
accuracy. The news-augmented model is rejected; the result does not support
an incremental-news claim. It is saved in
`model/artifacts/evaluation/annual_drug_news_ablation.json` and is
reproducible with `model/scripts/evaluate_annual_drug_news.py`.

## 2026-09-14 Pass 415: Monthly News-Aligned Drug Check

The monthly HHS Arkansas pharmacy-NDC panel was tested with a fixed 15-drug
selection. Month `t` news and pharmacy history predicted the observed target
in month `t+1`; training covered 2018-01 through 2021-12 and testing covered
2022-01 through 2024-12. The panel has better temporal alignment than the
annual CMS experiment, but only 7 drugs met the minimum 30 training rows, 25
test rows, and three-state-variation checks. Across 240 held-out rows,
history-only scored 62.08% exact / 62.12% balanced accuracy, while
history-plus-news scored 55.00% / 55.86%. The result fails the 80% gate and
does not show news uplift. It is saved in
`model/artifacts/evaluation/monthly_drug_news_ablation.json` and is
reproducible with `model/scripts/evaluate_monthly_drug_news.py`.

## 2026-09-14 Pass 416: Validation-Selected Monthly News Ablation

The monthly ablation was tightened: drugs were selected using training data
through 2020-12, 2021 was reserved for regularization selection, and 2022-01
through 2024-12 remained untouched for testing. Four drugs met the minimum
30 training, 10 validation, and 25 test rows plus three-state variation.
Across 135 held-out rows, history-only scored 51.85% exact / 50.96% balanced
accuracy, while history-plus-news scored 53.33% / 51.54%. This is a small
positive movement but does not meet the 10-15-drug or 80% gate. A follow-up
training-only news-feature ranking selected the number of news variables on
the validation year; its history-plus-news score was 51.85% exact / 50.90%
balanced, equal to the history-only exact score. The result remains
unpromoted and is stored in
`model/artifacts/evaluation/monthly_drug_news_ablation.json`.

## 2026-09-14 Pass 417: FDA Per-NDC Supply Audit

The existing FDA shortage archive was checked as a possible longer-history
news-aligned supply target. After restricting the candidate selection to the
pre-test window and applying the restored news overlap, the candidate NDC
histories exposed only two observed supplier-pressure states in training.
That is insufficient for a defensible per-drug five-state model. No labels
were invented, no test-period target selection was used, and no supply-plus-
news result was promoted. The discarded exploratory runner made no retained
model or data changes.

## 2026-09-14 Pass 419: Extended Arkansas Newspaper Demand Ablation

The local 3DLNews2 panel was used to extend news history to 2013-2022. A
fixed 15-drug CMS annual demand experiment trained on 2013-2018, reserved
2019 as validation context, and tested target years 2020-2022 across 45
held-out drug-year rows. History-only scored 97.78% exact / 98.25% balanced
accuracy, while adding 3DLNews keyword counts scored 77.78% / 77.22%.
Because the overlap provides only three test transitions per drug and early
news coverage is sparse, the result is an ablation rather than a promoted
metric. It demonstrates that the current news features do not add validated
annual drug-demand lift. The artifact is
`model/artifacts/evaluation/annual_3dlnews_drug_ablation.json`.

## 2026-09-14 Pass 420: Aggregate Context Utility Evaluation

The existing HHS monthly context evaluator was rerun on real Arkansas
pharmacy-NDC demand joined to FDA shortage, Arkansas 3DLNews, CDC FluView,
and RESP-NET inputs. It produced 59 chronological folds, 13,552 held-out
rows, and 1,619 drugs. Mean WAPE was 0.5933 for the history-only base and
0.5996 with the context bundle, a +0.0063 deterioration. Within-five-percent
accuracy was 8.25% versus 8.32%, and context did not improve every fold. The
result is not an incremental-news success and remains unpromoted. The artifact
is `model/artifacts/evaluation/hhs_monthly_context_utility_metrics.json`.

## 2026-09-14 Pass 418: Shared Monthly Drug-State Scale

The monthly ablation was rerun using one low/medium/high demand scale learned
from training-period targets across the selected drugs, matching the state
definition used by the successful annual CMS evaluator. Ten drugs met the
minimum row checks, producing 344 held-out drug-month rows. History-only
scored 68.60% exact / 65.67% balanced accuracy; history-plus-news scored
65.12% / 63.45%. The shared scale fixes the candidate-count limitation but
does not meet the 80% gate, and news reduces accuracy. The result remains
unpromoted in `model/artifacts/evaluation/monthly_drug_news_ablation.json`.

## 2026-09-14 Pass 421: Qualified Signal Surface Rebuilt

The existing qualified forecast builder was rerun with the available county
demand test panel and current public source files. It materialized 43,366
rows across 14 target types with zero duplicate keys, including 1,212
statewide annual drug-demand rows and 3,303 regional annual drug-demand rows.
The output is `model/artifacts/forecasts/qualified_metric_forecasts.csv.gz`.
This makes the proxy signals consumable but does not alter the negative
news-uplift results or satisfy the learned end-to-end completion gates.

## 2026-09-14 Pass 422: Metric Audit Regenerated

The contract-based metric audit was regenerated after rebuilding the qualified
forecast surface. It now recognizes 1 qualified proxy metric, passes target
validity and operational-surface checks, and confirms 43,366 forecast rows
with no duplicate keys. Coverage breadth and learned end-to-end/news accuracy
remain incomplete; no completion status was changed.

## 2026-09-14 Pass 423: NNDSS Arkansas History Audit

The checked-in CDC NNDSS extract was audited for a possible five-to-eight-
disease route. Despite source metadata describing 2022-present coverage, the
Arkansas rows in the local file contain only 2025 and 2026 observations;
2022-2024 rows are national. The file therefore has no Arkansas training
history for a chronological disease forecast. The attempted evaluator was
discarded without retaining code or generating labels/results.

## 2026-09-14 Pass 424: CDC Archive Availability Clarified

Official CDC documentation was checked to distinguish source availability from
local capture. CDC states that archived weekly NNDSS tables exist from 2014
onward, but the current `x9gk-5huc` extract in this repository still has only
2025-2026 Arkansas rows. The disease README now records this distinction and
the archived-table URL. No historical values were fabricated or treated as
zeroes.

## 2026-09-14 Pass 425: Part A Head/Signal Clarification

The Part A artifact was re-read to document the distinction between its 20
news input columns and its 11 broader promotion-candidate output heads. The
heads use month `t` inputs to predict month `t+1`, train through 2021-12, and
test CDC targets from 2022-01 through 2026-01 and HHS targets through
2024-12. Eight of the eleven candidates reach 70% exact accuracy; three CDC
age heads are retained only by the broader candidate gate. This clarification
does not change any metrics or promote the news layer.

## 2026-09-14 Pass 426: National NNDSS Multi-Disease Screen

The checked-in national NNDSS rows were screened as a possible alternative to
the Arkansas disease route. Using the official `m1` current-week count, 2022
through 2023 as training and calendar year 2024 as a chronological holdout,
12 disease labels had at least 70 non-null training observations and at least
three observed training values. Each label had 51 usable held-out weekly
transitions (49 for one label). Prior-week history was compared with prior
week history plus the dated monthly news features. No disease reached the
80% three-state accuracy requirement. The best history-only exact accuracy
was 84.31% for Campylobacteriosis, but its balanced accuracy was 61.05%; the
news version fell to 68.63%. Shigellosis reached 84.31% with news and
Giardiasis 72.55%, but neither met the full gate. This route is rejected for
promotion because it has only national 2022--2024 history and lacks the
required 5--8 qualifying disease signals.

## 2026-09-14 Pass 427: FDA Breadth Rerun Deferred

The existing `evaluate_shortage_pressure.py` was started once to regenerate
the full FDA rolling artifact. It remained CPU-bound for approximately twelve
minutes and produced no partial output, so it was stopped cleanly to avoid an
unbounded prototype run. No source data, metrics, or forecast artifacts were
changed. The previously recorded FDA evidence remains the authoritative
result until a bounded per-NDC rerun is added.

## 2026-09-14 Pass 428: Fifteen Generic Supply Labels

The FDA shortage archive was screened at the generic/formulation level to
avoid counting 15 NDC packages as 15 drugs. Fifteen distinct labels were
selected using observed-month counts through 2023-12 only. The existing
next-month supplier-count evaluator produced 145 rolling folds, 6,243
held-out transitions, and 98.776% mean within-five-percent accuracy for all
15 labels. A bounded five-feature dated-news ablation also produced 98.776%;
validation selected persistence in every fold, so there was no measured news
lift. This result is a supply proxy only: FDA national supplier reporting for
NDC evidence exposed to Arkansas, not pharmacy inventory or dispensing, and
the labels include related formulations.

That exploratory result was not retained as a final artifact because its
selection cutoff overlapped earlier scored folds.

## 2026-09-14 Pass 429: Chronology Correction for Generic Supply Screen

The preceding 15-label screen was rejected as a final result because its
selection cutoff (2023-12) overlapped earlier scored folds. It was replaced
by a chronology-safe screen: labels were selected through 2015-12, with a
minimum 36 observed training months, and scored only from 2016-01 onward.
This yielded 12 distinct generic/formulation labels, 124 rolling folds, and
4,245 held-out next-month transitions. History-only and history-plus-five-news
both achieved 98.3938% mean within-five-percent accuracy. The news comparison
selected persistence in every fold, so no news lift is claimed. The corrected
artifact is `model/artifacts/evaluation/chronology_safe_12_generic_supply_news_screen.json`.

The result was made reproducible with
`model/scripts/evaluate_generic_supply_news.py`; rerunning it reproduced the
same fold count, held-out row count, and aggregate metrics. Syntax, JSON, and
whitespace checks passed, and no evaluator process remains running.

## 2026-09-14 Pass 430: Supply-Event Feasibility Check

For the same chronology-safe labels and post-2015 scoring window, the
next-month supplier-count panel contained zero transitions from no active
suppliers to an active shortage and only 1.6% non-equal supplier-count
transitions. A news-event accuracy metric on this panel would therefore be
dominated by class imbalance and would not be a defensible unexpected-event
test. No event model or promotion claim was added.

## 2026-09-14 Pass 431: Neural Supply/News Comparison

The reproducible generic supply evaluator was extended to exercise the
existing NumPy `NeuralSignalBlender` within each chronological fold. The MLP
used the same six supply-history features and five dated news features, was
selected against persistence on the validation window, and was refit only on
pre-test data. Its aggregate result was identical to the regression and
persistence paths: 98.3938% within-five-percent accuracy across 124 folds and
4,245 held-out transitions. The neural path is now tested in this target
family, but no incremental news or neural lift is claimed.

## 2026-09-14 Pass 432: CDC Archive Mapping Deferred

The official CDC Stacks NNDSS collection was inspected and confirmed to expose
downloadable historical weekly table text files, including 2018--2021 items.
A bounded page-ID crawl was attempted for selected disease terms, but CDC
Stacks throttled the requests and no complete ID map was returned. The crawl
was stopped cleanly; no partial archive data was ingested and no disease
metric was changed. The current local NNDSS extract therefore remains the
authoritative disease dataset.

## 2026-09-14 Pass 433: Active-Goal Documentation Reconciled

`PROJECT_GOAL.md` was updated to reference the chronology-safe 12-label FDA
supply screen and its reproducible 98.39% within-five-percent result. The
documentation explicitly retains the limitations: the target is an upstream
supplier-count proxy, persistence dominates, and the dated news and NumPy
neural paths have not shown incremental lift. No direct-inventory or
news-causality claim was added.

## 2026-09-14 Pass 434: Generic Supply Forecast Surface

The chronology-safe FDA evaluator was rerun after adding its explicit forecast
output. It wrote 12 labeled next-observed-month rows to
`model/artifacts/forecasts/chronology_safe_12_generic_supply_news_next_month.csv.gz`.
Each row records the latest available feature month, forecast month, predicted
active supplier count, five lagged news features, and the selected persistence
method. Latest observations are not available for every generic label, so the
rows have different forecast periods; this prevents overstating the file as a
single-date live forecast. Metrics remained unchanged: 124 folds, 4,245
held-out transitions, and 98.3938% within five percent for history-only, news,
and neural paths.

## 2026-09-14 Pass 435: Statewide ARCOS News Evaluation

The real DEA ARCOS Arkansas retail distribution summary was aggregated across
ZIP3 for the 15 best-covered controlled-substance codes. A new minimal
evaluator used training-only per-drug low/mid/high terciles, prior-quarter
history, and quarterly 3DLNews2 Arkansas keyword counts to predict the next
quarter. Drug selection used data through 2018; scoring was restricted to
2013-Q1 through 2022-Q4 so missing post-2022 news was not converted to zero.
Across 9 chronological four-quarter folds, the selected system scored 61.14%
exact / 50.83% balanced accuracy, compared with 43.40% / 33.33% for
persistence. The news logistic candidate scored 62.76% / 53.89%, showing
modest real lift but failing the 80% three-state gate. The result is in
`model/artifacts/evaluation/arcos_state_news_metrics.json`; no promotion claim
was made because ARCOS is a distribution proxy rather than inventory or
dispensing.

## 2026-09-14 Pass 436: Weekly Disease-Pressure Screen

The existing real CDC FluView weekly panel was screened for next-week numeric
and three-state predictions. National ILI reached 91.51% exact / 80.72%
balanced accuracy, and national WILI reached 86.91% / 77.82%, across seven
chronological year folds. These are two influenza measures rather than two
distinct diseases, and Arkansas versions had lower balanced accuracy, so the
screen does not satisfy the requested 5–8-disease breadth. National numeric
within-five-percent accuracy was only 19.32% for ILI and 16.68% for WILI. No
promotion claim was added.

## 2026-09-14 Pass 438: ARCOS Transition-Baseline Check

An exploratory transition-aware state baseline was tested on the same 15-drug
statewide ARCOS panel. State transition modes were learned only from each
fold's fit window and selected against persistence on validation data. Across
9 chronological folds it scored 35.16% exact / 29.39% balanced accuracy,
below persistence at 43.40% / 33.33%. It was rejected and no architecture
change was retained.

## 2026-09-14 Pass 437: FDA Observed-Supplier Eventfulness Check

The same 12 chronology-safe FDA generic/formulation labels were screened using
the explicit observed-supplier-count field rather than active shortage
suppliers. The panel contained 1,997 label-month rows and only 22 non-equal
month-to-month transitions (1.10%), versus 25 active-supplier transitions
(1.25%). The alternative target is therefore also persistence-dominated and
was rejected for news evaluation; no metric or promotion claim was added.

## 2026-09-14 Pass 439: Direct Learned FDA Model Audit

The chronology-safe 12-label FDA evaluator was extended to report direct
learned forecasts separately from validation-selected persistence. Across 124
folds and 4,245 held-out transitions, direct history-only ridge reached
83.8467% within five percent; direct ridge with five dated news features
reached 82.3409%. The direct news model therefore clears the prototype 70%
numeric floor but does not improve over direct history. The direct neural
model reached 0% within five percent and was not promoted. The selected
persistence/news/neural paths and their limitations remain separately reported.

## 2026-09-14 Pass 440: Direct News Forecast Materialization

The 12-row FDA prototype forecast surface was rerun with the direct ridge
news-model prediction included beside the persistence-selected prediction.
The rows retain feature and forecast periods, the five lagged news variables,
and explicit model-method fields. The direct forecast is fit on all observed
transitions for signal inspection; evaluation metrics continue to come only
from the chronological held-out protocol. No live-date claim was added for
labels whose latest source observation is older than the archive maximum.

## 2026-09-14 Pass 441: NDC9 Breadth Check

The FDA evaluator was extended with an NDC9 selection mode. The 12 selected
NDC9s produced 98.19% within-five-percent accuracy for the validation-selected
system across 4,464 held-out transitions. Inspection of source generic names
showed that these NDC9s represent only three active ingredients: leucovorin,
fentanyl, and atropine. The result is retained as a data audit but rejected as
evidence for 10--15 independent drugs.

## 2026-09-14 Pass 442: Regression Verification

The focused regression suite completed with 48 passing tests and 2 failures
in the pre-existing project-status tests. Those failures are caused by absent
or stale ignored artifacts (`research_exhaustion.json` and the expected
14-metric audit), not by the new FDA evaluator. No fabricated artifact or test
contract change was introduced.

## 2026-09-14 Pass 443: Full-Panel Annual News Check

The full 1,548-drug Arkansas CMS statewide panel was tested with eight dated
annual 3DLNews groups. The chronological screen covered five held-out years,
2018--2022, and selected regularization on each validation year. History-only
averaged 77.56% exact / 78.02% balanced five-state accuracy; the raw-news
model averaged 42.25% / 42.29%. The raw annual news features overfit and were
rejected. No result was promoted and no new artifact was retained.

## 2026-09-14 Pass 444: Full CMS NLP Drug-Demand Result

The existing dated 20-column NLP signal surface was aggregated annually with a
log1p transform and evaluated against the full CMS Arkansas state-by-generic
drug panel. Feature year *t* predicted target year *t+1*; training and
validation remained chronological, and test feature years were 2021--2023
(targets 2022--2024). The run covered 1,368 generic drugs and 3,575 held-out
rows across 3 test years. History-only scored 68.5035% exact / 69.2473%
balanced five-state accuracy. History plus NLP news scored 80.6434% /
80.9560%, clearing the prototype 80% three-state gate and improving over
history. The artifact is
`model/artifacts/evaluation/cms_partd_full_news_metrics.json`, reproduced by
`model/scripts/evaluate_cms_partd_news.py`. It remains a demand proxy, not
direct inventory or dispensing truth.

## 2026-09-14 Pass 445: CMS NLP Forecast Surface

The reproducible full-panel CMS/NLP evaluator now writes
`model/artifacts/forecasts/cms_partd_full_news_next_year.csv.gz`. It contains
1,324 Arkansas generic-drug rows using 2024 observed demand and dated NLP
features to predict 2025 low-to-high demand states. Each row retains all 20
news inputs plus the model, source, and state-definition fields. The forecast
surface is explicitly a demand-proxy research output, not direct inventory or
dispensing truth; held-out evaluation remains the separate 2021--2023
chronological result.
