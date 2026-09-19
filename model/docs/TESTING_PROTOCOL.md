# Publishable Testing Protocol

This is the canonical evaluation procedure for the current public evidence
surface. It deliberately evaluates compatible targets separately; it does not
combine quarterly demand, county demand, and supplier shortage into one score.

## 1. Freeze and verify data

The versioned suite is under
`data/targeted_additions/publishable_test_dataset/`. Before fitting any model,
`evaluate_publishable_test_dataset.py` reads `manifest.json`, verifies every
SHA-256 hash, and checks the declared row grain and consecutive target period.
The current suite contains:

- Arkansas NDC9 to next-quarter Medicaid utilization and shortage state;
- Arkansas county-drug to next-year Medicare Part D demand;
- national FDA supplier-NDC to next-month shortage continuation, restricted to
  Arkansas-exposed NDCs.

These are observed public targets. They are not substitutes for unavailable
weekly county-pharmacy inventory or county-by-supplier dispensing labels.

## 2. Split chronologically

No random row split is permitted. The current fixed split is:

| Task | Train | Validation | Held-out test |
| --- | --- | --- | --- |
| Arkansas NDC9 quarter | years through 2019 | 2020 | 2021 onward |
| County-drug year | years through 2020 | 2021 | 2022 onward |
| Supplier-NDC month | years through 2020 | 2021 | 2022 |

Validation selects hyperparameters, transformations, thresholds, and blend
weights. The held-out test is read only for the final report. For model
development, the corresponding rolling-origin evaluators must be preferred:
each fold trains on earlier periods, selects on a later validation window, and
scores the next future window.

## 3. Score against explicit baselines

Numeric demand uses WAPE, MAE, RMSE, sMAPE, R2, and the fraction within 5%,
10%, and 20% relative error. It is compared with persistence and seasonal or
history-only baselines. Binary shortage tasks report raw accuracy, balanced
accuracy, AUROC, AUPRC, Brier score, precision, recall, and F1. Raw accuracy
cannot establish success for rare shortage labels. Five-state demand scores
also report decrease/stable/increase class counts and the persistence state
baseline on every rolling fold, so majority-class accuracy cannot support a
promotion claim.

## 4. Promotion gates

The revised project contract is documented in `PROJECT_GOAL.md`. Each metric
must have at least 25 valid chronological held-out samples, be numeric or a
minimum five-state output, and exceed 65% accuracy. Numeric accuracy is the
fraction within 5% relative error; state accuracy is exact state agreement.
The stronger 75% gates are tracked separately: at least one weekly/monthly
metric, one per-drug metric across 100 or more drugs, and one per-Arkansas-
region metric must each reach 75%. Incompatible targets are never averaged.
Failed or under-labeled proposals are discarded as metrics rather than counted
as context successes. A result must also beat its task baseline under the
task-specific rolling gate.

### Event true-positive route

Event-oriented metrics may additionally qualify when raw accuracy is at least
65% and precision among predicted event cases is at least 80%. The event
definition is fixed before scoring. For a state signal, the evaluator receives
an explicit set of event states; the current five-state high zone is `{3, 4}`.
For a numeric signal, it receives an explicit threshold; predicted values at
or above that threshold are event announcements. A state event is correct only
when the predicted state exactly equals the observed state. A numeric event is
correct when the prediction is within 5% relative error of the observed value;
the observed value does not also have to cross the announcement threshold. The denominator is the number of
predicted event cases, so this is event precision, not recall and not majority
class accuracy. If no event is predicted, the score is undefined and the
metric cannot qualify through this route.

The machine-readable implementation is
`arkansas_pharma_signal.event_accuracy.score_event_predictions`. Every
operational metric records `event_definition` and `event_metrics`; metrics for
which event semantics are not meaningful explicitly use `not_applicable`.
The four coverage gates are evaluated independently: geography, drug,
supplier, and disease/symptom. Each passing category must expose its required
filter key and pass either the original 75% route or the event route.

## 5. Leakage and provenance controls

Feature timestamps must precede target timestamps. Annual or revised sources
are joined only from the last completed period. Right-censored supplier rows
are excluded from continuation scoring, not treated as negatives. Every
promoted result records source URLs, retrieval dates, target semantics,
geography, supplier resolution, cadence, hashes, fold definitions, and the
exact feature list. Proxies such as FluView, ARCOS, NADAC, news, and FDA
shortage records remain explicitly typed as proxies or inputs when they do not
measure the target directly.

The machine-readable metric-library audit additionally requires
`source_license_access`, `feature_timestamp_boundary`, and
`missingness_censoring` for every candidate, including rejected candidates.
This prevents a high score from being separated from the access restrictions,
publication lag, target semantics, or censoring rules that produced it.

## Reproduction

The full repository regression suite is run from the repository root:

```bash
.venv/bin/pytest -q
```

The publishable benchmark suite is run with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_test_dataset.py
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_test_dataset.py \
  --rolling --output model/artifacts/evaluation/publishable_test_suite_rolling_metrics.json
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_model.py --rolling
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_publishable_model.py \
  --rolling --validation-window-years 2 \
  --output model/artifacts/evaluation/publishable_model_rolling_residual_2yval_metrics.json
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_whole_system.py --end-to-end --rolling
```

The first command is the minimum whole-suite verification. A model is not
called publishable from a pointwise result alone when the rolling evaluator is
available. The second command is the public-suite rolling report; it keeps
statewide, county, supplier, and ARCOS distribution tasks separate and applies
no composite score.
The fourth command is the canonical whole-system report: it nests
the compatible public tasks and the layered demand result, and intentionally
sets `composite_score` to `null` because the targets do not share semantics.
It also validates and records the qualified metric feature-store artifact,
including source/output hashes, row counts, and the supplier-preserving join
grain. This component remains separate and is never included in a composite
accuracy score.

The learned evaluator now reports the actual validation-selected convex blend
of raw ridge and deep stack. Existing `publishable_model_rolling*.json`
artifacts generated before this correction are historical and must not be
used as post-correction learned-model evidence until regenerated.

Fixed and rolling artifacts also include `target_adequacy`. It records row
counts, cadence, geography, supplier resolution, right-censoring, and the
observed-shortage-onset sample check. The current suite has only 101
uncensored non-shortage supplier feature rows and 96 observed next-month
shortage positives, so onset modeling is flagged thin. The requested Arkansas
county x supplier x drug x week/month inventory target is explicitly marked
unavailable; no synthetic labels are created.

The suite separately evaluates the DEA ARCOS ZIP3 distribution proxy at the
exact next-quarter grain. It currently contains 14,137 transitions and 15
rolling folds. Its rolling improvement over the best temporal baseline is
16.70%, but this result is not an all-drug pharmacy-demand or inventory claim.
The contractual within-5-percent rate is also reported separately: the
selected ARCOS model currently reaches 26.82% across rolling test rows, while
the best naive baseline reaches 41.05%. WAPE improvement therefore does not
qualify ARCOS under the revised accuracy contract.

The metric-library audit is run with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/audit_metric_library.py
```

The strict project-status audit is run with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/audit_project_status.py
```

It reports proxy-library readiness separately from learned end-to-end
architecture readiness. The project cannot be marked complete when the
qualified proxy gates pass but the learned model's 75% contract or the legacy
publishability gates fail.

It rejects binary-only shortage targets, missing within-5-percent or exact
state accuracy, fewer than 25 held-out rows, fewer than three rolling folds,
and insufficient drug or Arkansas-region coverage for the Part 1 gates. A
candidate is never promoted from pooled raw accuracy when its state space is
imbalanced or its baseline is stronger.

The Part 1 coverage gates are also evaluated at the claimed grain. The
per-drug gate counts only individual drugs with at least 25 held-out rows and
at least 75% exact or within-5-percent accuracy; merely having 100 drugs in a
pooled panel is insufficient. The regional gate similarly requires all five
Arkansas DHS regions to individually meet 75%, not just a pooled regional
average. These counts are recorded as
`drug_count_at_or_above_75_percent` and
`region_count_at_or_above_75_percent` in `metric_library_audit.json`.

The current generated state candidates use five states. Historical three-state
outputs remain compatibility artifacts and are not promotion evidence. The
qualified shortage-pressure candidate uses the observed FDA supplier count
directly as ordered states: `0` none, `1` one supplier, `2` two suppliers,
`3` three suppliers, and `4` four-or-more suppliers.

Legacy three-state rows remain available in evaluator artifacts for comparison,
but `external_metric_output.validate_metric_rows` rejects their
`legacy_three_state_proxy` status. They cannot be serialized into the qualified
metric forecast surface or attached to the metric feature store. This prevents
historical high scores from bypassing the current five-state minimum.

- `ndc_monthly_shortage_pressure_state`: 145 monthly rolling folds,
  419,806 held-out rows, 2,551 NDCs, 92.09% exact accuracy, and 80.38%
  balanced accuracy across five observed states. Its high-state precision is
  72.77%, so it qualifies through the stricter raw route, not the event route.
  It is national FDA supplier-count evidence, not county inventory truth.
- `national_weekly_fluview_respiratory_pressure_state`: 7 weekly folds,
  342 held-out weeks, 84.08% exact accuracy, and 68.93% balanced accuracy.
  It is the qualified national five-state target; the Arkansas three-state
  and Arkansas five-state FluView variants remain rejected compatibility
  artifacts.
- `arkansas_region_annual_demand_state`: 6 annual folds, 21,304 held-out
  region-drug rows, all 5 Arkansas DHS regions and 1,308 drugs, 92.01% exact
  accuracy, and 92.0% balanced accuracy. This is annual context and does not
  replace a weekly/monthly regional inventory label.
- `ndc_monthly_recall_pressure_state`: 10 wider monthly folds, 275,296
  held-out rows, 3,877 NDCs, 99.99% exact accuracy, and 99.57% balanced
  accuracy. The result is persistence-dominated and represents recall
  severity, not pharmacy availability.
- `arkansas_weekly_hospital_influenza_admission_pressure_state`: 4 weekly
  folds, 187 held-out transitions, 74.16% exact accuracy, and 68.76% balanced
  accuracy. This is a hospital-utilization proxy, not pharmacy dispensing.
- CDC VSRR overdose pressure is not in the qualified forecast surface. Its
  current snapshot has 46 apparent rolling folds and 1,023 transitions, but
  every historical row carries the same later `data_as_of` vintage. Those
  apparent scores are therefore not publishable chronological evidence and
  are rejected rather than counted toward the metric library.

The operational forecast surface is built with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/build_qualified_metric_forecasts.py
```

The resulting `qualified_metric_forecasts.csv.gz` currently contains 46,911
rows across 14 qualified targets, including 1,585 current shortage NDC-month
rows, 3,032 current recall NDC-month rows, 52 therapeutic-class rows, and
3,680 current region-drug annual rows. It contains no hospital-influenza
target; the statewide NSSP influenza target is represented as a current
disease-pressure row.
Stale NDC and region-drug pairs are excluded; no missing supplier or county
allocation is synthesized.
`source_freshness` preserves the latest observation period used as a feature;
it is not rewritten to the forecast period, so source publication lag remains
visible to downstream consumers.
Qualified forecast horizons are cadence-specific day estimates: weekly `7`,
monthly `30`, quarterly `91`, and annual `365`. They describe the period
between the feature observation and the predicted period and must be positive.

Qualified persistence rows intentionally set `confidence`, `risk_score`, and
interval bounds to null. They are point state forecasts, not calibrated
probabilities or prediction intervals. Every row carries
`uncertainty_status=not_estimated` and `calibration_status=not_calibrated`.
These fields must remain explicit until a leakage-safe calibration procedure
is evaluated on held-out folds; consumers must not interpret the old `0.50`
placeholder as confidence.

Incremental utility experiments are separate from metric promotion:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_signal_utility.py
```

The current FluView/wastewater result is negative: two folds, 162 held-out
rows, and a -7.48 percentage-point balanced-accuracy change after adding
wastewater. It is retained as context only.

The recall-to-shortage utility result is mixed: 10 folds, 12,188 held-out
rows, 240 NDCs, and a +0.91 percentage-point mean balanced-accuracy change
after adding recall state, but not every chronological fold improved. It is
not used as an incremental feature claim.

The FluView/hospital-admission utility experiment is also negative: 602
aligned weekly rows, four folds, and a balanced-accuracy change from 84.07%
to 83.11% (delta -0.96 percentage points). The hospital metric remains a
standalone utilization proxy, not a demonstrated incremental feature.

The public Arkansas NDC-demand utility experiment is also required for any
claim that external context improves a regression. After correcting the NDC9
normalization, it uses four chronological folds and 64,116 held-out
NDC-quarter rows. History-only Ridge mean WAPE is `0.2588` versus `3.6351`
after adding 21 prior disease/news variables (delta `+3.3764`), so the
current context bundle is rejected for incremental utility. The disease-only
family changes WAPE by `+0.01848` and the news-only family by `+2.54259`;
neither family passes the all-fold improvement check. The matched nonlinear
control reaches mean WAPE `0.2401` history-only versus `0.2412` with all
context variables (delta `+0.00116`), again with no
all-fold context improvement. Run it with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_nonlinear_demand_utility.py
```
The Ridge family ablation is run with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_public_demand_context_utility.py
```

The monthly HHS pharmacy-NDC ablation also includes the lagged national
RESP-NET RSV rate, shifted by one calendar month before joining the target.
It uses 59 rolling monthly folds and 13,552 held-out drug-month rows. The
history-only mean WAPE is `0.59325`; the six-feature context bundle reaches
`0.59956` (delta `+0.00630`). Within-5-percent coverage changes from
`0.08247` to `0.08319`, but not every fold improves, so RESP-NET is rejected
as an incremental utility feature. It remains an independently qualified
external disease-pressure proxy. Run it with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_hhs_monthly_context_utility.py
```

The Arkansas APCD claim-count report was tested as an additional lagged
market-activity feature over its actual 2018-07 through 2021-06 coverage. The
experiment uses 12 chronological folds and 2,382 held-out HHS drug-month rows.
The seven-feature context bundle reaches mean WAPE `0.75392` versus the
history-only `0.75742` (delta `-0.00350`), but does not improve every fold and
within-5-percent coverage falls from `0.09258` to `0.09114`. Relative to the
existing six-feature bundle on the same folds (`0.75332`), APCD activity
worsens mean WAPE by `0.00060`. It is therefore retained as an independently
qualified activity proxy but rejected as demonstrated incremental utility for
the public HHS regression. Run it with:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/evaluate_apcd_monthly_context_utility.py
```

The historical weather ablation is run with:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli --root . evaluate-weather-utility
```

It compares the existing FluView autoregressive/seasonal state model with the
same chronological folds plus current-week county-aggregated weather. The
current artifact contains seven folds and 342 held-out weeks; the weather
augmentation is rejected when its mean balanced-accuracy delta is negative.

The hospital-influenza weather ablation is run with:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli --root . evaluate-hospital-weather-utility
```

It uses the hospital target's native rolling-origin folds and joins weather to
the current hospital week before predicting the next week. The current
artifact contains four folds and 187 held-out weeks.

The FEMA disaster ablation is run with:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli --root . evaluate-hospital-disaster-utility
```

Declarations are aggregated by current Monday week and county events are
summarized statewide. No-event weeks receive structural zero values; the
source is not interpolated. The current artifact contains four folds and 187
held-out weeks.

The metric-library audit now records `baseline_skill` for every candidate. This
includes the mean model-minus-persistence balanced-accuracy delta, whether all
folds beat persistence, and an `incremental_utility_status`. A
`qualified_proxy` therefore means that the external target is predictable and
testable; it does not mean that the signal has improved a private pharmacy
inventory regression. Only a separate leakage-safe utility experiment can
establish that stronger claim.

Qualified forecast rows can be converted to regression covariates with
`metric_feature_store.attach_qualified_metric_features`. The adapter preserves
`forecast_period`, geography, drug, and supplier keys, records source metadata,
rejects unqualified targets or duplicate grains, and emits explicit missingness
indicators without imputing a missing signal. It never broadcasts a national
NDC metric to an Arkansas county or pharmacy automatically. The attachment
function is an integration seam; the current public benchmarks do not claim
pharmacy-level utility because they lack a matching private inventory target.
The default feature-store grain includes `supplier`; supplier-by-NDC rows
therefore remain separate features rather than being silently collapsed.
The adapter also supports an explicit `join_mode="context"`: only targets
with metadata-declared source and destination keys may project to a broader
local frame. Missing destination geography keys remain all-one missingness;
they are never inferred or filled.

To build the reproducible artifact used by downstream regressions:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/build_metric_feature_store.py
```

This validates `qualified_metric_forecasts.csv.gz`, writes
`qualified_metric_feature_store.csv.gz`, and records source/output hashes and
feature metadata in `qualified_metric_feature_store.json`.
For downstream local frames, pass `join_mode="context"` to
`attach_qualified_metric_features` only when the declared projection policy is
appropriate. The generated metadata records which features projected and
which were skipped because local geography, supplier, or ZIP3 keys were not
available.

The modular model's optional metric-context vector has 16 positions: one value
and one missingness indicator for each historical context feature, in the
declared feature order. The opt-in historical training ablation uses only the dated FDA shortage archive
and FDA enforcement event sources through the feature origin, with exact-NDC
joins. Unmatched NDCs and sources without a point-in-time vintage remain
missing; current-only qualified forecast rows are never joined to past labels.
The current panel coverage is sparse (1,826 of 44,071 rows), so context
utility must be established by an explicit ablation and cannot be inferred
from feature presence. The context path is disabled by default after the
current ablation failed to improve the held-out demand objective.

Before those rows are written, `external_metric_output.validate_metric_rows`
rejects unknown targets, non-qualified promotion statuses, non-finite values,
empty period/semantics fields, and state predictions outside the required
five-state set `0`, `1`, `2`, `3`, `4`. Legacy three-state artifacts do not pass
this gate under the revised contract. The historical feature store covered
legacy targets, including the supplier x NDC recall state and the quarterly
ARCOS ZIP3 distribution-pressure state; the operational store now contains
14 qualified targets. Supplier-filterable
rows use a dedicated `supplier` key; `labeler` is not used as a substitute.
The feature-store adapter reapplies the value and target checks when rows are
loaded directly from disk, so callers cannot bypass the serializer contract.

### Modular state-head contract

The modular trainer must not report a five-state score unless its thresholds
were fit on the training slice and all scored labels are in `{0, 1, 2, 3, 4}`.
The promoted state semantics are five ordered next-period log-demand-change
bands;
absolute demand level is only an explicit ablation.
`state_thresholds` and `state_loss_weight` are persisted with checkpoints;
`validation_state` and `test_state` contain exact and balanced accuracy. A
degenerate training slice uses the unavailable sentinel `-1`, skips the state
loss for those rows, and reports zero scored state rows rather than inventing
a class boundary. The artifact also scores a persistence state comparator
using the same training-fitted thresholds. This is a structural validity
check, not evidence that the quarterly demand state is a weekly or
pharmacy-level inventory target. It additionally scores the numeric point
forecast after applying the same state thresholds, so classifier-head and
numeric-forecast state performance cannot be conflated.

### Live weather geography test

The weather refresh path is tested at point/date grain. Tests verify that an
explicit five-digit Arkansas county FIPS survives the hourly-to-daily
aggregation, that non-Arkansas FIPS is rejected, and that an untagged point
does not acquire a county value. A single point is not eligible for county
metrics without an independently sourced county-to-point mapping.

The mapping build is accepted only when it contains exactly 75 unique
Arkansas county FIPS values and finite latitude/longitude for every row. The
batch refresh requires one returned NWS county key per registry row; partial
responses fail the refresh and are not written as a complete county surface.

Historical weather rows are tested separately from forecast rows. The loader
must deduplicate station/date records and preserve missing measurements. The
aggregator must emit only observed periods, retain the county/station join
keys, and expose observed-day count and coverage ratio. It must not fill gaps
or claim that nearest-station values are county ground truth.

The `attach_weather_context` seam is tested with an exact
`cadence + period_start + period_end + county_fips` key. A target county with
no matching weather period must receive an explicit availability flag and
missing values; it must not receive another county's context.
