# MediTrack Revised Project Goal

## Scope

MediTrack converts freely accessible, real-time-usable information into a
filterable library of external signals that can be joined to individual
pharmacy inventory-regression models. It does not need to observe or claim
local pharmacy on-hand inventory. Signals may describe demand pressure,
distribution, shortage exposure, acquisition cost, disease activity, weather,
news, economic conditions, policy, recalls, or other pharmacy-relevant state.

The existing code-level layers remain required. News and structured inputs must
pass through independently testable extraction, geography, temporal, economic,
political, and cross-domain connection layers before final signal heads.
Hard-coded decision rules are not acceptable as the predictive architecture.

## Active Prototype Expansion

The current expansion keeps this architecture and targets one of two surfaces:

1. supply or demand predictions for at least 10-15 drugs in Arkansas, an
   Arkansas county/region, or the United States; or
2. diagnosis/occurrence-rate predictions for at least 5-8 diseases in one of
   those geographies.

News remains a primary external input and must be evaluated as part of the
larger NLP/neural signal layer. For this prototype, a state output must have
at least three ordered states and reach 80% exact accuracy, or a numeric output
must reach 70% of observations within 5% relative error. These are prototype
targets, not permission to relax the stricter established metric-library
contract below. Every claim still requires real historical data, chronological
held-out testing, per-target counts, baselines, and a logged promotion
decision.

The current CMS annual Arkansas state-by-generic-drug result meets the
multi-drug state-accuracy scale as a demand proxy. A new full-panel,
news-enabled evaluation covers 1,368 generic drugs and 3,575 held-out rows;
the 20 NLP-signal model reaches 80.64% exact / 80.96% balanced five-state
accuracy versus 68.50% / 69.25% history-only. A chronology-safe FDA shortage
screen now provides 12
generic/formulation supply-proxy labels at 98.39% within-5% accuracy across
4,245 held-out next-month transitions. The FDA regression, persistence, and
existing NumPy neural paths all tie; that target remains persistence-dominated.
The CMS annual result above is the current news-enabled prototype candidate.
Its runner also materializes 1,324 Arkansas generic-drug signals for the 2025
forecast year from 2024 observations; these remain demand proxies rather than
direct inventory forecasts.

The same evaluator now also reports direct learned models without the
persistence fallback: history-only ridge reaches 83.85% within-5% accuracy and
ridge with five news features reaches 82.34% across the same 4,245 transitions.
This clears the prototype numeric floor for the learned supply model, but news
slightly reduces accuracy and the neural path does not clear the floor.

The statewide ARCOS expansion now evaluates 15 controlled-substance codes with
quarterly Arkansas news. Across 9 chronological folds, the news candidate
reached 62.76% exact / 53.89% balanced three-state accuracy, versus 43.40% /
33.33% for persistence. This is genuine news lift but fails the 80% state
gate; ARCOS remains a distribution proxy rather than pharmacy inventory.

## Final Metric Contract

Every established metric must satisfy all of the following:

1. **Filterability:** output rows expose stable keys for the applicable period,
   drug or drug class, Arkansas county/region or other geography, supplier when
   available, metric name, value, uncertainty, freshness, and provenance.
2. **Operational input access:** every model input is free and obtainable from a
   documented public source at application time, with publication lag recorded.
3. **Pharmacy relevance:** a research paper, official data definition, or clear
   causal/operational connection must support its use as an inventory-regression
   covariate.
4. **Temporal usefulness:** weekly or monthly prediction is preferred. Annual
   or quarterly metrics are allowed only when they provide a useful lower-
   frequency context signal and are labeled accordingly.
5. **Testability:** the target must have at least 25 valid held-out test samples
   under a chronological, leakage-safe protocol. Metrics without adequate
   public labels are discarded as model variables, not treated as successes.
6. **State or numeric output:** numeric outputs are the preferred representation
   and should be used whenever the source supports a meaningful quantity. A
   categorical state output must have at least five ordered or clearly defined
   states. Binary, three-state, and four-state outputs do not satisfy the final
   contract, though historical experiments may remain documented as rejected
   or legacy evidence.
7. **Uncertainty honesty:** uncalibrated point/state forecasts must expose
   `uncertainty_status=not_estimated` and `calibration_status=not_calibrated`;
   confidence values and intervals may be populated only after held-out
   calibration or coverage testing.

## Accuracy Gates

Numeric accuracy means the prediction is within 5% of the observed target.
State accuracy means the predicted state exactly matches the observed state.
All results must be reported by fold, not only as a pooled average.

For event-oriented signals, a second acceptance route is allowed without
reducing the five-state minimum. A state signal must declare its event zone;
for the current five-state ordered outputs, states `3` and `4` are the high
event zone. A numeric signal must declare a numeric event threshold. The
event score is precision among rows whose prediction announces that event:
state predictions must exactly match the observed state, while numeric
predictions must be within the existing 5% relative-error tolerance. For a
numeric signal, the observed value does not also need to cross the event
threshold: the threshold defines when the model announces risk, and the error
margin defines whether that announcement was correct. Empty predicted-event
sets are undefined and never count as 80%.

The event route requires both raw accuracy of at least 65% and event true-
positive precision of at least 80%. The original 75% raw accuracy route
continues to qualify a metric without relying on event precision. Balanced
accuracy remains reported for state outputs and the minimum five-state
coverage rule remains mandatory.

Current event definitions are:

| Metric | Event zone | Event applicable |
| --- | --- | --- |
| FDA shortage supplier count | numeric count `>=1` | yes; within 5% of observed count |
| FDA NDC shortage-pressure state | states `3` or `4` (three or four-plus active shortage suppliers) | yes, high supplier-pressure state |
| NADAC observed price | none; economic level, not an event | no |
| Arkansas regional demand state | states `3` or `4` | yes, high regional demand |
| Arkansas county demand state | states `3` or `4` | yes, high county demand |
| Arkansas statewide Part D drug demand state | states `3` or `4` | yes, high drug demand |
| ATC therapeutic-class demand state | states `3` or `4` | yes, high therapeutic-class demand |
| Arkansas APCD claim-activity state | states `3` or `4` | yes, high observed claim activity |
| FDA NDC recall severity | numeric severity `>=1` | yes; within 5% of observed severity |
| FDA supplier-NDC recall severity | numeric severity `>=1` | yes; within 5% of observed severity |
| ARCOS ZIP3/drug distribution state | states `3` or `4` | yes, high distribution pressure |
| National FluView respiratory state | states `3` or `4` | yes, high respiratory pressure |
| National RESP-NET RSV hospitalization state | states `3` or `4` | yes, high hospitalization pressure |
| Arkansas NSSP influenza ED state | states `3` or `4` | yes, high influenza ED pressure |

These definitions are part of the target contract and are scored only on
chronological held-out predictions. They do not assert that a proxy directly
measures pharmacy inventory.

Part 1 requires at least:

- one weekly or monthly metric with at least 75% accuracy;
- one per-drug metric meeting 75% accuracy across 100 or more drugs; and
- one per-Arkansas-region metric meeting 75% accuracy.

Each of the following signal categories must also have at least one metric
that passes either the 75% raw route or the 65% raw plus 80% event route:
geographically divided output, drug-divided output, supplier-divided output,
and disease/symptom-divided output. A category is not satisfied by a metric
that lacks the relevant filter key.

Part 3 requires every established metric to exceed 65% accuracy under its
qualified test protocol. A metric that fails this floor is removed or retained
only as an explicitly non-predictive context input, never counted toward the
metric library.

These gates are applied separately by compatible target and grain. No
composite accuracy may hide failure in a particular drug, region, period, or
signal family.

Historical three-state evaluators may remain in the repository as rejected
research evidence, but their rows are not promoted, serialized as qualified
metrics, or attached as model features. Only numeric or five-state outputs can
enter the current operational metric surface.

## Completion Condition

The goal is not complete until a documented, research-backed search through
public government sources, GitHub, Kaggle, and relevant research-paper datasets
has exhausted additional pharmacy-relevant metrics that have both sufficient
labels and a proper test procedure. “Unavailable local inventory truth” is no
longer a blocker; unsupported or untestable proposed signals remain excluded.

Completion is audited in three layers: qualified external-proxy readiness,
learned end-to-end architecture readiness, and research exhaustion. The older
annual/binary/modular publishability checks remain visible as compatibility
diagnostics, but are not active completion gates when their target contract
has been superseded by the five-state or numeric protocol. Passing proxy
coverage or accuracy gates alone does not complete the project while research
exhaustion is unproven.

## Required Reporting

Each metric artifact must report its source and license/access note, target
semantics, feature timestamp boundary, cadence, geography, drug/supplier
resolution, missingness and censoring treatment, sample count, baseline, fold
metrics, uncertainty/calibration where applicable, and promotion status.
The metric audit must also explicitly mark whether the target is a direct
pharmacy observation. Current public targets are required to declare
`target_is_direct_pharmacy_observation=false` and
`target_claim_scope=proxy_only_not_pharmacy_inventory`; a proxy score must not
be presented as a fill, backorder, supplier allocation, or on-hand label.
