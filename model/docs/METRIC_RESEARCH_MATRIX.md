# Pharmacy-Relevant Metric Research Matrix

## Review note — 2026-09-14

The matrix is a promotion ledger, not a count of available features. The
current code exposes more candidate signals than qualified metrics, including
ARCOS distribution, disease surveillance, recall/shortage, news, weather,
economics, trade, and supplier context. New non-Arkansas signals may enrich an
Arkansas forecast only when source geography, publication time, exposure
mapping, and held-out results are preserved.

This matrix is the gate between a plausible feature and an established
MediTrack metric. A source can be useful as a live input without being a valid
held-out target. The target must have at least 25 chronological test samples
and must satisfy the revised contract in `PROJECT_GOAL.md`.

## Evidence From Research

| Evidence | What it supports | What it does not provide |
| --- | --- | --- |
| [Predicting drug shortages using pharmacy data and machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10009839/) | Pharmacy demand-side data and reported shortage data can predict impactful shortages; the study reports a one-month-ahead four-class shortage task | The pharmacy data and labels are not a freely downloadable Arkansas dataset |
| [NRC archive of the same shortage study](https://publications-cnrc.canada.ca/eng/view/object/?id=f87b692f-1927-4188-b7c5-496e28c76a05) | A four-state shortage class reached 69% accuracy one month ahead using data from 22 Canadian pharmacies | It is Canadian, pharmacy-proprietary, and does not establish Arkansas performance |
| [Medication Sales and Syndromic Surveillance](https://wwwnc.cdc.gov/eid/article/12/3/05-0573_article) | Weekly medication sales can forecast influenza-like illness at national and regional scales; reported correlations were 0.85–0.96 one to three weeks ahead | The underlying sales panel is not an open Arkansas pharmacy feed |
| [Value of Pharmacy-Based Influenza Surveillance](https://pmc.ncbi.nlm.nih.gov/articles/PMC4604937/) | Near-real-time antiviral dispensing is a plausible local disease-demand proxy | The Ontario dispensing data came from a large private pharmacy panel |
| [Climate-informed respiratory pharmaceutical demand](https://link.springer.com/article/10.1007/s10651-026-00731-8) | Weekly respiratory pharmaceutical demand is associated with weather and neighboring-region spillovers | The 1,200-pharmacy Greek ERP panel is not public or Arkansas-specific |
| [ShortageSim dataset and code](https://github.com/Lemutisme/ShortageSim) | 2,925 FDA shortage events support national shortage trajectory and policy-simulation research | It is not local pharmacy inventory, and simulated trajectories are not observed pharmacy outcomes |
| [New York Fed GSCPI methodology](https://www.newyorkfed.org/research/staff_reports/sr1017) and [pharmaceutical inventory-driver research](https://pmc.ncbi.nlm.nih.gov/articles/PMC10091338/) | A public monthly index measures global transportation/manufacturing pressure, while pharmaceutical inventory research identifies disruption conditions as inventory drivers | GSCPI has no drug, supplier, or Arkansas allocation and must not be treated as pharmacy inventory truth |

## Public Candidate Status

The Arkansas APCD monthly pharmacy-claim-count report is a qualified
five-state activity proxy by reporting entity. Its local normalized panel has
1,140 rows across 37 entities and 36 months (2018-07 through 2021-06). A
fit-history quantile persistence evaluation produced 488 held-out transitions
across three non-overlapping rolling folds: 91.83% exact state accuracy,
92.48% balanced accuracy, and 95.73% precision for predictions in states 3-4.
This target is
not drug-specific and has no supplier or inventory outcome; it is exposed only
as an activity context signal.

The HHS NDC panel was also linked to NLM RxNorm/RxClass historical NDC
relations. Claims are fractionally allocated across multiple three-character
ATC groups rather than assigned by an arbitrary first relation. The scored
view contains 74 mapped groups, of which 71 have complete scored transitions; the next-month five-state evaluator used 57
chronological folds and 3,104 held-out transitions, reaching 73.84% exact and
74.56% balanced accuracy. This is a therapeutic-class demand proxy for the
mapped NDC subset, not a diagnosis or inventory label; unmapped NDCs are
excluded, mapped NDCs cover 42.41% of HHS claim lines, and the crosswalk is
separately manifest-backed.

| Candidate metric | Public target grain | Input availability | Testability | Status |
| --- | --- | --- | --- | --- |
| FDA shortage continuation/onset risk | National NDC9 x supplier x month | FDA/openFDA daily | Existing monthly panel; onset slice is thin | Keep as supplier-context metric; do not claim local inventory |
| FDA drug-recall pressure | National recalled product/NDC x event date | FDA recall downloads/openFDA daily | Termination-aware NDC-month panel built from local FDA enforcement archive | Qualified numeric recall-severity proxy; not pharmacy availability |
| NADAC acquisition-cost pressure | National NDC x week | CMS weekly and annual public data | Reconstructed 2021-2025 Arkansas-exposed panel; 3 rolling folds, 1.11M held-out transitions | Rejected as a learned metric: persistence WAPE 0.00135 versus model 0.00368; retain as research-only price context |
| NADAC relative acquisition-cost change state | National NDC x week | CMS weekly public data | 3 chronological folds and 1,113,045 held-out transitions; fit quantiles collapse at zero and only two effective movement states are observed | Rejected under the five-state contract; the level-price proxy remains qualified |
| Arkansas FluView respiratory pressure | Arkansas/state x week | CDC/Delphi weekly | Seven chronological folds and 342 held-out weeks; five-state exact accuracy 74.42% but balanced accuracy 56.72% | Rejected under the five-state balanced-accuracy floor; retain as input/context and legacy three-state evidence |
| National FluView respiratory pressure | United States national x week | CDC/Delphi weekly | Seven chronological folds and 342 held-out weeks; five-state exact accuracy 84.08% and balanced accuracy 68.93%, with all five states represented | Qualified weekly upstream respiratory proxy; not pharmacy dispensing or Arkansas allocation |
| National RESP-NET RSV hospitalization pressure | United States RESP-NET aggregate x week | CDC RESP-NET Socrata API | Four chronological folds, 187 held-out weeks; five-state exact accuracy 83.31%; high-state precision 84.06%; observed weekly rate per 100,000 | Qualified weekly disease-pressure proxy; not Arkansas-local and not pharmacy dispensing |
| Arkansas statewide Part D demand by generic drug | Arkansas state x generic drug x year | CMS public geography-by-drug annual files | Six chronological folds, 6,998 held-out rows, 1,548 drugs; 89.09% exact / 89.35% balanced five-state accuracy; states 3-4 precision 95.30% | Qualified annual drug-filterable demand proxy; no supplier or county allocation and not inventory truth |
| Arkansas pharmacy NDC monthly demand | Arkansas pharmacy-taxonomy billing NPI x NDC x month, aggregated statewide | HHS Medicaid Provider Spending by NDC, joined to local NPPES | 57 rolling folds, 12,998 held-out transitions, 51.41% exact, 9.10% within 5%, 65.52% high-zone precision | Rejected as a promoted metric; retained as training-only pharmacy demand context because HHS suppression prevents zero imputation and the learned/state gates fail |
| Arkansas county pharmacy NDC monthly demand | County x NDC x month inferred from pharmacy billing NPI practice city | HHS Medicaid Provider Spending by NDC + NPPES + Census-geocoder crosswalk | 57 rolling folds, 8,813 held-out transitions, 43.59% exact, 11.06% within 5%, 53.14% high-zone precision | Rejected; county mapping is usable as context but the forecast is not sufficiently predictable |
| Arkansas pharmacy NDC labeler monthly demand | NDC labeler code x month, aggregated statewide | HHS Medicaid Provider Spending by NDC + NPPES pharmacy taxonomy | 21 rolling folds, 2,556 held-out transitions, 58.79% exact, 10.28% within 5%, 75.31% high-zone precision | Rejected; labeler is not a supplier and the raw event-route floor fails |
| Arkansas statewide monthly pharmacy claim-line demand | Arkansas statewide x month, aggregated from pharmacy-taxonomy billing providers | HHS Medicaid Provider Spending by NDC + NPPES pharmacy taxonomy | 57 rolling folds, 57 held-out months, 63.16% exact, 7.02% within 5%, 73.91% high-zone precision | Rejected; fails both the 65% raw floor and 80% elevated-event route |
| Arkansas monthly observed pharmacy-provider capacity | Arkansas statewide x month, maximum observed pharmacy-taxonomy billing-provider count | HHS Medicaid Provider Spending by NDC + NPPES pharmacy taxonomy | 57 rolling folds, 57 held-out months, 56.14% exact, 10.53% within 5%, 52.17% high-zone precision | Rejected; provider observation count is a context measure, not inventory capacity, and fails both routes |
| Monthly HHS pharmacy-demand external-context utility | Arkansas NDC x month; history-only versus prior-month shortage, news, and respiratory context | HHS NDC panel plus FDA shortage archive, Arkansas news metadata, and CDC FluView | 59 rolling folds, 13,552 held-out transitions, history-only WAPE 0.5933 versus context WAPE 0.5996; context delta +0.0063 and did not improve every fold | Rejected as demonstrated incremental utility; inputs remain available for future private-inventory validation |
| National-minus-Arkansas FluView divergence | Paired United States/Arkansas x week | CDC/Delphi weekly | Seven chronological folds and 341 held-out paired transitions; five-state exact accuracy 54.94% and balanced accuracy 43.36% | Rejected; derived divergence is not sufficiently predictable |
| Arkansas wastewater respiratory pressure | Site/region x week | CDC NWSS public data | Local 2022-2026 site panel has 3,397 consecutive transitions; persistence raw state accuracy exceeds 84% for collapsed influenza states, but balanced accuracy is below 60% | Rejected under balanced-state safeguard; retain as input/context candidate |
| Arkansas regional wastewater pressure | Arkansas DHS region x pathogen x week, derived from county-served sites | CDC NWSS public data plus Census county naming and the existing Arkansas region crosswalk | Level state screen: influenza 195 rows, 80.00% raw / 27.31% balanced / 75.00% event precision; change-state screen: 62.05% raw / 20.97% balanced / 64.77% event precision; RSV and SARS-CoV-2 are also below the gates | Rejected under balanced/event gates; retained as a reproducible research candidate, not a qualified regional metric |
| National wastewater pathogen pressure | United States pathogen x week, derived from public site means | CDC NWSS public data | Five-state event screen: influenza 58.58% raw / 73.21% high-zone precision, RSV 56.41% / 51.28%, SARS-CoV-2 76.92% / 61.29%; all have 5 states but none reaches the 80% event route | Rejected; national disease signals remain research-only context |
| Weather disruption/respiratory pressure | Station/county/region x day or week | NOAA public data | Requires matched public pharmacy-relevant target | Input family; not independently established yet |
| Historical weather as FluView feature | Arkansas statewide x week | Local NOAA/NCEI daily summaries mapped to Census county points; live NWS refresh | 7 rolling folds, 342 held-out weeks; baseline mean balanced accuracy 75.48%, weather-augmented 72.78%, delta -2.70 percentage points | Rejected as incremental utility for the current FluView proxy; retain as an input candidate for other targets |
| Historical weather as hospital-influenza feature | Arkansas x week | Local NOAA/NCEI daily summaries mapped to Census county points | 4 rolling folds, 187 held-out weeks; baseline mean balanced accuracy 73.85%, weather-augmented 62.99%, delta -10.86 percentage points; every fold worsened | Rejected as incremental utility for the qualified hospital proxy |
| FEMA disaster pressure as hospital-influenza feature | Arkansas x week, sourced from county declarations | FEMA OpenFEMA declarations, current-week county aggregation | 4 rolling folds, 187 held-out weeks; baseline mean balanced accuracy 73.85%, disaster-augmented 72.21%, delta -1.64 percentage points | Rejected as incremental utility; retain sparse county event context |
| Arkansas news disruption index | Arkansas x month | Public article metadata and live news sources | Requires a direct target or validated event label | Input family; no standalone metric promotion |
| DEA ARCOS distribution pressure | Arkansas ZIP3 x controlled drug x quarter | DEA public reports | 14,137 transitions and 15 rolling folds | Qualified quarterly distribution proxy; not supplier-specific |
| DEA ARCOS distribution pressure state | Arkansas ZIP3 x controlled drug x quarter | DEA public reports | 11,503 held-out transitions, 15 rolling folds, 39 controlled-substance codes and 85 ZIP3s; fit-only five-quantile states produce 90.01% exact and 89.38% balanced accuracy | Qualified quarterly five-state distribution proxy; persistence-dominated and not pharmacy inventory |
| CMS county/drug demand | Arkansas county x drug x year | CMS public data | Existing annual county artifact | Qualified annual demand context; below weekly/monthly priority |
| CMS county/drug demand five-state output | Arkansas county x drug x year | CMS public data | 6 chronological folds, 139,197 held-out rows, 1,326 drugs, 73 counties; 74.33% exact / 74.12% balanced accuracy | Qualified annual five-state demand proxy; 28 counties meet the 75% exact-accuracy threshold; no supplier or inventory claim |
| CDC dispensing-rate maps | State/county annual opioid, buprenorphine, and naloxone classes | CDC public tables based on IQVIA Xponent | Only three broad drug groups and annual 2019-2024 coverage | Retain as external research context; not a sufficiently detailed new metric |
| CMS Medicare Part D opioid prescribing by geography | State/county/ZIP annual opioid prescribing rates | CMS public-use 2024 CSV | One annual vintage was available in the current public file; no 25-sample chronological forecast test can be constructed | Rejected as a new metric; useful descriptive context only |
| CDC/CMS outpatient antibiotic dispensing | State or region x antibiotic class x year, with age/sex/provider-specialty variants | CDC annual reports and public CMS Part D extracts | Official CDC documentation describes annual retail-pharmacy volumes; 2011-2024 provides fewer than 25 chronological samples per state/class target and no weekly/monthly Arkansas target | Rejected as an established metric; retain as annual context only |
| ONC/Surescripts county e-prescribing activity | Arkansas county x e-prescribing adoption/use period | ONC public county CSV/API | Official dataset covers December 2008-April 2014 only; it is not a real-time input today, has no NDC/drug target, and measures network adoption rather than dispensing volume | Rejected under live-input and drug-specific target criteria; retain as historical research evidence |
| Arkansas APCD monthly pharmacy claim counts | Potential dispensing pharmacy/NDC/date x month, subject to approved release | [Arkansas APCD](https://www.arkansasapcd.net/Home/) public documentation and requester portal | Official materials confirm pharmacy claims and current releases, but claim records require an approved data request; no unrestricted downloadable panel was found, so no reproducible 25-sample forecast test exists | Rejected for the public-data architecture; a future approved data partnership could materially improve local targets |
| Arkansas PDMP controlled-substance dispensing | Arkansas pharmacy/prescription x week | Arkansas Department of Health PDMP | The program contains dispensing records and weekly reporting, but access is authorized rather than an unrestricted public dataset | Rejected for the public-data architecture; a future approved data partnership could materially improve local targets |
| CDC NNDSS Arkansas disease pressure | Arkansas disease label x week, current-week provisional cases | CDC public weekly table; current local extract begins in 2025 and uses `m1` current-week counts | Screened with three chronological short-window folds: 200 held-out rows across 9 labels, no label reached 65% balanced accuracy and only one label exhibited all three states | Rejected; insufficient history and predictive performance |
| CDC acute respiratory illness activity | Arkansas statewide symptom activity x week | CDC public Socrata API, live weekly | Current API response contains one Arkansas observation (`2026-08-08`), so no 25-sample chronological test can be constructed | Rejected until historical Arkansas vintages are publicly accessible |
| CDC Arkansas hospital respiratory admissions | Arkansas x COVID/flu/RSV x week | CDC NHSN HRD public API, weekly | Legacy three-state influenza has 4 folds and 187 held-out transitions; the required five-state screen is 59.09% exact / 47.79% balanced with 73.12% high-zone precision; numeric influenza is 9.32% within 5% | No final disease/symptom metric promoted; COVID/RSV also fail state coverage or balanced-accuracy safeguards |
| CDC NSSP ED trajectories | Arkansas county/HSA x respiratory pathogen x week | CDC public NSSP API, weekly | 15,352 Arkansas rows across 76 county labels and 23 HSA labels; regional pathogen percentages are absent and regional trend labels are `Data Unavailable` | Rejected for regional metric; statewide rows are not a new regional target |
| CDC NSSP statewide pathogen ED pressure | Arkansas statewide x COVID/flu/RSV x week | CDC public NSSP API, weekly | Influenza: 10 chronological folds, 130 held-out rows, all five states, 73.08% exact accuracy and 80.00% high-zone precision; COVID 68.46% / 55.56%, RSV 63.85% / 81.25% | Influenza qualified through event route; COVID/RSV rejected; output retains pathogen filter |
| CDC provisional county overdose deaths | Arkansas county x rolling-12-month overdose deaths x month | [CDC VSRR Socrata dataset](https://data.cdc.gov/National-Center-for-Health-Statistics/VSRR-Provisional-County-Level-Drug-Overdose-Death-/gb4e-yj24), public API | County-level monthly rows exist, but counts are suppressed for many county-months, values are rolling 12-month provisional counts, and the available archive is one `data_as_of=2026-07-05` snapshot with no historical vintages | Rejected as a publishable forecast target: a leakage-safe historical next-month test cannot be reconstructed; retain only as a clearly lagged live-context candidate |
| CDC FluVaxView retail-pharmacy vaccination volume | US national x pharmacy setting x week | CDC public table based on IQVIA pharmacy claims | Direct pharmacy activity, but the downloadable table currently ends with the 2023-2024 season and has no current Arkansas pharmacy allocation | Research evidence only; rejected as a real-time metric/input |
| CMS Part D formulary/pharmacy-network PUF | Monthly plan x NDC x service-area county/NPI network | CMS monthly public-use files; NDC formulary, county locator, NPI network, cost-share and dispensing-fee fields | Monthly cadence and rich pharmacy-relevant fields, but the CMS terms restrict derivative works/competitive offerings; it is plan access/network structure, not observed dispensing or inventory | Rejected for the current model pending legal-use clearance; do not download into training artifacts |
| FDA FAERS adverse-event pressure | National drug/NDC x quarter | FDA quarterly FAERS/AEMS files or openFDA; public domain/CC0, product and reaction reports | Long history and drug identifiers, but quarterly cadence, at least three-month publication lag, voluntary reporting and usage-dependent reporting volume, no Arkansas pharmacy geography, and no direct dispensing/stock label | Rejected as an established inventory covariate; retain as future pharmacovigilance context only |
| Synthetic Kaggle pharmacy sales | Non-Arkansas simulated daily sales | Downloadable but synthetic/uncertain provenance | Can test algorithms only | Excluded from publishable metric targets |
| Public GitHub pharmacy sales examples | Usually non-Arkansas or undocumented | Varies | Not a defensible Arkansas target | Excluded unless provenance and license are verified |

## Current Research Conclusion

The strongest freely testable path is a signal library rather than a single
local inventory target: FDA supplier shortage context, NADAC price pressure,
weather/disease-derived respiratory pressure, ARCOS distribution pressure, and
CMS demand context. Pharmacy-specific literature supports the direction, but
the literature also confirms that the most direct labels usually come from
private pharmacy or hospital systems. Each candidate must therefore be
evaluated independently; literature relevance alone cannot satisfy the
MediTrack accuracy gates.

The current public-source search also found a potentially useful CMS monthly
formulary/network file with NDC, service-area county, and pharmacy-NPI fields.
It is not introduced despite its technical suitability: the published CMS
terms include restrictions on derivative works and competitive offerings, and
the file describes plan access/network structure rather than observed pharmacy
dispensing. This is a documented legal/provenance exclusion, not an
unsearched candidate.

FDA FAERS/AEMS was separately screened. FDA describes the source as quarterly
post-marketing adverse-event and medication-error reports; openFDA notes that
release can lag by three months or more and that common products naturally
generate more reports because they are used more. Those properties make a
raw report-count state a biased pharmacovigilance signal rather than a
defensible pharmacy-demand or inventory target. It is excluded until a
separate exposure-normalized, pharmacy-linked label is available.

## Historical and Rejected State Candidates

The binary FDA shortage continuation target was not promoted because its raw
accuracy was driven by an almost-always-positive label and its balanced
accuracy was approximately 0.50. A derived monthly NDC pressure state is more
useful and remains directly observable from the same public archive:

| Metric | State definition | Rolling evidence | Status |
|---|---|---|---|
| NDC monthly shortage pressure | five states: `0` none, `1` one, `2` two, `3` three, `4` four-or-more active FDA suppliers | 145 folds, 419,806 held-out rows, 2,551 NDCs; 92.09% mean exact accuracy and 80.38% balanced accuracy; high-zone precision 72.77% | qualified proxy through the 75% raw route |
| Historical Arkansas weekly respiratory pressure | training-tertile `low`, `mid`, `high` state over CDC FluView WILI | Superseded by the current five-state screen; retained as context/legacy evidence | rejected under the current contract |
| Historical Arkansas region annual demand state | training-tertile `low`, `mid`, `high` state over CMS Part D demand claims | Superseded by the current five-state regional target | rejected legacy representation |
| Historical NDC monthly recall pressure | legacy three-state recall class encoding | Superseded operationally by numeric recall severity | rejected legacy representation |
| Historical supplier x NDC monthly recall pressure | legacy three-state recall class encoding keyed by FDA recalling firm | Superseded operationally by numeric supplier-NDC severity | rejected legacy representation |
| Global monthly supply-chain pressure state | New York Fed GSCPI `<0`, `0-1`, `>1` standardized pressure bands | 91 rolling folds, 272 held-out months; 85.35% exact but 32.17% balanced accuracy | Rejected: state imbalance fails the balanced-state gate; retain GSCPI as an input candidate, not an established metric |
| Arkansas quarterly Medicaid prescription demand | Arkansas Medicaid FFS prescription count | 21 rolling folds, 83 held-out quarters; 46.83% within-5% accuracy; persistence 48.02% | Rejected: direct public demand target fails numeric gate and learned model does not beat persistence |
| Quarterly next-period demand-change state | low/mid/high tertiles of `log1p(target) - log1p(observed demand)` from the training slice | Small six-layer model, 5 epochs, 2,597 held-out rows; classifier 42.78% exact / 40.01% balanced; numeric forecast 39.55% / 35.85%; neutral no-change persistence 35.66% / 33.33% | Rejected: below 65% state floor and only modest lift over persistence; quarterly cadence is also below the weekly/monthly priority |
| National monthly shortage breadth state | Number of distinct NDCs with active FDA shortages, encoded low/mid/high | 146 rolling folds, 437 held-out months; 100.00% exact but 33.33% balanced accuracy because the target state persists within short test blocks | Rejected: persistence-dominated and fails balanced-state gate |

The first metric is national FDA evidence restricted to the Arkansas-exposed
drug universe; it has no local county allocation. The second is Arkansas
statewide respiratory surveillance, not pharmacy dispensing. Both are
filterable external covariates and satisfy the three-state and held-out-row
requirements, but neither proves pharmacy inventory accuracy.

The recall metric remains explicitly limited: FDA provides downloadable firm
recall data from 2009 onward, but a missing recall record does not establish
that a pharmacy had stock available. The qualified result is therefore a
recall-severity covariate, not a direct availability or inventory forecast.

## Incremental Utility Screening

The aligned Arkansas FluView/wastewater experiment contains 336 weekly rows
and two chronological test folds. Adding current-week mean influenza
wastewater activity to the FluView state features reduced mean balanced
accuracy from 84.45% to 76.97% (delta -7.48 percentage points), so wastewater
is not promoted as an established metric or claimed to improve the model.
It remains available as a monitored input family for future retraining.

A second cross-domain ablation joined normalized FDA recall severity to the
five-state shortage-pressure panel. It retained 240 NDCs, 10 monthly folds,
and 12,188 held-out rows. Adding recall state changed mean balanced accuracy
from 60.97% to 61.87% (delta +0.91 percentage points), but not every fold
improved. Recall remains a standalone context metric and is not claimed to
add reliable incremental shortage-prediction skill.

The machine-readable audit now exposes this distinction in `baseline_skill`:
the qualified standalone proxies are retained for covariate experimentation,
while persistence dominance or a negative cross-signal ablation is explicitly
reported and cannot be interpreted as demonstrated pharmacy-regression lift.

The CDC NNDSS Arkansas weekly table was also screened as a possible disease-
specific demand proxy. The current extract contains only 2025-2026 Arkansas
observations. Using the official `m1` current-week count, training-tertile
low/mid/high states, and three chronological short windows produced 200
held-out rows across nine labels. No label reached 65% balanced accuracy, and
only one label showed all three states in the scored rows. It is rejected until
a longer, stable historical extract is available.

The [CDC Weekly Hospital Respiratory Data](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy)
was then screened because it directly represents weekly hospital utilization.
The Arkansas extract contains 314 rows from 2020-2026. The influenza
admission-pressure state produced four chronological folds, 187 held-out
transitions, 74.16% exact accuracy, and 68.76% balanced accuracy, so it is
promoted as a weekly utilization proxy. COVID admission states are too
imbalanced (49.13% balanced accuracy), and RSV has only two complete folds and
38.43% balanced accuracy; both are rejected.

The aligned FluView/hospital ablation contains 602 weekly rows and four
chronological folds. Adding current-week influenza hospital admissions reduced
balanced accuracy from 84.07% to 83.11% (delta -0.96 percentage points), so
the hospital series is not currently claimed to add incremental FluView skill.

The public Arkansas NDC demand ablation is the closest available test of
pharmacy-regression utility. After correcting the NDC9 normalization in the
publishable panel, the experiment contains 64,116 held-out NDC-quarter rows
and four chronological folds. A history-only Ridge model has mean WAPE
0.2588; adding 21 prior disease/news variables increases mean WAPE to 3.6351
(delta +3.3764).
The context family is therefore not promoted as incremental utility. The
experiment is retained as a negative public-demand result, not as evidence
that the signals cannot help private inventory models under different data.
Family decomposition found no positive subgroup: six disease variables
increased mean WAPE by 0.01848, while fourteen news variables increased mean
WAPE by 2.54259. Neither family improved every fold.
As a model-family control, a compact HistGradientBoosting regressor improved
the history-only mean WAPE to 0.2401, but adding all 21 context variables
increased it to 0.2412 (delta +0.00116). This indicates that the negative
context result is not solely a linear-model limitation.

The [CDC NSSP ED trajectory dataset](https://data.cdc.gov/Public-Health-Surveillance/NSSP-Emergency-Department-Visit-Trajectories-by-St/rdmq-nq56)
was screened for a detailed Arkansas weekly signal. Although it exposes
county/HSA identifiers, the retrieved Arkansas rows contain pathogen
percentages only for the statewide `All` row; all regional influenza trend
labels are `Data Unavailable`. It cannot support the required regional
held-out test and is rejected as a regional metric.

The [CDC FluVaxView pharmacy vaccination table](https://data.cdc.gov/Vaccinations/Weekly-Cumulative-Estimated-Number-of-Influenza-Va/83ng-twza)
was also screened. It is unusually relevant because its pharmacy setting is
based on IQVIA retail-pharmacy claims and provides weekly national activity;
the [CDC dashboard](https://www.cdc.gov/fluvaxview/dashboard/adult-vaccinations-administered.html)
documents the pharmacy setting and weekly cadence. The public table's latest
season is 2023-2024, however, so it cannot be consumed as a free real-time
input today. It also has no Arkansas pharmacy allocation. It is retained as
research support for vaccine-demand relevance, not introduced as a metric or
used to claim current pharmacy utility.

NADAC was re-audited using official CMS annual snapshots. The reproducible
panel builder in `model/scripts/build_nadac_historical_panel.py` normalized
the 2023-2025 files, filtered them to the Arkansas exposure bridge, and
combined them with the local 2021-2022 weekly history. The resulting panel
contains 1,932,824 NDC-date observations for 9,051 NDCs from 2021-01-06
through 2025-12-31. Three chronological folds produced 1,113,045 held-out
transitions. The model's mean WAPE was 0.00368 versus 0.00135 for persistence,
and the model failed to beat persistence in every fold. It is therefore
rejected despite sufficient sample size; it remains a national acquisition-
cost context variable, not a local inventory or supplier metric.
