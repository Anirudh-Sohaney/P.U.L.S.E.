# Input Variable Audit

## Review note — 2026-09-14

This audit remains the authoritative classification of model inputs. The
repository has since added or exposed broader adapters for ARCOS distribution,
CDC respiratory and wastewater signals, HHS provider-NDC spending, supplier
context, and universal forecast filtering. These additions do not change the
live/training-only classification automatically: each source still requires a
documented refresh path and a point-in-time-safe join. Arkansas is the primary
geography, while national, bordering-state, and global observations remain
explicitly contextual.

Date: 2026-08-18

Scope: every input family currently represented in `data/final_data/VARIABLE_LIST.md`
(404 distinct variables, 6,479,152 feature observations) plus the targeted Arkansas
additions consumed by the model. This audit precedes code changes; it classifies
inputs by whether a real-time application could obtain the raw observation **today**
from a free/public source with a documented refresh cadence.

## Classification definitions

| Class | Meaning |
|---|---|
| `LIVE_INPUT` | Raw observation obtainable today from a free/public source with a documented refresh cadence of daily or faster. Safe to feed a real-time application with at most a day of staleness. |
| `NEAR_REAL_TIME_INPUT` | Raw observation obtainable today from a free/public source with a documented refresh cadence of weekly to monthly, or daily data with a multi-day publication lag. Usable in a real-time app only with explicit staleness handling. |
| `LOCAL_PHARMACY_HISTORY_INPUT` | A pharmacy supplies its own current dispensing/inventory history at forecast time. This is operational for a downstream pharmacy model, but is not a free/public external input and must not be represented as one. |
| `PERIODIC_TRAINING_ONLY` | Raw observation obtainable today but published annually (or slower) with a long publication lag, or a historical-only series no longer refreshed. Valid for training/backtesting; not a live feature. |
| `STATIC_IDENTITY_CONTEXT` | Static or near-static catalog (drug/NDC dictionaries, crosswalks, rosters, registries). Changes rarely; used as identity/context, not a time-varying signal. |
| `DERIVED_TRAINING_VARIABLE` | Deterministic transform (lag, rolling mean/std, change, seasonal anomaly/baseline) or point-in-time prior/context join of a raw observation. Not obtainable as a raw observation; recomputed from the underlying raw series. |

Criterion applied: *can a real-time application obtain the raw observation today
from a free/public source with a documented refresh cadence?* If yes and cadence is
daily-or-faster → `LIVE_INPUT`; weekly/monthly → `NEAR_REAL_TIME_INPUT`; pharmacy-
supplied current history → `LOCAL_PHARMACY_HISTORY_INPUT`; annual or historical-only
→ `PERIODIC_TRAINING_ONLY`; catalog → `STATIC_IDENTITY_CONTEXT`; transform →
`DERIVED_TRAINING_VARIABLE`.

Derived availability is inherited from the raw feature family rather than
inferred from the transformed name alone. For example, a FluView lag remains
operationally rebuildable, while a PM2.5 annual lag, Census annual change, or
historical COVID lag remains training-only. This prevents a transform from
silently bypassing the raw source's cadence and publication-lag constraints.

The machine-readable registry in
`model/arkansas_pharma_signal/input_sources.py` records source URL, cadence,
publication lag, source availability, and refresh-adapter status for every
near-real-time variable. WHO fields whose public refresh endpoint could not be
verified are instead recorded in `RESEARCH_ONLY_SOURCE_REGISTRY`. This keeps
historical artifacts traceable without treating a landing page as an input
feed. `all_sources_documented` and `all_adapters_ready` are separate fields:
public availability alone does not authorize an operational forecast.

The input summary exposes the same distinction as
`disposition_operational_ready` versus `operational_ready`. The forecast guard
uses the latter and fails closed when any source adapter is not integrated;
research runs may still use the former with an explicit training-only override.

Point-in-time fields prefixed `prior_` are explicitly derived variables. Their
suffix may contain a source token such as `news`, `ili`, or `shortage`, but that
does not make the materialized prior value a raw live feed. At forecast time the
underlying source must be refreshed and the prior join recomputed without using
the target period.

## Current input families

### chronic_health — CDC PLACES (11 variables)

| Field | Value |
|---|---|
| Source | CDC PLACES 2024 release (`cdc_places_2024`) |
| URL | https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-Place-Data-202/epbn-9bv3 |
| Cadence | Annual; 2022 reference year, released ~2 years later |
| Geography | County (Arkansas) |
| Classification | `PERIODIC_TRAINING_ONLY` |
| Leakage/staleness | ~2-year publication lag; prevalence is a slow-moving population trait, not a live demand signal |
| Disposition | Keep for training context. **Not live.** Do not feed a real-time application as a current observation. |

### demand_price — CMS Part D, Medicaid SDUD, NADAC, openFDA (26 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `cms_partd` | fills, beneficiaries, claims, drug cost, cost share (ge65/tot) | Annual, 2013–2022 | national; state | `PERIODIC_TRAINING_ONLY` |
| `cms_medicare_quarterly_partd` | latest national Part D claims, spending, beneficiaries | Quarterly snapshot; local file covers 2024 and 2025 Q1-Q3 | national | `NEAR_REAL_TIME_INPUT` context only |
| `medicaid_sdud` | reimbursed amounts, prescriptions, units, suppressed rows | Quarterly, 2012–2025 locally; CMS annual refresh | state/NDC | `NEAR_REAL_TIME_INPUT` for periodic target refresh only (not a real-time pharmacy feature; publication lag is documented) |
| `nadac` | per-unit min/mean/max price, row count | Weekly | national | `NEAR_REAL_TIME_INPUT` |
| `openfda_enforcement` (+ `_recent`) | `recall_active` | Daily/weekly updates | national | `LIVE_INPUT` |
| `openfda_shortages` (+ `_recent`) | `shortage_active` | Daily/weekly updates | national | `LIVE_INPUT` |

URLs:

- CMS Part D Prescribers by Provider and Drug: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug
- Medicaid SDUD: https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data
- NADAC: https://www.medicaid.gov/medicaid/prescription-drugs/pharmacy-pricing/index.html
- openFDA Drug Shortages: https://open.fda.gov/apis/drug/drugshortages/
- openFDA Drug Enforcement (recalls): https://open.fda.gov/apis/drug/enforcement/

Leakage/staleness:

- **CMS Part D is annual with a ~1.5–2 year publication lag. Explicitly non-live.** The
  panel window ends 2022; a real-time app cannot obtain 2026 Part D demand today.
- CMS Medicare Quarterly Part D is a public quarterly snapshot, but the current
  file is a finalized-2024 plus partial-2025 release rather than a complete
  historical quarter-by-quarter panel. It is latest national context only; it
  is never interpolated, used as an Arkansas label, or used to backfill history.
- Medicaid SDUD is quarterly and revised with late reports; usable as a quarterly
  pressure signal with documented lag, not a live feed.
- NADAC weekly prices are current and free; near-real-time.
- openFDA shortages/enforcement are updated continuously. Both their explicit
  `openfda_*` source names and the canonical inventory variables
  `shortage_active` and `recall_active` are classified as `LIVE_INPUT`, because
  `VARIABLE_LIST.md` records their openFDA provenance.

Disposition: keep Part D and SDUD for training; promote only `shortage_active`,
`recall_active`, and NADAC toward live use.

### derived — transforms of raw series (228 variables)

| Raw source | Variables | Raw cadence | Geography | Classification |
|---|---|---|---|---|
| `epa_airdata_annual_conc_by_monitor` | PM2.5, ozone lags/rolling/anomaly | Annual | monitor | `PERIODIC_TRAINING_ONLY` (raw) + `DERIVED_TRAINING_VARIABLE` (transforms) |
| `bls_public_api` | unemployment, labor force, employment, CPI lags/rolling/anomaly | Monthly | state; national | `NEAR_REAL_TIME_INPUT` (raw) + `DERIVED_TRAINING_VARIABLE` (transforms) |
| `delphi_fluview` | ILI, num_ILI, wILI lags/rolling/anomaly | Weekly | national; state | `NEAR_REAL_TIME_INPUT` (raw) + `DERIVED_TRAINING_VARIABLE` (transforms) |
| `fred_ppi` | pharma PPI lags/rolling/anomaly | Monthly | national | `NEAR_REAL_TIME_INPUT` (raw) + `DERIVED_TRAINING_VARIABLE` (transforms) |
| `noaa_ncei_daily_summaries` | temperature, precipitation, wind lags/rolling/anomaly | Daily | station | `LIVE_INPUT` (raw) + `DERIVED_TRAINING_VARIABLE` (transforms) |
| `cdc_covid_county_transmission` | COVID cases/transmission/test lags/rolling/anomaly | Daily, 2020–2022 only | county | `PERIODIC_TRAINING_ONLY` (historical-only series) + `DERIVED_TRAINING_VARIABLE` |

URLs:

- EPA AQS annual concentration by monitor: https://aqs.epa.gov/aqsweb/airdata/download_files.html
- BLS Public Data API v2: https://www.bls.gov/developers/
- Delphi Epidata FluView: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
- FRED (PPI PCU32543254): https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCU32543254
- NOAA NCEI Daily Summaries (GHCND): https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- CDC COVID county transmission: https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-County-Level-of-Community-Tran/8396-v7yb

Leakage/staleness:

- **COVID county transmission series ended 2022-10-18. Historical-only. Explicitly non-live.**
- EPA annual air data has a multi-month publication lag; annual cadence → training only.
- NOAA daily summaries are published with ~1–2 day lag; live-usable.
- BLS monthly series publish ~3 weeks after month end; near-real-time with documented lag.
- All `_change_`, `_lag_`, `_rolling_`, `_seasonal_` variables are deterministic
  transforms; they inherit the raw source's class for availability but are themselves
  `DERIVED_TRAINING_VARIABLE`.

Disposition: keep transforms for training; the only live-usable raw series in this
family are NOAA daily weather and (with lag) BLS/FRED/FluView.

### disasters — FEMA openFEMA (2 variables)

| Field | Value |
|---|---|
| Source | `fema_openfema` |
| URL | https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2 |
| Cadence | Event-driven; declarations posted as issued |
| Geography | county_or_area |
| Classification | `LIVE_INPUT` (event feed, near-real-time publication) |
| Leakage/staleness | Disaster declarations are retrospective (issued after the event); usable as a live event flag, not a forecast of the event |
| Disposition | Keep as live event context; do not treat declaration date as event onset |

### disease_global — WHO FluNet/FluID, GHO, WUENIC, OWID, DON (55 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `who_flunet` | influenza subtype counts, specimens | Weekly | country | `NEAR_REAL_TIME_INPUT` (weekly, multi-week lag) |
| `who_fluid` | ILI/ARI cases, intensity, spread | Weekly | country | `PERIODIC_TRAINING_ONLY` (public feed unverified) |
| `who_don` | `outbreak_event` | Event-driven | global_event | `PERIODIC_TRAINING_ONLY` (historical artifact; refresh unverified) |
| `who_gho` | immunization coverage, life expectancy | Annual | country | `PERIODIC_TRAINING_ONLY` |
| `wuenic` | immunization estimates | Annual | country | `PERIODIC_TRAINING_ONLY` |
| `owid` | life expectancy, under-5 mortality | Annual | country | `PERIODIC_TRAINING_ONLY` |

URLs:

- WHO FluNet: https://www.who.int/tools/flunet
- WHO FluID: https://www.who.int/tools/fluid
- WHO Disease Outbreak News: https://www.who.int/emergencies/disease-outbreak-news
- WHO GHO OData API: https://www.who.int/data/gho/info/gho-odata-api
- WUENIC: https://www.who.int/teams/immunization-vaccines-and-biologicals/immunization-analysis-and-insights/global-monitoring/immunization-coverage/who-unicef-estimates-of-national-immunization-coverage
- OWID: https://ourworldindata.org/

Leakage/staleness: WHO documents weekly FluNet/FluID reporting, but the
required public FluID object was not present in the verified UAT mart and the
production-style endpoint did not return during review. GHO/WUENIC/OWID are
annual estimates. Global disease series are context for a US/Arkansas pharmacy
model, not live demand inputs.

Disposition: keep WHO-derived artifacts for training only until a stable,
tested, point-in-time public refresh endpoint is verified.

### disease_us — CDC COVID, NNDSS, Delphi FluView, RESP-NET (17 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `cdc_covid_county_transmission` | cases/100k, transmission level, test % | Daily, 2020–2022 only | county | `PERIODIC_TRAINING_ONLY` (historical-only) |
| `nndss_weekly` | current-week cases, cumulative YTD, 52-week max | Weekly | national | `NEAR_REAL_TIME_INPUT` |
| `delphi_fluview` | ILI, num_ILI, wILI, age buckets, providers | Weekly | national; state | `NEAR_REAL_TIME_INPUT` |

URLs:

- CDC COVID county transmission: https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-County-Level-of-Community-Tran/8396-v7yb
- NNDSS Weekly Data: https://data.cdc.gov/NNDSS/NNDSS-Weekly-Data/x9gk-5huc
- Delphi Epidata FluView: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
- CDC RESP-NET Rates and Clinical Data: https://data.cdc.gov/Public-Health-Surveillance/RESP-NET-Rates-and-Clinical-Data/kvib-3txy

Leakage/staleness: **COVID county series is historical-only (ended 2022-10-18);
explicitly non-live.** NNDSS and FluView are weekly with ~1–2 week lag; near-real-time.

Disposition: keep COVID for training; NNDSS/FluView/RESP-NET near-real-time.

### economics_us — BLS, FRED, USAspending (9 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `bls_public_api` | unemployment rate, labor force, employment, CPI | Monthly | state; national | `NEAR_REAL_TIME_INPUT` |
| `fred_ppi` | pharma producer price index | Monthly | national | `NEAR_REAL_TIME_INPUT` |
| `usaspending` | contract obligations | Fiscal year, 2011–2021 | national | `PERIODIC_TRAINING_ONLY` |

URLs:

- BLS Public Data API v2: https://api.bls.gov/publicAPI/v2/timeseries/data/
- FRED: https://fred.stlouisfed.org/docs/api/fred/
- USAspending API: https://api.usaspending.gov/

Leakage/staleness: BLS/FRED monthly with ~3-week lag; near-real-time. **USAspending
fiscal-year series ends 2021 and is historical-only; explicitly non-live.** Note the
panel contains historical-only sub-series: `arkansas_employed_population`,
`arkansas_labor_force`, `arkansas_unemployed_population`, and
`consumer_price_index_medical_care` all end 2009-12-01 and are
`PERIODIC_TRAINING_ONLY` even though the same BLS source is live for other series.

Disposition: keep; monthly series near-real-time, fiscal-year and 2000–2009
sub-series training-only.

### Machine-enforced operational selection

The name-level audit is enforced by
`model/arkansas_pharma_signal/input_contract.py` and
`datasets.feature_columns_for_mode(view, "operational")`. A derived feature is
not automatically operational: if its name identifies a historical or
periodic source family, it is excluded even though its representation is a
lag, rolling value, or log transform. This prevents fields such as
`medicaid_exact_rx_log`, `medicaid_rx_q4_log`, and `arcos_grams_lag` from
entering a live model merely because they are derived.

On the current 392-column production feature surface, operational mode keeps
363 columns and excludes 29. The excluded set contains 20 periodic raw/static
fields and 9 derived variables whose source is non-operational. Local
`y_last` demand history and transforms of live weather/disease feeds remain
eligible. Unknown names continue to fail closed, and the contract is covered
by the input-contract and next-period test suites.

The standard `forecast` CLI also fails closed when a serialized model contains
periodic or historical-only features. Such a model must be retrained in
operational mode; the explicit `--allow-training-only-features` override is
available only for documented historical/research runs and is recorded in
forecast metadata through the non-operational input contract.
The metadata also records `forecast_mode` as either `operational` or
`research_training_only`, so downstream consumers can reject the latter.
The same fail-closed guard and metadata are applied to `forecast-universal`,
which invokes the serialized demand and shortage-risk models while building
the county x drug x supplier contract.

The modular Transformer/graph training path uses a separate 38-field history
contract. Local pharmacy history fields (`value`, `value_last`, rolling
history, quarter indicators, and demand-change transforms) are explicitly
operational for the eventual pharmacy-level application because the pharmacy
supplies them itself. The audit now records these under
`LOCAL_PHARMACY_HISTORY_INPUT`, separate from free/public feeds. Only three
modular fields are genuinely non-operational periodic fields (unemployment,
GSCPI, and tariff). Its serialized `input_contract` therefore sets
`forecast_mode` to `research_training_only` rather than implying that the
300M-class checkpoint is deployable with free live inputs.
The standalone modular trainer exposes `--operational-only`, which fails
closed under this condition instead of producing a misleading operational
checkpoint.

### environment — EPA AQS, NOAA NCEI (16 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `epa_airdata_annual_conc_by_monitor` | PM2.5, ozone mean/max/obs% | Annual | monitor | `PERIODIC_TRAINING_ONLY` |
| `noaa_ncei_daily_summaries` | temperature, precipitation, wind, snow, humidity, extreme days | Daily | station | `LIVE_INPUT` |

URLs:

- EPA AQS: https://aqs.epa.gov/aqsweb/airdata/download_files.html
- NOAA NCEI GHCND: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

Leakage/staleness: EPA annual data has multi-month lag; training-only. NOAA daily
summaries publish with ~1–2 day lag; live-usable. `humidity` ends 2024-12-31 and is
historical-only within an otherwise live source.

Disposition: NOAA daily weather is the strongest current live input family; EPA
annual air quality is training-only.

### healthcare — CMS Medicare Geographic Variation, Medicaid CHIP (17 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `cms_medicare_geographic_variation` | utilization, spending, beneficiary count, MA participation | Annual, 2014–2024 | county; national | `PERIODIC_TRAINING_ONLY` |
| `medicaid_chip_performance_indicator` | enrollment, applications, call-center metrics | Monthly | state | `NEAR_REAL_TIME_INPUT` |

URLs:

- CMS Medicare Geographic Variation: https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparison/medicare-geographic-variation-national-by-drg
- Medicaid/CHIP Performance Indicator: https://www.medicaid.gov/medicaid/quality-of-care/performance-measurement/adpqm/index.html

Leakage/staleness: Medicare Geographic Variation is annual with long lag; training-only.
Medicaid CHIP enrollment is monthly; near-real-time.

Disposition: keep; annual Medicare training-only, monthly CHIP near-real-time.

### policy — WITS TRAINS (1 variable)

| Field | Value |
|---|---|
| Source | `wits_trains` |
| URL | https://wits.worldbank.org/tariff/trains/en/ |
| Cadence | Annual, 2012–2022 |
| Geography | country |
| Classification | `PERIODIC_TRAINING_ONLY` |
| Leakage/staleness | Annual tariff data with multi-year lag; slow-moving policy context |
| Disposition | Keep for training; not live |

### population — Census county estimates, CDC PLACES (12 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `census_county_population_estimates` | population, births, deaths, migration, rates | Annual | county; region | `PERIODIC_TRAINING_ONLY` |
| `cdc_places_2024` | `population_total` (shared source) | Annual | county | `PERIODIC_TRAINING_ONLY` |

URL: https://www.census.gov/programs-surveys/popest.html

Leakage/staleness: annual estimates with ~1-year lag; slow-moving denominators.
Training-only.

Disposition: keep as training denominators; not live.

### supply_chain — NY Fed GSCPI, RWI/ISL, EIA, World Bank (8 variables)

| Source | Variables | Cadence | Geography | Classification |
|---|---|---|---|---|
| `nyfed_gscpi_monthly_2012_2022` | GSCPI | Monthly, 2012–2022 | global | `PERIODIC_TRAINING_ONLY` (historical-only window) |
| `rwi_isl_container_throughput_monthly_2012_2022` | container throughput | Monthly, 2012–2022 | global | `PERIODIC_TRAINING_ONLY` (historical-only window) |
| `eia_crude_spot_prices_monthly_2012_2022` | crude spot price | Monthly, 2012–2022 | global | `PERIODIC_TRAINING_ONLY` (historical-only window) |
| `worldbank_pink_sheet_monthly_2012_2022` | commodity price | Monthly, 2012–2022 | global | `PERIODIC_TRAINING_ONLY` (historical-only window) |

URLs:

- NY Fed GSCPI: https://www.newyorkfed.org/research/policy/gscpi
- RWI/ISL Container Throughput Index: https://www.isl.org/en/containerindex
- EIA Open Data: https://www.eia.gov/opendata/
- World Bank Pink Sheet: https://www.worldbank.org/en/research/commodity-markets

Leakage/staleness: **All four series are stored only for the 2012–2022 window.
The underlying sources are live monthly feeds today, but the local variables are
historical-only snapshots. Explicitly non-live as stored.** If refreshed, GSCPI and
container throughput become `NEAR_REAL_TIME_INPUT` (monthly cadence).

Disposition: keep for training; refresh from live sources if monthly supply-chain
pressure is wanted as a near-real-time feature.

### trade — CEPII BACI (6 variables)

| Field | Value |
|---|---|
| Source | `cepii_baci` |
| URL | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 |
| Cadence | Annual, 2000–2024 |
| Geography | country |
| Classification | `PERIODIC_TRAINING_ONLY` |
| Leakage/staleness | Annual bilateral trade with ~1–2 year publication lag |
| Disposition | Keep for training; not live |

### Targeted Arkansas additions (model inputs, not in VARIABLE_LIST)

| Source | Role | Cadence | Classification |
|---|---|---|---|
| `cms_partd_prescriber_provider_drug` | Arkansas provider-drug demand proxy | Annual, 2013–2024 | `PERIODIC_TRAINING_ONLY` |
| `medicaid_sdud` (local `data/S_D`) | Quarterly prescription counts | Quarterly | `NEAR_REAL_TIME_INPUT` |
| `dea_arcos_arkansas` | Controlled-substance retail distribution | Annual PDFs | `PERIODIC_TRAINING_ONLY` |
| `fda_ndc_directory` | NDC product/package catalog | Current snapshot | `STATIC_IDENTITY_CONTEXT` |
| `fda_shortages_recalls_current` | Shortage + enforcement records | Daily/weekly | `LIVE_INPUT` |
| `disease_surveillance_current` | FluView, NNDSS, wastewater (AR) | Weekly | `NEAR_REAL_TIME_INPUT` |
| `arkansas_news_3dlnews` | Monthly news keyword counts | Monthly | `NEAR_REAL_TIME_INPUT` (local corpus; refresh cadence not documented) |
| `nppes_provider_locations` | Provider geography | Monthly refresh | `STATIC_IDENTITY_CONTEXT` |
| `arkansas_pharmacy_roster` | Pharmacy facilities by city/year | Annual | `STATIC_IDENTITY_CONTEXT` |
| `fda_establishments` (DRLS) | Establishment registration | Annual snapshot | `STATIC_IDENTITY_CONTEXT` |

## Explicit non-live markers (required)

The following are **not** live inputs under the stated criterion:

1. **Annual CMS Part D** (`cms_partd`, `cms_partd_prescriber_provider_drug`) — annual
   publication with ~1.5–2 year lag; panel ends 2022/2024.
2. **CDC PLACES / chronic prevalence** (`cdc_places_2024`) — annual, 2022 reference
   year released ~2 years later.
3. **Historical-only economic series** — `usaspending` (ends 2021),
   `arkansas_employed_population` / `arkansas_labor_force` /
   `arkansas_unemployed_population` / `consumer_price_index_medical_care` (end
   2009-12-01), `humidity` (ends 2024-12-31), all `*_monthly_2012_2022` supply-chain
   snapshots, `cdc_covid_county_transmission` (ends 2022-10-18).
4. **Static catalogs** — FDA NDC directory, FDA DRLS establishments, NPPES provider
   locations, pharmacy roster, drug dictionary, RxNav/NDC catalog, ZIP-HSA crosswalk.
   These are `STATIC_IDENTITY_CONTEXT`, not time-varying signals.

## Candidate live inputs (research-backed)

Sources a real-time application could add today, all free/public with documented
cadence. None are currently in the feature store.

| Candidate | Source URL | Cadence | Geography | Access | Notes / risk |
|---|---|---|---|---|---|
| NWS weather + alerts | https://api.weather.gov (docs: https://www.weather.gov/documentation/services-web-api) | Alerts ~2 min; observations ~20 min lag; forecasts hourly | Point/grid, county (UGC) | Free, no key (User-Agent required) | Authoritative US alerts plus point-to-grid hourly forecasts; the live adapter preserves issuance/validity timestamps. Archive beyond 7 days requires NCEI. Live weather replaces/augments NOAA daily summaries at higher frequency. |
| FDA shortages | https://open.fda.gov/apis/drug/drugshortages/ | Continuous/daily | national (drug-level) | Free | Already in store as `shortage_active`; promote to live refresh. |
| FDA recalls/enforcement | https://open.fda.gov/apis/drug/enforcement/ | Daily | national (drug-level) | Free | Already in store as `recall_active`; promote to live refresh. |
| CDC NSSP respiratory (syndromic) | https://www.cdc.gov/nssp/ (BioSense/ESSENCE) | Near-real-time (daily) | State/county (ED visits) | **Access-gated**: state/local health dept approval | Strongest respiratory early signal, but not open download; requires partnership. Mark as candidate with access caveat, not guaranteed free. |
| CDC wastewater (NWSS) | https://data.cdc.gov/d/atcp-73re (export: https://data.cdc.gov/api/v3/views/atcp-73re/export.csv?accessType=DOWNLOAD) | Weekly, Fridays | Site/state/national | Free, public | WVAL for SARS-CoV-2, Flu A, RSV; already partially in store via `cdc_wastewater_ar_site_weekly`; refresh weekly. |
| CDC FluView (ILINet) | https://cmu-delphi.github.io/delphi-epidata/api/fluview.html | Weekly | National/state | Free | Already in store via `delphi_fluview`; refresh weekly. |
| CDC RESP-NET RSV | https://data.cdc.gov/Public-Health-Surveillance/RESP-NET-Rates-and-Clinical-Data/kvib-3txy | Weekly | National RESP-NET aggregate | Free | Integrated via Socrata API; national disease-pressure context only, not Arkansas-local. |
| BLS | https://api.bls.gov/publicAPI/v2/timeseries/data/ | Monthly (~3-week lag) | State/national | Free, key optional | Already in store; near-real-time with documented lag. |
| News (GDELT 2.0) | https://www.gdeltproject.org/data.html | 15-minute updates | Global, georeferenced | Free | Replaces/augments local 3DLNews corpus with documented cadence; volume normalization required (coverage grows over time). |
| Supply-chain indicators | NY Fed GSCPI https://www.newyorkfed.org/research/policy/gscpi; RWI/ISL https://www.isl.org/en/containerindex; EIA https://www.eia.gov/opendata/; World Bank Pink Sheet https://www.worldbank.org/en/research/commodity-markets | Monthly (EIA crude daily) | Global | Free | Underlying sources are live; local 2012–2022 snapshots must be refreshed to become near-real-time. |

## Summary

| Class | Families represented | Live-usable today |
|---|---|---|
| `LIVE_INPUT` | openFDA shortages/enforcement, FEMA disasters, NOAA daily weather | Yes — tested refresh adapters |
| `NEAR_REAL_TIME_INPUT` | NADAC, BLS, FRED, FluView, NNDSS, RESP-NET, Medicaid CHIP, wastewater, news corpus | Yes — with documented lag handling and adapter gates |
| `PERIODIC_TRAINING_ONLY` | WHO FluID/DON artifacts, CMS Part D, CDC PLACES, EPA AQS, CMS Medicare GeoVar, Census population, BACI, WITS, USAspending, historical-only sub-series, 2012–2022 supply-chain snapshots, COVID county | No — training/backtest only |
| `STATIC_IDENTITY_CONTEXT` | FDA NDC, FDA DRLS, NPPES, pharmacy roster, drug dictionaries, crosswalks | No — identity context |
| `DERIVED_TRAINING_VARIABLE` | All `_lag_`, `_rolling_`, `_change_`, `_seasonal_` transforms and point-in-time `prior_` joins | Recomputed from raw series; operational status inherits the raw source |

Accuracy note: cadence and URL claims above are taken from the cited official
documentation and the local source manifests; they were not re-verified by live
downloads in this audit. Refresh cadence of the local 3DLNews corpus is not
documented and should be recorded before it is treated as a live feed.
