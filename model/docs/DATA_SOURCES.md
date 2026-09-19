# Data Sources

This is an Arkansas-first inventory. Arkansas sources receive priority for
geography and evaluation, while national, neighboring-state, and global
sources are retained when they provide useful upstream context. A source is
not an approved feature merely because it is public: its publication lag,
coverage, identity mapping, leakage boundary, and chronological test result
must be recorded.

This model uses local real-data assets already present in `data/`. Additional public sources may be added only when cited and reproducible. Synthetic data is prohibited.

## Local Canonical Data

| Source path | Role |
|---|---|
| `data/final_data/features/external_state_features.csv.gz` | Combined numeric external-state feature store across disease, weather, economics, trade, supply-chain, population, demand/price, disasters, and derived variables. |
| `data/targeted_additions/epidemiology_annotation/cirad_padiweb_annotation.ods` | Public CIRAD/PADI-web expert-annotated animal-disease news sentences; evaluation-only benchmark for the generic outbreak trigger, not a human-disease training source. |
| `data/targeted_additions/padi_expert/data/*.tab` | Public PADI-web animal-disease article-event labels produced with help from two epidemiologists; evaluation-only transfer benchmark, not pharmacy truth. |
| `data/targeted_additions/padi_web_weak/*.tab` | 70,707 weak article-relevance labels used only for the learned Layer 1 relevance model; not event or forecast truth. |
| `data/targeted_additions/biocaster/` | Archived BioCaster event frames and 30 still-retrievable source pages; evaluation-only outbreak-trigger recall benchmark. Missing/redirected historical pages are excluded. |
| `data/targeted_additions/band/` | BAND held-out human-health outbreak-news NER and QA test split from the authors' public repository; evaluation-only recall benchmark, never training data. |
| `data/targeted_additions/daniel/` | DAnIEL English token-level epidemic-event test split; evaluation-only disease-entity recall benchmark, never training data. |
| `data/final_data/events/events.csv.gz` | Structured shortage, recall, enforcement, disaster, and event rows. |
| `data/final_data/entities/entities.csv.gz` | Entity registry for geography, drug, ingredient, global/national/state, and related nodes. |
| `data/final_data/entities/relationships.csv.gz` | Graph edges such as drug-to-ingredient and geography relationships. |
| `data/final_data/entities/drug_dictionary.json` | Drug and active ingredient dictionary. |
| `data/final_data/entities/rxnav_ndc_catalog.csv.gz` | RxNav/NDC drug mapping. |
| `data/final_data/entities/dartmouth_zip_hsa_hrr.csv.gz` | ZIP to HSA/HRR health-service geography crosswalk. |

## Targeted Arkansas Additions

| Source path | Role |
|---|---|
| `data/targeted_additions/cms_partd_prescriber_provider_drug/data/arkansas_partd_provider_drug_by_year.csv.gz` | Arkansas provider-drug demand proxy: claims, 30-day fills, cost, beneficiaries, prescriber city/type, generic and brand drug, 2013-2024. |
| `data/targeted_additions/cms_partd_geography_drug/data/arkansas_partd_geography_drug_by_year.csv.gz` | Locally normalized CMS Part D geography-by-drug panel: Arkansas state FIPS 05, generic drug, annual total claims, 2013-2024; 14,275 rows. No county or supplier field. |
| `data/targeted_additions/cms_partd_geography_drug/source_manifest.json` | Source scope, suppression handling, local normalization path, and CMS geography-by-drug provenance. |
| `data/targeted_additions/hhs_medicaid_provider_spending_ndc/raw/medicaid-provider-spending-ndc.csv.zip` | Locally downloaded HHS July 2026 Medicaid Provider Spending by NDC source; 2018-01 through 2024-12, provider-NDC-month, with CMS suppression limitations. |
| `data/targeted_additions/hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_ndc_monthly.csv.gz` | Arkansas NPPES pharmacy-taxonomy billing-provider slice of the HHS source: 28,386 observed NDC-month rows, 2,540 NDCs; training/evaluation candidate only after failing the five-state monthly accuracy gate. |
| `data/targeted_additions/hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_county_ndc_monthly.csv.gz` | Same HHS slice allocated through the provenance-bearing city-to-county crosswalk: 24,050 observed county-NDC-month rows across 72 mapped counties; rejected after county-level rolling evaluation. |
| `model/artifacts/evaluation/arkansas_pharmacy_labeler_monthly_demand_metrics.json` | Reproducible labeler-code screening: 21 rolling folds, 2,556 held-out rows; rejected because raw five-state accuracy is 58.79% and labeler is not a supplier. |
| `data/targeted_additions/hhs_medicaid_provider_spending_ndc/source_manifest.json` | HHS source checksum, NPPES taxonomy filter, date range, scope, and suppression limitations. |
| `data/targeted_additions/cms_medicare_quarterly_partd/data/medicare_quarterly_partd_spending_by_drug.csv` | CMS public Medicare Part D quarterly snapshot: national drug-level claims, spending, beneficiaries, and manufacturer context. Current local snapshot covers 2024 and 2025 Q1-Q3; context only, not an Arkansas target or historical quarterly panel. |
| `data/S_D/data/combined/*.json.gz` (`source.source_id == medicaid_sdud`) | Arkansas Medicaid State Drug Utilization Data quarterly prescription counts by drug, annualized into real Medicaid demand-pressure features. |
| `data/demand_price/data/nadac_ndc_weekly.csv.gz` | CMS NADAC weekly national average acquisition-cost observations by NDC; used as a price-pressure input, never as a demand or shortage label. |
| `model/artifacts/evaluation/nadac_price_metrics.json` | Held-out national NADAC transition benchmark; persistence is retained as a baseline, not a learned architecture claim. |
| `model/artifacts/evaluation/nadac_arkansas_exposed_metrics.json` | Held-out NADAC benchmark restricted to NDC9s observed in the Arkansas Medicaid exposure panel; national acquisition-cost proxy only. |
| `model/artifacts/evaluation/nadac_arkansas_exposed_rolling_metrics.json` | Original rolling exact-seven-day NADAC benchmark; retained for provenance. |
| `model/artifacts/evaluation/nadac_historical_reconstruction_rolling_metrics.json` | Three-fold 2021-2025 CMS NADAC reconstruction; rejected because persistence beats the model in every fold. |
| `data/targeted_additions/cms_sdud_recent/source_manifest.json` | Manifest for opt-in 2023-2025 Arkansas CMS SDUD annual refreshes; 2024 is partial and the files are periodic targets, not live features. |
| CMS `State Drug Utilization Data` annual CSVs | Official annual state/NDC target refresh. The opt-in `fetch_medicaid_sdud_csv` adapter filters Arkansas, excludes suppressed rows, aggregates prescriptions by product and quarter, and records retrieval/source metadata. `evaluate-quarterly` and `forecast-quarterly` accept repeated `--live-sdud-url` flags for multiple years. It is not treated as a real-time feature. |
| `data/targeted_additions/dea_arcos_arkansas/data/arcos_arkansas_retail_summary.csv.gz` | Arkansas ZIP3/quarter controlled-substance retail distribution proxy, 2006-2025; parsed from [DEA retail summaries](https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html). Broader ARCOS references include the [Nature dataset](https://www.nature.com/articles/s41597-024-03534-3) and [Indiana metadata](https://maps.indiana.edu/metadata/Demographics/Health_Opioid_Data_ARCOS_DEA.html). |
| `model/artifacts/evaluation/arcos_distribution_state_metrics.json` | Leakage-safe quarterly low/mid/high ARCOS distribution-pressure evaluation: 15 folds, 11,503 held-out rows, 39 controlled-substance codes, 85 ZIP3s. |
| `model/arkansas_pharma_signal/features.py::build_arcos_annual_features` | Drug-level annual ARCOS grams, ZIP3 breadth, and prior-year growth joined into the annual demand panel; distribution proxy only. |
| `data/targeted_additions/fda_ndc_directory/data/fda_ndc_products_current.csv.gz` | Current FDA NDC product catalog with labeler, active ingredient, strength, route, dosage form, application, marketing category. |
| `data/targeted_additions/fda_ndc_directory/data/fda_ndc_packages_current.csv.gz` | Current FDA NDC package catalog. |
| `data/targeted_additions/fda_shortages_recalls_current/data/fda_shortages_current.csv.gz` | Current/recent FDA shortage records. |
| `data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv` | Dated Internet Archive reconstruction of national FDA shortage status by NDC9/supplier, 2012-2026; right-censored and evaluation-only. |
| `data/targeted_additions/fda_shortage_archive/data/arkansas_medicaid_shortage_exposure.csv` | Arkansas Medicaid-observed NDC9-quarter exposure joined to national FDA shortage state; local exposure target, not local inventory truth. |
| `model/artifacts/evaluation/arkansas_exposed_supplier_shortage_metrics.json` | Censor-safe monthly supplier-drug evaluation restricted to NDC9s observed in Arkansas Medicaid exposure; national supplier evidence, not county inventory. |
| `model/artifacts/evaluation/recall_demand_utility_metrics.json` | Four-fold FDA recall-context ablation on the corrected Arkansas NDC9-quarter demand panel; not demonstrated as incremental utility because the lagged variant does not improve every fold. |
| `model/artifacts/evaluation/shortage_recall_utility_metrics.json` | Ten-fold five-state FDA shortage-pressure ablation with recall state; mixed mean lift but no all-fold improvement, so not promoted as individual-pharmacy utility. |
| `model/artifacts/evaluation/county_region_rolling_metrics.json` | Region-stratified rolling annual county-drug demand benchmark using the same leakage-safe contract as the statewide evaluator; regional reporting only, not county-by-supplier inventory. |
| `model/arkansas_pharma_signal/geography.py` | Arkansas DHS/TEFRA five-region county-FIPS partition; deterministic reporting geography, not an inferred demand cluster. |
| `data/targeted_additions/publishable_test_dataset/` | Versioned grain-preserving evaluation suite: Arkansas NDC9-quarter demand/shortage, county-drug annual demand, and supplier-NDC monthly continuation, with manifest hashes, deterministic county regions, and lag policy. |
| `model/docs/TARGET_AVAILABILITY.md` | Evidence audit separating public direct targets, operational inputs, and unavailable weekly/monthly county-by-supplier pharmacy labels. |
| `model/arkansas_pharma_signal/universal_forecast.py` | Filterable forecast contract with target cadence, geography, supplier resolution, direct-pharmacy-observation flag, and promotion status on every output row. |
| `model/arkansas_pharma_signal/metric_feature_store.py` | Grain-preserving adapter from qualified forecast rows to downstream regression features; default keys include period, geography, county, drug, labeler, and supplier. |
| `model/scripts/build_metric_feature_store.py` | Reproducible command that validates the qualified forecast surface and writes the feature store plus source/output hash metadata. |
| `model/artifacts/evaluation/quarterly_rolling_1y_metrics.json` | Six-fold one-year quarterly demand evaluation; includes the separate requested 75% numeric/state accuracy gate and promotion status. |
| `data/targeted_additions/fda_shortages_recalls_current/data/fda_enforcement_2023_current.csv.gz` | Recent FDA enforcement/recall records. |
| `data/targeted_additions/disease_surveillance_current/data/cdc_fluview_ar_national_weekly.csv.gz` | Arkansas and national weekly ILI from Delphi/CDC FluView. |
| `model/artifacts/evaluation/weekly_fluview_ar_proxy_metrics.json` | Seven-fold strict next-week Arkansas WILI proxy evaluation; healthcare-utilization proxy only, not pharmacy dispensing truth. |
| `data/targeted_additions/disease_surveillance_current/data/cdc_nndss_ar_national_weekly.csv.gz` | Arkansas and national weekly notifiable disease signals. |
| `data/targeted_additions/disease_surveillance_current/data/cdc_wastewater_ar_site_weekly.csv.gz` | Arkansas site-level wastewater viral activity for respiratory pathogens. |
| `data/targeted_additions/arkansas_news_3dlnews/data/arkansas_3dlnews_monthly_keyword_counts.csv.gz` | Monthly Arkansas news keyword counts by health/supply/disaster group. |
| `data/targeted_additions/arkansas_news_3dlnews/data/arkansas_3dlnews_targeted_article_metadata.csv.gz` | Arkansas news metadata for targeted articles. |
| `data/targeted_additions/nppes_provider_locations/data/arkansas_nppes_provider_locations.csv.gz` | Arkansas provider geography and identifiers. |
| `data/targeted_additions/arkansas_pharmacy_roster/data/` | Historical Arkansas pharmacy/facility roster outputs when present. |
| `data/targeted_additions/arkansas_pharmacy_roster/data/arkansas_pharmacy_facilities.csv.gz` | Official Arkansas pharmacy-directory facilities by city/year; used for pharmacy access exposure counts. |
| `data/targeted_additions/cdc_vsrr_arkansas/data/cdc_vsrr_arkansas_county_monthly.json` | Current CDC provisional county overdose-count snapshot filtered to Arkansas; suppressed rows remain absent and are never interpreted as zero. No historical publication vintages are available, so it is not a publishable forecast target. |
| `data/targeted_additions/cdc_vsrr_arkansas/data/source_manifest.json` | Retrieval URL, timestamp, row counts, SHA-256, and suppression/rolling-count limitations for the CDC VSRR archive. |
| Legacy news-model output, when supplied externally | Output of the separately maintained 20-signal FLAN-T5-small news model: monthly Arkansas/national public-news proxy signals. Consumed only through `news_only_adapter.py`; not direct pharmacy inventory or dispensing truth. The legacy directory is not required for the current checkout. |
| `model/artifacts/news/news_only_catalog_features.csv.gz` and `.json` | Validated annual bridge artifact, including source/output SHA-256 hashes, signal IDs, row count, and provenance URLs. |

### NADAC historical coverage qualification

CMS states that NADAC is updated weekly and publishes monthly and weekly files
([CMS NADAC operations](https://www.medicaid.gov/medicaid/nadac)). The local
`nadac_ndc_weekly.csv.gz` artifact contains 2,719,680 rows but only 109 distinct
`as_of_date` values from 2013-11-28 through 2022-12-28. It remains qualified as
a near-real-time input when refreshed from CMS. A strict seven-day transition
benchmark was nevertheless possible for the dense 2021-2022 portion: the
learned ridge layer lost to persistence nationally (`0.672%` versus `0.247%`
WAPE) and on the Arkansas Medicaid-exposed NDC subset (`0.493%` versus
`0.162%`). The learned price target is therefore rejected as an improvement;
the persistence result is retained only as a baseline and is not a shortage,
demand, or local-inventory claim.

## Official Public Sources Cited

- FDA/openFDA Drug Shortages API: `https://open.fda.gov/apis/drug/drugshortages/`
- openFDA Drug NDC Directory API overview: `https://open.fda.gov/apis/drug/ndc/`
- FDA National Drug Code Directory: `https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory`
- openFDA Drug Shortages download endpoint: `https://open.fda.gov/apis/drug/drugshortages/download/`
- FDA FAERS/AEMS quarterly data: `https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files` (screened, not used as an inventory metric)
- openFDA drug adverse-event API: `https://open.fda.gov/apis/drug/event/` (quarterly and lagged; screened, not used)
- openFDA API overview: `https://open.fda.gov/apis/`
- CMS Medicare Part D Prescribers by Provider and Drug: `https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug`
- CMS Medicare Quarterly Part D Spending by Drug: `https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-quarterly-part-d-spending-by-drug`
- CMS Monthly/Quarterly Part D Formulary and Pharmacy Network PUF: `https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/monthly-prescription-drug-plan-formulary-and-pharmacy-network-information` (screened, not used; published terms restrict derivative/competitive use)
- CMS Formulary/Pharmacy Network terms: `https://data.cms.gov/sites/default/files/2023-11/350202c5-3699-442e-9a5d-448741dc4a5b/TermsandConditionsforUsePharm042009.pdf`
- Delphi Epidata FluView endpoint for CDC ILINet: `https://cmu-delphi.github.io/delphi-epidata/api/fluview.html`
- CDC FluView overview/methods: `https://www.cdc.gov/fluview/overview/index.html`
- CDC wastewater respiratory virus surveillance: `https://www.cdc.gov/wastewater/respiratory-viruses/national.html`
- CDC wastewater data catalog export for SARS-CoV-2, influenza A, and RSV WVAL: `https://catalog.data.gov/dataset/cdc-wastewater-viral-activity-level-for-sars-cov-2-influenza-a-and-rsv`
- CDC RESP-NET RSV weekly rate snapshot: `data/targeted_additions/cdc_respnet_rsv_current/data/respnet_rsv_overall_weekly.json`; source API `https://data.cdc.gov/resource/kvib-3txy`, filtered to RSV-NET, Overall age/race/sex, Overall state, observed weekly rate.
- ASHP/FDA shortage parameters context: `https://www.ashp.org/drug-shortages/current-shortages/fda-and-ashp-shortage-parameters`
- ASPE analysis of historical FDA shortage status: `https://aspe.hhs.gov/reports/drug-shortages-2018-2023`
- ASPE/NCBI methods report: `https://www.ncbi.nlm.nih.gov/books/NBK611681/`
- PADI-web expert-annotated event corpus: `https://doi.org/10.57745/99SNOZ`
- CIRAD/PADI-web fine-grained annotation corpus: local ODS under
  `data/targeted_additions/epidemiology_annotation`; publication and guidelines
  are recorded in the review log.

### Live operational refresh: NWS Arkansas alerts

- Endpoint: `https://api.weather.gov/alerts/active?area=AR`
- Adapter: `arkansas_pharma_signal.live_inputs.fetch_nws_arkansas_alerts`
- Cadence: event-driven; NWS active-alert feed is intended for current alerts
- Output: alert ID, event, headline, severity, urgency, certainty, onset,
  expiry, affected-area description, UGC zones, and retrieval metadata
- Limitation: weather/disaster context only; it is not a demand, supply, or
  shortage label and does not modify historical training artifacts

### Live operational refresh: NWS hourly point forecasts

- Endpoints: `https://api.weather.gov/points/{latitude},{longitude}` followed by
  the returned `forecastHourly` grid endpoint
- Adapter: `arkansas_pharma_signal.live_inputs.fetch_nws_hourly_forecast`
- Cadence: hourly forecast periods; the point response identifies the official
  forecast office/grid for the requested coordinate
- Output: valid-time interval, temperature, precipitation probability,
  relative humidity, wind speed/direction, day/night flag, issuance time, and
  point coordinates
- Provenance: retrieval time, points URL, forecast URL, issuance time, and
  forecast validity bounds are returned in metadata
- Limitation: this is an operational weather context input only. It has no
  independently qualified pharmacy target and must not be backfilled into
  historical evaluation without a timestamped archive.

`build_nws_daily_weather_features` is the deterministic feature boundary for
hourly responses. It aggregates only observed forecast periods by UTC date
and point, emits an hour-count completeness field, and leaves absent periods
missing rather than imputing a full day.

Opt-in CLI refresh:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . refresh-weather --latitude 34.7465 --longitude -92.2896
```

The command writes raw and daily files under `model/artifacts/live/` and
provenance under `model/artifacts/metadata/`; it does not modify historical
training or evaluation artifacts.

## Data Rules

1. Do not synthesize rows, labels, inventory, locations, suppliers, or disease observations.
2. Numeric values may be extracted from real text when the source and extraction rule are recorded.
3. Derived features are allowed only when they are deterministic transformations of real observations, such as lags, rolling means, deltas, rates, and anomalies.
4. Missing values must remain missing or be handled by model-native missingness indicators. Do not backfill unobserved source data into fake observations.
5. Every new data source must add a source manifest with source URL, retrieval time, license/access note, and transformation note.

### Historical FDA shortage archive qualification

ASPE reports that it reconstructed 2018-2023 NDC-level shortage histories from
dated Internet Archive snapshots of the FDA downloadable shortage workbook and
shortage-detail pages. This is a credible candidate for a future monthly
shortage target, but the reconstructed row-level artifact is not published by
ASPE and the archive retrieval was not reproducible in the current environment.
It is therefore research evidence only and is not included in training,
backtesting, or the universal forecast. Current FDA/openFDA status remains
context and event evidence; a missing event is not treated as confirmed
availability.

An additional reproducibility check queried Common Crawl indexes from 2022
through 2025 for the FDA shortage page named by the ASPE method. All four
queries returned no captures, so no archived snapshot was retrieved or added.
This does not invalidate the ASPE analysis; it means the row-level historical
artifact cannot currently be regenerated from an openly retrievable archive in
this environment.

### Reproducible historical FDA archive acquisition

The Wayback CDX endpoint was subsequently reachable and returned 99 deduplicated
CSV captures for the FDA downloadable shortage endpoint. MediTrack retained 97
captures from April 2020 through July 20, 2026, with raw-byte checksums in
`data/targeted_additions/fda_shortage_archive/raw/SHA256SUMS`. The builder under
`data/targeted_additions/fda_shortage_archive/scripts/` extracts NDCs, initial
posting dates, observed resolution dates, and constructs a monthly national
status panel. Unresolved products are right-censored; absence after the final
capture is never converted to availability.

This panel is a valid public supply-duration/continuation target, but it is not
an Arkansas-local shortage label. Its first compact temporal evaluation was
dominated by persistent positive states: raw accuracy was approximately 99.5%,
while balanced accuracy was approximately 50% and AUROC was below 0.50. It is
therefore retained for external supply-layer research and not used to claim
publishable Arkansas forecasting skill.

The panel was later extended with the available 2024-2026 captures. The current
artifact covers January 2012 through July 2026 and contains 254,316 rows and
2,551 NDC9 products. The extension is suitable for recent-regime scoring but
does not change the target's national, right-censored semantics.
### Live operational refresh: openFDA Drug Shortages

- Endpoint: `https://api.fda.gov/drug/shortages.json`
- Adapter: `arkansas_pharma_signal.live_inputs.fetch_openfda_shortages`
- Cadence: daily according to the openFDA endpoint documentation
- Use: opt-in `forecast-universal --live-fda-shortages`; historical S_D files remain the evaluation source
- Required provenance: endpoint, retrieval date, and `meta.last_updated` are written to universal forecast metadata
- Limitation: FDA-reported shortage status is not pharmacy inventory and does not provide Arkansas county allocation

### Live operational context: National Weather Service hourly forecasts

- Endpoint: `https://api.weather.gov/points/{latitude},{longitude}` followed by the returned `forecastHourly` endpoint
- Adapter: `arkansas_pharma_signal.live_inputs.fetch_nws_hourly_forecast`
- Feature boundary: `arkansas_pharma_signal.live_inputs.build_nws_daily_weather_features`
- Cadence: hourly source, deterministic daily aggregation
- Geography rule: a point forecast is point-level unless a verified Arkansas county FIPS is supplied by the caller; the adapter never infers or broadcasts county weather
- Limitation: Census internal points provide a reproducible query location, not county-wide weather truth; the model must not interpret a point forecast as an area average

The county registry is now available at
`model/artifacts/geography/arkansas_county_weather_points.csv` and is built by
`build-weather-geography` from the Census 2025 county Gazetteer. Census
provides representative internal points, so the resulting metric remains
county-keyed point context rather than an area-average weather observation.

Use `refresh-county-weather` to query one NWS forecast per registry row. The
command fails on an incomplete county response and writes hourly/daily live
artifacts only after the complete surface is returned.

### Historical county weather observations

- Local source: `data/final_data/raw_downloads/noaa_daily_summaries_*.json`
- Normalizer: `arkansas_pharma_signal.historical_weather.load_noaa_daily_summaries`
- Aggregator: `arkansas_pharma_signal.historical_weather.build_county_weather_panel`
- Output: `model/artifacts/weather/arkansas_county_weather_weekly_monthly.csv.gz`
- Cadence: observed weekly and monthly rows from daily station summaries
- Spatial method: each Census county internal point is assigned its nearest available Arkansas station; station ID, coordinates, distance, and observed-day coverage are retained
- Missingness: no interpolation or imputation; partial periods remain visible through `weather_coverage_ratio`
- Limitation: nearest-station distance reaches approximately 149 km in the current eight-station local source, so county rows are contextual proxies and not direct county weather observations

The `attach_weather_context` integration seam joins these rows only on exact
cadence, period, and county keys. It is a feature-preparation utility, not a
claim that the current annual training model has been retrained with weather.
