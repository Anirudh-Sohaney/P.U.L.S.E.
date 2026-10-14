# Data directory guide

This folder contains public-source extracts, normalized features, source
catalogs, ETL scripts, and a separate synthetic pharmacy benchmark. It mixes
different vintages and levels of readiness: a source being listed here does
not mean it was successfully downloaded, is current, or is a validated model
signal. Use each folder's `README.md`, `sources.md`, `source_manifest.json`,
`schema.json`, and coverage/status files for the detailed record-level contract.

Public data in this repository are contextual proxies. They do **not** provide
direct pharmacy inventory, dispensing, wholesaler allocation, or backorder
truth unless a source explicitly says so. The synthetic benchmark is labeled
separately below and must not be described as observed transactions.

## Folder architecture

| Folder | Contents |
|---|---|
| `arkansas/` | Legacy Arkansas corpus payload. `data.json` is stored as a Git LFS pointer; retrieve its payload with Git LFS before using it. |
| `crosswalks/` | Drug, company, industry, and geography identity links (for example NDC/RxNorm, ATC, LEI, NAICS, ZIP-to-HSA/HRR); includes extractors and source/access status. Some sources are partial or pending. |
| `demand_price/` | Historical/current drug utilization, drug prices, Medicare/Medicaid spending, NDC, shortage, and recall extracts, with extraction scripts. |
| `disease_global/` | Global disease and epidemiology observations and source catalog; includes WHO, ECDC, OWID, DHS, and related public series. |
| `disease_us/` | U.S. disease surveillance package, including FluView, NNDSS, and PLACES source adapters, schemas, and extraction status. |
| `economics_global/` | Country-level economic and institutional context (GDP, inflation, FX, industrial activity, uncertainty, and related indicators). |
| `economics_us/` | U.S. economic and pharmaceutical-manufacturing indicators, manifests, schemas, and extraction/validation tools. |
| `final_data/` | A historical assembled feature/event store: `features/` has domain and combined feature tables; `events/` has shortage, recall, and disaster events; `entities/` has entity/crosswalk records; `raw_downloads/` keeps inputs for this assembly; `sources/` and `quality/` preserve provenance and validation. Newer targeted extracts are not automatically merged into this build. |
| `international/` | International-source catalog and notes for country-level context. |
| `pharma_infrastructure/` | Pharmacy, provider, facility, manufacturer, and healthcare-system infrastructure source catalog. |
| `S_D/` | Annual pharmaceutical supply/demand records, primarily public utilization and supply-event data; organized by year and source with a drug dictionary, schema, and validation files. |
| `sanctions/` | Current OFAC SDN and consolidated sanctions lists, normalized data, scripts, and coverage metadata. The Global Sanctions Database (GSDB) is documented as unavailable, not included data. |
| `shocks_events/` | Source catalog for acute shocks and disruption events (such as disasters, recalls, shortages, and outbreak events). |
| `supply_chain/` | Monthly logistics, freight, commodity-input, and port activity indicators plus extraction notes. |
| `tariff_policy/` | Tariff and trade-policy records, schemas, manifests, and extract/validation scripts; check the manifest for which sources were actually retrieved. |
| `trade/` | Bilateral commodity trade and trade-network data, including BACI/CEPII and Eurostat-oriented extractors; large raw caches and some generated subsets are local/ignored. |
| `targeted_additions/` | Arkansas-focused and expanded public extracts grouped by source/dataset; see the inventory below. Each manifest records source status and coverage. |
| `synthetic_pharmacy_data/` | Deterministic generated clinic sales, event tags, validation, benchmark code, provenance, and benchmark results. This is the only explicitly synthetic pharmacy-sales benchmark in this folder. |
| `scripts/` | Shared extraction, text/download, and final-data assembly utilities. |

### `targeted_additions/` inventory

These are individual datasets or source-specific packages; actual files and
coverage vary by subfolder. The named agency is the source owner/provider,
not a claim that every requested dataset was obtained.

| Data package(s) | What the data represents | Principal source(s) |
|---|---|---|
| `cms_medicare_quarterly_partd`, `cms_partd_prescriber_provider_drug`, `cms_partd_geography_drug` | Medicare Part D drug spending/utilization by quarter, prescriber/provider, geography, and drug | U.S. Centers for Medicare & Medicaid Services (CMS) |
| `cms_sdud_recent` | State Medicaid outpatient drug utilization | CMS State Drug Utilization Data (SDUD) |
| `cms_nadac_historical` | National average drug acquisition cost price history | CMS NADAC |
| `hhs_medicaid_provider_spending_ndc` | Arkansas Medicaid provider/NDC spending, including a derived county allocation | HHS/CMS; NPPES and Census geography used for allocation |
| `nppes_provider_locations`, `arkansas_pharmacy_roster`, `arkansas_apcd_claim_counts` | Provider/practice locations, pharmacy facilities, and monthly Arkansas pharmacy-claim counts by submitting entity (an activity proxy, not drug-, pharmacy-, or inventory-level outcomes) | CMS NPPES; Arkansas State Board of Pharmacy directories; Arkansas APCD |
| `dea_arcos_arkansas` | Controlled-substance distribution summaries by time, drug code, and Arkansas geography; a distribution proxy, not pharmacy dispensing | DEA ARCOS |
| `fda_ndc_directory`, `fda_establishments` | Product/package identity and registered establishment/manufacturer reference data | FDA NDC Directory and FDA establishment data |
| `fda_shortage_archive`, `fda_shortages_recalls_current` | Historical/current shortage and enforcement/recall events | FDA shortage listings and openFDA enforcement data |
| `disease_surveillance_current`, `cdc_hospital_respiratory_current`, `cdc_nssp_ar_current`, `cdc_respnet_rsv_current`, `cdc_vsrr_arkansas` | Weekly disease activity, wastewater, emergency/hospital respiratory activity, RSV, and vital-statistics context | CDC, CDC Socrata, FluView/Delphi, and source-specific public releases |
| `arkansas_news_3dlnews`, `gdelt_news_signals`, `news_article_corpus` | Arkansas news metadata/keyword aggregates and broader article/event records | 3DLNews2 and GDELT; GDELT normalization was rate-limited, so its availability is incomplete |
| `biocaster`, `eventepi`, `band`, `daniel`, `padi_web_weak`, `padi_expert`, `epidemiology_annotation` | Disease/outbreak news corpora, event extraction examples, and weak/expert annotations for NLP evaluation | BioCaster, EventEpi/WHO DON/ProMED-derived materials, BAND, DAnIEL, and PADI-web research datasets; licenses and included subsets vary |
| `rxnorm_ndc_atc` | Drug-code and therapeutic-class crosswalks | NLM RxNorm/RxNav, FDA NDC, and WHO ATC resources |
| `publishable_test_dataset` | Curated chronological model-evaluation inputs; not a pharmacy transaction source | Arkansas Medicaid NDC9, FDA shortage, CDC FluView, Arkansas news, county CMS demand, and DEA ARCOS inputs; see its README for exact contract and provenance |

## Data families and sources

The source lists under each domain directory are the authoritative catalogs;
the table below summarizes their principal contents and provenance.

| Family | Data represented | Main source examples |
|---|---|---|
| Drug utilization and prices | Drug-level volume/spending, acquisition prices, NDC/product identity, shortage and recall context | CMS Part D, Medicaid SDUD, CMS NADAC, HHS/CMS provider-NDC, FDA NDC/openFDA, WHO GHED, OECD and EMA source catalogs |
| U.S. and global disease | Flu-like illness, notifiable diseases, mortality, outbreaks, prevalence, immunization, and wastewater surveillance | CDC/FluView/Delphi, CDC Socrata/NNDSS/PLACES, WHO FluNet/FluID/GHO/Disease Outbreak News, ECDC, OWID, DHS, WUENIC; some historical/proprietary series are pending |
| Arkansas/provider infrastructure | Provider and facility locations, pharmacy roster, claims counts, NPI and drug identifiers | CMS NPPES, Arkansas State Board of Pharmacy, Arkansas APCD, FDA, DEA, NLM, Census and Dartmouth Atlas |
| U.S./global economics | Prices, unemployment, GDP, FX, inflation, industrial production, institutions, and uncertainty | BLS, World Bank WDI, IMF, OECD, Penn World Table, Maddison Project, BIS, UNCTAD, V-Dem, EPU/TPU/GPR, and related catalogs |
| Supply chain, trade, and policy | Freight/ports, input prices, bilateral trade flows, tariff exposure, sanctions, and disruption events | NY Fed GSCPI, IMF PortWatch, BTS, EIA, World Bank Pink Sheet, RWI/ISL, CEPII BACI/Gravity, Eurostat Comext, USITC/Census, and OFAC |
| Assembled signals | Long-format numeric features with geography, observation time, horizon, source/quality metadata; separate event and entity tables | Built from the public-source extracts above; see `final_data/quality/build_manifest.json` and source metadata for this particular build |

Some large source payloads are Git LFS pointers; run `git lfs pull` to retrieve
their contents where available. A pointer is not the underlying dataset.

Important availability caveats: DEA ARCOS has a documented 2011 gap; GDELT
article expansion was throttled; several crosswalks require credentials or
interactive downloads; GSDB was not obtained; some source-family folders hold
catalogs/manifests but no normalized observation files. Inspect each
`source_manifest.json`, coverage file, and README before relying on a source.
Do not treat a missing or suppressed value as zero, or a public proxy as a
direct Arkansas pharmacy outcome.

## Synthetic pharmacy data generation

`synthetic_pharmacy_data/generate_synthetic_pharmacy.py` creates
`arkansas_clinic_daily_pharmacy_sales.csv`. It uses the fixed random seed
`20250915` to generate **32,880 rows**: 30 named medicines for every day from
2023-01-01 through 2025-12-31, representing one hypothetical local-government
clinic in Arkansas. The CSV contains date, state, facility type, medicine,
therapeutic class, units sold, unit price, revenue, a stockout flag, and
`injected_event_tags`.

The generator combines assumed product baselines and price trends with
Poisson count noise, a weekday clinic schedule, seasonal allergy/respiratory
patterns, and dated scenario pulses for events such as flu/COVID activity,
access changes, and supply/outbreak responses. Public reporting and citations
in `synthetic_guide.md` inform scenario timing; effect sizes, clinic scale,
product mix, and resulting sales are **modeling assumptions**, not measured
Arkansas dispensing or patient records. Event tags expose which generated
effects apply to each row. The fixed seed makes regeneration deterministic.

Regenerate and validate the sales file:

```bash
python data/synthetic_pharmacy_data/generate_synthetic_pharmacy.py
python data/synthetic_pharmacy_data/validate_synthetic_dataset.py
```

The website's `demand_inventory_snapshot.csv` is another deterministic
demonstration artifact derived from this sales history; it is not observed
inventory. Benchmark split, model comparisons, and limitations are described
in `benchmark_protocol.md`; current aggregate and per-medicine outputs are in
`current_benchmark_results.md` and `per_drug_benchmark_results.md`.

The benchmark provenance also references a frozen signal file under the
repository's local-only `test/` directory. That directory is intentionally
not distributed in this repository, so reproducing the signal-enhanced
benchmark from a clean checkout requires obtaining that signal input
separately. Do not carry old metrics over to a regenerated sales file or
present synthetic benchmark performance as real-world clinical or pharmacy
evidence.
The website ships a separate definitions-only manifest for its 1,312-ID API
contract. It contains no historical signal values and does not replace this
benchmark input.
