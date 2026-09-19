# Targeted Additions — Arkansas-First and Expanded Health Data Sources

Supplemental datasets normalized for MediTrack research, with Arkansas as the
primary geography and national, neighboring-state, and global context where
local data is unavailable. Each subdirectory contains `data/` (normalized
`.csv.gz`), `raw/` (source files), a `README.md`, and a `source_manifest.json`.

The source inventory is intentionally broader than Arkansas demand. It covers
public disease and outbreak indicators, FDA supply events, pharmacy and
provider geography, controlled-substance distribution, news, weather,
economics, trade, sanctions, and infrastructure. A source being present does
not make it a qualified model signal: coverage, publication lag, geography,
drug identity, leakage risk, and chronological held-out performance must still
be checked.

## Expansion sources under review

The DEA ARCOS material is retained as a distribution proxy, not as pharmacy
dispensing or inventory truth. The review includes the DEA retail summaries,
the Washington Post ARCOS visualizations and database documentation, the
Nature Scientific Data ARCOS dataset, Indiana's metadata record, and the
provided Northern District of Ohio report. These sources can support wider
state and ZIP3 distribution features, subject to licensing, reproducibility,
and time-coverage checks.

Useful references:

- [DEA retail drug summary reports](https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html)
- [ARCOS data explorer](https://wpinvestigative.github.io/arcos/)
- [Nature Scientific Data ARCOS dataset](https://www.nature.com/articles/s41597-024-03534-3)
- [Indiana ARCOS metadata](https://maps.indiana.edu/metadata/Demographics/Health_Opioid_Data_ARCOS_DEA.html)

These references complement the locally parsed Arkansas ARCOS table; they do
not replace official-source validation or create labels for local pharmacy
inventory.

## Dataset index

| Family | Data file | Rows* | Coverage | Agencies |
|---|---|---|---|---|
| CMS Part D prescriber/provider/drug | `cms_partd_prescriber_provider_drug/data/arkansas_partd_provider_drug_by_year.csv.gz` | 3,425,234 | 2013–2024 | CMS |
| NPPES provider locations | `nppes_provider_locations/data/arkansas_nppes_provider_locations.csv.gz` | 97,528 | Aug 2026 snapshot | CMS |
| NPPES practice location reference | `nppes_provider_locations/data/arkansas_nppes_practice_location_reference.csv.gz` | 8,717 | Aug 2026 snapshot | CMS |
| HHS Medicaid Provider Spending by NDC | `hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_ndc_monthly.csv.gz` | 28,386 | 2018-01–2024-12 | HHS/CMS |
| HHS Medicaid Provider Spending by NDC, county allocation | `hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_county_ndc_monthly.csv.gz` | 24,050 | 2018-01–2024-12 | HHS/CMS/NPPES/Census |
| FDA NDC directory — products | `fda_ndc_directory/data/fda_ndc_products_current.csv.gz` | 115,223 | current | FDA |
| FDA NDC directory — packages | `fda_ndc_directory/data/fda_ndc_packages_current.csv.gz` | 216,542 | current | FDA |
| FDA drug shortages | `fda_shortages_recalls_current/data/fda_shortages_current.csv.gz` | 1,637 | current | FDA |
| FDA enforcement (recalls) | `fda_shortages_recalls_current/data/fda_enforcement_2023_current.csv.gz` | 3,168 | 2023-01-01→now | FDA |
| AR pharmacy facility roster | `arkansas_pharmacy_roster/data/arkansas_pharmacy_facilities.csv.gz` | 11,716 | 2008–2022 | ASBP |
| DEA ARCOS retail (AR) | `dea_arcos_arkansas/data/arcos_arkansas_retail_summary.csv.gz` | 4,106 | 2006–2025 | DEA |
| FluView ILI surveillance | `disease_surveillance_current/data/cdc_fluview_ar_national_weekly.csv.gz` | 1,522 | 2012–2026 weekly | CDC/Delphi |
| NNDSS weekly notifiable diseases | `disease_surveillance_current/data/cdc_nndss_ar_national_weekly.csv.gz` | 27,612 | 2022–2026 | CDC |
| Wastewater WVAL (AR sites) | `disease_surveillance_current/data/cdc_wastewater_ar_site_weekly.csv.gz` | 3,536 | 2020–2026 | CDC |
| Wastewater WVAL (derived national) | `disease_surveillance_current/data/cdc_wastewater_national_weekly.csv.gz` | 711 | 2020–2026 | CDC (derived) |
| 3DLNews2 AR news (targeted) | `arkansas_news_3dlnews/data/arkansas_3dlnews_targeted_article_metadata.csv.gz` | 1,246 | 2013–2022 | 3DLNews2 |
| 3DLNews2 AR news — monthly keyword counts | `arkansas_news_3dlnews/data/arkansas_3dlnews_monthly_keyword_counts.csv.gz` | 507 | 2013–2022 | 3DLNews2 |

\* Row counts exclude header. GDELT produced one raw timeline response but did not complete normalized outputs because the DOC API repeatedly returned 429 throttling responses; 3DLNews2 Arkansas newspaper files were retained as the normalized news fallback.

## Schema conventions

Every normalised CSV carries `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes` columns identifying provenance and any transformation caveats. All times UTC.

## Sources

- CMS Part D: `https://data.cms.gov/provider-data/` (Part D Prescriber PUF)
- NPPES: `https://download.cms.gov/nppes/NPPES_Data_Dissemination_August_2026_V2.zip`
- FDA NDC: `https://www.accessdata.fda.gov/cder/ndctext.zip`
- FDA openFDA: `https://api.fda.gov/drug/...` (shortages, enforcement)
- ASBP: Arkansas State Board of Pharmacy directory PDFs via OCLC CONTENTdm
- DEA: ARCOS retail drug summaries, Report 1
- FluView: Delphi Epidata `source=fluview`
- NNDSS: `https://data.cdc.gov/dataset/NNDSS-Weekly-Data/x9gk-5huc`
- Wastewater: `https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level... /atcp-73re`
- GDELT: DOC 2.0 API `https://api.gdeltproject.org/api/v2/doc/doc`
- 3DLNews2: `https://g-e840e1.746abe.8540.data.globus.org/1-Google/1-Newspaper/preprocessed_state/AR/`

## Sources attempted but not available

- **ASBP license verification search** (per-license) — behind an Azure WAF JS challenge; not bypassed. Facility data obtained from official directory PDFs instead.
- **ARCOS Report 1, year 2011** — DEA returns an error page (34 KB) for every 2011 report; gap.
- **2007 AR pharmacy directory PDF** — scanned images only, no embedded text; OCR of a personal-pharmacist roster was not pursued.
- **NNDSS 2012–2021 weekly** — the public `x9gk-5huc` dataset begins 2022; earlier weekly NNDSS not exposed by this source.
- **healthdata.gov wastewater `x5bf-y9if`** — catalog pointer now returns "non-tabular" / no row access; the live dataset is `atcp-73re` on data.cdc.gov.
- **GDELT DOC API article-list/timeline expansion** — repeatedly returned 429 rate-limit responses after the first timeline request. `gdelt_news_signals/raw/timeline_arkansas_pharmacy.json` and `gdelt_news_signals/scripts/fetch_gdelt_doc.py` are retained, but normalized GDELT data was not produced. Normalized Arkansas news signals were obtained from 3DLNews2 instead.
