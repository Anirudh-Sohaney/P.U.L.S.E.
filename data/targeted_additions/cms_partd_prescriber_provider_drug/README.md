# CMS Part D Prescribers - by Provider and Drug (Arkansas)

## Source
- Official name: Medicare Part D Prescribers - by Provider and Drug
- Official URL: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug
- Machine-readable catalog: https://data.cms.gov/data.json

## Retrieval
- Retrieved at (UTC): 2026-08-12
- Method: enumerated 12 per-year distributions from data.cms.gov/data.json (`Medicare Part D Prescribers - by Provider and Drug` dataset, distribution entries titled `...NPIBN.csv`), then streamed each full CSV and kept only rows where `Prscrbr_State_Abrvtn == 'AR'`.
- Download tool: custom script `scripts/download_partd_arkansas.py` using Python stdlib `urllib` + `csv`, chunkless row streaming.
- Example URL used (2024): https://data.cms.gov/sites/default/files/2026-05/0ae165f4-eb44-495d-8cac-67f4571b6b83/MUP_DPR_RY26_P04_V10_DY24_NPIBN.csv (full list in `source_manifest.json`).

## Files
- `raw/arkansas_partd_{2013..2024}.csv` — AR-filtered rows per year (original full CSVs are 1.4–4.1 GB/year and were NOT retained; see limitations).
- `data/arkansas_partd_provider_drug_by_year.csv.gz`

## Fields kept
`Prscrbr_NPI`, `Prscrbr_Last_Org_Name`, `Prscrbr_First_Name`, `Prscrbr_City`, `Prscrbr_State_Abrvtn` (`AR`), `Prscrbr_Type` (specialty), `Brnd_Name`, `Gnrc_Name`, `Tot_Clms`, `Tot_30day_Fills`, `Tot_Drug_Cst`, `Tot_Benes`, plus added `year`, `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Filter used
- State filter: `Prscrbr_State_Abrvtn == 'AR'`.
- Filtering performed during streaming download, before any local persistence of full files.

## Row counts by year
2013: 265,038 · 2014: 270,034 · 2015: 274,350 · 2016: 277,826 · 2017: 282,027 · 2018: 281,884 · 2019: 284,671 · 2020: 281,701 · 2021: 284,274 · 2022: 292,177 · 2023: 307,793 · 2024: 323,459 · **Total: 3,425,234**

## License / access notes
- CMS public data; public domain / no access restrictions. Source: CMS research data (Part D Prescribers), subject to CMS data use policy (https://www.cms.gov/about-cms/agency-information/aboutwebsite/using-website/website-policies).

## Known limitations
- Dataset has **no ZIP** field — provider city/state only; no ZIP preserved.
- Full multi-GB original CSVs not retained (disk budget); only AR-filtered raw subsets are stored.
- `Prscrbr_NPI` is the prescribing NPI; not all prescribers are licensed to prescribe in AR (based on claim address of record).
- 2013–2020 files were decoded with `utf-8-sig` and `errors='replace'` (2020 file had non-UTF-8 bytes); affected characters are replacement chars.
- Latest available year is 2024 (data release 2026-05).