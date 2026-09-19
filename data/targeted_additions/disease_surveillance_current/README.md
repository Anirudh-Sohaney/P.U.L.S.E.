# Disease Surveillance — Arkansas & National (current)

Weekly infectious-disease surveillance signals for Arkansas (`ar`) and US national (`nat`), 2012 through latest available. Three normalized datasets, all in `data/`.

## Files

| File | Rows | Coverage | Notes |
|---|---|---|---|
| `data/cdc_fluview_ar_national_weekly.csv.gz` | 1,522 | 2012–2026 (latest epiweek) | Delphi FluView ILI outpatient surveillance, weekly, ar + nat. Latest published issue per epiweek. `ili`/`wili` = ILI%, `num_ili`, `num_patients`, `num_providers`. |
| `data/cdc_nndss_ar_national_weekly.csv.gz` | 27,612 | 2022–2026 | NNDSS weekly provisional notifiable-disease counts (m1–m4 monthly columns) for Arkansas + US RESIDENTS. Source dataset starts 2022; 2012–2021 not available in this source. |
| `data/cdc_wastewater_ar_site_weekly.csv.gz` | 3,536 | 2020–2026 | CDC wastewater site-level weekly Viral Activity Level (WVAL 0–10 + category) for Arkansas sites, per pathogen (SARS-CoV-2, Influenza A virus, RSV). |
| `data/cdc_wastewater_national_weekly.csv.gz` | 711 | 2020–2026 | **Derived** national weekly WVAL = mean `site_wval` across all reporting US sites per `week_end`/`pathogen_target`. Not an official CDC national series. |

`data/cdc_nndss_ar_national_weekly.csv.gz` uses `label` for the reported disease; `m1`–`m4` plus `*_flag` columns ("-" = not reported / N not applicable, "-1" or similar suppression notes) follow the CDC table conventions described in the source metadata.

## Schema conventions

All rows carry `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`. `epiweek` is the CDC MMWR epiweek integer (YYYYWww). `week_end` in wastewater files is the Saturday ending the surveillance week.

## Raw

- `raw/delphi_fluview_ar_nat.json` / `.csv` — API response + flattened table
- `raw/nndss_weekly_ar_nat.json` — Socrata rows
- `raw/cdc_wastewater_wval_all.json` / `.csv` — all-site WVAL pull (526,007 rows) used to derive the national weekly mean

## Sources

- Delphi FluView API: `https://delphi.cmu.edu/epidata/api.php?source=fluview&regions=ar,nat&epiweeks=201201-202653`
- CDC NNDSS Weekly Data (Socrata): `https://data.cdc.gov/dataset/NNDSS-Weekly-Data/x9gk-5huc`
- CDC archived weekly and annual tables: `https://www.cdc.gov/nndss/infectious-disease/weekly-and-annual-disease-data-tables.html`
- CDC Wastewater Viral Activity Level (Socrata): `https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re`

## Limitations

- **NNDSS years:** the current `x9gk-5huc` extract is described as 2022–present, but the checked-in Arkansas rows contain only 2025–2026; 2022–2024 rows are national. CDC documents separate archived weekly tables from 2014 onward, but those historical Arkansas tables have not yet been ingested here and must not be inferred as zeroes.
- **National WVAL:** no official national jurisdiction series is published in `atcp-73re`; the national file is a computed mean of site-level WVAL (see formula above).
- **FluView revisions:** only the latest `issue` (revision) per epiweek is retained; backcast revisions are not included.
- The referenced healthdata.gov dataset `x5bf-y9if` returned "no row or column access to non-tabular tables" (broken/legacy pointer, now consolidated into `atcp-73re`).

Script: `scripts/fetch_disease_surveillance.py` (re-runnable).
