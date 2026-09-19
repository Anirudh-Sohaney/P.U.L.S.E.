# Demand & Price Extraction

This directory now contains cleaned, source-linked extracts for the demand and price source set.

## What was extracted

| Source | Output | Status | Transform |
|---|---|---|---|
| Medicaid State Drug Utilization Data | `data/medicaid_sdud_state_quarter.csv.gz` | ok | state/year/quarter aggregate |
| NADAC | `data/nadac_ndc_weekly.csv.gz` | partial | row-level cleaned CSV |
| Medicare Part D PUFs | `data/cms_partd_geography_drug.csv.gz` | ok | row-level cleaned CSV |
| openFDA Drug Shortages | `data/openfda_shortages_2012_2022.csv.gz` | ok | row-level cleaned CSV |
| ASHP Shortages API | `data/ashp_shortages_RESTRICTED.txt` | restricted | access status documented; no public bulk extract |
| EMA Shortages Catalogue | `data/ema_shortages_2012_2022.csv.gz` | ok | row-level cleaned CSV; current catalogue is not historical 2012-2022 |
| openFDA Enforcement/Recalls | `data/openfda_enforcement_2012_2022.csv.gz` | ok | row-level cleaned CSV |
| openFDA NDC Directory | `data/openfda_ndc_2012_2022.csv.gz` | ok | row-level cleaned CSV with marketing-start-date filter |
| MSH Intl Medical Products Price Guide | `data/msh_price_guide_2015_NOT_EXTRACTED.txt` | blocked | PDF downloaded; image-only extraction blocked without OCR |
| OECD pharma consumption + key indicators | `data/oecd_pharma_2012_2022.csv.gz` | ok | row-level cleaned CSV |
| WHO GHED | `data/who_ghed_2012_2022.csv.gz` | ok | xlsx sheet1 -> CSV filtered to years |
| IQVIA MIDAS / Xponent | `data/iqvia_MIDAS_Xponent_PROPRIETARY.txt` | proprietary | access status documented; paid subscription required |

## Notes

- Medicaid SDUD is aggregated to state/year/quarter/utilization because the raw annual files are too large to keep verbatim in the current workspace.
- Other sources are preserved at row level with selected columns normalized into CSV/GZ outputs.
- ASHP and IQVIA are explicitly catalogued as access-controlled sources: ASHP requires a key/commercial license and IQVIA is proprietary/paid.
- The MSH PDF is downloaded and retained, but is image-only and cannot be cleanly tabulated without OCR tooling; no fabricated rows are emitted.
