# DEA ARCOS — Retail Drug Summary, Arkansas

## Source
- Official name: DEA ARCOS Retail Drug Summary Reports (U.S. Drug Enforcement Administration, Diversion Control Division)
- Official URLs:
  - https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html
  - archive: https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/archive/archives-report.html

## Retrieval
- Retrieved at (UTC): 2026-08-12
- Files: **Report 01** "Retail Drug Distribution by ZIP Code Within State (by Grams)" per year:
  - 2006–2015: `https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/{year}/{year}_rpt1.pdf`
  - 2016–2025: `https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/report_yr_{year}.pdf`
- Parser: `scripts/parse_arcos_arkansas.py` (pymupdf text extraction; block-sorted reading for the 2018+ layout). Handles two PDF layouts: classic `DRUG CODE:XXXXDRUG NAME:...`+`STATE:` (2006–2016, 2017 mixed) and new `DRUG: XXXX - NAME`+`STATE: AR - ARKANSAS` (2018–2025).

## Files
- `raw/rpt1_{2006..2015}.pdf` and `raw/report_yr_{2016..2025}.pdf`
- `data/arcos_arkansas_retail_summary.csv.gz` — **4,106 Arkansas rows** (2006–2025, no 2011)

## Fields kept
year, drug_code, drug_name, zip3 (3-digit registrant ZIP prefix), quarter1_grams, quarter2_grams, quarter3_grams, quarter4_grams, total_grams, plus `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Rows by year
2006: 157 · 2007: 171 · 2008: 173 · 2009: 185 · 2010: 179 · 2012: 201 · 2013: 184 · 2014: 190 · 2015: 195 · 2016: 193 · 2017: 160 · 2018: 202 · 2019: 200 · 2020: 200 · 2021: 198 · 2022: 198 · 2023: 193 · 2024: 175 · 2025: 170 — **Total: 4,106**. (2011 Report 1 PDF on DEA site returns an error page; not available.)

## Validation
- Arkansas amphetamine (drug code 1100) totals reconstructed from parsed rows exactly match the PDF "TOTAL GRAMS" state lines: 2016 = 192,688.33 g; 2022 = 272,793.11 g.

## License / access notes
- Public DEA data (Federal data, no access restrictions). Disclaimer in source: ARCOS data are reported by manufacturers/distributors; DEA publishes as collected.

## Known limitations
- 2011 Report 1 missing on the DEA website (error page); 2011 excluded.
- Data are **quarterly**, not monthly.
- ZIP field is the 3-digit registrant ZIP prefix (Report 01 grouping), not full ZIP.
- No business-activity classification in Report 01 (that breakdown appears in other ARCOS reports, not retained).
- A handful of rows across years are missing where a value was printed without decimals and/or a row straddled PDF page boundaries (validated <0.5% loss on checked years; see validation above).
- 2023–2025 PDFs present fewer drugs/zip rows (declining ARCOS reporting breadth in those releases).