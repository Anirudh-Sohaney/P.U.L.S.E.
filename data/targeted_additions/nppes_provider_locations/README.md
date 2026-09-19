# NPPES NPI Downloadable File (Arkansas providers)

## Source
- Official name: NPPES National Provider Identifier (NPI) Data Dissemination Files
- Official URL: https://download.cms.gov/nppes/NPI_Files.html

## Retrieval
- Retrieved at (UTC): 2026-08-12
- File: `NPPES_Data_Dissemination_August_2026_V2.zip` (1,151,460,412 bytes), monthly full replacement, V.2, dated 2026-08-10
- URL used: https://download.cms.gov/nppes/NPPES_Data_Dissemination_August_2026_V2.zip
- Also saved (raw only): `NPPES_Deactivated_NPI_Report_081026_V2.zip` (monthly deactivation XLSX)
- Extraction tool: `scripts/extract_nppes_arkansas.py` (Python stdlib `zipfile`+`csv`, streamed; the 11.6 GB `npidata_*.csv` was never held fully in memory).

## Files
- `raw/NPPES_Data_Dissemination_August_2026_V2.zip` — full monthly zip
- `raw/NPPES_Deactivated_NPI_Report_081026_V2.zip`
- `raw/arkansas_nppes_providers.csv` — AR-filtered rows from `npidata_pfile_20050523-20260809.csv`
- `data/arkansas_nppes_provider_locations.csv.gz` — 97,528 AR records
- `data/arkansas_nppes_practice_location_reference.csv.gz` — 8,717 AR secondary-practice-location rows from `pl_pfile_20050523-20260809.csv`

## Fields kept (npidata)
NPI, Entity Type Code, legal/last/first/middle name, credential text, business mailing address line1 + city/state/postal, business practice location address line1/line2 + city/state/postal, enumeration date, last update date, deactivation reason code, deactivation date, reactivation date, 15 healthcare taxonomy codes + primary-switch flags, 15 license-number state codes. Plus `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Filters used
- npidata: kept rows where business **mailing** state OR business **practice location** state equals `AR`.
- practice-location reference: kept rows where secondary practice location state equals `AR`.
- Pharmacy/prescriber relevance: no taxonomy filter applied (all Arkansas NPIs retained; taxonomy columns let downstream filter to pharmacies/prescribers).

## Row counts
- AR NPI records: **97,528**
- AR secondary practice location reference rows: **8,717**

## License / access notes
- Public data from CMS/NPPES (Centers for Medicare & Medicaid Services). No restrictions on use. Issuance of an NPI does not validate licensure.

## Known limitations
- Taxonomy **codes** only (not descriptions) in CSV; descriptions are in the `NPPES_Data_Dissemination_CodeValues.pdf` included in the zip (not extracted to data files).
- V.2 format: extended field lengths for name fields (see fileheader saved in manifest).
- Practice location reference file contains only NON-primary (secondary) locations; primary locations live in the main npidata file.
- Deactivation report (XLSX) saved as raw only; not normalized (deactivation dates also present in npidata rows).
- Address text is as-reported by providers; Arkansas filter depends on provider-reported state fields.