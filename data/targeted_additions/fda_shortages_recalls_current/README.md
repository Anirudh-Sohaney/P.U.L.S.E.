# FDA Drug Shortages & Enforcement (Recall) Data — Current

## Sources
- Official name: openFDA API (U.S. FDA)
- Official URLs:
  - Shortages: https://api.fda.gov/drug/shortages.json
  - Enforcement: https://api.fda.gov/drug/enforcement.json
  - API docs: https://open.fda.gov/apis/drug/

## Retrieval
- Retrieved at (UTC): 2026-08-12
- API URL examples:
  - `https://api.fda.gov/drug/shortages.json?limit=1000&skip=0`
  - `https://api.fda.gov/drug/enforcement.json?limit=1000&skip=0&search=report_date:[20230101 TO 20261231]`
- Pagination: `limit=1000`, `skip` increments of 1000 until fewer than 1000 rows returned; 0.4 s delay between pages.
- Tool: `scripts/fetch_fda_openfda.py`.

## Files
- `raw/drug_shortages.json` — **1,637** current shortage records (all).
- `raw/drug_enforcement.json` — **3,168** enforcement/recall records, report_date 2023-01-01 → present.
- `data/fda_shortages_current.csv.gz`
- `data/fda_enforcement_2023_current.csv.gz`

## Fields kept
Shortages: status, generic_name, brand_name, company_name, presentation, availability, related_ndc, drug_category, is_affected, start_date, update_date, reason, therapy_area, openfda sub-object (skipped), etc. (all shortage JSON fields except `openfda`).
Enforcement: recall_number, recalling_firm, report_date, reason_for_recall, product_type, status, distribution_pattern, classification, code_info, product_description, product_quantity, city, state, country, initial_firm_notification, voluntary_mandated, openfda sub-object (skipped), etc.

## Filters used
- Shortages: none (all records returned by API).
- Enforcement: `report_date:[20230101 TO 20261231]`.

## License / access notes
- Public data from U.S. FDA. openFDA Terms of Service apply (free API, data is public). Disclaimer: openFDA data are unvalidated.

## Known limitations
- List/dict fields (e.g., `openfda`, multi-valued fields) are stored JSON-serialized in the CSVs.
- Enforcement `report_date` filter uses API's reported recall date; 3,168 of total matched.
- Shortage API did not return a `meta.results.total` count; 1,637 fetched (all pages).