# FDA National Drug Code (NDC) Directory — Current

## Sources
- Official name: National Drug Code Directory
- Official URLs:
  - https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory
  - https://www.fda.gov/drugs/development-approval-process-drugs/national-drug-code-database-background-information
- Downloadable data file: `ndctext.zip` (SQL flat files `product.txt`, `package.txt`) at https://www.accessdata.fda.gov/cder/ndctext.zip

## Retrieval
- Retrieved at (UTC): 2026-08-12
- Command/URL used: `curl -L https://www.accessdata.fda.gov/cder/ndctext.zip` (also saved `ndcxls.zip` from https://www.accessdata.fda.gov/cder/ndcxls.zip, which contained identical content).
- Normalization: Python `pandas.read_csv(sep='\t', encoding='latin-1')`; added `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Files
- `raw/ndctext.zip` (10,749,009 bytes), `raw/ndcxls.zip`
- `data/fda_ndc_products_current.csv.gz` — **115,223 product records**
- `data/fda_ndc_packages_current.csv.gz` — **216,542 package records**

## Fields kept (product)
PRODUCTID, PRODUCTNDC, PRODUCTTYPENAME, PROPRIETARYNAME, PROPRIETARYNAMESUFFIX, NONPROPRIETARYNAME, DOSAGEFORMNAME, ROUTENAME, STARTMARKETINGDATE, ENDMARKETINGDATE, MARKETINGCATEGORYNAME, APPLICATIONNUMBER, LABELERNAME, SUBSTANCENAME (active ingredients), ACTIVE_NUMERATOR_STRENGTH, ACTIVE_INGRED_UNIT, PHARM_CLASSES, DEASCHEDULE, NDC_EXCLUDE_FLAG, LISTING_RECORD_CERTIFIED_THROUGH.

## Fields kept (package)
PRODUCTID, PRODUCTNDC, NDCPACKAGECODE, PACKAGEDESCRIPTION, STARTMARKETINGDATE, ENDMARKETINGDATE, NDC_EXCLUDE_FLAG, SAMPLE_PACKAGE.

## Filters used
- None (full current directory). `NDC_EXCLUDE_FLAG` and listing-status fields retained so downstream can filter; `product.txt` includes discontinued products.

## License / access notes
- Public domain data from FDA (U.S. government work). No access restrictions; FDA open data policy applies.

## Known limitations
- "Current" listing snapshot as of retrieval date; NDC directory is updated continuously.
- Files decoded as `latin-1` (FDA source uses 8-bit encoding; non-ASCII characters decoded accordingly).
- Date fields are `YYYYMMDD` strings, not ISO dates.
- The `ndcxls.zip` and `ndctext.zip` downloads were byte-identical at retrieval time.