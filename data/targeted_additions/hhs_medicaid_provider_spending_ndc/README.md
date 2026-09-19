# HHS Medicaid Provider Spending by NDC

This family contains the locally downloaded July 2026 HHS Open Data release
and an Arkansas pharmacy-taxonomy extraction. The source is public and covers
January 2018 through December 2024 at billing-provider NPI x prescribing-
provider NPI x NDC x claim month.

The extraction keeps rows whose billing NPI appears in the local CMS NPPES
Arkansas file with at least one taxonomy code beginning `3336` (pharmacy). It
then aggregates `TOTAL_CLAIM_LINES` by month and NDC. This is Medicaid/CHIP
claim activity, not all-payer dispensing, supplier allocation, or inventory.
HHS suppresses small cells; missing rows are not treated as zero demand.

The candidate was tested with 57 rolling-origin folds and 12,998 held-out
consecutive transitions. It failed promotion with 51.41% five-state exact
accuracy, 9.10% numeric within-5% accuracy, and 65.52% precision for states
3-4. It remains available as training-only context.

An additional county panel uses the provenance-bearing NPPES practice-city
and Census-geocoder crosswalk. It contains 24,050 observed county-NDC-month
rows across 72 mapped counties. Its 8,813 held-out transitions scored 43.59%
exact, 11.06% within 5%, and 53.14% precision for states 3-4, so it is also
training-only context rather than a promoted metric.

Source: https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/
