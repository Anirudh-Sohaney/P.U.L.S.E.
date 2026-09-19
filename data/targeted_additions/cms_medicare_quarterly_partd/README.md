# CMS Medicare Quarterly Part D Spending by Drug

This directory stores the public CMS Medicare Quarterly Part D Spending by
Drug snapshot used as a periodic national demand-pressure context source.
The current snapshot contains finalized 2024 and partial 2025 aggregates; it
is not a historical quarterly panel. The model therefore uses it only as
latest context when the coverage timestamp is available before scoring.

Source: https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-quarterly-part-d-spending-by-drug

The source reports prescription claims, spending, beneficiaries, and
manufacturer-level rows. ``Overall`` rows are retained by the loader to avoid
double-counting manufacturer and total rows. CMS warns that recent quarterly
data are preliminary and may change as claims mature.
