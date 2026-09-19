# US Economics Sources

US state/national economic and industrial data: employment, wages, production, prices, construction, contracts, firm dynamics. Provides the US economic-health state and pharma manufacturing footprint.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| FRED + ALFRED | https://fred.stlouisfed.org/docs/api/fred/ | 800k+ series, national→county | ALFRED = point-in-time vintages | Free API | Mandatory for leak-free backtest |
| BEA Regional | https://www.bea.gov/data/economic-accounts/regional | US | state income by industry | Free API + zip CSV (some tables discontinued 2024) | State economic health |
| BLS QCEW | https://www.bls.gov/cew/downloadable-data-files.htm | US, quarterly | employment + wages | Free bulk | Pharma mfg footprint over time |
| BLS LAUS / CES / CPI / PPI | https://www.bls.gov/developers/ | US | state unemployment; PPI by NAICS incl. pharma | Free API | Price + labor channel |
| Census CBP | https://www.census.gov/programs-surveys/cbp.html | US, annual | establishments (325411/325412/325414) | Free API | US facility density |
| Census ASM / Economic Census | https://www.census.gov/programs-surveys/asm.html | US | industry shipments, capex by state | Free | Production capacity |
| Census Construction / VIP | https://www.census.gov/construction/c30/c30index.html | US | manufacturing structure construction spend | Free | Factory buildout proxy |
| Fed Industrial Production G.17 | https://www.federalreserve.gov/releases/g17/ | US, monthly | NAICS 3254 pharma production index | Free | Direct pharma output signal |
| Philly Fed State Coincident Indexes | https://www.philadelphiafed.org/surveys-and-data/regional-analysis | US, monthly | state activity index | Free | State-node health |
| USAspending | https://api.usaspending.gov/ | US, 2008– | every federal contract by NAICS/PSC/vendor/place | Free API | Gov medical demand + supplier ID |
| Census BDS | https://www.census.gov/programs-surveys/bds.html | US | firm births/deaths | Free | Firm-exit hazard priors |

## Review notes

- **FRED/ALFRED is mandatory**: point-in-time vintages prevent lookahead bias in backtests.
- **Fed G.17 NAICS 3254** is the closest thing to a direct US pharma output signal.
- **QCEW + CBP** track the pharma manufacturing footprint (employment, establishments).
- **USAspending** links government medical procurement to suppliers — demand + supplier identification.
- BDS gives firm-exit priors; Construction/VIP proxies new factory buildout.