# Supply-chain data extract

Extraction window: 2012-01 through 2022-12, matching `sources.md`. PortWatch and BTS begin in 2019 because those source series do not cover the earlier window.

## Files

- [NY Fed GSCPI](data/nyfed_gscpi_monthly_2012_2022.csv) — 132 monthly observations; current latest-vintage column in the NY Fed interactive CSV (vintage `Aug-26`).
- [EIA crude spot prices](data/eia_crude_spot_prices_monthly_2012_2022.csv) — 264 monthly observations: Brent (`RBRTE`) and WTI (`RWTC`), dollars per barrel.
- [World Bank Pink Sheet](data/worldbank_pink_sheet_monthly_2012_2022.csv) — 9,372 tidy commodity-month observations, nominal US dollars and published units.
- [RWI/ISL container throughput](data/rwi_isl_container_throughput_monthly_2012_2022.csv) — 132 monthly observations with total and North Range original, seasonally adjusted, and trend-cycle indices (2015=100).
- [BTS freight indicators](data/bts_freight_indicators_2019_2022.csv) — 11,514 observations across the published indicator series, retaining the source, unit, and value fields.
- [IMF PortWatch](data/imf_portwatch_country_monthly_2019_2022.csv) — 8,640 country-month observations aggregated from daily records; sums preserve vessel class, total port calls, imports, and exports.

## Provenance

- NY Fed: https://www.newyorkfed.org/research/policy/gscpi
- IMF PortWatch: https://portwatch.imf.org/
- BTS: https://www.bts.gov/freight-indicators
- EIA: https://www.eia.gov/opendata/
- World Bank Pink Sheet: https://www.worldbank.org/en/research/commodity-markets
- RWI/ISL: https://www.rwi-essen.de/en/research-advice/research-unit/macroeconomics-and-public-finance/hightlight-topic/rwi-isl-container-throughput-index
- OECD ICIO: https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html
- Eurostat FIGARO: https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables
- ADB MRIO: https://www.adb.org/what-we-do/data/regional-input-output-tables
- UNCTADstat: https://unctadstat.unctad.org/datacentre/

## Not silently substituted

The OECD ICIO archive was identified, but its ZIP download was blocked by the host's Cloudflare challenge in this environment. UNCTADstat and the ADB page also rejected direct automated retrieval, and FIGARO requires selecting a specific Eurostat table/API slice. No fabricated or third-party replacement values were inserted for those four source families.

The extractor is reproducible from [extract_supply_chain.py](extract_supply_chain.py).
