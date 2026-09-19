# Supply Chain: Logistics, Inputs & Structure Sources

Logistics disruption, input costs, and inter-country production structure. These capture how shocks propagate: port congestion, freight, energy/commodity input prices, and input-output linkages.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| IMF PortWatch | https://portwatch.imf.org/ | 1,600+ ports, daily, 2019– | port calls, chokepoint disruption | Free API | Best logistics shock feed |
| UNCTAD port call stats | https://unctadstat.unctad.org/datacentre/ | Global, quarterly | port performance, liner connectivity | Free | Structural port capacity |
| BTS Freight Indicators | https://www.bts.gov/freight-indicators | US | freight volume, rates | Free | US inland logistics |
| NY Fed GSCPI | https://www.newyorkfed.org/research/policy/gscpi | Global, monthly | supply chain pressure index | Free | Global congestion state |
| RWI/ISL Container Throughput | https://fred.stlouisfed.org/ | Global, monthly | container throughput | Free via FRED | Trade volume nowcast |
| EIA Open Data | https://www.eia.gov/opendata/ | Global + US state | energy prices/production | Free API | Input-cost channel |
| World Bank Pink Sheet | https://www.worldbank.org/en/research/commodity-markets | Global | commodity prices | Free | Raw material channel |
| OECD ICIO / TiVA | https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html | 80 economies, 1995–2022 | inter-country sector input-output | Free zip | Shock propagation weights |
| FIGARO (Eurostat) | https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables | EU + partners | IC supply-use tables | Free | EU propagation |
| ADB MRIO | https://www.adb.org/what-we-do/data/regional-input-output-tables | Asia | regional input-output | Free | Asia propagation |

## Review notes

- **IMF PortWatch** is the best logistics shock feed (daily, port-level, 2019–).
- **NY Fed GSCPI** is a clean monthly global congestion state.
- **EIA + Pink Sheet** feed the input-cost channel (energy, raw materials).
- **OECD ICIO / FIGARO / ADB MRIO** provide the inter-country input-output structure — how a shock in one sector/country propagates to pharma.