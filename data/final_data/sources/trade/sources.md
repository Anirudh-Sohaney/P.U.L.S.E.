# Trade Sources

Primary bilateral trade data: who ships what drug/commodity to whom, at what value and quantity. These provide the trade edges and the trade-flow labels for the model.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| BACI (CEPII) | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 | 200 countries, 1995– | HS6 exporter×importer×year, value + quantity, reconciled | Free bulk CSV | Primary bilateral trade edges + label |
| UN Comtrade | https://comtradeplus.un.org/ | 200+ reporters, 1962– | HS2–HS6, monthly/annual | Free preview API (rate-limited); bulk paid | Raw fallback, monthly frequency |
| CEPII Gravity | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8 | 200+ country pairs, 1948– | distance, RTA, colonial ties, GDP | Free | Static edge features |
| IMF DOTS | https://data.imf.org/ | ~190 countries, monthly | bilateral merchandise trade | Free | Higher-frequency trade nowcast |
| Census Intl Trade API | https://www.census.gov/data/developers/data-sets/international-trade.html | US, 2010– monthly | US state × NAICS × destination country | Free API | Rare state-level trade exposure |
| USITC DataWeb | https://www.usitc.gov/applications/dataweb/about | US, 1989– | HTS10, duty rates, districts, ports | Free account + API | US import detail + duties |
| Eurostat Comext | https://ec.europa.eu/eurostat/web/international-trade-in-goods/database | 27+ EU countries, monthly | CN8 | Free bulk | EU-side trade |

## Review notes

- **BACI is the workhorse**: reconciled, consistent bilateral HS6 flows with both value and quantity. Start here.
- **Comtrade** is the raw upstream source; use it for monthly frequency and as fallback when BACI lags.
- **Census Intl Trade + USITC** give US-specific detail (state exposure, HTS10, duties) — valuable for the US side.
- **Eurostat Comext** covers the EU side at CN8.
- Gravity, DOTS are edge features / nowcasts, not primary labels.