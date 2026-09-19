# Tariff & Policy Sources

Tariffs, non-tariff measures, and trade-restriction events. These capture the policy channel: how duties and export/import restrictions shift cost and availability of drug inputs and finished products.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| WITS / UNCTAD TRAINS | https://wits.worldbank.org/ | ~200 countries, 1988– | HS6 MFN + preferential tariffs, NTMs | Free bulk | Tariff/landed-cost channel |
| WTO IDB/CTS | https://tao.wto.org/ | WTO members | tariff-line | Free | Tariff cross-check |
| Global Trade Alert | https://globaltradealert.org/data-center | Global, 2008– | every trade measure, dated, sector+product+country pair | Free non-commercial + API | Best export-restriction label |
| EUI/GTA/WB COVID trade policy DB | https://globalgovernanceprogramme.eui.eu/covid-19-trade-policy-database-food-and-medical-products/ | Global, 2020– | medical + food trade measures | Free | Medical-specific restriction labels |
| ITC Market Access Map COVID tracker | https://www.macmap.org/covid19 | Global | temporary measures | Free | Supplement |
| US HTS | https://hts.usitc.gov/ | US | current + historical duty rates | Free | Tariff schedule ground truth |

## Review notes

- **Global Trade Alert** is the best dated, product-level export-restriction label feed (2008–).
- **EUI COVID DB** is medical-specific — directly relevant to pharma supply shocks in 2020+.
- **WITS/TRAINS** provides the structural tariff channel; **US HTS** is the ground truth for US duties.
- WTO IDB/CTS and MACMap are cross-checks/supplements.