# Pharma Infrastructure Sources

Facility-, firm-, and substance-level regulatory and manufacturing structure: registrations, inspections, compliance actions, approvals, GMP status, production statistics. Core nodes and edges of the supply network.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| FDA Drug Establishment Registration | https://www.fda.gov/drugs/guidance-compliance-regulatory-information/drug-establishments-current-registration-site | Global facilities serving US | FEI, name, address, country, registered operations | Free download | Core facility node |
| FDA DMF Index | https://www.fda.gov/drugs/drug-master-files-dmfs/drug-master-file-dmf-index | Global API makers | holder ↔ substance ↔ status ↔ date | Free quarterly CSV | API→firm edges |
| FDA Inspections Classification DB | https://datadashboard.fda.gov/ora/ | 2009– | facility × date × NAI/VAI/OAI | Free + API | Facility health labels |
| FDA Compliance Actions / Warning Letters / Import Alerts | https://datadashboard.fda.gov/ | Global | dated enforcement per firm/facility | Free | Regulatory event labels |
| Orange Book | https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book | US | product ↔ applicant ↔ patent/exclusivity | Free | Market structure, source count |
| Purple Book | https://purplebooksearch.fda.gov/ | US | biologics ↔ license structure | Free | Biologic market structure |
| EudraGMDP | https://eudragmdp.ema.europa.eu/ | EU + inspected 3rd countries | GMP certs, mfg auth, non-compliance | Free search | Ex-US facility nodes |
| EDQM CEP database | https://extranet.edqm.eu/publications/recherches_CEP.shtml | Global | substance ↔ holder ↔ site | Free | Ex-US API edges |
| WHO Prequalification | https://extranet.who.int/prequal/ | Global | prequalified products + mfg sites | Free | LMIC supply nodes |
| Eurostat PRODCOM | https://ec.europa.eu/eurostat/web/prodcom/database | EU | production volume of pharma products | Free | Rare true output data |
| UNIDO INDSTAT | https://stat.unido.org/data/table | 100+ countries | ISIC 2100: output, value added, employment, capex, establishments | Some free, INDSTAT4 licensed | National pharma capacity |
| OECD STAN | https://www.oecd.org/en/data/datasets/stan-database.html | 60+ countries | output, capital stock, investment | Free | Capital-stock trend |
| OECD FDI by industry | https://data-explorer.oecd.org/ | 50+ countries | bilateral FDI into chemicals & pharma | Free | Investment channel |
| UNCTAD FDI / greenfield | https://unctadstat.unctad.org/ | 200+ countries | greenfield FDI values by sector | Free (fDi Markets paid) | New factory pipeline |
| GSRS | https://gsrs.ncats.nih.gov/ | Global | substance identity, UNII | Free | Substance crosswalk |

## Review notes

- **FDA Establishment Registration** = core facility node list; **Inspections + Compliance Actions** = facility health labels.
- **DMF Index + EDQM CEP** map API makers to substances — the upstream supply edges.
- **PRODCOM / INDSTAT / STAN** are rare true output/capacity data (EU, UNIDO, OECD).
- Orange/Purple Books define US market structure and competitor counts.
- GSRS provides UNII substance identity for cross-referencing.