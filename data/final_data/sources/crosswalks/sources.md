# Crosswalks & Identity Sources

Identity resolution and concordance: linking drugs (NDC/ATC), firms (LEI), geographies (ZIP/HRR), and trade codes (HS/NAICS). These make the other categories joinable into one graph.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| RxNorm / RxNav API | https://lhncbc.nlm.nih.gov/RxNav/APIs/ | US | NDC ↔ ingredient ↔ RxCUI | Free API | Product identity resolution |
| WHOCC ATC/DDD Index | https://atcddd.fhi.no/atc_ddd_index/ | Global | ATC ↔ DDD | Free | Therapeutic class axis |
| Census NAICS concordances | https://www.census.gov/naics/ | US/global | NAICS ↔ ISIC ↔ SIC | Free | Industry axis |
| WITS product concordances | https://wits.worldbank.org/product_concordance.html | Global | HS ↔ SITC ↔ BEC ↔ ISIC | Free | Trade↔industry join |
| GLEIF LEI | https://www.gleif.org/en/lei-data/gleif-golden-copy | Global | legal entity + parent hierarchy | Free | Firm ownership graph |
| HUD USPS Crosswalk | https://www.huduser.gov/portal/datasets/usps_crosswalk.html | US | ZIP ↔ tract ↔ CBSA | Free | Pharmacy geo resolution |
| Dartmouth Atlas | https://data.dartmouthatlas.org/ | US | ZIP ↔ HSA ↔ HRR | Free | Healthcare market geography |

## Review notes

- **RxNorm + ATC/DDD** are the core drug-identity resolvers (NDC↔ingredient↔class).
- **GLEIF LEI** builds the firm ownership/parent hierarchy — needed to attribute facilities to ultimate parents.
- **HUD + Dartmouth** map pharmacies to markets (CBSA/HSA/HRR).
- **Gap**: no public ATC↔HS6 concordance exists — linking trade flows to drug products requires building one (or proxying via NAICS 3254 ↔ HS).
- This folder is the glue; everything else joins through it.