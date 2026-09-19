# Demand & Price Sources

Drug-specific utilization, reimbursement, price, and shortage data. The demand labels and price-shock signals. Includes the sources already used in `S_D` plus additional price/global benchmarks.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| Medicaid State Drug Utilization Data | https://data.medicaid.gov/dataset/ | US, 1991– | state × NDC × quarter: Rx count, units, $ | Free bulk | Best free geographic demand label |
| NADAC | https://data.medicaid.gov/dataset/ | US, 2014– | national average drug acquisition cost | Free | Price-shock label |
| Medicare Part D PUFs | https://data.cms.gov/provider-summary-by-type-of-service | US, 2013– | state/NPI × drug × year | Free | Demand + prescriber geography |
| openFDA Drug Shortages | https://open.fda.gov/apis/drug/drugshortages/ | US, 2012– | drug, reason, dates | Free API/bulk | Primary shortage label |
| ASHP Shortages API | https://github.com/ASHP-Software/drugShortagesDoc | US | earlier + richer than FDA | Key on request | Earlier shortage label |
| EMA Shortages Catalogue | https://www.ema.europa.eu/en/human-regulatory/overview/public-health-threats/drug-shortages | EU | dated shortage events | Free | EU shortage label |
| openFDA Enforcement/Recalls | https://open.fda.gov/apis/drug/enforcement/ | US, 2004– | recall class, firm, reason, distribution states | Free bulk | Facility-level failure events |
| openFDA NDC Directory | https://open.fda.gov/apis/drug/ndc | US | NDC ↔ product | Free daily | Product crosswalk |
| MSH Intl Medical Products Price Guide | https://msh.org/resources/international-medical-products-price-guide/ | Many LMICs | reference prices by molecule | Free | Global price benchmark |
| OECD Health: pharma spending + DDD consumption | https://data-explorer.oecd.org/ | 38+ countries | pharma spend, DDD consumption per 1000/day | Free | Cross-country demand normalization |
| WHO Global Health Expenditure DB | https://apps.who.int/nha/database | 190 countries | health + pharma spend | Free | Country demand capacity |
| IQVIA MIDAS / Xponent | https://www.iqvia.com/ | Global | gold-standard Rx volume | Paid, expensive | Validation only |

## Review notes

- **Medicaid SDUD + Medicare Part D** are the best free geographic demand labels (already in `S_D`).
- **openFDA Shortages + ASHP + EMA** are the shortage event labels (FDA = primary, ASHP = earlier/richer, EMA = EU side).
- **NADAC** captures price shocks at the acquisition-cost level.
- **MSH + OECD + WHO** provide global price and consumption benchmarks for cross-country normalization.
- IQVIA is the gold standard but paid — use only for validation, never as a core feature.