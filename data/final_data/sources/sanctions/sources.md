# Sanctions Sources

Entity- and country-level sanction state. Sanctions directly block trade in pharmaceutical inputs and finished goods, so they are supply-shock drivers.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| OFAC SDN / Consolidated | https://sanctionslist.ofac.treas.gov/ | Global | entity-level, dated | Free bulk | Firm-node sanction state |
| Global Sanctions Data Base | https://www.globalsanctionsdatabase.com/ | 1950–, 200+ countries | sanction episodes, type, sender/target | Free | Country-node sanction history |

## Review notes

- **OFAC SDN** is the operational list: match firm/facility names to sanction status.
- **GSDB** provides long-run country-pair sanction episodes for structural priors.
- Small category but high leverage: a sanctioned firm is a broken supply edge.