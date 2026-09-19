# Global Disease & Epidemiology Sources

Global disease burden, surveillance, and health-system data: incidence/prevalence, outbreaks, influenza, immunization, mortality. Cross-country demand priors and outbreak event labels.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| IHME GBD | https://vizhub.healthdata.org/gbd-results/ | 204 countries, 1990– | incidence/prevalence/DALY, 370+ diseases, age/sex | Free non-commercial CSV (100k rows/req) | Global disease burden priors |
| GHDx | https://ghdx.healthdata.org/ | Global | catalog of underlying datasets | Free | Source discovery |
| WHO GHO | https://www.who.int/data/gho/info/gho-odata-api | 194 countries | health indicators via OData API | Free | Country health-system state |
| WHO FluNet / FluID | https://www.who.int/tools/flunet | 100+ countries, weekly | virologic + syndromic influenza | Free | Global respiratory signal |
| WHO Disease Outbreak News | https://www.who.int/emergencies/disease-outbreak-news | Global | dated outbreak events | Free (scrapeable) | Outbreak event labels |
| ECDC Surveillance Atlas | https://atlas.ecdc.europa.eu/ | 30 EU/EEA | 50+ diseases weekly/annual | Free | EU epi panel |
| ProMED-mail | https://promedmail.org/ | Global, 1994– | expert outbreak reports | Free archive | Early-warning text corpus |
| DHS Program | https://dhsprogram.com/data/ | 90+ countries | household health surveys | Free w/ registration | LMIC health baseline |
| WUENIC immunization coverage | https://www.who.int/teams/immunization-vaccines-and-biologicals/immunization-analysis-and-insights/global-monitoring/immunization-coverage/who-unicef-estimates-of-national-immunization-coverage | 195 countries, 1980– | vaccine coverage | Free | Vaccine demand base |
| Our World in Data | https://ourworldindata.org/ | Global | tidy CSVs of many above | Free | Fastest ingest path |
| Human Mortality Database | https://mortality.org/ | 41 countries | weekly mortality | Free | Excess-mortality shock proxy |

## Review notes

- **IHME GBD** is the global disease burden backbone (370+ diseases, age/sex).
- **WHO DON + ProMED** are the early-warning outbreak event feeds — natural text-corpus companions to the news data.
- **FluNet** is the global respiratory signal; **WUENIC** the vaccine demand base.
- **OWID** is the fastest ingestion path for many of these series.
- **HMD** provides weekly mortality for excess-mortality shock proxies.