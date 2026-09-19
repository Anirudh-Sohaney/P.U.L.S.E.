# US Disease & Epidemiology Sources

US disease burden and syndromic surveillance: influenza-like illness, ED visits, notifiable conditions, wastewater, mortality, chronic prevalence. These drive drug demand (respiratory, vaccine, oncology, etc.).

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| Delphi Epidata API (CMU) | https://cmu-delphi.github.io/delphi-epidata/ | US | unified ILINet/NSSP/claims/COVIDcast, with vintages | Free API | Start here; vintage-aware |
| CDC FluView / ILINet | https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html | US, weekly | state/region weekly ILI % | Free | Respiratory demand driver |
| CDC NSSP ED Visits | https://data.cdc.gov/ | US, 2022– | county/state weekly ED visits by syndrome | Free Socrata API | Highest-resolution syndromic |
| Project Tycho | https://www.tycho.pitt.edu/ | US, 1888–2017 | weekly disease counts, 3.6M records | Free API | Unmatched disease history |
| CDC NNDSS Weekly | https://data.cdc.gov/browse?q=NNDSS | US, 1996– | state × week × ~120 conditions | Free Socrata | Broad notifiable coverage |
| CDC WONDER | https://wonder.cdc.gov/ | US, 1968– | county mortality by cause | Free | Outcome + demographic base |
| CDC NWSS Wastewater | https://data.cdc.gov/browse?q=wastewater | US, 2020– | site/county pathogen levels | Free | Leading epi indicator |
| WastewaterSCAN | https://data.wastewaterscan.org/ | US | multi-pathogen wastewater | Free | Pathogen lead signal |
| CDC PLACES | https://www.cdc.gov/places/ | US | census-tract chronic disease prevalence | Free | Pharmacy-local exposure weights |
| BRFSS | https://www.cdc.gov/brfss/ | US, 1984– | state behavioral/chronic disease | Free | Chronic demand base |
| NHANES / NHIS / MEPS | https://www.cdc.gov/nchs/ | US | individual clinical + Rx use | Free | Utilization microdata |
| HCUPnet | https://datatools.ahrq.gov/hcupnet | US | hospital/ED discharge stats | Free (some paid) | Acute care demand |
| SEER | https://seer.cancer.gov/ | US | cancer incidence by registry | Free w/ agreement | Oncology demand |
| CDC NIS vaccination coverage | https://data.cdc.gov/ | US, state | immunization coverage | Free | Vaccine demand |

## Review notes

- **Delphi Epidata** is the recommended entry point: unified, vintage-aware syndromic/claims data.
- **NSSP ED visits** is the highest-resolution syndromic feed (2022–); **FluView/ILINet** the classic respiratory driver.
- **Project Tycho** gives unmatched historical disease series (1888-2017).
- **NWSS + WastewaterSCAN** are leading indicators for respiratory surges.
- PLACES/BRFSS/NHIS/MEPS provide chronic-disease and utilization baselines; SEER for oncology.