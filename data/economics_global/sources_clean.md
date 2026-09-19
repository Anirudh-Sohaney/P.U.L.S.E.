# Cleaned Global Economics Sources

## Focus Window

- `2012–2022` is the primary evaluation/use window.
- Data outside that window is retained only as context.

## Structured Source List

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| World Bank WDI | https://data.worldbank.org/ | 217 countries, 1960– | ~1,400 annual indicators | Free API | Country-node baseline |
| IMF IFS / WEO / BOPS / GFS | https://data.imf.org/ | ~190 countries | monthly–annual: FX, rates, CPI, reserves, IP, fiscal | Free | Macro-distress state |
| OECD Data Explorer | https://data-explorer.oecd.org/ | 38+ countries | monthly MEI, CLI, PPI, industrial production | Free API | Leading indicators |
| Penn World Table 11 | https://www.rug.nl/ggdc/productivity/pwt/ | 180+ countries, 1950– | real GDP, capital stock, TFP | Free | Long-run structural priors |
| Maddison Project | https://www.rug.nl/ggdc/historicaldevelopment/maddison/ | 169 countries, year 1– | long-run GDP/capita | Free | Deep history normalization |
| BIS Statistics | https://data.bis.org/ | 60+ countries | credit, debt service, FX, property | Free | Financial-crisis channel |
| UNCTADstat | https://unctadstat.unctad.org/ | 200+ countries | FDI, commodity prices, maritime, GVC | Free | FDI + maritime |
| V-Dem | https://v-dem.net/data/ | 202 countries, 1789– | 500+ political/institutional indicators | Free | Institutional/regulatory capacity |
| World Uncertainty Index | https://worlduncertaintyindex.com/ | 143 countries, quarterly | text-derived uncertainty | Free | Ready-made news-derived benchmark |
| EPU / TPU / GPR indices | https://www.policyuncertainty.com/ | 20+ countries, monthly/daily | policy + trade + geopolitical risk | Free | Baseline your LLM index must beat |
| World Bank Enterprise Surveys | https://www.enterprisesurveys.org/ | ~150 countries | firm-level constraints incl. pharma mfg | Free | Firm-node priors |

## Notes

- `WDI + IMF` form the baseline country state.
- `OECD` adds monthly leading indicators.
- `EPU/TPU/GPR` are the benchmark uncertainty indices.
- `V-Dem` proxies regulatory and institutional capacity.
- Enterprise Surveys provide firm-level priors including pharma manufacturing constraints.
