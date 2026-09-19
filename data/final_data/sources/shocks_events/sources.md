# Shocks & Events Sources

Conflict, disaster, and event data: geo-coded political violence, disasters, and global event streams. These are supply-shock drivers (war, natural disaster, unrest) that break production and logistics.

**Focus window: 2012–2022.** All sources evaluated/used primarily for this period; data outside it is context only.

| Source | URL | Coverage | Granularity | Access | Role in model |
|---|---|---|---|---|---|
| GDELT 2.0 | https://www.gdeltproject.org/data.html | Global, 1979– | CAMEO events, actors, geo, tone, 15-min updates | Free (BigQuery) | Densify your 500k news corpus |
| ACLED | https://acleddata.com/data/ | 200+ countries, 1997– | geo-coded political violence | Free open-access account + API | Conflict channel |
| UCDP GED | https://ucdp.uu.se/downloads/ | Global, 1989– | village-day organized violence | Free | Conflict channel (curated) |
| EM-DAT | https://www.emdat.be/ | Global, 1900– | disasters: type, deaths, damage | Free w/ registration | Natural hazard channel |
| NOAA NCEI | https://www.ncei.noaa.gov/ | Global | earthquakes, storms, temperature | Free | Facility hazard exposure |

## Review notes

- **GDELT** is the high-density event stream — can densify the existing 500k news corpus with machine-readable events.
- **ACLED + UCDP** are the curated conflict channels (ACLED richer, UCDP more rigorous).
- **EM-DAT** covers disasters; **NOAA** provides physical hazard exposure for facility risk scoring.
- These are shock *drivers* — they feed supply-risk, not demand.