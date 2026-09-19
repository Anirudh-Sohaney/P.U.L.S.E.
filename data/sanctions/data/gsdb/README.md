# GSDB (Global Sanctions Data Base) — placeholder

Data NOT yet obtained. GSDB distributes data only via email request — no
public direct download or mirror exists (checked 2026-08-09).

## How to obtain

1. Go to https://www.globalsanctionsdatabase.com/data/
2. Fill "Request the GSDB Data" form (email, name, affiliation, country,
   intended use). Form posts to mailchimp:
   `globalsanctionsdatabase.us2.list-manage.com/subscribe/post?u=ca659499444cdde6885f22d5d&id=3898dea0ef`
3. Data is emailed. Latest release GSDB-R4: 1950–2023, 1,547 sanction cases,
   two versions (case-specific + dyadic).
4. Drop received files here, then update `../source_manifest.json` and
   `../coverage.json` (status -> downloaded, record counts).

## Citation

Felbermayr, Kirilakha, Syropoulos, Yalcin, Yotov (2020), "The Global
Sanctions Data Base", *European Economic Review* 129.

## Expected files (from R4)

- `GSDB_V4.csv` (case-specific) and/or dyadic version
- `Var_Description.xlsx` (variable dictionary)