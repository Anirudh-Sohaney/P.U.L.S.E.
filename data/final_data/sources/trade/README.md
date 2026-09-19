# Trade Data

Primary bilateral trade data: who ships what drug/commodity to whom, at what
value and quantity. Provides trade edges and trade-flow labels for the model.

**Focus window: 2012–2022.**

## Layout

```
trade/
├── sources.md            # source catalog (URLs, coverage, access, role)
├── source_manifest.json  # machine-readable record of what was downloaded
├── coverage.json         # per-source download status + window coverage
├── cache/                # raw downloads (gitignored)
│   ├── BACI_HS92_V202601.zip     # 2.4G, CEPII BACI HS92 full 1995-2022
│   ├── Gravity_csv_V202211.zip   # 207M, CEPII Gravity V202211
│   ├── comext_intra_eu.json      # Eurostat Comext intra-EU (SITC agg)
│   ├── comext_extra_eu.json      # Eurostat Comext extra-EU (SITC agg)
│   └── manifest.json
├── data/                 # extracted/subset data (gitignored)
│   ├── baci/             # baci_hs92_{2012..2022}.csv, one per year
│   ├── gravity_window.csv        # Gravity rows 2012-2022 only (635,040)
│   ├── gravity_countries.csv     # Gravity country reference table
│   ├── comext_intra_eu.csv       # flattened long-form (122,688 rows)
│   └── comext_extra_eu.csv       # flattened long-form (56,688 rows)
└── scripts/              # download + normalize (committed)
    ├── download_cepii.py
    ├── download_eurostat.py
    ├── normalize_baci.py
    ├── normalize_gravity.py
    └── normalize_eurostat.py
```

## Data formats

**BACI** (`data/baci/baci_hs92_YYYY.csv`): columns `t,i,j,k,v,q` =
year, exporter (ISO3 numeric), importer (ISO3 numeric), HS6 code,
value (1000 USD), quantity (tonnes). Reconciled bilateral flows — the
workhorse for trade edges and labels.

**Gravity** (`data/gravity_window.csv`): country-pair × year rows with
distance (distw_harmonic, dist, distcap), contiguity, common language,
colonial ties, RTA membership, GDP, etc. Static edge features.

**Comext** (`data/comext_*.csv`): long-form `freq,indic_et,sitc06,partner,
geo,time,value`. Note: the Eurostat dissemination API serves SITC Rev.4
aggregates, not CN8. CN8 granularity requires the bulk Comext download
(see sources.md).

## Pipeline

```bash
python3 scripts/download_cepii.py      # -> cache/ (BACI + Gravity zips)
python3 scripts/download_eurostat.py   # -> cache/ (Comext JSON)
python3 scripts/normalize_baci.py      # -> data/baci/ (2012-2022 only)
python3 scripts/normalize_gravity.py   # -> data/gravity_window.csv
python3 scripts/normalize_eurostat.py  # -> data/comext_*.csv
```

All scripts re-runnable; skip existing files unless `--force`.

## Status

Downloaded: BACI HS92 V202601, Gravity V202211, Eurostat Comext (SITC agg).
Pending (see coverage.json): UN Comtrade (rate-limited preview API, bulk
paid), IMF DOTS (API not public), Census Intl Trade (API key required),
USITC DataWeb (account required), Eurostat Comext CN8 bulk (registration).