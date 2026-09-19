# US Economics Data Package (partial extraction)

Reproducible extraction of the US economic and pharmaceutical-manufacturing indicators catalogued in [`sources.md`](sources.md).

## Scope

- **Reference window:** 2012–2022 inclusive.
- **Geography:** US national and state where the source supports it.
- **No imputation:** missing, suppressed, unavailable, and credential-gated observations are not replaced with invented values.
- **Provenance:** every normalized record carries its source, source record identifier, retrieval time, and the raw response is retained in `cache/`.

## Layout

```text
economics_us/
├── README.md
├── sources.md                    # source rationale and model roles
├── source_manifest.json          # machine-readable extraction plan
├── schema.json                   # canonical JSONL record schema
├── .env.example                  # optional API-key names only
├── .gitignore
├── scripts/
│   ├── common.py                 # stdlib HTTP, caching, records, I/O
│   ├── extract.py                # source extractors and CLI
│   └── validate.py               # structural and range validation
├── cache/                        # immutable raw HTTP responses (local)
├── data/by_source/<source>/      # normalized JSONL, partitioned by period
└── metadata/
    ├── extraction_status.json    # result of the most recent run
    └── validation_errors.json    # result of validation
```

Raw downloads and generated data are intentionally kept local by `.gitignore` where they may be large. The manifest, code, and schema remain reviewable and reproducible; the local `metadata/extraction_status.json` and `metadata/validation_errors.json` are generated audit artifacts and are preserved in the working folder but ignored by Git.

## Quick start

The extractor uses only Python 3.10+ standard-library modules; no package installation is required.

```bash
cd economics_us
python3 scripts/extract.py --list
python3 scripts/extract.py --source public
# or run one source:
python3 scripts/extract.py --source fred_ppi
python3 scripts/validate.py
```

`public` attempts sources that do not require user credentials. A failed endpoint does not stop the other sources: the reason is recorded in `metadata/extraction_status.json`. Re-running uses cached responses unless `--refresh` is supplied.

Credential-gated sources are enabled by environment variables. Never put keys in this repository:

```bash
export FRED_API_KEY='...'
export BEA_API_KEY='...'
export BLS_API_KEY='...'
export CENSUS_API_KEY='...'
python3 scripts/extract.py --source fred_alfred --source bea_regional --source bls_api --source census_cbp
```

The BLS public API currently accepts low-volume unauthenticated requests, so `bls_api` is included in `public`; `BLS_API_KEY` is optional and is sent only when present.

## Implemented extraction targets

| Source ID | Normalized target | Access | Status without credentials |
|---|---|---|---|
| `fred_ppi` | Monthly PPI for pharmaceutical and medicine manufacturing (`PCU32543254`) | FRED public CSV | attempted |
| `fred_alfred` | Year-end vintage observations for the same series | FRED/ALFRED API | skipped unless `FRED_API_KEY` |
| `bea_regional` | State GDP, all-industry line from `SAGDP9N` | BEA Regional API | skipped unless `BEA_API_KEY` |
| `bls_api` | LAUS state unemployment, CES manufacturing employment, CPI, PPI | BLS Public Data API | attempted |
| `bls_qcew` | State QCEW pharma industry (`3254`) annual employment, establishments, wages | BLS QCEW API CSV | attempted |
| `census_cbp` | State CBP pharma establishments, employment, payroll (`3254`) | Census API (key may be required by year/environment) | attempted; current run failed |
| `census_asm` | ASM pharma production/capital variables | documented endpoint; bulk schema varies | not automated; recorded |
| `census_construction` | Manufacturing construction spending/VIP | documented source; table selection varies | not automated; recorded |
| `fed_g17` | G.17 pharma industrial-production series candidates | public FRED graph CSV candidates | attempted |
| `philly_state_coincident` | State coincident indexes | Philadelphia Fed publication files | not automated; recorded |
| `census_bds` | Firm births/deaths | Census BDS bulk/API | not automated; recorded |
| `usaspending` | Annual federal obligations for NAICS `3254`, FY 2012–2022 | USAspending public API | attempted |

The extractor never silently substitutes a different series for a requested target. Candidate G.17 series are tried explicitly and the selected series is recorded in metadata. This is a **partial extraction package**: ASM, Construction/VIP, Philadelphia Fed state indexes, and BDS are documented but intentionally not automated until their publication schemas are pinned; G.17 remains unresolved when no verified series identifier is found.

## Data contract

Normalized records are UTF-8 JSON Lines. Each record has a period, geography, metric, numeric value, unit, and nested provenance. See [`schema.json`](schema.json). Source-native responses are not discarded: they remain in `cache/` and are addressed by deterministic filenames.

`value: null` is not emitted as an observation. Suppression codes and non-numeric source values are represented in the status/validation metadata rather than converted to zero.

## Interpretation cautions

- Public FRED graph CSV is the current revised series; it is useful for descriptive data but is **not** a point-in-time vintage. Use `fred_alfred` with year-end vintages for a leak-aware backtest, and retain the vintage field.
- BLS QCEW and CBP include confidentiality/suppression conventions. Suppressed cells are skipped, never interpreted as zero.
- USAspending periods are federal fiscal years (`FY2012` is 2011-10-01 through 2012-09-30), not calendar years.
- Payroll and spending units are source-native and are declared in each record.
- ASM, Construction/VIP, BDS, and Philadelphia Fed files are intentionally marked as not automated until their current publication schema is pinned; no placeholder data is produced.

## Reproducibility

1. Review `source_manifest.json` and `sources.md`.
2. Run extraction with a clean cache or `--refresh`.
3. Inspect `metadata/extraction_status.json` for HTTP failures, credentials, counts, and raw files.
4. Run `python3 scripts/validate.py` and inspect `metadata/validation_errors.json`.
5. Preserve the cache and status file alongside any downstream model snapshot.
