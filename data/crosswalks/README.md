# Crosswalks & Identity Data

Reproducible extraction package for the identity and concordance sources catalogued in
[`sources.md`](sources.md). The reference window is **2012–2022**.

## What is included

| Source | Output | Status |
|---|---|---|
| RxNorm/RxNav | `data/rxnav_ndc_catalog.csv` and `data/rxnav_version.json` | Current public NDC codelist extracted; RxCUI/ingredient lookup is available for supplied NDCs. Historical RxNorm bulk releases require a UMLS account. |
| WHO ATC/DDD | `data/WHO_ATC_PENDING.txt` | Pending: the official complete machine-readable index and historical vintages are distributed through the WHOCC portal under its access/licensing terms. |
| Census NAICS | `data/census_naics_*.csv` | Downloaded by the script when the Census concordance page exposes the official XLSX links; XLS/XLSX files are parsed without third-party packages. |
| WITS | `data/WITS_PENDING.txt` | Pending: the concordance portal requires an interactive download/account workflow; no stable unauthenticated bulk endpoint was assumed. |
| GLEIF LEI | `data/gleif_lei_records.jsonl` | **Partial public API extract**: default run writes the bounded first page (100 records); `--gleif-all` requests the complete current API population. |
| HUD USPS | `data/HUD_USPS_PENDING.txt` | Pending: HUD states that crosswalk downloads/API require login/token access. |
| Dartmouth Atlas | `data/dartmouth_zip_hsa_hrr.csv` | **Partial public extract**: 2018–2019 ZIP→HSA→HRR CSV ZIP vintages normalized to long CSV; older official `.xls` vintages are retained in `cache/` pending a legacy-XLS parser. |

No synthetic observations are emitted. Sources that are current-only, gated, or
not available as a stable public bulk file are documented as pending rather than
represented by invented rows.

## Layout

```text
crosswalks/
├── README.md
├── sources.md                         # human-readable source catalog
├── source_manifest.json                # machine-readable status and provenance
├── schema.json                         # harmonized downstream record contract
├── data/                               # generated local extracts and access notes
├── cache/                              # raw downloads (ignored by Git)
└── scripts/
    ├── download.py                     # public downloads, cache, and manifest
    ├── normalize.py                    # raw files → normalized CSV/JSONL
    └── validate.py                     # structural/data-quality checks
```

## Run

Only Python 3.10+ standard-library modules are required:

```bash
python3 scripts/download.py
python3 scripts/normalize.py
python3 scripts/validate.py

# Optional: resolve only the NDCs you need through the public RxNav API.
# The input CSV must contain an `ndc` column.
python3 scripts/normalize.py --rxnav-ndc-input my_ndcs.csv
```

`schema.json` defines the harmonized record shape for downstream joins. The
source-native CSV/JSONL outputs retain source-specific columns; their exact
column dictionaries and transformations are documented in `data/normalization_status.json`
and this README. The downloader is conservative and skips existing cache files. Use `--refresh`
to replace them. The GLEIF extractor defaults to one API page (100 records) to avoid an
unbounded multi-hundred-megabyte download; use `python3 scripts/download.py --gleif-all`
when a complete current LEI population is required. `--no-network` writes the
three pending-access notes only; it does not refresh downloaded source status.

## Interpretation

- Crosswalk weights are source-native address/allocation or relationship weights;
  they are not probabilities unless the source explicitly defines them that way.
- Dartmouth files are annual vintages. The official page lists 2012–2019, but this
  run normalized only the 2018–2019 CSV ZIP files; 2012–2017 legacy `.xls` files are
  retained in `cache/` and marked `pending_parser` rather than lossy-parsed. The page
  does not by itself cover 2020–2022.
- HUD 2012 Q1–2022 Q4 files use 2010 Census geographies. They are quarterly snapshots,
  not annual data, and require HUD access credentials in the current portal.
- GLEIF is a current API population. API records include registration and validation
  dates, but this extract is not a historical point-in-time reconstruction.
- RxNav's all-NDC endpoint is a codelist, not an NDC→RxCUI mapping. Use
  `--rxnav-ndc-input` for an explicit, rate-limited NDC→RxCUI→ingredient lookup;
  the API does not provide a practical unauthenticated bulk mapping export.

## Citations

See [`sources.md`](sources.md) and `source_manifest.json` for source URLs, retrieval
provenance, transformations, and documented gaps.
