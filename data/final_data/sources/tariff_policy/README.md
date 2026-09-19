# Tariff & Policy Data Package

Reproducible extraction package for the tariff, non-tariff measure, and trade-restriction
sources catalogued in [`sources.md`](sources.md).

**Reference window: 2012–2022 inclusive.** Values or events outside this window are
not written to normalized outputs. An event that begins before 2012 but remains active
in the window is retained when the source provides an end date or active interval.

## What is included

| Source | Normalized output | Default status |
|---|---|---|
| WITS / UNCTAD TRAINS | `data/wits_trains_tariffs.csv` | bounded query configured; one USA/world HS6 panel demonstrated |
| WTO IDB/CTS | `data/wto_idb_cts.csv` | pending portal export/API subscription |
| Global Trade Alert | `data/global_trade_alert.csv` | pending licensed/API or manual export |
| EUI/GTA/WB COVID database | `data/eui_covid_trade_measures.csv` | pending release file pinning |
| ITC Market Access Map COVID tracker | `data/macmap_covid_measures.csv` | pending registered/interactive export |
| US HTS | `data/us_hts.csv` | pending annual release download pinning |

This is a **partial extraction package**: the bounded WITS demonstration is extracted;
the remaining five sources await their authorized export/release files. No placeholder
observations are created. If a source is unavailable through a stable, public
machine-readable endpoint, the extractor writes a pending note under `data/` and records
the reason in `metadata/extraction_status.json`. `source_manifest.json` is the planned
source catalog; the latest run's statuses in `metadata/extraction_status.json` are
authoritative for what was actually extracted.

## Layout

```text
tariff_policy/
├── README.md
├── sources.md                         # human-readable source catalog
├── source_manifest.json                # machine-readable provenance and status
├── schema.json                         # canonical tariff/event record contract
├── .gitignore
├── cache/                              # immutable raw HTTP responses (local)
├── data/                               # normalized CSVs and pending notes (local)
├── metadata/                           # generated extraction/validation reports
└── scripts/
    ├── common.py                       # HTTP caching, parsing, provenance helpers
    ├── extract.py                      # source extractors and CLI
    └── validate.py                     # structural and window checks
```

## Quick start

Only Python 3.10+ standard-library modules are required:

```bash
cd tariff_policy
python3 scripts/extract.py --list
python3 scripts/extract.py --public
python3 scripts/validate.py
```

`--public` runs the safe default: it writes pending notes for sources that need
credentials or a pinned export and does not issue an unbounded query. To query WITS,
provide explicit dimensions and opt in:

```bash
python3 scripts/extract.py --source wits_trains \
  --reporter 840 --partner 000 --product 300490 --start-year 2012 --end-year 2022
```

The WITS query uses the official TRAINS API and preserves the raw response in `cache/`.
The product is an HS6 code; `000` means world partner in the WITS convention. Use one
or more explicit dimensions rather than attempting a full global pull, which WITS
explicitly disallows. Repeat the command for a planned, bounded panel and retain the
manifest alongside the output.

Optional local files can be normalized after a manual download:

```bash
python3 scripts/extract.py --source wto_idb_cts --input /path/to/export.csv
python3 scripts/extract.py --source global_trade_alert --input /path/to/export.csv
python3 scripts/extract.py --source eui_covid --input /path/to/export.csv
python3 scripts/extract.py --source macmap_covid --input /path/to/export.csv
python3 scripts/extract.py --source us_hts --input /path/to/export.csv
```

The manual-file path is deliberately explicit: source-native columns are retained and
mapped only when their meaning can be identified. A content-addressed copy and SHA-256
hash of each supplied CSV are saved under `cache/manual_inputs/`. Unsupported or
ambiguous files fail with a useful message rather than being silently misclassified.

## Data contract

The canonical schema supports two families of records:

- `tariff`: annual applied or scheduled rates, usually HS6 and reporter/partner scoped.
- `measure`: dated trade-policy interventions, including export restrictions and import
  facilitations.

Normalized files are UTF-8 CSV. Source-native raw downloads remain in `cache/` when the
extractor fetched them. Every row includes `source_id`, `source_url`, `retrieved_at`,
`source_record_id`, and `is_imputed=false`. Source-native columns not needed by the
canonical contract are preserved in `source_payload_json`.

## Interpretation and known limits

- TRAINS tariff observations use the HS nomenclature reported by WITS (`nomencode`),
  which must be respected when joining HS6 products across revisions.
- MFN and preferential rates are separate rows; they must not be averaged together.
- Trade-policy events may overlap across GTA, the EUI COVID database, and MACMap. They
  are kept source-separately; downstream deduplication is a modeling decision.
- The COVID-specific sources are primarily 2020–2022 and therefore do not provide a
  full 2012–2022 panel. Their `window_coverage` records this honestly.
- Current or annual HTS schedules are not historical event series. Preserve the
  schedule year and source revision.
- No values are imputed and no pending source is represented by empty synthetic rows.

## Reproducibility and provenance

1. Review `sources.md` and `source_manifest.json`.
2. Run `python3 scripts/extract.py --public` or a bounded source command.
3. Inspect `metadata/extraction_status.json` and preserve `cache/` with any snapshot.
4. Run `python3 scripts/validate.py`; inspect `metadata/validation_errors.json`.
5. Cite the source release, retrieval date, and any source-specific license/access terms
   in downstream work.
