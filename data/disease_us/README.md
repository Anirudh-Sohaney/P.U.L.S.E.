# US Disease and Epidemiology Data Package

Reproducible extraction of public epidemiology data for the 2012–2022 reference window.

## What is extracted

The first extraction run covers three public, machine-readable datasets:

| Extractor | Data | Native source |
|---|---|---|
| `delphi_fluview` | Weekly ILI counts, percentages, provider counts, and age bands for the national series and US states/DC | Delphi Epidata FluView API |
| `nndss_weekly` | Weekly provisional notifiable-disease counts, national by default | CDC Socrata dataset `x9gk-5huc` |
| `places_2022` | PLACES 2022 release place-level prevalence estimates and confidence limits | CDC Socrata dataset `epbn-9bv3` |

The remaining catalogued sources are documented in `source_manifest.json` and are marked pending when they require a custom query, account, agreement, or a pinned bulk-file schema. No placeholder observations are created.

## Layout

```text
.
├── data/by_source/<source>/observations.jsonl  # normalized long-format observations
├── cache/                                      # raw API responses, content-addressed
├── metadata/extraction_status.json             # counts, errors, and pending sources
├── metadata/validation_errors.json             # validation results
├── scripts/common.py                           # HTTP, caching, JSONL helpers
├── scripts/extract.py                          # extraction CLI and adapters
├── scripts/validate.py                         # structural and range checks
├── schema.json                                 # normalized record contract
└── source_manifest.json                        # targets and source identifiers
```

## Run

Only Python 3.10+ standard-library modules are required:

```bash
python3 scripts/extract.py --list
python3 scripts/extract.py --source all_public
python3 scripts/validate.py
```

To refresh cached responses:

```bash
python3 scripts/extract.py --source all_public --refresh
```

NNDSS defaults to the national `US RESIDENTS` reporting area to keep the first run manageable. To request all reporting areas, add `--nndss-all-geographies`.

## Normalized format

Each JSONL row is one metric for one period and geography. Native wide rows are deliberately flattened so metrics can be filtered and compared consistently:

```json
{
  "source_id": "delphi_fluview",
  "dataset_id": "fluview",
  "period": {"year": 2012, "epiweek": 201201, "issue": 201740},
  "geography": {"level": "state", "id": "ca", "name": "ca"},
  "metric": "wili",
  "value": 3.50589,
  "unit": "percent",
  "dimensions": {"lag": 300, "release_date": "2017-10-24"},
  "provenance": {"source_url": "...", "retrieved_at": "...", "raw_file": "..."}
}
```

Missing numeric values are omitted as observations; source flags and native metadata remain in `dimensions`. No imputation or zero-filling is performed.

## Interpretation cautions

- Delphi FluView is weekly ILI surveillance and is subject to backfill and issue/vintage revisions.
- NNDSS counts are provisional and may be revised by later weekly publications.
- PLACES values are model-based estimates, not direct case counts. The 2022 release is retained as the named release; its native `year` field is preserved.
- `value` is numeric where the source supplies a numeric value. Units are normalized only when unambiguous; native field names remain in `metric` and `dimensions`.
