# Global Disease and Epidemiology Data Package

Reproducible global disease and epidemiology extraction for the 2012–2022
reference window. The authoritative source catalog is `sources.md`; machine-
readable access status is `source_manifest.json`.

## Extracted observations

The current package contains normalized JSONL observations under
`data/by_source/<source>/observations.jsonl` from every source with a verified
public machine-readable path. The current public observation outputs total
**5,428,009** rows; run `scripts/validate.py` for the authoritative count in
`metadata/validation_errors.json`.

| Source | Extractor | Coverage in this package |
|---|---|---|
| Our World in Data | `owid` | Life expectancy and under-five mortality |
| WHO FluNet | `who_flunet` | Weekly virologic surveillance |
| WHO FluID | `who_fluid` | Weekly syndromic surveillance |
| WHO GHO | `who_gho` | 15 selected life-expectancy/immunization indicators |
| WHO Disease Outbreak News | `who_don` | Dated outbreak events |
| WUENIC | `wuenic` | National vaccine coverage and source-native counts |
| DHS Program | `dhs` | Public API indicator observations from 124 surveys |
| ECDC Surveillance Atlas | `ecdc_surveillance_atlas` | `CURRENT.GENERAL` dataset, all observations returned by the public `I,Q` query across 59 topics; 20 of 79 cataloged topics returned no rows |
| Human Mortality Database | `hmd` | Four official public summary workbooks; weekly registered files remain gated |

All extracted records are filtered to 2012–2022, retain native dimensions, and
include source URL, retrieval timestamp, and cached raw response path. WHO GHO
contains the 15 configured indicators; ECDC contains the public
`CURRENT.GENERAL` dataset query, not every Atlas dataset. Missing values are
omitted; no imputation or zero-filling is performed.

## Sources requiring user access or a formal data workflow

- **IHME GBD:** the GBD Results Tool requires an authenticated/user-selected
  query; no supported anonymous data API is available. Public catalog metadata
  is captured by `scripts/catalog.py`.
- **GHDx:** public catalog pages are captured, but record downloads redirect to
  authenticated access. GHDx is a discovery catalog rather than a distinct
  observation series.
- **ProMED-mail:** the current public site exposes site structure and media
  assets, but not an anonymous historical article export/API. Full archive
  extraction requires the provider’s permitted access tier or contract.
- **Human Mortality Database:** the weekly mortality files require free
  registration/login, but four official public summary workbooks are extracted
  by `hmd`. The package labels this as `public_summary_only`; it does not claim
  the registered weekly files are present.

These unavailable portions are not represented with fabricated placeholder
observations. Their exact status and links are in `source_manifest.json` and
`metadata/catalog_status.json`.

## Layout

```text
.
├── data/by_source/<source>/observations.jsonl  # normalized observations
├── cache/                                      # exact raw downloads
├── metadata/extraction_status.json             # extraction results/status
├── metadata/catalog_status.json                # public catalog captures
├── metadata/validation_errors.json             # validation report
├── scripts/common.py                           # HTTP/cache/JSONL/XLSX helpers
├── scripts/extract.py                          # source extractors
├── scripts/catalog.py                           # restricted-source metadata capture
├── scripts/validate.py                         # structural validation
├── schema.json                                 # normalized record contract
└── source_manifest.json                         # catalog and access status
```

## Reproduce

Python 3.10+ standard-library modules are sufficient:

```bash
python3 scripts/extract.py --list
python3 scripts/extract.py --source all_public
python3 scripts/catalog.py
python3 scripts/validate.py
```

Use `--refresh` to re-fetch raw responses. Source selection is repeatable:

```bash
python3 scripts/extract.py --source who_flunet
python3 scripts/extract.py --source who_gho
python3 scripts/extract.py --source ecdc_surveillance_atlas
```

For a custom year range, pass the same bounds to extraction and validation.
Large public sources are cached by URL/request and written atomically.

## Normalized record

```json
{
  "source_id": "who_flunet",
  "dataset_id": "VIW_FNT",
  "period": {"year": 2022},
  "geography": {"level": "country", "id": "USA", "name": "United States"},
  "metric": "INF_A",
  "value": 123,
  "unit": "source_native",
  "dimensions": {"whoregion": "AMR", "iso_week": "52"},
  "provenance": {
    "source_url": "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv",
    "retrieved_at": "2026-08-10T00:00:00Z",
    "raw_file": "cache/who_flunet_<sha256-prefix>.csv"
  }
}
```
