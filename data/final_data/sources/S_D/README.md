# Pharmaceutical Supply/Demand Dataset (S_D)

Drug-specific pharmaceutical demand and supply information for 2000-2022.
Each observation = drug compound/product + geographic location + quantity/demand
metric + date, with full provenance and explicit missingness.

## Layout

```
S_D/
├── README.md
├── schema.json            canonical record schema
├── source_manifest.json   source URLs, formats, coverage, extraction status
├── coverage.json          per-source year coverage and documented gaps
├── data/
│   ├── combined/          2000.json .. 2022.json (all sources merged)
│   ├── by_source/         per-source yearly files
│   ├── drug_dictionary.json
│   ├── unmapped_records.json
│   └── validation_errors.json
├── scripts/               download, normalize, build, validate
└── cache/                 source downloads
```

All final data files are valid UTF-8 JSON arrays. Years with no observations
contain `[]`.

## Sources (priority order)

1. **Medicaid State Drug Utilization Data** (`medicaid_sdud`) — quarterly, state-level, 2019-2022 (partial; subset of states).
2. **NHS England prescribing** (`nhs_prescribing`) — not extracted in this run.
3. **CMS Medicare Part D Spending by Drug** (`cms_part_d`) — national, 2020-2022.
4. **DEA ARCOS** (`dea_arcos`) — PDFs downloaded 2016-2022, not parsed.
5. **FDA drug recalls** (`fda_recalls`) — events, 2004-2022.
6. **FDA drug shortages** (`fda_shortages`) — events, 2012-2022.

## Metrics

Records preserve source-native metrics and units. Monetary values are stored
separately and never treated as quantity. See `schema.json` for the full list.

## Known limitations

- 2000-2003 have no drug-level data from these sources (combined files are `[]`).
- Medicaid 2000-2018 not bulk-extracted; ARCOS and NHS not parsed.
- ARCOS grams are supply-flow, not patient consumption.
- Recalls/shortages are supply-disruption events, not quantities.
- No synthetic or interpolated values; `is_imputed` is always `false`.

## Reproduce

```bash
cd data/S_D/scripts
python3 download_fda_recalls.py
python3 download_fda_shortages.py
python3 download_cms.py
python3 download_medicaid.py          # downloads large CSVs into cache/
python3 download_arcos.py             # downloads PDFs into cache/
python3 download_nhs.py               # probes NHSBSA API
python3 normalize_drugs.py            # RxNorm resolution -> drug_dictionary.json
python3 build_combined_dataset.py     # -> data/combined/*.json
python3 validate_dataset.py           # -> validation_errors.json + summary
```

## Resulting counts

Final validation run (0 errors):

| Source | Records |
|---|---|
| medicaid_sdud | 6,516,406 |
| cms_part_d | 58,585 |
| fda_recalls | 14,983 |
| fda_shortages | 789 |
| **Total** | **6,590,763** |

- Years covered: 2012-2022 (Medicaid prescriptions by state, CMS Part D,
  FDA recalls, FDA shortages). 2006-2011 have FDA recall events only.
  2000-2005, 2008, 2009 combined files are empty arrays (no free official
  drug-level source in this pipeline covers those years).
- Unique canonical drugs: 13,611; unique original drug names: 31,420.
- Unique geographies: 53 (states + national).
- Mapped: 3,416,343; unmapped: 3,174,420 (preserved in `unmapped_records.json`).
- Metrics: prescription_count, part_d_claims, thirty_day_fills,
  part_d_total_drug_cost, part_d_beneficiaries, recall_event, shortage_active.
- Validation: 0 errors (see `data/validation_errors.json`).
