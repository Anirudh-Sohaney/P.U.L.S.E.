# Sanctions Data

Entity- and country-level sanction state. Sanctions directly block trade in
pharmaceutical inputs and finished goods, so they are supply-shock drivers.

**Focus window: 2012–2022.** Data outside it is context only.

## Layout

```
sanctions/
├── README.md
├── source_manifest.json   source URLs, access method, formats, role
├── coverage.json          per-source status, records, documented gaps
├── data/
│   ├── ofac_sdn.csv           19,199 SDN entities (lists pipe-joined)
│   ├── ofac_consolidated.csv  481 consolidated non-SDN records
│   └── gsdb/                  placeholder for GSDB (email-request only)
├── scripts/
│   ├── download_ofac.py       fetch current OFAC files via SLS API
│   └── normalize_ofac.py      SDN.XML -> ofac_sdn.csv; CONS_PRIM.CSV copy
└── cache/                     raw downloads (gitignored)
```

## Sources

1. **OFAC SDN List** (`ofac_sdn`) — entity-level, current snapshot
   (publish 2026-08-07). Firm-node sanction state: match firm/facility names.
   Downloaded: `SDN.CSV`, `SDN.XML`.
2. **OFAC Consolidated List** (`ofac_consolidated`) — non-SDN lists
   (NS-PLC, SSI, FSE, CAPTA, MBS, NSCMIC...). Non-blocking directives.
   Downloaded: `CONS_PRIM.CSV`, `CONSOLIDATED.XML`.
3. **GSDB** (`gsdb`) — country-pair sanction episodes 1950–2023 (R4,
   1,547 cases). **Not yet obtained**: distributed only via email request
   form at https://www.globalsanctionsdatabase.com/data/ (no public mirror).
   Submit request, drop received files in `data/gsdb/`, update manifest.

## Access notes

- OFAC moved to a JS app; old `sanctionslist.ofac.treas.gov/sdn.csv` URLs
  now 403. Current flow: POST `{host}/api/PublicationPreview/SdnList` (or
  `ConsolidatedList`) with `{}` -> file list; download via
  `{host}/api/PublicationPreview/exports/{fileName}` (302 -> presigned S3).
- OFAC snapshot has no per-entry start dates; historical state requires the
  delta archive (`/api/PublicationPreview/GetDeltaFileArchive`).

## Rebuild

```bash
python3 scripts/download_ofac.py   # re-fetch current files
python3 scripts/normalize_ofac.py  # regenerate data/*.csv (self-checks counts)
```

## Citations

- U.S. Treasury OFAC, Specially Designated Nationals and Blocked Persons List.
- U.S. Treasury OFAC, Consolidated Sanctions List.
- Felbermayr, Kirilakha, Syropoulos, Yalcin, Yotov (2020), "The Global
  Sanctions Data Base", *European Economic Review* 129.