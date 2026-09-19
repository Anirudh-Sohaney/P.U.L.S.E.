# final_data

Built at `2026-08-11T20:39:17+00:00` from `data_infra.md`, local normalized extracts, and official public downloads. Documentation review: `2026-09-14`.

This assembly remains the shared feature and event foundation for the
Arkansas-first model. Newer targeted additions may have later snapshots and
are not silently merged into this historical build. Refresh metadata and
source-specific READMEs are authoritative for current availability.

## Structure

- `features/<domain>/<domain>_features.csv.gz` and `.jsonl.gz`: canonical long-format numeric feature rows.
- `features/external_state_features.csv.gz` and `.jsonl.gz`: combined feature store across domains.
- `events/events.csv.gz` and `.jsonl.gz`: structured event records for shortages, recalls, and FEMA disaster declarations.
	- `entities/`: entity graph and copied entity/crosswalk reference files used for drug, NDC, HRR, LEI, and related joins.
- `sources/`: copied source manifests, READMEs, source notes, extraction statuses, and validation reports.
- `raw_downloads/`: newly downloaded official API responses used only by this final assembly.
	- `quality/`: validation, coverage, source coverage, extraction issues, run counts, and build manifest.

## Feature Contract

Every feature row includes `variable_id`, `value`, `geography_*`, `observation_time`, `forecast_horizon`,
and source/quality metadata required by `data_infra.md`. Values are numeric floats wherever the source
provides a numeric observation. Categorical observations are either transformed into deterministic numeric
indicators with the native category retained in `dimensions_json`, or stored in `events/`.

No imputation, fake county disease estimates, or fake daily values are produced.

## Counts

- Feature rows by domain: `{"chronic_health": 825, "demand_price": 574411, "derived": 4573510, "disasters": 1348, "disease_global": 26816, "disease_us": 220471, "economics_us": 1575, "environment": 596766, "healthcare": 7835, "policy": 11, "population": 14224, "supply_chain": 10560, "trade": 450800}`
- Event rows: `20153`

## Newly Downloaded Sources

	- CDC PLACES 2024 county release, Arkansas county population and chronic-disease measures.
	- FEMA OpenFEMA Disaster Declarations Summaries for Arkansas declarations from 2012 onward.
	- NOAA NCEI Daily Summaries for geographically distributed Arkansas weather stations from 2000 onward.
	- EPA AirData annual AQI and monitor concentration files for Arkansas from 2000 onward.
	- CMS Medicare Geographic Variation and Medicaid/CHIP enrollment public files.
	- BLS Public Data API unemployment, CPI, and pharmaceutical PPI series from 2000 onward.
	- openFDA drug shortages and enforcement/recall APIs for recent FDA supply events.
	- CEPII BACI HS92 U.S. pharmaceutical import/export exposure by partner country and HS product.
	- Derived lag, change, rolling, and seasonal anomaly features for appropriate historical series.

	Attempted but limited: Census ACS 2024 county profile and Census International Trade API returned key-gated
	HTML responses in this environment. CDC PLACES was retained for county population/chronic health, and CEPII
	BACI was used for official public U.S. pharmaceutical trade fallback. FDA inspection metadata was reachable
	through HHS, but its JSON resource returned 403 and the archived Excel distribution returned 404; openFDA
	enforcement/recall events were retained as the official accessible FDA supply-chain substitute.

See `quality/build_manifest.json` for detailed statuses and source paths.
