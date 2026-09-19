# Third-Pass Expansion Report

Generated from the validated third-pass `final_data/` build completed on 2026-08-11.

## Summary

The third pass extended the existing Arkansas pharmaceutical external-state feature store in place.

Major changes:

- Added Arkansas county-level CDC COVID-19 community transmission observations.
- Added derived lag/change/rolling/seasonal features for county COVID-19 disease signals.
- Extended CEPII BACI pharmaceutical trade from 2012-2024 back to 2000-2024.
- Added U.S. Census Bureau downloadable county population estimates and components for Arkansas counties, 2010-2025.
- Added Arkansas regional population aggregates using Arkansas planning and development districts recognized in AR Code section 14-166-202.
- Added 8 Arkansas region entities and county-to-region relationships.
- Expanded BLS economic variables from 5 to 9 observed variables.
- Added computed quality metadata files:
  - `quality/historical_coverage.json`
  - `quality/geographic_coverage.json`
  - `quality/temporal_continuity.json`
  - `quality/disease_coverage.json`

## Final Dataset Totals

- Feature rows: 6,479,152
- Event rows: 20,153
- Variables: 404
- Entities: 29,923
- Relationships: 7,589
- Disease observations: 247,287
- Disease/event disease identifiers: 1,421
- Arkansas counties: 75
- Arkansas regions: 8
- Countries represented in feature geography rows: 227

## Validation

Validation passed.

- Duplicate record IDs: 0
- Invalid dates: 0
- Implausible numeric values: 0

Validation file: `quality/validation_report.json`

## Expansion Compared With Previous Build

| Metric | Previous build | Third-pass build | Change |
|---|---:|---:|---:|
| Feature rows | 3,925,916 | 6,479,152 | +2,553,236 |
| Variables | 302 | 404 | +102 |
| Event rows | 20,153 | 20,153 | 0 |
| Disease observations | 51,251 | 247,287 | +196,036 |
| Disease identifiers | 1,420 | 1,421 | +1 |
| Arkansas counties | 75 | 75 | 0 |
| Arkansas disease county coverage | 0 counties in `disease_us` | 75 counties in `disease_us` | +75 counties |
| Arkansas regions | 0 | 8 | +8 |
| Entities | 29,915 | 29,923 | +8 |
| Relationships | 7,506 | 7,589 | +83 |
| Earliest overall observation | 2000-01-01 | 2000-01-01 | unchanged |
| Earliest population observation | 2022-01-01 | 2010-01-01 | extended 12 years |
| Earliest trade observation | 2012-01-01 | 2000-01-01 | extended 12 years |

## Feature Family Coverage

| Family | Rows | Variables | Period | Frequency | Geography | Mean continuity |
|---|---:|---:|---|---|---|---:|
| derived | 4,573,510 | 228 | 2000-01-02 to 2026-08-08 | annual, daily, monthly, weekly/annual | county, monitor, national, state, station | 91.67% |
| environment | 596,766 | 16 | 2000-01-01 to 2026-08-08 | annual, daily | monitor, station | 95.3668% |
| demand_price | 574,411 | 25 | 2012-01-01 to 2026-08-10 | weekly, quarterly, annual, event-like | national, state | 100.0% |
| trade | 450,800 | 6 | 2000-01-01 to 2024-01-01 | annual | country | 88.8786% |
| disease_us | 220,471 | 17 | 2012-01-02 to 2022-12-26 | daily, weekly/annual | county, national, state | 91.3263% |
| disease_global | 26,816 | 52 | 2012-01-01 to 2022-01-01 | weekly/annual/event-like | aggregate, country, global event | 97.0297% |
| population | 14,224 | 12 | 2010-01-01 to 2025-01-01 | annual | county, region | 98.5563% |
| supply_chain | 10,560 | 8 | 2012-01-01 to 2022-12-01 | monthly | global | 100.0% |
| healthcare | 7,835 | 17 | 2014-01-01 to 2025-01-02 | annual, monthly | county, national, state | 99.1035% |
| economics_us | 1,575 | 9 | 2000-01-01 to 2026-07-01 | monthly, fiscal year | national, state | 99.8953% |
| disasters | 1,348 | 2 | 2013-01-29 to 2026-01-24 | event | county/area | 100.0% |
| chronic_health | 825 | 11 | 2022-01-01 | annual | county | 100.0% |
| policy | 11 | 1 | 2012-01-01 to 2022-01-01 | annual | country | 100.0% |

## Third-Pass Data Added

### CDC COVID-19 County Transmission

Source:

- CDC United States COVID-19 County Level of Community Transmission Historical Data.

Added variables:

- covid_cases_per_100k_7_day
- covid_percent_test_results_reported
- covid_community_transmission_level

Coverage:

- Arkansas county daily observations.
- All 75 Arkansas counties represented.
- Native community transmission level retained in `dimensions_json`.
- Ordinal transformation used only for numeric modeling compatibility:
  - low = 1
  - moderate = 2
  - substantial = 3
  - high = 4

The disease layer now contains county-level Arkansas disease observations rather than only statewide/national disease rows.

### Census Population Estimates

Source:

- U.S. Census Bureau county population estimates downloadable CSVs.

Added variables:

- population_total
- population_growth_abs
- births
- deaths
- natural_change
- international_migration
- domestic_migration
- net_migration
- birth_rate
- death_rate
- natural_change_rate
- net_migration_rate

Coverage:

- Arkansas county observations: 2010-2025
- Arkansas regional aggregate count variables: 2010-2025
- All 75 Arkansas counties retained.
- 8 Arkansas regions retained.

No Census API key was used. This bypassed the recurring Census API Missing Key response by using official downloadable CSV files.

### Arkansas Regions

Regions are based on Arkansas planning and development districts recognized in AR Code section 14-166-202.

Added regions:

- Northwest Arkansas Economic Development District
- North Central Arkansas Economic Development District
- Northeast Arkansas Economic Development District
- Southeast Arkansas Economic Development District
- Southwest Economic Development District of Arkansas
- Western Arkansas Economic Development District
- West Central Arkansas Economic Development District
- Central Arkansas Economic Development District

Each Arkansas county maps to exactly one region.

### Trade

Source:

- CEPII BACI HS92.

Expanded:

- Previous period: 2012-2024
- New period: 2000-2024

Variables retained:

- pharmaceutical_import_value
- pharmaceutical_export_value
- pharmaceutical_import_volume
- pharmaceutical_export_volume
- country_import_share
- country_export_share

Trade remains U.S.-level partner-country pharmaceutical exposure, not Arkansas-specific trade.

### Economics

Source:

- BLS Public Data API.

Added or expanded variables:

- arkansas_unemployment_rate
- arkansas_unemployed_population
- arkansas_employed_population
- arkansas_labor_force
- arkansas_healthcare_social_assistance_employment
- arkansas_private_education_health_services_employment
- national_unemployment_rate
- consumer_price_index_all_items
- consumer_price_index_medical_care
- consumer_price_index_medical_care_commodities
- consumer_price_index_prescription_drugs
- pharma_producer_price_index

Coverage:

- 2000-2026 where each BLS series supports the range.

## New Quality Metadata

### `quality/historical_coverage.json`

Contains per-variable/source/frequency/geography coverage fields:

- current_start
- current_end
- verified_public_start
- verified_public_end
- native_frequency
- geography
- historical_gap
- expansion_possible
- source
- action_taken
- quality_rank

### `quality/geographic_coverage.json`

Contains family-level geographic counts for:

- Arkansas statewide
- Arkansas counties
- Arkansas regions
- U.S. states
- countries
- national
- global
- stations
- monitors

### `quality/temporal_continuity.json`

Contains per-series continuity metrics:

- actual observations
- expected observations
- continuity percentage
- longest continuous run
- number of gaps
- longest gap in days

Also contains family-level mean continuity percentages.

### `quality/disease_coverage.json`

Contains disease identifier coverage:

- disease_id
- disease_name
- first_date
- last_date
- number_of_observations
- variables
- sources
- geographies

## Remaining Issues

- Census ACS detailed demographics remain incomplete. The Census API still returned key-gated HTML in this environment, but county population estimates were successfully retrieved through official Census downloadable CSVs.
- Arkansas county-level disease coverage improved substantially for COVID-19, but most other reportable diseases still lack county-level public rows in the integrated sources.
- Chronic disease remains limited to the CDC PLACES estimate year currently extracted.
- Arkansas-specific pharmaceutical trade was not created because the reliable public trade fallback is national U.S. BACI partner-country trade.
- FDA facility inspection/compliance data remains blocked by public endpoint access failures already documented in `quality/extraction_issues.json`.
- To stay within workspace disk constraints, per-domain JSONL mirror files are not retained. The required combined JSONL file is retained at `features/external_state_features.jsonl.gz`.

