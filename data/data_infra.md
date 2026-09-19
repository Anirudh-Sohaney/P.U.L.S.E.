Arkansas Pharmaceutical External-State Data Architecture
Status: Final  
System: Arkansas pharmaceutical forecasting API  
Purpose: Provide pharmacy-specific forecasting models with a
standardized, continuously updated representation of external conditions
that can influence medicine demand, medicine supply, or both.
---
1. System objective
The system maintains an external-state data layer for Arkansas
pharmacies.
It does not predict a pharmacy's inventory directly.
It produces structured variables describing:
disease activity and disease trends;
environmental conditions;
healthcare utilization and access;
population and demographic structure;
economic conditions;
pharmaceutical supply and shortage conditions;
manufacturer and facility health;
trade and transportation exposure;
government, regulatory, and reimbursement conditions;
disasters and other acute disruptions.
A pharmacy combines these variables with its own historical inventory,
dispensing, purchasing, lead-time, and patient data.
The intended downstream relationship is:
``` text
Arkansas External State
        +
Pharmacy Historical State
        ↓
Pharmacy-specific forecasting model
        ↓
future demand / inventory requirement
```
The external API therefore supplies exogenous/contextual variables,
not a universal pharmacy forecast.
This architecture is grounded in pharmaceutical forecasting research
showing that historical demand alone can omit hidden factors, and that
supply-chain information, contextual variables, seasonal disease
outbreaks, inventory information, and population dynamics can improve
pharmaceutical forecasting. See:
Bilal et al. (2025), The missing puzzle piece: contextual insights
for enhanced pharmaceutical supply chain forecasting:
https://doi.org/10.1080/00207543.2025.2546028
Zhu et al. (2021), Demand Forecasting with Supply-Chain Information
and Machine Learning: Evidence in the Pharmaceutical Industry:
https://doi.org/10.1111/poms.13426
Bertolotti et al. (2024), A prediction framework for pharmaceutical
drug consumption using short time-series:
https://doi.org/10.1016/j.eswa.2024.124265
Shukar et al. (2021), Drug Shortage: Causes, Impact, and Mitigation
Strategies: https://doi.org/10.3389/fphar.2021.693426
Aronson, Heneghan & Ferner (2023), Drug shortages. Part 2: Trends,
causes and solutions: https://doi.org/10.1111/bcp.15843
Cornelissen et al. (2025), Causes and management of drug shortages:
a scoping review: https://doi.org/10.1016/j.sapharm.2025.02.004
---
2. Design principles
2.1 External state, not direct inventory prediction
The API must not contain a variable whose meaning is "this pharmacy will
need 37 units of drug X next week."
Instead it contains variables such as:
``` text
influenza_activity
influenza_growth_4w
hospital_utilization_pressure
amoxicillin_shortage_status
manufacturer_health
trade_import_change
local_disaster_pressure
```
The pharmacy model determines how those variables affect its own
inventory.
2.2 Every variable has five required properties
Every variable must have:
``` text
variable_id
value
geography
observation_time
forecast_horizon
```
It must also have metadata:
``` text
source
source_timestamp
release_frequency
coverage_start
coverage_end
data_quality
missingness
transformation
```
2.3 Observations must be available before the prediction target
The API must prevent temporal leakage.
If a pharmacy forecast is generated at time `t`, the API may only expose
information that was publicly observable at or before `t`.
For retrospective training, the system must reconstruct the historical
information state as it existed at that date rather than using today's
revised value.
2.4 Raw observations and derived variables are separate
For every external feature:
``` text
raw observation
    ↓
normalized observation
    ↓
derived feature
```
Example:
``` text
CDC weekly ILI activity
    ↓
Arkansas ILI level
    ↓
ILI 1-week change
ILI 4-week change
ILI anomaly vs seasonal baseline
ILI forecast: next 1 week
ILI forecast: next 4 weeks
```
Raw observations are retained permanently.
2.5 Variable values are not forced into a universal -1 to +1 scale
Different variables have different semantics.
Use:
rates for rates;
counts for counts;
percentages for percentages;
index values for indices;
probabilities for probabilities;
categorical states for categorical states;
standardized scores only where a derived index is explicitly
defined.
The downstream model may standardize them.
---
3. Geographic architecture
The system uses a hierarchical geography.
Tier 0 --- Arkansas
The entire state.
Tier 1 --- Arkansas counties
All 75 Arkansas counties.
This is the primary local geographic unit.
Tier 2 --- Arkansas health/service regions
County-level data are aggregated into stable Arkansas regions for
variables that are not available at county level.
The canonical regions are:
``` text
Northwest Arkansas
North Central Arkansas
Northeast Arkansas
West Central Arkansas
Central Arkansas
East Central Arkansas
Southwest Arkansas
South Central Arkansas
Southeast Arkansas
```
County-level data always take precedence over regional aggregates.
Tier 3 --- United States
All 50 states + DC.
Used for:
national disease conditions;
state disease conditions;
pharmaceutical shortages;
healthcare policy;
state economic conditions;
national pharmaceutical supply conditions.
Tier 4 --- Foreign countries
Countries are represented individually when they have a documented
relationship to U.S. pharmaceutical supply, pharmaceutical
manufacturing, pharmaceutical trade, or a major relevant
disease/environmental pathway.
No arbitrary list of countries is used.
Countries enter the external-state graph through measurable
relationships such as:
``` text
U.S. pharmaceutical imports
API imports
finished-drug imports
known manufacturer location
known supplier relationship
trade volume
```
Tier 5 --- Global
Used for:
global disease events;
major geopolitical events;
global commodity disruptions;
global pharmaceutical supply disruptions.
---
4. Time architecture
The API maintains variables at the finest stable frequency supported by
the source.
The forecast horizons are standardized to:
`text H1 = next 7 days H2 = next 28 days H3 = next 90 days`
Not every variable is predicted at every horizon.
Frequency rules
Daily
Use for:
FDA drug shortages;
FDA recalls/enforcement;
acute disasters;
weather;
air quality;
rapidly changing supply events.
Derived daily variables:
`text current level 1-day change 7-day change 7-day rolling average 7-day anomaly next-7-day risk`
Weekly
Use for:
influenza;
RSV;
other weekly infectious-disease surveillance;
acute healthcare utilization where available.
Derived weekly variables:
`text current level 1-week change 4-week change seasonal anomaly next-week forecast next-4-week forecast`
Monthly
Use for:
trade;
employment;
Medicaid enrollment;
macroeconomic indicators;
pharmaceutical import/export activity.
Derived monthly variables:
`text current level month-over-month change 3-month change 12-month change 12-month seasonal anomaly next-month expectation next-3-month expectation where justified`
Annual
Use for:
population;
demographic composition;
chronic disease prevalence;
long-term economic structure;
healthcare access structure.
Annual variables are treated as slowly changing state variables, not
short-term forecasts.
---
5. Layer A --- Disease and epidemiological state
Disease activity is the primary external demand-side family.
Research supports seasonal disease activity and epidemiological
uncertainty as pharmaceutical-demand drivers.
5.1 Acute infectious diseases
The initial production disease set is:
``` text
influenza
COVID-19
RSV
pneumococcal disease
pertussis
measles
mumps
rubella
varicella
hepatitis A
hepatitis B
hepatitis C
HIV
syphilis
gonorrhea
chlamydia
tuberculosis
meningococcal disease
haemophilus influenzae
salmonellosis
campylobacteriosis
shigellosis
cryptosporidiosis
giardiasis
norovirus-related illness where surveillance supports it
```
The system may add diseases when a public longitudinal source exists and
the disease has a plausible medication-demand pathway.
5.2 Variables per disease
For each disease/geography pair:
`text disease_cases disease_rate disease_growth_1w disease_growth_4w disease_growth_12w disease_seasonal_anomaly disease_activity_index`
Where surveillance permits:
`text hospitalization_rate hospitalization_growth death_rate death_growth`
Forecast variables
For diseases with sufficiently stable weekly observations:
`text disease_activity_next_1w disease_activity_next_4w`
These are forecasts generated from the disease series, not from the news
model.
The news/LLM system may later provide an independent early-warning
feature.
5.3 Arkansas-specific disease geography
Use county-level values only when the source publishes sufficiently
consistent county-level data.
Otherwise:
`text Arkansas state     ↓ regional proxy`
must not be fabricated.
A missing county observation is represented as missing, with a
source/coverage flag.
5.4 Primary public sources
CDC NNDSS
CDC provides weekly provisional infectious-disease data and annual
finalized data. Weekly data are available from 2014-present, with
historical weekly tables extending further through CDC archives; annual
tables extend to the 1950s.
Source: https://www.cdc.gov/nndss/infectious-disease/
CDC FluView / ILINet
Weekly influenza-like-illness activity is available by state and CBSA.
Source: https://www.cdc.gov/fluview/
CDC NREVSS
Weekly RSV laboratory tests and detections are publicly available.
Source:
https://data.cdc.gov/Laboratory-Surveillance/Respiratory-Syncytial-Virus-Laboratory-Data-NREVSS/52kb-ccu2
Arkansas Department of Health
ADH publishes weekly viral respiratory disease reports covering
influenza, COVID-19, and RSV activity in Arkansas.
Source:
https://healthy.arkansas.gov/programs-services/diseases-conditions/communicable-diseases/influenza-flu/
ADH influenza surveillance also captures geographic and demographic
information from reported cases, although the public reporting system
does not expose a complete population-level case series.
Source:
https://healthy.arkansas.gov/programs-services/data-statistics-registries/influenza-surveillance-web-reporting/
---
6. Layer B --- Chronic disease and baseline health state
Acute surveillance does not describe the population's long-term
medication burden.
Use annual county-level prevalence variables for conditions that
influence persistent pharmaceutical demand.
Variables
``` text
diabetes_prevalence
hypertension_prevalence
hyperlipidemia_prevalence
coronary_heart_disease_prevalence
stroke_prevalence
asthma_prevalence
COPD_prevalence
depression_prevalence
chronic_kidney_disease_prevalence
arthritis_prevalence
obesity_prevalence
tobacco_use_prevalence
```
Source
CDC PLACES provides county, place, tract, and ZCTA estimates for health
outcomes, risk behaviors, preventive services, and social determinants.
Current and prior releases are publicly downloadable.
Source: https://www.cdc.gov/places/tools/data-portal.html
The 2024 county release provides nationwide county-level estimates.
Source:
https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2024-releas/d3i6-k6z5
PLACES includes the 2016-2019 500 Cities predecessor series and later
PLACES releases.
These variables are annual baseline state variables, not weekly outbreak
indicators.
---
7. Layer C --- Population and demographic state
Population structure changes the expected medication demand of a
geography.
Variables
``` text
population_total

population_age_0_4
population_age_5_17
population_age_18_34
population_age_35_64
population_age_65_plus

population_growth_1y
population_growth_5y

net_migration
birth_rate
death_rate

population_density
```
Where available:
``` text
sex_distribution
race_ethnicity_distribution
household_size
```
These are contextual variables and must not be interpreted as causal
medication predictors individually.
Sources
U.S. Census population estimates provide county totals, components of
change, and demographic characteristics. Current county totals cover
2020-2025, with historical vintages available through Census archives;
population-estimate series extend much farther historically.
Source:
https://www.census.gov/programs-surveys/popest/data/data-sets.html
ACS 5-year estimates provide annual county-level demographic,
socioeconomic, housing, insurance, and related measures from 2009-2024.
Source: https://www.census.gov/data/developers/data-sets/acs-5year.html
---
8. Layer D --- Healthcare utilization and access
This is the bridge between population/disease conditions and medication
use.
Variables
At county/state/HRR levels where publicly available:
``` text
medicare_beneficiary_count
medicare_per_capita_spending
outpatient_utilization
inpatient_utilization
emergency_department_utilization
hospitalization_rate
physician_service_utilization
home_health_utilization
post_acute_utilization
```
Additional access variables:
``` text
uninsured_rate
medicaid_enrollment
medicaid_enrollment_growth
medicare_enrollment
primary_care_access
preventive_service_use
usual_source_of_care
cost_related_care_avoidance
```
Sources
CMS Medicare Geographic Variation data provide annual state-, county-,
and hospital-referral-region-level demographic, spending, use, and
quality indicators.
Current data extend through 2024.
Source:
https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county
CMS Medicaid enrollment data are available monthly at state level. The
MBES series begins in 2014, and the public monthly data series is
available through Medicaid.gov.
Source:
https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data
Source:
https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360
CMS Part D Prescription Drug Profiles contain aggregated prescription
claims, drug characteristics, prescriber characteristics, demographics,
and payment information for Medicare Part D beneficiaries.
Source:
https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/prescription-drug-profiles-puf
CMS prescriber-level Part D data historically provide drug-specific
prescription counts and costs by prescriber, subject to privacy
suppression.
Source:
https://www.cms.gov/newsroom/fact-sheets/updated-prescriber-level-medicare-data
---
9. Layer E --- Economic state
Economic variables are included because affordability, healthcare
utilization, employment, population income, and pharmaceutical
purchasing conditions can change with economic conditions.
County-level variables
``` text
personal_income_per_capita
personal_income_growth
employment
unemployment
unemployment_rate
employment_growth
wage_income
poverty_rate
median_household_income
```
State/national variables
``` text
GDP
GDP_growth
inflation
CPI
unemployment_rate
employment_growth
consumer_price_pressure
healthcare_expenditure
```
Healthcare-specific economic variables
``` text
Medicaid_enrollment
uninsured_rate
Medicare_spending_per_capita
healthcare_spending_per_capita
```
Sources
BEA county personal income provides annual county personal income,
including wages, proprietors' income, dividends, interest, rents, and
government benefits.
Source: https://www.bea.gov/data/income-saving/personal-income-by-county
BEA county GDP provides county-level gross domestic product.
Source: https://bea.gov/data/gdp/gdp-by-county
BLS Local Area Unemployment Statistics provides state, county, and local
unemployment and labor-force measures.
Source: https://www.bls.gov/lau/data.htm
ACS provides annual 5-year county estimates for poverty, income,
insurance coverage, employment, and related socioeconomic
characteristics.
Source: https://www.census.gov/data/developers/data-sets/acs-5year.html
---
10. Layer F --- Environmental and weather state
Environmental conditions can affect disease activity, respiratory
illness, heat-related illness, disaster exposure, and healthcare
utilization.
Variables
Daily source variables:
``` text
temperature_mean
temperature_max
temperature_min
precipitation
snowfall
wind
humidity
```
Derived variables:
``` text
heat_index
cold_pressure
extreme_heat_days_7d
extreme_cold_days_7d
precipitation_anomaly_7d
temperature_anomaly_7d
```
Air-quality variables:
``` text
PM2_5
ozone
AQI
PM2_5_anomaly
ozone_anomaly
```
Derived health-pressure variables:
``` text
heat_health_pressure
respiratory_air_quality_pressure
extreme_weather_pressure
```
The derived pressure variables must be deterministic transformations of
the underlying measurements, not subjective LLM outputs.
Sources
NOAA Climate Data Online provides free historical daily, monthly,
seasonal, and annual weather and climate observations including
temperature, precipitation, wind, and related measures.
Source: https://www.ncdc.noaa.gov/cdo-web/
EPA Air Quality System provides monitor-level daily and aggregate
air-quality data through an API.
Source: https://aqs.epa.gov/aqsweb/documents/data_api.html
---
11. Layer G --- Disaster and acute disruption state
Disasters can simultaneously alter:
local healthcare utilization;
medication demand;
transportation;
facility access;
pharmaceutical distribution;
population location.
Variables
For each disaster/geography:
``` text
disaster_active
disaster_type
disaster_severity
disaster_area
population_affected
infrastructure_disruption
healthcare_disruption
transportation_disruption
```
Derived:
``` text
disaster_pressure_7d
disaster_pressure_28d
```
Do not use a generic "natural disaster = 1" variable.
The event must retain its type and geography.
Source
FEMA OpenFEMA disaster declarations provide federally declared disasters
from 1953 onward, including major disasters, emergencies, and
fire-management assistance, with geographic areas.
Source:
https://www.fema.gov/about/openfema/disaster-declarations-summaries
---
12. Layer H --- Pharmaceutical supply and shortage state
This is a core component.
Drug shortages are multifactorial. Published reviews identify
manufacturing problems, raw-material shortages, logistics, business
decisions, regulatory/political actions, recalls, just-in-time
inventory, unexpected demand increases, epidemics, and market changes as
contributors.
Therefore supply variables must be independent from disease-demand
variables.
Drug-level variables
For each drug/active ingredient where FDA data support identification:
``` text
shortage_status
shortage_duration
shortage_onset
shortage_resolution
shortage_reason
shortage_supply_level
shortage_demand_level
```
Derived:
``` text
shortage_active
shortage_age_days
shortage_risk_7d
shortage_risk_28d
shortage_risk_90d
```
Recall variables
``` text
recall_active
recall_class
recall_scope
recall_date
recall_duration
```
Derived:
``` text
recall_pressure_7d
recall_pressure_28d
```
Source
FDA's Drug Shortages API covers 2012-present and is updated daily.
Source: https://open.fda.gov/apis/drug/drugshortages/
FDA defines a drug shortage as a situation in which U.S. demand or
projected demand exceeds supply, and identifies manufacturing quality
problems, production delays, raw-material/component delays, unexpected
demand increases, and discontinuations among causes.
Source:
https://www.fda.gov/drugs/drug-shortages/frequently-asked-questions-about-drug-shortages
FDA's openFDA enforcement/recall data cover publicly releasable recall
records from 2004-present and are updated weekly.
Source: https://open.fda.gov/apis/drug/enforcement/
---
13. Layer I --- Manufacturer and manufacturing-facility state
Manufacturer health is represented as an observable
compliance/operational-risk state.
It is not treated as a claim about the company's financial health
unless a public quantitative financial source exists.
Facility variables
For each relevant drug manufacturer/facility:
``` text
facility_location
facility_product_count
facility_drug_role
facility_inspection_status
facility_last_inspection
facility_OAI_count
facility_VAI_count
facility_NAI_count
facility_warning_letter_count
facility_recall_count
facility_import_action_count
```
Derived:
``` text
facility_compliance_risk
facility_recent_compliance_risk
facility_disruption_pressure
```
A manufacturer/facility risk score must be computed from observed FDA
events.
Example deterministic construction:
``` text
OAI = high-risk event
VAI = moderate-risk event
NAI = no-action event
recent warning letter = elevated risk
recent product recall = elevated risk
```
The exact numerical mapping is defined in the feature-engineering
specification and versioned. It must not be changed silently.
Source
FDA drug inspection/compliance data provide facility-specific inspection
classifications and related enforcement information.
FDA's drug data dashboard contains pharmaceutical inspections,
compliance, recalls, and import actions from 2009-present.
Source:
https://www.fda.gov/drugs/guidance-compliance-regulatory-information/pharmaceutical-inspections-and-compliance
FDA inspection classifications are:
``` text
NAI = No Action Indicated
VAI = Voluntary Action Indicated
OAI = Official Action Indicated
```
Source:
https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/inspection-basics/inspection-classifications
The FDA inspection database is updated weekly but is not a complete
census of every FDA inspection; this limitation must be retained in
metadata.
Source:
https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/inspection-classification-database
---
14. Layer J --- International trade and pharmaceutical supply exposure
The system must not infer that a country matters merely because it is
politically important.
A country enters this layer when measurable trade or
pharmaceutical-supply exposure exists.
Variables
At state/country/commodity level:
``` text
import_value
export_value
import_volume
export_volume
import_growth_1m
import_growth_3m
import_growth_12m
export_growth_1m
export_growth_3m
export_growth_12m
```
For pharmaceutical-relevant HS codes:
`text pharmaceutical_import_value pharmaceutical_import_volume pharmaceutical_import_share pharmaceutical_import_country_concentration`
Derived:
`text trade_disruption_pressure import_dependency country_supply_exposure`
Time resolution
Monthly.
Do not create daily trade variables when the source is monthly.
Sources
Census International Trade provides monthly U.S. imports and exports by
state, trading partner, and Harmonized System code.
State-level HS exports are available monthly from 2002-present.
State-level HS imports are available monthly from 2008-present.
Source:
https://www.census.gov/foreign-trade/statistics/historical/hs.html
The Census API provides monthly state-level trade by HS code and trading
partner.
Source: https://api.census.gov/data/timeseries/intltrade.html
USITC DataWeb provides U.S. trade data and maintains electronic trade
data from 1989 onward.
Source: https://www.usitc.gov/applications/dataweb/faqs
---
15. Layer K --- Policy, regulation, reimbursement, and market state
Policy changes can alter medication utilization without a corresponding
disease change.
Variables
``` text
Medicaid_enrollment
Medicaid_enrollment_growth
Medicaid_policy_event
Medicare_policy_event
FDA_approval_event
FDA_label_change
FDA_warning_event
drug_recall_event
drug_discontinuation_event
prescribing_restriction_event
reimbursement_change_event
public_health_program_event
```
Where numeric public data exist:
``` text
policy_effective_date
affected_population
affected_drug_class
affected_geography
```
Policy/news-derived events are represented as event records rather than
arbitrary continuous scores.
The event system converts them into state variables only when a
deterministic relationship exists.
---
16. Layer L --- News-derived event state
The news corpus is a supplementary sensing system.
News is not itself a numeric economic/disease/supply variable.
The LLM converts articles into structured events.
Required event schema
``` json
{
  "event_id": "...",
  "event_type": "...",
  "entity_id": "...",
  "geography": "...",
  "start_time": "...",
  "reported_time": "...",
  "severity": 0.0,
  "confidence": 0.0,
  "expected_duration": "...",
  "source_id": "...",
  "affected_entities": [],
  "affected_drugs": [],
  "affected_diseases": []
}
```
Allowed event types include:
``` text
disease_outbreak
disease_growth
manufacturing_disruption
facility_shutdown
facility_fire
facility_damage
product_recall
drug_shortage
raw_material_disruption
transport_disruption
port_disruption
trade_restriction
tariff_change
sanction
regulatory_change
healthcare_facility_closure
hospital_overload
natural_disaster
economic_disruption
labor_strike
public_health_campaign
```
The LLM identifies the event.
The deterministic data layer determines which known entities, drugs,
locations, and relationships are affected.
---
17. Entity and relationship graph
The system requires a persistent entity graph.
Entity classes
``` text
disease
drug
active_ingredient
manufacturer
manufacturing_facility
supplier
country
state
county
healthcare_facility
distributor
port
trade_route
policy
disaster
```
Relationship classes
``` text
manufactures
produces
contains
supplies
imports
exports
distributes
located_in
depends_on
affected_by
treats
associated_with
substitutes
competes_with
```
This graph is what allows an upstream event to propagate without
requiring the LLM to memorize every possible connection.
Example:
``` text
Factory X
    └── manufactures → API Y
                           └── used_in → Drug Z
                                           └── supplied_to → US market
                                                            └── Arkansas
```
If Factory X is disrupted, the system updates the Factory X state and
propagates supply-risk features through known relationships.
The LLM does not directly invent the downstream pharmaceutical
consequence.
---
18. Feature classes exposed to pharmacies
The API exposes five classes of external features.
18.1 Current-state features
Examples:
``` text
influenza_activity
RSV_activity
unemployment_rate
drug_shortage_status
manufacturer_compliance_risk
PM2_5
```
18.2 Trend features
Examples:
``` text
influenza_growth_1w
influenza_growth_4w
unemployment_growth_3m
drug_shortage_change_7d
import_growth_3m
```
18.3 Anomaly features
Examples:
``` text
influenza_anomaly_vs_5y_seasonal_baseline
temperature_anomaly
trade_import_anomaly
hospitalization_anomaly
```
18.4 Forward external-state features
Examples:
``` text
influenza_activity_next_1w
influenza_activity_next_4w
shortage_risk_next_4w
disaster_pressure_next_7d
```
These are generated only for variables with sufficient historical data
and a validated forecasting model.
18.5 Event features
Examples:
``` text
new_recall_7d
manufacturing_disruption_7d
trade_restriction_28d
disease_outbreak_7d
```
Event variables preserve the event's identity and metadata.
---
19. Final API output contract
A pharmacy request is:
``` http
GET /v1/external-state
    ?county=05
    &as_of=2026-08-10
    &horizon=28d
```
The API returns a versioned feature vector.
Conceptual structure:
``` json
{
  "schema_version": "1.0",
  "as_of": "2026-08-10",
  "geography": {
    "state": "AR",
    "county": "Benton"
  },

  "disease": {
    "influenza": {
      "activity": 72.1,
      "growth_1w": 8.4,
      "growth_4w": 31.2,
      "seasonal_anomaly": 19.3,
      "next_1w": 76.5,
      "next_4w": 81.2
    }
  },

  "healthcare": {
    "utilization_index": 61.4,
    "hospital_pressure": 57.1,
    "medicare_spending_per_capita": 12453
  },

  "population": {
    "population_total": 321000,
    "population_growth_1y": 1.8,
    "population_age_65_plus_pct": 13.2
  },

  "economy": {
    "unemployment_rate": 3.7,
    "unemployment_change_3m": 0.2,
    "personal_income_per_capita": 61200
  },

  "environment": {
    "pm25": 7.2,
    "temperature_anomaly_7d": 2.1,
    "extreme_heat_days_7d": 1
  },

  "supply": {
    "amoxicillin_shortage": 0,
    "amoxicillin_shortage_risk_28d": 0.12
  },

  "manufacturing": {
    "relevant_manufacturer_risk": 0.31
  },

  "trade": {
    "pharmaceutical_import_change_3m": -4.7,
    "pharmaceutical_import_anomaly": -8.2
  },

  "events": [
    {
      "event_type": "manufacturing_disruption",
      "severity": 0.73,
      "confidence": 0.91,
      "affected_entity": "manufacturer_id"
    }
  ]
}
```
The numeric values above are schema examples only. Production values are
populated exclusively from the actual data pipeline.
---
20. Relevance filtering
The underlying state may contain tens of thousands of variables.
A pharmacy does not receive all of them.
The API performs deterministic relevance filtering using:
``` text
pharmacy geography
drug inventory
drug classes
known suppliers
known manufacturers
patient population if supplied
historical pharmacy demand if supplied
```
The API can therefore return:
``` text
global state
      ↓
US state
      ↓
Arkansas
      ↓
county
      ↓
drug/manufacturer dependency graph
      ↓
pharmacy-relevant variables
```
This is preferable to sending every pharmacy an identical
100,000-dimensional vector.
---
21. Target forecasting horizons
The API is designed around the downstream pharmacy forecasting problem.
7-day horizon
Primary inputs:
``` text
acute disease trends
weather
air quality
disasters
new recalls
new shortages
acute healthcare pressure
recent events
```
28-day horizon
Primary inputs:
``` text
disease trajectory
seasonality
healthcare utilization
shortage trajectory
manufacturer risk
trade movement
economic movement
policy changes
```
90-day horizon
Primary inputs:
``` text
seasonality
population
economic trends
trade trends
manufacturer state
longer-term disease trends
planned policy changes
structural healthcare conditions
```
---
22. Historical training representation
For every timestamp `t`, the dataset contains:
``` text
external_state(t)
```
and the pharmacy model later supplies:
``` text
pharmacy_state(t)
```
The forecasting target is:
``` text
pharmacy_demand(t + H)
```
or:
``` text
pharmacy_inventory_requirement(t + H)
```
where `H` is 7, 28, or 90 days.
The external-state system is therefore evaluated independently:
``` text
Does external_state(t)
contain information that improves
pharmacy_demand(t + H)?
```
---
23. Data-quality requirements
Every feature has:
``` text
source_id
source_name
source_url
source_frequency
coverage_start
coverage_end
last_observation
publication_lag
revision_status
geographic_resolution
missingness
confidence
transformation_version
```
No feature enters the production API without these fields.
Source hierarchy
Priority:
``` text
1. Federal/state administrative data
2. Federal/state surveillance systems
3. Official regulatory data
4. Official trade/economic data
5. Peer-reviewed derived datasets
6. Reputable secondary datasets
7. News-derived events
```
News is therefore a sensor for events, not the authoritative source
when an authoritative quantitative source exists.
Example:
``` text
News:
"Flu cases are surging."

CDC:
actual weekly ILI activity = X

Production state:
CDC measurement = primary variable
news event = supplementary early-warning event
```
---
24. Public-data availability matrix
---
Variable family  Primary public                Frequency  Reliable historical
source                                              coverage
---
Population       Census                           Annual              decades
Population  
Estimates
Demographics     Census ACS                       Annual            2009-2024
5-year
Chronic disease  CDC PLACES                       Annual   2016-2024 releases
prevalence
Influenza        CDC                              Weekly      long historical
activity         FluView/ILINet                            series; production
baseline 2014+
Notifiable       CDC NNDSS                 Weekly/annual weekly 2014+, annual
diseases                                                              decades
RSV              CDC NREVSS                       Weekly           multi-year
Arkansas         ADH                              Weekly   current historical
respiratory                                                           reports
activity
Medicare         CMS Geographic                   Annual           multi-year
utilization      Variation
Medicare         CMS Part D PUFs                  Annual           multi-year
prescription  
utilization
Medicaid         Medicaid.gov                    Monthly                2014+
enrollment
Income           BEA county                       Annual      long historical
personal income                                       series
County GDP       BEA                              Annual           multi-year
Unemployment     BLS LAUS                        Monthly      long historical
series
Weather          NOAA CDO           Daily/monthly/annual             decades+
Air quality      EPA AQS                           Daily              decades
Disasters        FEMA                    Event/daily API                1953+
Drug shortages   FDA openFDA                       Daily                2012+
Drug recalls     FDA openFDA                      Weekly                2004+
Drug facility    FDA                      Weekly/current                2009+
compliance
Pharmaceutical   Census                          Monthly       2002+ state HS
trade            International                           exports; 2008+ state
Trade                                             HS imports
U.S. trade       Census/USITC                    Monthly          1992+/1989+
depending product
News events      Internal corpus            event-driven      ~2000s-present
based on corpus
---
25. Variables explicitly excluded
The following are not part of the core external-state architecture
unless a reliable public longitudinal source becomes available.
25.1 Arbitrary "sentiment"
Do not expose:
``` text
news_sentiment = -0.73
```
as a production pharmaceutical feature.
Sentiment has no stable pharmaceutical interpretation.
25.2 Generic country importance
Do not expose:
``` text
country_importance = 83
```
unless the value is derived from measurable pharmaceutical/trade
exposure.
25.3 Unverified manufacturer "health"
Do not infer:
``` text
Company X financial health = 41
```
from news sentiment.
Use observable variables such as:
``` text
inspection classification
recall activity
warning letters
shortage reports
production disruption
trade exposure
```
25.4 Daily disease variables when only weekly surveillance exists
Do not interpolate weekly disease observations into fake daily case
counts.
25.5 County-level disease estimates that do not exist
Do not manufacture county-level disease activity from state-level data.
The system records the actual geographic resolution of the source.
---
26. Core hypothesis to be tested
The architecture exists to test one empirical hypothesis:
> **Adding external-state variables derived from public epidemiological,
> environmental, healthcare, economic, regulatory, trade, supply-chain,
> manufacturing, disaster, and news-event data improves out-of-sample
> pharmacy demand/inventory forecasting over historical pharmacy data
> alone.**
The comparison must contain at least:
```text Model A: pharmacy historical data only
Model B: pharmacy historical data + non-news external state
Model C: pharmacy historical data + non-news external state +
news-derived event state

    The contribution of the API is demonstrated only if B and/or C improves out-of-sample forecasting under leakage-controlled evaluation.

    ---

    # 27. Final architecture

    ```text
                        PUBLIC WORLD
                             │
         ┌───────────────────┼────────────────────┐
         │                   │                    │
      Disease             Economy             Supply
         │                   │                    │
     Weather             Healthcare           Trade
         │                   │                    │
     Disasters            Policy            Manufacturers
         │                   │                    │
     News ──────────────────┴────────────────────┘
                             │
                             ▼
                    RAW DATA INGESTION
                             │
                             ▼
                    NORMALIZATION LAYER
                             │
                             ▼
                  ENTITY / RELATION GRAPH
                             │
                             ▼
                 EXTERNAL STATE ENGINE
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
       Current state       Trends          Forecasts
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ARKANSAS STATE API
                             │
                  relevance filtering
                             │
                             ▼
                    PHARMACY-SPECIFIC
                     EXTERNAL VECTOR
                             │
                             ▼
                LOCAL PHARMACY MODEL
                             │
            historical demand + inventory
            + external state
                             │
                             ▼
                    INVENTORY FORECAST

The system's defining principle is:
The API models the world outside the pharmacy. The pharmacy model
decides what that world means for its own inventory.
---
28. Primary research and data references
Pharmaceutical forecasting
Bilal, A. I., Tabar, B. R., Hewage, H. H., Bititci, U., & Fenta, T.
(2025). The missing puzzle piece: contextual insights for enhanced
pharmaceutical supply chain forecasting. International Journal of
Production Research. https://doi.org/10.1080/00207543.2025.2546028
Zhu, X., Ninh, A., Zhao, H., & Liu, Z. (2021). Demand Forecasting
with Supply‐Chain Information and Machine Learning: Evidence in the
Pharmaceutical Industry. Production and Operations Management, 30,
3231-3252. https://doi.org/10.1111/poms.13426
Bertolotti, F., Schettini, F., Ferrario, L., Bellavia, D., &
Foglia, E. (2024). A prediction framework for pharmaceutical drug
consumption using short time-series. Expert Systems with
Applications, 253, 124265.
https://doi.org/10.1016/j.eswa.2024.124265
Fourkiotis, K. P., & Tsadiras, A. (2024). Applying Machine Learning
and Statistical Forecasting Methods for Enhancing Pharmaceutical
Sales Predictions. Forecasting.
https://doi.org/10.3390/forecast6020018
Chen, X., Lu, G., Zhang, H., & Wan, J. (2026). Knowledge
graph-enhanced deep learning for pharmaceutical demand forecasting.
Scientific Reports, 16. https://doi.org/10.1038/s41598-026-40107-3
Drug shortages and supply chains
Shukar, S., Zahoor, F., Hayat, K., et al. (2021). Drug Shortage:
Causes, Impact, and Mitigation Strategies. Frontiers in
Pharmacology, 12, 693426. https://doi.org/10.3389/fphar.2021.693426
Aronson, J., Heneghan, C., & Ferner, R. (2023). Drug shortages.
Part 2: Trends, causes and solutions. British Journal of Clinical
Pharmacology, 89, 2957-2963. https://doi.org/10.1111/bcp.15843
Cornelissen, N., Zielhuis, S., van den Bemt, P. M. L. A., & van den
Bemt, B. (2025). Causes and management of drug shortages: a scoping
review. Research in Social and Administrative Pharmacy.
https://doi.org/10.1016/j.sapharm.2025.02.004
Pava, M. L. S. L., & Tucker, E. L. (2023). Effects of geopolitical
strain on global pharmaceutical supply chain design and drug
shortages. European Journal of Operational Research, 327, 641-654.
https://doi.org/10.1016/j.ejor.2023.11.030
Arkansas-specific evidence
Lindner, S. R., Hart, K., Manibusan, B., McCarty, K. J., Meinhofer,
A., & Cunningham, P. (2023). State- and County-Level Geographic
Variation in Opioid Use Disorder, Medication Treatment, and
Opioid-Related Overdose Among Medicaid Enrollees. JAMA Health
Forum, 4. https://doi.org/10.1001/jamahealthforum.2023.4018
Toth, M., Moore, P., Tant, E., et al. (2020). Early impact of the
implementation of Medicaid episode-based payment reforms in
Arkansas. Health Services Research.
https://doi.org/10.1111/1475-6773.13316
Mahashabde, R. V., Shrikhande, M., Han, X., Martin, B., Elhassan,
N., & Hayes, C. (2022). Concordance of opioid exposure in all-payer
claims databases with prescription drug monitoring program database
using Arkansas as a case example. Health Services Research.
https://doi.org/10.1111/1475-6773.14016
Golden, W., Peng, C., Shoptaw, E. J., et al. (2025). Measurement of
Practice-Level Antibiotic Utilization in a Medicaid Patient-Centered
Medical Home Program. Annals of Family Medicine, 23, 407-411.
https://doi.org/10.1370/afm.2410
Public data
CDC NNDSS: https://www.cdc.gov/nndss/infectious-disease/
CDC FluView: https://www.cdc.gov/fluview/
CDC NREVSS RSV:
https://data.cdc.gov/Laboratory-Surveillance/Respiratory-Syncytial-Virus-Laboratory-Data-NREVSS/52kb-ccu2
Arkansas Department of Health influenza/respiratory surveillance:
https://healthy.arkansas.gov/programs-services/diseases-conditions/communicable-diseases/influenza-flu/
CDC PLACES: https://www.cdc.gov/places/
Census population estimates:
https://www.census.gov/programs-surveys/popest/data/data-sets.html
Census ACS:
https://www.census.gov/data/developers/data-sets/acs-5year.html
CMS Medicare Geographic Variation:
https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county
CMS Part D Prescription Drug Profiles:
https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/prescription-drug-profiles-puf
Medicaid enrollment:
https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data
BEA county personal income:
https://www.bea.gov/data/income-saving/personal-income-by-county
BEA county GDP: https://bea.gov/data/gdp/gdp-by-county
BLS LAUS: https://www.bls.gov/lau/data.htm
NOAA Climate Data Online: https://www.ncdc.noaa.gov/cdo-web/
EPA AQS API: https://aqs.epa.gov/aqsweb/documents/data_api.html
FEMA OpenFEMA:
https://www.fema.gov/about/openfema/disaster-declarations-summaries
FDA Drug Shortages: https://open.fda.gov/apis/drug/drugshortages/
FDA Enforcement/Recalls: https://open.fda.gov/apis/drug/enforcement/
FDA pharmaceutical inspections/compliance:
https://www.fda.gov/drugs/guidance-compliance-regulatory-information/pharmaceutical-inspections-and-compliance
Census International Trade:
https://www.census.gov/foreign-trade/statistics/historical/hs.html
Census International Trade API:
https://api.census.gov/data/timeseries/intltrade.html
USITC DataWeb: https://www.usitc.gov/applications/dataweb/faqs
