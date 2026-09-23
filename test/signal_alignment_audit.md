# Synthetic-driver / model-signal alignment audit

This is an audit of benchmark construction, not evidence of real Arkansas demand.

- Sales rows: 32,880
- Candidate signal columns: 1,312
- Source signal rows used: 3,951
- Columns with two or fewer distinct daily values: 1,236

## Event-tag lift in the generated sales

| Tag | Rows | Mean units on | Mean units off | Lift |
|---|---:|---:|---:|---:|
| covid_summer_2025_outpatient_pulse | 217 | 30.171 | 5.007 | 6.03x |
| covid_summer_2024_outpatient_pulse | 224 | 20.147 | 5.070 | 3.97x |
| school_return | 1,464 | 18.145 | 4.568 | 3.97x |
| covid_summer_2023_outpatient_pulse | 189 | 15.000 | 5.116 | 2.93x |
| flu_2023_24 | 1,177 | 12.989 | 4.882 | 2.66x |
| dated_supply_outbreak_response | 8,412 | 9.166 | 3.800 | 2.41x |
| 2022_23_tripledemic_carryover | 180 | 11.267 | 5.139 | 2.19x |
| clinic_inventory_cap | 193 | 10.689 | 5.140 | 2.08x |
| rsv_winter | 3,249 | 9.140 | 4.738 | 1.93x |
| winter_respiratory | 3,971 | 8.112 | 4.769 | 1.70x |
| monday_catchup | 4,710 | 7.628 | 4.762 | 1.60x |
| friday_short_day | 4,680 | 6.612 | 4.934 | 1.34x |
| flu_2024_25 | 1,320 | 6.755 | 5.106 | 1.32x |
| spring_allergy | 552 | 5.009 | 5.175 | 0.97x |
| medicaid_unwinding_2023 | 3,285 | 4.640 | 5.232 | 0.89x |
| jan_jul_repricing | 180 | 4.356 | 5.177 | 0.84x |
| covid_waves | 3,288 | 3.536 | 5.354 | 0.66x |
| amoxicillin_shortage | 151 | 3.377 | 5.181 | 0.65x |
| shortage_procurement_premium | 151 | 3.377 | 5.181 | 0.65x |
| antibiotic_supply_pressure | 120 | 2.458 | 5.183 | 0.47x |
| diabetes_demand_growth | 1,462 | 0.642 | 5.384 | 0.12x |
| h5n1_monitoring | 122 | 0.582 | 5.190 | 0.11x |
| weekend_closed | 9,390 | 0.434 | 7.067 | 0.06x |
| OTC_competition_discount | 1,009 | 0.308 | 5.327 | 0.06x |
| naloxone_otc_switch | 1,009 | 0.308 | 5.327 | 0.06x |
| GLP1_procurement_premium | 731 | 0.114 | 5.288 | 0.02x |
| GLP1_shortage | 731 | 0.114 | 5.288 | 0.02x |

## Most variable signal columns

| Signal | Standard deviation | Nonzero days | Unique values |
|---|---:|---:|---:|
| model_external_state_feature_medicare_beneficiary_count_annual_national_US | 3.31e+07 | 732 | 3 |
| model_external_state_feature_adult_medicaid_enrollment_monthly_state_AR | 1.898e+05 | 701 | 3 |
| model_external_state_feature_medicaid_chip_total_enrollment_monthly_state_AR | 1.343e+05 | 1,066 | 4 |
| model_external_state_feature_medicaid_enrollment_monthly_state_AR | 1.206e+05 | 1,066 | 4 |
| model_external_state_feature_medicaid_chip_child_enrollment_monthly_state_AR | 6.819e+04 | 1,066 | 4 |
| model_external_state_feature_chip_enrollment_monthly_state_AR | 1.389e+04 | 1,066 | 4 |
| model_external_state_feature_medicare_per_capita_spending_annual_national_US | 6220 | 732 | 3 |
| model_external_state_feature_medicaid_chip_new_applications_monthly_state_AR | 3546 | 1,066 | 4 |
| model_external_state_feature_outpatient_utilization_annual_national_US | 2233 | 732 | 3 |
| model_external_state_feature_home_health_utilization_annual_national_US | 1014 | 732 | 3 |
| model_external_state_feature_emergency_department_utilization_annual_national_US | 276.4 | 732 | 3 |
| model_external_state_feature_inpatient_utilization_annual_national_US | 105.3 | 732 | 3 |
| model_external_state_feature_consumer_price_index_all_items_monthly_national_US | 51.63 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_lag_1_monthly_national_US | 51.53 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_rolling_mean_4_monthly_nationa | 51.35 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_lag_4_monthly_national_US | 51.16 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_rolling_mean_12_monthly_nation | 50.86 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_lag_12_monthly_national_US | 50.34 | 1,066 | 36 |
| model_external_state_feature_consumer_price_index_all_items_seasonal_baseline_monthly_nati | 37.07 | 1,066 | 36 |
| model_external_state_feature_post_acute_utilization_annual_national_US | 24.65 | 732 | 3 |
| model_news_output_arkansas_anti_infective_national_supply_monthly_state_AR | 23.76 | 1,066 | 22 |
| model_external_state_feature_nadac_per_unit_max_weekly_national_US | 16.89 | 1,066 | 2 |
| model_news_output_national_supply_chain_risk_monthly_state_AR | 16.68 | 1,066 | 22 |
| model_news_output_national_supply_stage_direction_monthly_state_AR | 16.68 | 1,066 | 22 |
| model_news_output_arkansas_anti_infective_composite_burden_monthly_state_AR | 15.3 | 1,005 | 18 |

## Strongest monthly associations with total sales

These are descriptive correlations, not causal claims.

| Signal | Correlation |
|---|---:|
| model_external_state_feature_arkansas_unemployment_rate_seasonal_baseline_monthly_state_AR | -0.418 |
| model_external_state_feature_consumer_price_index_all_items_rolling_std_4_monthly_national | -0.316 |
| model_external_state_feature_national_unemployment_rate_seasonal_baseline_monthly_national | -0.307 |
| model_external_state_feature_consumer_price_index_all_items_change_4_monthly_national_US | -0.307 |
| model_external_state_feature_arkansas_unemployment_rate_lag_12_monthly_state_AR | -0.298 |
| model_external_state_feature_dual_eligible_rate_annual_national_US | 0.289 |
| model_external_state_feature_post_acute_utilization_annual_national_US | 0.285 |
| model_external_state_feature_home_health_utilization_annual_national_US | 0.275 |
| model_external_state_feature_inpatient_utilization_annual_national_US | 0.273 |
| model_external_state_feature_emergency_department_utilization_annual_national_US | 0.273 |
| model_external_state_feature_medicare_beneficiary_count_annual_national_US | 0.271 |
| model_external_state_feature_outpatient_utilization_annual_national_US | 0.269 |
| model_external_state_feature_medicare_advantage_participation_rate_annual_national_US | 0.268 |
| model_external_state_feature_national_unemployment_rate_rolling_std_12_monthly_national_US | 0.265 |
| model_external_state_feature_medicare_per_capita_spending_annual_national_US | 0.263 |
| model_external_state_feature_national_unemployment_rate_seasonal_anomaly_monthly_national_ | 0.252 |
| model_external_state_feature_arkansas_unemployment_rate_change_12_monthly_state_AR | 0.250 |
| model_external_state_feature_arkansas_unemployment_rate_seasonal_anomaly_monthly_state_AR | 0.247 |
| model_news_output_arkansas_anti_infective_supply_policy_pressure_monthly_state_AR | 0.226 |
| model_external_state_feature_arkansas_unemployment_rate_rolling_mean_4_monthly_state_AR | 0.216 |
| model_news_output_arkansas_anti_infective_supply_chain_monthly_state_AR | 0.210 |
| model_news_output_arkansas_anti_infective_anesthesia_policy_monthly_state_AR | 0.209 |
| model_external_state_feature_national_unemployment_rate_rolling_mean_4_monthly_national_US | 0.206 |
| model_external_state_feature_arkansas_unemployment_rate_change_4_monthly_state_AR | -0.204 |
| model_external_state_feature_consumer_price_index_all_items_lag_4_monthly_national_US | 0.197 |

## Interpretation

The benchmark should not claim that every generated event is represented by a useful model signal. Annual or near-static columns cannot explain short outbreak windows, and duplicated derived columns should be deduplicated before counting independent evidence. A fair next benchmark either adds documented, time-varying event inputs or reports that the signals did not improve the event-driven synthetic demand.
