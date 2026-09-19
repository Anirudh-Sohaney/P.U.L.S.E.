# Data Sufficiency Assessment

Answer: **yes, the available data is now sufficient** to train a mixed forecasting system for Arkansas pharmaceutical demand, supply-risk, and shortage-risk forecasts across a broad set of regions, suppliers/facilities, drugs, and disease drivers.

This is not a claim that the final model will be perfect or that true inventory can be observed directly. It is a claim that the dataset is no longer merely plausible or thin: it now has enough supervised labels, Arkansas-specific demand history, facility/provider geography, supplier/manufacturer identifiers, disease/weather/news drivers, and shortage/recall outcomes to train and evaluate a practical ensemble of regression models, neural networks, and language-model/text encoders.

## What Changed

- Arkansas drug demand is now direct and high-volume: CMS Part D provider-drug-year rows add 3,425,234 Arkansas observations across 2013-2024, 16,237 NPIs, 1,603 generic drugs, and 310 prescriber cities.
- Arkansas controlled-substance demand is regionally richer: DEA ARCOS adds 4,106 Arkansas ZIP3/drug/year-quarter rows across 2006-2025 for 39 drug codes and 85 ZIP3s.
- Supplier/facility geography is now present: NPPES adds 97,528 Arkansas provider/location records and 8,717 secondary practice-location rows; ASBP directories add 11,716 pharmacy, hospital pharmacy, institutional pharmacy, and specialty pharmacy facility rows.
- Drug/manufacturer mapping is now stronger: FDA NDC adds 115,223 product rows and 216,542 package rows with labeler, product, active ingredient, route, dosage form, marketing category, and application fields.
- Shortage/supply labels are current: FDA shortages add 1,637 shortage records and FDA enforcement adds 3,168 recall/enforcement records from 2023-2026, supplementing the existing shortage/recall history in `final_data`.
- Disease surveillance is current enough for demand shocks: FluView covers Arkansas/national weekly ILI from 2012-2026; NNDSS adds Arkansas/national weekly notifiable diseases from 2022-2026; CDC wastewater adds Arkansas site-level and derived national weekly WVAL for SARS-CoV-2, influenza A, and RSV through 2026.
- News/event inputs are usable: 3DLNews2 Arkansas files add 13,689 raw article records from 2013-2022 and normalized keyword metadata/counts for 14 usable health/supply/disaster groups. GDELT was attempted but throttled.

## Modeling Coverage

- **Regression/time-series models:** enough structured numeric history for drug-level demand, shortage/recall event labels, weather, disasters, disease surveillance, prices, population, healthcare capacity, trade, and supply-chain indicators.
- **Neural networks:** enough multi-source panel data for embeddings over drug, NPI, facility, city/county/ZIP3, disease, supplier/labeler, and lagged external signals.
- **Language-model components:** enough text and metadata for a small domain classifier or embedding model over Arkansas news, FDA recall/shortage text, FDA distribution/reason fields, WHO disease outbreak text, and source documentation. A large LM can be used as a feature extractor or classifier; the data does not require training a frontier-scale LM from scratch.

## Remaining Caveats

- Direct pharmacy inventory and wholesaler allocation data are still not public, so the forecast should be framed as demand, supply-risk, and shortage-risk, not observed shelf inventory.
- FDA shortage labels are national. Arkansas-specific shortage impact must be learned by combining national shortage/recall labels with Arkansas demand, provider/facility geography, ARCOS demand, disease, weather, and news signals.
- ASBP live license verification was blocked by WAF; official directory PDFs are historical through 2022.
- NNDSS public consolidated data starts in 2022; older weekly disease history is covered primarily by FluView, COVID county history, global disease sources, and existing final_data disease features.
- 3DLNews2 keyword matching is intentionally broad and should be cleaned or modeled with a text classifier before production use.

## Bottom Line

The answer to the driving question is **qualified yes for research/prototype training**, but not yet for a validated weekly regional production metric. The strongest immediate target is a carefully labeled signal library that forecasts:

- per drug or drug class,
- per Arkansas city/county/ZIP3/HSA/region depending on source resolution,
- per provider/pharmacy/facility group where identifiers are available,
- per supplier/labeler/manufacturer when FDA NDC/shortage/recall fields map cleanly,
- per disease or disease family using FluView, NNDSS, wastewater, COVID, chronic-health, and news/disaster signals.

The new regional wastewater screen demonstrates the distinction: its
influenza output reaches 80.00% pooled exact accuracy, but only 27.31%
balanced accuracy and 75.00% high-zone precision. It is therefore retained as
research context, not counted as a passed metric or evidence of local pharmacy
inventory predictability.
