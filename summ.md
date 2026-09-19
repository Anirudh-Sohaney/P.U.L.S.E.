# Arkansas-First Pharmaceutical Forecasting Project Summary

Last reviewed: 2026-09-14. Arkansas remains the primary geography, but the
model is being expanded to use national, neighboring-state, and global context
without relabeling those observations as Arkansas pharmacy truth.

## Project Objective
Build an Arkansas-focused pharmaceutical forecasting system that can:
- Combine real, readily available inputs such as news, disease surveillance, weather, supply-chain signals, and drug identity data.
- Forecast pharmaceutical demand, supply disruption risk, and shortage impact at a drug- and region-aware level.
- Stay under a practical parameter budget by using a hybrid of regression, neural, and language-model-derived features rather than a single giant model.
- Produce strong computed evidence that the model would improve real pharmacy inventory prediction.

## Code Base Directory
- Main model code: `model/arkansas_pharma_signal/`
- Project docs: `model/docs/`
- Tests: `model/tests/`
- Artifacts:
  - `model/artifacts/panel/`
  - `model/artifacts/evaluation/`
  - `model/artifacts/forecasts/`
  - `model/artifacts/metadata/`
  - `model/artifacts/trained/`

## What Was Built
The project now contains a multi-layer forecasting stack with:
- Annual Arkansas city-drug demand modeling from CMS Part D and external real-world features.
- A strict next-period evaluation framework with train/validation/test time splits.
- A validated quarterly Arkansas Medicaid demand model that works as a stronger near-term signal.
- A quarterly forecast artifact for Medicaid prescription pressure in Arkansas.
- A shortage-risk evaluator with validation-selected candidate ranking and top-k metrics.
- A guarded Medicaid feature layer that distinguishes exact-match and broader identifier-bridge variants.

## Core Model Surfaces
### Annual demand path
- Predicts next-year demand for Arkansas city-drug rows.
- Uses history, disease, news, supply, provider, and identity layers.
- Includes a calibrated convex blend selected on validation WAPE.

### Quarterly Medicaid path
- Predicts next-quarter Arkansas Medicaid prescription demand by drug.
- Uses historical quarterly persistence, moving averages, quarterly seasonality, and real event layers.
- This is the strongest deployable near-term signal in the project.

### Shortage-risk path
- Predicts next-period shortage event risk.
- Evaluates candidate scores and selects by validation top-k recall, then AUPRC, then Brier.
- Includes diagnostic test-best reporting so hindsight performance is not mistaken for deployable performance.

## Real Data Used
The model uses only real data from the workspace, including:
- CMS Part D provider-drug annual data.
- Arkansas Medicaid SDUD quarterly prescription-count records.
- FDA NDC products and packages.
- FDA shortages and recalls.
- Arkansas and national disease surveillance.
- Arkansas news signals.
- External state and national economic/supply indicators.
- Entity and relationship graphs for drug identity and mappings.

## Important Computed Results
### Annual demand
- Best naive baseline: `city_drug_last`
- Best annual model: `calibrated_ridge_blend`
- Best current WAPE: `0.144370`
- Strongest naive WAPE: `0.146271`
- Improvement over strongest naive: `1.2994%`
- Status: better than baseline, but far below the 10% gate for a publishable inventory-impact claim

### Quarterly Medicaid demand
- Best quarterly model remains publishable on its own.
- Latest rolling one-year evidence previously cleared the quarterly gate.
- Quarterly model is the strongest operationally useful signal in the project.

### Medicaid annual feature layer
- Exact-match Medicaid coverage:
  - `124,926 / 544,070` rows
  - `22.96%` row coverage
  - `30.09%` claim-weighted coverage
  - `182 / 1,582` matched annual drug keys
- Audited bridge Medicaid coverage:
  - `228,107 / 544,070` rows
  - `41.93%` row coverage
  - `55.54%` claim-weighted coverage
  - `401 / 1,582` matched annual drug keys
- Guarded selector result:
  - Validation chose the `all` Medicaid variant
  - Best test WAPE with guarded selector: `0.144370`
  - The bridge improved coverage, but the overall annual gain still remains small

### Shortage risk
- Annual shortage-risk selection remains weak.
- The rolling shortage-risk path captured `0 / 1,535` positives for the selected model across six folds.
- Diagnostic hindsight selection captured only `20 / 1,535`.
- Status: not publishable, not marketable as an inventory-impact claim.

## Current Issues
The project is not finished in the sense of the original objective. The main unresolved issues are:
- Annual demand improvement is still too small to support a strong inventory-impact claim.
- Medicaid bridge coverage is much better than exact-match coverage, but broader coverage did not translate into a large strict-test gain.
- Shortage-risk prediction remains weak in the rolling setting, especially for top-k capture.
- The project still lacks a truly strong pharmacy-adjacent shortage-impact label such as:
  - backorder events,
  - substitution events,
  - local fill disruptions,
  - supplier allocation records,
  - higher-frequency shortage-status histories.
- Exact annual model selection still depends on validation gains that are modest, not decisive.
- The focused expert-gold, input-contract, and ARCOS evaluation suite now
  passes 52 tests after restoring the real Git LFS data files and adding the
  annual ARCOS feature join. The full suite and forecast gates remain the next
  verification step.
- The latest strict operational annual evaluation used 173 features and
  achieved WAPE `0.145785` versus `0.146271` for the strongest naive baseline
  (`0.33%` improvement). This is measurable but not a sufficient promotion
  result.

## Expansion policy

The existing model is preserved. Expansion proceeds through reusable,
provenance-bearing adapters for disease, news, FDA supply events, ARCOS
distribution, weather, economics, trade, sanctions, and provider/supplier
geography. Each candidate signal must retain its source and publication time,
join at an explicit drug/geography/supplier grain, and beat a simple
chronological baseline before promotion. Signals that are sparse, stale,
leaky, or below the declared accuracy gate remain research-only.

## What Is Actually Strong
- The quarterly Medicaid demand model is real, stable, and useful.
- The project has a rigorous no-leakage evaluation design.
- The code now distinguishes:
  - exact-match Medicaid features,
  - audited identifier-bridge Medicaid features,
  - selected production candidate,
  - diagnostic hindsight candidate.
- The documentation and tests are broad enough to keep the modeling claims honest.

## Current Artifacts of Note
- Annual panel: `model/artifacts/panel/panel.csv`
- Quarterly Medicaid panel: `model/artifacts/panel/medicaid_sdud_quarterly_panel.csv`
- Annual evaluation: `model/artifacts/evaluation/metrics.json`
- Annual leaderboard: `model/artifacts/evaluation/leaderboard.csv`
- Quarterly forecast: `model/artifacts/forecasts/quarterly_forecast.csv`
- Quarterly evidence: `model/artifacts/evaluation/quarterly_rolling_1y_metrics.json`
- Rolling shortage-risk evidence: `model/artifacts/evaluation/risk_rolling_metrics.json`

## Tests and Verification
- The model test suite passes.
- The project includes strict schema and leakage tests.
- The evaluation artifacts are regenerated after each major change.
- The current code base is consistent with the evidence written into the docs and JSON artifacts.

## Bottom Line
This project now has:
- A credible Arkansas quarterly Medicaid demand signal.
- A guarded annual demand model that incorporates real Medicaid features.
- A rigorous shortage-risk evaluation framework.
- Strong documentation and artifact discipline.

It does not yet have:
- Strong enough annual demand lift,
- Strong enough shortage-risk proof,
- Or enough inventory-impact evidence to honestly claim the full objective is complete.
