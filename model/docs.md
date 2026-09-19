# Production Model Documentation

The `/model/` directory has been transitioned from a testing/experimental sandbox to a final production-ready state.

## Final Deliverable
The primary entry point is `model/prod_pipeline.py`. 
When executed, this script:
1. Fetches real-world input data (via APIs and placeholders for News).
2. Performs full-data model training (merging all historical training and testing splits into one final model).
3. Produces a unified prediction dictionary containing:
   - 20 base news sentiment signals (monthly)
   - 18 Arkansas ATC therapeutic demand state signals (monthly)
   - 1,368 CMS Part D drug demand state signals (annual)
4. Outputs the results to `model/final_predictions.json`.

## Data Sources
- **HHS Medicaid Provider Spending**: Monthly drug billing data.
- **CMS Medicare Part D**: Annual prescribers summary.
- **GDELT News Events**: Pre-processed monthly sentiment features.

## Cleanup
All outdated `evaluate_*.py` and `build_*.py` testing scripts have been removed to keep the codebase clean, simple, and strictly aligned with production delivery. The historical rigorous evaluation metadata remains located in `model/docs/`.
