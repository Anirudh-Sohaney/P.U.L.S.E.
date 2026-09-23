# Pulse — Pharmacy Demand and Replenishment Demo

A per-account demonstration of demand forecasting and inventory replenishment planning, with **XGBoost as the numerical prediction model**.

> **Development status:** Scientific proof-of-concept. Upload only synthetic or appropriately de-identified data; this is not an operational pharmacy system.

## Demo scope

- Train one direct next-14-day XGBoost model per medication from prior daily sales, matching the selected `/test/` benchmark architecture.
- Use the full frozen catalog of 1,312 candidate signals, selecting five per drug on training-period absolute correlation.
- Supply each selected signal at 1-, 7-, and 14-day lags only after its source period closes.
- Pair the forecast with the account's latest uploaded on-hand inventory to demonstrate stockout timing and replenishment math.
- Keep the sales, inventory, forecast, and evaluation assumptions auditable.

## Repository Structure

```text
website/
├── backend/              # API/server (Python)
├── frontend/             # Web UI (React/TypeScript)
├── ml/                   # Data/feature/model pipeline (XGBoost)
├── data/                 # Local development data
├── tests/                # Automated tests
├── scripts/              # Utility and debug scripts
├── docs.md               # Full project specifications
├── .env.example          # Environment-variable template
├── Dockerfile            # Container build
├── docker-compose.yml    # Multi-service orchestration
├── Makefile              # Common commands
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

1. Copy `.env.example` to `.env` and configure secrets.
2. `docker-compose up` or follow Makefile targets.
3. Run `make test` to verify the pipeline.

## Model Rule

XGBoost is the primary numerical prediction model for this project. An LLM may optionally be used later to explain model outputs, summarize evidence, or provide a natural-language interface, but it should not silently replace the validated numerical prediction model.

## Demand and inventory demo

Create an account, then upload a sales CSV with `date,drug_name,units_sold` and an inventory CSV with `date,drug_name,on_hand_units` (optional `on_order_units`). Sales need at least 130 consecutive daily rows per medication, including zero-sale days. The backend stores uploads, the trained model, and forecasts in that account's directory. The repository's synthetic Arkansas sales CSV can be used for a demo.

The backend trains a separate XGBoost regressor per drug on the direct total units sold over the next 14 calendar days. It selects up to five signals per drug from 1,312 dated candidates by absolute correlation on training data only, then uses each selected signal's 1-, 7-, and 14-day lags. Historical lag, rolling-statistic, calendar, prior-price, and prior-stockout features and XGBoost parameters match `test/run_publishable_benchmark.py`; price and stockout default to zero if absent in an upload. The 14-day total is allocated across days using recent day-of-week sales so the dashboard can show 1-day and 7-day estimates; those shorter-horizon figures are not separately validated models.

The replenishment policy uses a seven-day lead-time scenario, 28-day trailing demand, and a safety-stock calculation. It is a planning illustration, not a purchasing recommendation.

The forecaster joins the long-format catalog at
`test/test_Signals/signals_2023_2025.csv.gz`, selects five signals separately
for each drug using training rows only, and applies 1-, 7-, and 14-day lags
after each source period closes. The catalog has 1,312 unique dated signals,
including 1,212 derived drug-demand states; its schema and scope are
documented in that directory's `README.md`.

## Safety / Data Rules

Do not commit patient-identifying information, private pharmacy datasets, API keys, passwords, `.env` files, raw production datasets, or unreviewed model artifacts containing sensitive data. Use synthetic, public, or appropriately de-identified data during development.

## Disclaimer

This project is a research/development system. Unless separately validated and authorized, predictions and replenishment quantities must not be represented as medical diagnoses, clinical decisions, guaranteed procurement requirements, or a replacement for pharmacists or other qualified professionals.
