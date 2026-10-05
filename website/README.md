# Pulse — Pharmacy Demand and Replenishment Demo

A per-account demonstration of demand forecasting and inventory replenishment planning, with **XGBoost as the numerical prediction model**.

> **Development status:** Scientific proof-of-concept. Upload only synthetic or appropriately de-identified data; this is not an operational pharmacy system.

The current browser workspace trains XGBoost locally and does not upload sales
or inventory CSVs. The public API, daily worker, source-coverage gaps, local
startup commands, and remaining deployment gates are documented in
[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md). The older server-side demand
routes are disabled by default and are not used by the current frontend.
The worker also records official quarterly Arkansas Medicaid prescription
counts at `/api/v1/signals/sources/sdud/latest`. Suppressed counts remain
unknown; published totals are explicitly labeled lower bounds and are not
fed into the historical monthly ATC model.

## Demo scope

- Train one direct next-14-day XGBoost model per medication from prior daily sales, matching the selected `/test/` benchmark architecture.
- Request up to three years of the 1,312-ID public catalog and select at most five eligible signals per drug using training-period absolute correlation.
- Supply selected signals at 1-, 7-, and 14-day lags only when their source timestamp is known and precedes the feature date.
- Pair the forecast with the latest browser-selected on-hand inventory to demonstrate stockout timing and replenishment math.
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

This laptop currently runs the API, daily worker, and built frontend as
enabled WSL systemd services. Open `http://localhost:3141`; use
`systemctl status pulse-api pulse-worker pulse-frontend` inside WSL to inspect
them. See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the source
schedule, local service files, and deployment limits. Stop these units before
using the Compose example below because they bind the same local ports.

1. Copy `.env.example` to `.env`. For a local demo, the backend generates a
   persistent signing key automatically; configure a unique secret before any
   hosted deployment.
2. Run `docker compose up --build` (or `docker-compose up --build` with the
   legacy Compose command).
3. Open `http://localhost:3141`; Nginx serves the built frontend and forwards
   `/api`, `/docs`, and `/openapi.json` to the backend. The API readiness endpoint
   is `http://localhost:8000/ready`. Compose binds both ports to the laptop's
   loopback interface. The image includes the 1,312-ID manifest and a
   checksum-pinned 2023–2025 historical catalog. Startup reads its 6,484
   source rows, stores 4,097 distinct records, and imports the verified older
   news bridge before the daily worker
   adds source updates. Legacy model output rows remain marked unusable for
   current `/latest` results. Use `--source PATH` only for a separately
   verified replacement catalog.
   The backend image installs the pinned `requirements-prod.lock` runtime and
   leaves server-side XGBoost out; browser training is bundled with the
   frontend. `requirements.txt` remains the fuller local development and
   legacy-model test environment.
4. Run `make test` to verify the backend and forecasting pipeline.

## Model Rule

XGBoost is the primary numerical prediction model for this project. An LLM may optionally be used later to explain model outputs, summarize evidence, or provide a natural-language interface, but it should not silently replace the validated numerical prediction model.

## Demand and inventory demo

The private workspace locks after 15 minutes without activity or when its
server session expires. An active browser training run is canceled if the
workspace locks; unlock and start the run again.
Use **Sign out everywhere** in the dashboard navigation to revoke all server
sessions for the account, including bearer tokens. This also locks the current
browser plan. Other devices will lock when they next check their session;
their encrypted local plan records remain on those devices and require a new
sign-in with the account password to unlock.

Create an account, then select a sales CSV with `date,drug_name,units_sold` and an inventory CSV with `date,drug_name,on_hand_units` (optional `on_order_units,expected_arrival_date`). Sales need at least 130 consecutive daily rows per medication, including zero-sale days. Inventory snapshots must cover at least 130 sales days per medication, using a same-day or previous-day snapshot, and the latest inventory date must match the last sales date. A single inventory snapshot is rejected because the model must train on past inventory as well as sales. The browser reads these files and trains locally; it does not upload the CSVs or model to the API. Outstanding orders count toward forecast coverage only after a dated expected arrival; orders without an arrival date remain visible but are excluded from available stock. The generated plan is encrypted in the browser's IndexedDB and can be unlocked with the account password. Run `python -m scripts.generate_browser_demo ../test_data` from `website/` to create clearly synthetic, paired demo histories.

The browser trains a separate XGBoost regressor per drug on the direct total units sold over the next 14 calendar days. It selects up to five available public signals per drug using training data only, then uses each selected signal's 1-, 7-, and 14-day lags. Historical sales lags, rolling statistics, calendar fields, optional prior price and stockout fields, and lagged inventory are features. The 14-day total is allocated across days using recent day-of-week sales so the dashboard can show 1-day and 7-day estimates; those shorter-horizon figures are not separately validated models.

The replenishment policy uses a seven-day lead-time scenario, 28-day trailing demand, and a safety-stock calculation. It is a planning illustration, not a purchasing recommendation.

The browser forecaster requests up to three years of the public signal catalog
through the API, as selected by the user, ending on the last sales date in the
uploaded file. This also supports historical backtests whose sales end before
today. Future-dated sales or inventory rows are rejected using the browser's
local calendar date. Saved plans with sales or any drug's inventory ending more
than one day ago are clearly labeled historical on the dashboard; their quantities are not current
order guidance. Only records with a known source
timestamp available by the feature date can enter training; missing values are
not filled with zero. The imported historical catalog has 1,312 signal IDs,
including 1,212 CMS Part D drug-demand states, but its latest drug states
target 2025. Its schema and scope are documented in
`test/test_Signals/README.md`.

## Safety / Data Rules

Do not commit patient-identifying information, private pharmacy datasets, API keys, passwords, `.env` files, raw production datasets, or unreviewed model artifacts containing sensitive data. Use synthetic, public, or appropriately de-identified data during development.

## Disclaimer

This project is a research/development system. Unless separately validated and authorized, predictions and replenishment quantities must not be represented as medical diagnoses, clinical decisions, guaranteed procurement requirements, or a replacement for pharmacists or other qualified professionals.
