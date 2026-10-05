# PULSE

PULSE is a research/demo project for pharmaceutical demand forecasting and pharmacy replenishment planning. It contains two related but separate systems:

1. **`website/`** — a React + FastAPI service. A separate daily worker records public source data; the browser trains a private per-drug XGBoost model from dated sales and inventory histories and displays illustrative replenishment calculations.
2. **`model/`** — an Arkansas-first public-data signal and forecasting research package. It builds/evaluates demand, shortage, disease, news, weather, supply, and related proxy metrics from the documented local data.

The website's synthetic pharmacy records are fabricated for testing, not observed pharmacy transactions. The public-data research model also does not observe pharmacy shelf stock, wholesaler allocation, backorders, or stockout ground truth. Neither output should be treated as a procurement instruction or clinical decision.

## Repository layout

```text
PULSE/
├── data/                  # Public-source, normalized, and synthetic datasets
│   ├── synthetic_pharmacy_data/  # Reproducible synthetic sales benchmark
│   └── ...                # Source families, manifests, crosswalks, and scripts
├── test_data/             # Convenient paired CSV inputs for the website demo
├── model/                 # Public-signal research model, tests, docs, artifacts
├── website/               # FastAPI backend, React frontend, and XGBoost demo
├── test/                  # Local-only benchmark/evaluation material; not in GitHub
├── summ.md                # Project notes/status (may be less current than code/docs)
└── README.md              # This guide
```

For each component's detailed architecture, see [model/README.md](model/README.md), [website/README.md](website/README.md), and [data/README.md](data/README.md).

## Prerequisites and external tools

For local development, install:

- **Git** to clone the repository.
- **Python 3.11** (3.10+ is required by the website package) and `pip` for the API/model environments.
- **Node.js 20 LTS** (includes `npm`) for the React/Vite frontend.
- **Git LFS** if you need data files tracked as LFS objects; some public-data payloads are not stored as ordinary Git blobs.

**Docker Desktop with Docker Compose v2** is optional. The current browser-training service includes a pinned historical signal catalog in the website build context. The older Python research workflow still uses a local benchmark path outside `website/`.

The older research demo needs no paid API key or live external service when its local inputs are available. A clean GitHub clone does **not** include the local-only `test/` benchmark directory. Its Python `DemandForecaster` expects:

```text
test/test_Signals/signals_2023_2025.csv.gz
```

The file contains the frozen dated signal catalog used by that evaluated research approach. It is excluded from GitHub; obtain the approved local benchmark artifact separately. Without it, the older `DemandForecaster(use_model_signals=True)` stops with a missing-catalog error. The current browser dashboard instead reads the API's bundled historical catalog and current verified source records. Do not substitute generated or differently-versioned signals and call the result the same benchmark.

To retrieve repository data stored with Git LFS (when you have cloned the repository):

```bash
git lfs install
git lfs pull
```

Git LFS is only needed for LFS-backed source assets; it does not restore the local-only `test/` catalog.

## Quick start: website with `test_data/`

This workflow uses the two paired synthetic CSVs in the repository. It runs the API and frontend natively so the backend can find both the local signal catalog and project files.

### 1. Install backend libraries

From the project root:

```bash
cd website
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS / WSL
source .venv/bin/activate
```

Then install the Python modules used by the website, including FastAPI, Uvicorn, XGBoost, pandas, NumPy, scikit-learn, authentication/security libraries, and test tools:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Install frontend packages

In a second terminal, from the project root:

```bash
cd website/frontend
npm ci
```

This checkout does not currently include a `package-lock.json`, so use `npm install` to resolve and install the frontend packages (React, Vite, TypeScript, charting, and UI libraries). If a lockfile is added later, prefer `npm ci` for a reproducible install.

### 3. Start the API

In the first terminal, with the website virtual environment activated and the current directory set to `website/`:

```bash
python -m backend.main
```

The API listens on `http://localhost:8000`. Check `http://localhost:8000/health` for health and `http://localhost:8000/docs` for the interactive API documentation. On first local startup, the backend creates a persistent signing key in `website/data/.jwt_secret`; keep that secret and the account database private. For a hosted deployment, configure a secure `SECRET_KEY` instead of relying on a development-generated secret.

### 4. Start the frontend

In the second terminal:

```bash
cd website/frontend
npm run dev -- --host 127.0.0.1
```

Open the Vite address shown in the terminal (normally `http://localhost:5173`). The Vite development server proxies `/api` requests to `http://localhost:8000`.

### 5. Generate paired synthetic histories and train

1. From `website/`, run `python -m scripts.generate_browser_demo ../test_data`.
2. Create an account in the local app and sign in.
3. Select `test_data/synthetic_browser_sales.csv` and `test_data/synthetic_browser_inventory.csv`.
4. Start training and review the forecast and replenishment table. The CSVs and model stay in the browser; the saved plan is encrypted on that device.

The generated files are synthetic and include 200 dated sales and inventory rows for each of three example drugs. The older `test_data/demand_inventory_snapshot.csv` contains one row per drug and is not a valid inventory history for browser training.

The browser trains a direct next-14-day XGBoost regressor per drug with lagged sales, dated inventory, and eligible public signals. The dashboard's 1- and 7-day quantities are allocated from the 14-day prediction using recent weekday patterns; they are not separately trained forecasts. The inventory/replenishment policy is illustrative only.

### Expected CSV contract

Sales requires these fields:

```text
date,drug_name,units_sold
```

Inventory requires:

```text
date,drug_name,on_hand_units
```

`date` or `as_of_date` is accepted as the inventory effective date, and `on_order_units` is optional. Each sales series must contain at least 130 consecutive daily rows per drug (including zero-sale days), without duplicate drug/date pairs. Each drug needs inventory snapshots covering at least 130 sales days, with each snapshot dated on the sales day or the preceding day. The latest inventory date must equal the final sales date for that drug.

### Stop the app

Stop each foreground process with `Ctrl+C`. User/account data and the generated local secret live in `website/data/`; do not commit real pharmacy data, credentials, database contents, or secrets.

## Optional Docker quick start

Docker Compose can build and start the website services:

```bash
cd website
docker compose up --build
```

The frontend is exposed at `http://localhost:3141`; the backend health endpoint is `http://localhost:8000/health`. Stop the stack with `Ctrl+C` and `docker compose down`.

The current API image includes the checksum-pinned historical catalog at
`website/catalog/historical_signal_catalog_2023_2025.csv.gz` and imports it on
startup. The optional legacy Python `DemandForecaster` still expects the
untracked `test/test_Signals/` benchmark artifact; it is disabled in the
production image. Use the native workflow above only when running that older
research path with `test_data/`.

## Running tests and builds

Run the website backend/model tests:

```bash
cd website
python -m pytest tests -q
```

Build the frontend:

```bash
cd website/frontend
npm run build
```

Run the research-model test suite from the repository root (install its dependencies first as described below):

```bash
python -m pytest model/tests -q
```

Passing software tests verifies tested behavior, not forecast superiority or real-world pharmacy impact.

## Running the separate `model/` research package

The `model/` package is distinct from the website's per-drug XGBoost forecast service. Its research pipelines build and evaluate public-data proxy metrics and broader signal surfaces; the website uses the account-specific synthetic sales/inventory workflow above.

Create a separate environment to keep the dependency sets independent:

```bash
cd model
python -m venv .venv
```

Activate `.venv` using the platform-specific commands above, then install the package and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the optional PyTorch deep-model path, install the `deep` extra as well. PyTorch installation can vary by operating system and accelerator; use the [official PyTorch install selector](https://pytorch.org/get-started/locally/) to choose the appropriate wheel, then install the project extra:

```bash
python -m pip install -e ".[dev,deep]"
```

From the repository root with that environment activated, inspect CLI commands and run the main workflows:

```bash
python -m arkansas_pharma_signal.cli --help
python -m arkansas_pharma_signal.cli --root . build-panel
python -m arkansas_pharma_signal.cli --root . train
python -m arkansas_pharma_signal.cli --root . evaluate --no-neural
python -m arkansas_pharma_signal.cli --root . forecast --max-rows 10000
python -m arkansas_pharma_signal.cli --root . audit-publishability
```

Use the `model/` README and `model/docs/` before interpreting specific forecasts, datasets, target definitions, or publication gates. Some source files are large or LFS-backed, and a command can only build outputs for the local inputs it actually has.

## Data, privacy, and interpretation

- `test_data/` contains synthetic data only. It is safe for the supplied demo, but it is not a record of a real Arkansas pharmacy or patient.
- `data/synthetic_pharmacy_data/` documents the synthetic sales-generation process and source hashes.
- Public research data are documented by `data/README.md`, source manifests, and `model/docs/DATA_SOURCES.md`. Public utilization and supply sources are proxies at their native geographic/time grain.
- Do not commit `.env`, `website/data/.jwt_secret`, `website/data/accounts.sqlite3`, user uploads, or private pharmacy/patient records.
- Model correlations and related news are contextual associations, not causal attributions. Forecasts and order quantities need qualified human review.
- PULSE is a development/research demo, not a validated clinical or procurement decision-support system.

## Documentation

- [Data guide](data/README.md) — data layout, source families, and synthetic generation.
- [Research model guide](model/README.md) — research-model architecture, workflow, outputs, and limitations.
- [Website guide](website/README.md) — website stack, account workflow, and XGBoost implementation.
- [Website specification](website/docs.md) — data contracts, feature/model behavior, testing, and deployment notes.
- [Synthetic data provenance](data/synthetic_pharmacy_data/PROVENANCE.md) and [synthetic data guide](data/synthetic_pharmacy_data/synthetic_guide.md).

## License

Private — All rights reserved.
