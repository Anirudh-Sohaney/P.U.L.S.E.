# Pulse

Arkansas-first pharmaceutical forecasting and pharmacy risk prediction
platform, expanding toward reusable national and global external signals.

## Repository Structure

```text
├── model/                  # Arkansas pharmaceutical forecasting (NLP/news signals, demand, shortage)
│   ├── arkansas_pharma_signal/   # Core model package
│   ├── scripts/                 # Evaluation and build scripts
│   ├── tests/                   # Model tests
│   └── docs/                    # Model documentation
├── data/                   # Supply/demand data (CMS Part D, FDA recalls, etc.)
├── website/                # Pharmacy risk prediction web platform
│   ├── backend/            # Python API
│   ├── frontend/           # React/TypeScript UI
│   ├── ml/                 # XGBoost prediction pipeline
│   ├── data/               # Local dev data
│   ├── tests/              # Website tests
│   └── docs.md             # Full specifications
├── summ.md                 # Project status summary
├── pytest.ini              # Root test configuration
└── README.md
```

## Components

### `model/` — Arkansas-first Pharmaceutical Forecasting
Multi-layer forecasting stack combining real-world inputs (news, disease
surveillance, weather, supply-chain signals, drug identity data, distribution,
trade, and economics) to forecast pharmaceutical demand, supply disruption
risk, and shortage impact. Arkansas is the primary high-resolution test bed;
broader signals are retained as contextual inputs. See `summ.md` and
`model/docs/` for current evidence and limitations.

### `website/` — Demand and replenishment demonstration
Locked React/Python demonstration that trains chronological per-drug XGBoost forecasts from the reproducible synthetic clinic-sales history, joins only time-safe news signals, and pairs the result with a transparent synthetic on-hand inventory scenario. It displays demand at 1, 4, 7, and 14 days, forecast stockout timing, and demonstration replenishment quantities. See `website/README.md`, `website/docs.md`, and `data/synthetic_pharmacy_data/synthetic_guide.md`.

### `data/` — Supply/Demand Data
CMS Part D quarterly data, FDA recalls, and other pharmaceutical datasets.

The current repository does not claim direct Arkansas pharmacy inventory truth. The website's on-hand inventory is explicitly synthetic and only demonstrates the connection between forecast demand and replenishment arithmetic. Public targets are proxies unless a source explicitly observes pharmacy fills, stockouts, backorders, or on-hand quantities.

### Synthetic pharmacy benchmark

`data/synthetic_pharmacy_data/` contains a reproducible single-clinic stress
test, not observed Arkansas pharmacy transactions. Its generator, event tags,
hashes, and untouched-test evaluation protocol are documented in
`synthetic_guide.md`, `PROVENANCE.md`, and `benchmark_protocol.md`. Results
from this artifact must not be generalized to multiple pharmacies without
multi-site or observed validation.

## Getting Started

### Model
```bash
cd model
pip install -e .
python -m arkansas_pharma_signal.cli --help
```

### Website
```bash
cd website
cp .env.example .env
docker-compose up
```

## Testing

```bash
# Model tests
cd model && pytest

# Website tests
cd website && pytest
```

## License

Private — All rights reserved.
