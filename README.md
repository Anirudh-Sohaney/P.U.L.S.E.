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

### `website/` — Pharmacy Risk Prediction Platform
XGBoost-based web platform for predicting pharmacy risks from structured data. React frontend, Python backend, full ML pipeline. See `website/README.md` and `website/docs.md`.

### `data/` — Supply/Demand Data
CMS Part D quarterly data, FDA recalls, and other pharmaceutical datasets.

The current repository does not claim direct pharmacy inventory truth. Public
targets are proxies unless a source explicitly observes pharmacy fills,
stockouts, backorders, or on-hand quantities.

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
