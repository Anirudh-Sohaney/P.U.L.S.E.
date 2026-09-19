# Implementation Contract

This document is the development contract for the model code in `model/`.

## Package Layout

Required files:

- `model/arkansas_pharma_signal/__init__.py`
- `model/arkansas_pharma_signal/config.py`
- `model/arkansas_pharma_signal/io.py`
- `model/arkansas_pharma_signal/features.py`
- `model/arkansas_pharma_signal/entities.py`
- `model/arkansas_pharma_signal/text_signals.py`
- `model/arkansas_pharma_signal/regression.py`
- `model/arkansas_pharma_signal/neural.py`
- `model/arkansas_pharma_signal/forecast.py`
- `model/arkansas_pharma_signal/evaluate.py`
- `model/arkansas_pharma_signal/cli.py`
- `model/tests/`
- `model/pyproject.toml`

## Dependency Constraints

The local `data/.venv` currently has pandas and numpy. It does not have scikit-learn, PyTorch, transformers, sentence-transformers, LightGBM, or XGBoost.

Required default implementation:

- Python 3.11+ compatible,
- pandas,
- numpy,
- no required heavyweight ML dependency.

Optional extras may add:

- scikit-learn for richer regression,
- PyTorch for neural networks,
- sentence-transformers or transformers for text embeddings.

If optional dependencies are absent, code must still run using deterministic numpy/pandas baselines and hashed text features.

## Data Inputs

Default config paths:

- feature store: `data/final_data/features/external_state_features.csv.gz`
- events: `data/final_data/events/events.csv.gz`
- entities: `data/final_data/entities/entities.csv.gz`
- relationships: `data/final_data/entities/relationships.csv.gz`
- CMS Part D Arkansas provider/drug/year: `data/targeted_additions/cms_partd_prescriber_provider_drug/data/arkansas_partd_provider_drug_by_year.csv.gz`
- FDA NDC products: `data/targeted_additions/fda_ndc_directory/data/fda_ndc_products_current.csv.gz`
- current FDA shortages: `data/targeted_additions/fda_shortages_recalls_current/data/fda_shortages_current.csv.gz`
- current FDA enforcement/recalls: `data/targeted_additions/fda_shortages_recalls_current/data/fda_enforcement_2023_current.csv.gz`
- Arkansas disease surveillance: `data/targeted_additions/disease_surveillance_current/data/`
- Arkansas news signals: `data/targeted_additions/arkansas_news_3dlnews/data/`
- DEA ARCOS retail distribution: `data/targeted_additions/dea_arcos_arkansas/data/arcos_arkansas_retail_summary.csv.gz`

## Minimum Viable Model

The first shippable model must:

1. Load real local datasets.
2. Build an annual Arkansas drug-demand panel from CMS Part D provider-drug data.
3. Map generic names to FDA NDC active ingredients and labelers where possible.
4. Build disease/news/weather/supply aggregate features from canonical external-state data.
5. Train a regression-style demand model using lagged demand and external features.
6. Train or compute a shortage/supply-risk model from FDA shortages/enforcement and supplier/drug exposure.
7. Implement a small neural-style numpy multi-task model or optional torch MLP if torch is installed.
8. Write a forecast CSV with at least 100 rows and support up to 10,000+ rows.
9. Write evaluation metrics and model metadata.
10. Include tests for schema, feature generation, and output contract.

## Forecast CLI

Required commands:

```bash
data/.venv/bin/python -m arkansas_pharma_signal.cli build-panel --root .
data/.venv/bin/python -m arkansas_pharma_signal.cli train --root .
data/.venv/bin/python -m arkansas_pharma_signal.cli forecast --root . --max-rows 10000
data/.venv/bin/python -m arkansas_pharma_signal.cli evaluate --root .
```

The package should also work with:

```bash
PYTHONPATH=model data/.venv/bin/python -m arkansas_pharma_signal.cli --help
```

## Output Locations

- `model/artifacts/panel/`
- `model/artifacts/trained/`
- `model/artifacts/forecasts/`
- `model/artifacts/evaluation/`
- `model/artifacts/metadata/`

Generated artifact files may be ignored by git if large. Source code and docs should remain tracked.

## Quality Gates

Before final review:

1. Run package help.
2. Build panel from local data.
3. Train model.
4. Generate at least 100 forecast rows.
5. Run evaluation.
6. Run tests.
7. Inspect output schema and confirm no synthetic rows were generated.
