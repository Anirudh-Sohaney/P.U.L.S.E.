# Pulse — Pharmacy Risk Prediction Platform

A professional web-based platform for predicting pharmacy/pharmaceutical risks from structured data using machine learning, with **XGBoost as the primary numerical prediction model**.

> **Development status:** Initial project scaffold. The model, data schema, and risk target must be defined and validated during development.

## Project Goals

- Build a reproducible pharmacy risk prediction pipeline.
- Use XGBoost as the primary tabular ML model.
- Validate and clean input data before inference.
- Prevent data leakage between training and evaluation.
- Provide risk scores/classes with transparent explanations.
- Build a professional pharmacy/healthcare-themed web interface.
- Keep frontend, backend, and ML components modular.
- Never fabricate real-world pharmacy data or model performance.

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

## Locked demand-forecast demo

The **Demo Data** page is linked to the fixed synthetic Arkansas daily
pharmacy-sales dataset in this repository. It trains (or loads) one
chronological XGBoost demand regressor per medication and provides a locked
14-day forecast series on the **Forecasts** page. Upload and retraining routes
are deliberately disabled for the demonstration.

The locked dataset has zero-unit days explicitly and covers 30 medications.
The demand route is separate from the legacy inventory risk-classification
workflow, so existing risk uploads and screens remain available.

When `model/artifacts/news/news_only_catalog_features.csv.gz` is present, the
forecaster joins its dated 20-signal production artifact only after the source
month closes. The fixed demo uses that integration to generate its locked
forecast artifact.

## Safety / Data Rules

Do not commit patient-identifying information, private pharmacy datasets, API keys, passwords, `.env` files, raw production datasets, or unreviewed model artifacts containing sensitive data. Use synthetic, public, or appropriately de-identified data during development.

## Disclaimer

This project is a research/development system. Unless separately validated and authorized, predictions should not be represented as medical diagnoses, clinical decisions, guaranteed outcomes, or a replacement for pharmacists or other qualified professionals.
