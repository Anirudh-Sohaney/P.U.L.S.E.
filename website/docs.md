# Pulse — Project Specifications

## Table of Contents

1. [Project Overview](#project-overview)
2. [Development Principles](#development-principles)
3. [Technology Stack](#technology-stack)
4. [Data](#data)
5. [Feature Engineering](#feature-engineering)
6. [Model](#model)
7. [Prediction](#prediction)
8. [Backend](#backend)
9. [Frontend](#frontend)
10. [Security](#security)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [Research](#research)
14. [AI Agent Rules](#ai-agent-rules)
15. [Master Plan](#master-plan)

---

## Project Overview

Build a professional web-based pharmacy risk prediction platform.

The platform accepts pharmacy-related structured data, validates and preprocesses it, generates model features, runs an XGBoost model, and presents a risk prediction through a professional pharmacy analytics dashboard.

## Development Principles

- Build incrementally.
- Keep components modular.
- Keep documentation close to the process it describes.
- Do not optimize for appearance before the data/model pipeline is credible.
- Do not invent requirements or results.
- Prefer reproducibility and testability.

**Priority:** Correctness > Data integrity > Model validity > Security > Usability > Visual polish > Extra features

## Technology Stack

- **ML Model:** XGBoost for primary numerical prediction
- **Backend:** Python API separating prediction logic from frontend
- **Frontend:** Modern web UI (React/TypeScript)
- **Version Control:** Git

Do not add frameworks or dependencies solely because they are popular. Choose simple, maintainable technologies that fit the project.

---

## Data

### Data Sources

Document every dataset or external source used by the project. For each source record: name, URL/reference, owner/provider, access method, license/terms, date accessed, variables used, known limitations, and whether data is public, synthetic, or user-provided. Never represent synthetic data as real pharmacy data.

### Data Schema

Define every accepted field with: field name, data type, description, unit, required/optional, valid range, missing-value policy.

Potential fields: medication_id, medication_name, inventory_quantity, prescription_volume, historical_demand, supplier_id, supplier_lead_time, historical_shortage.

### Data Ingestion

Pipeline: Receive data → Identify format/source → Parse data → Check schema → Validate → Pass valid data to preprocessing → Record ingestion status/errors. Support structured inputs such as CSV where appropriate.

### Data Validation

Validate: required columns, data types, missing values, impossible/invalid values, duplicate records, dates, category consistency, cross-field consistency. Invalid required data must not be silently sent to the model. Return actionable validation messages.

### Data Cleaning

Define reproducible cleaning procedures: type conversion, duplicate handling, missing-value handling, outlier handling when justified, date normalization, category normalization. Do not alter important values without a documented rule. Cleaning performed during training must be reproducible during inference.

### Data Storage

Define how raw, processed, and prediction data are stored. Separate: raw input data, cleaned data, feature data, model artifacts, prediction records. Minimize storage of sensitive information. Never commit secrets or private pharmacy/patient data.

---

## Feature Engineering

### Overview

Convert validated pharmacy data into model-ready features. Every feature should have: definition, source field(s), calculation, unit, availability at prediction time, reason for inclusion. Avoid features that leak future information.

### Feature Pipeline

Validated data → feature calculations → feature validation → final model matrix → XGBoost. The same feature-generation logic must be used during training and prediction. Save the feature definition/version with each model version.

### Feature Types

- **Inventory Features:** Current stock levels, stock turnover rates
- **Demand Features:** Prescription volumes, historical demand patterns
- **Supplier Features:** Lead times, reliability metrics
- **Historical Features:** Time-series trends, seasonal patterns

---

## Model

### Overview

XGBoost is the primary prediction model. The exact target must be explicitly defined before final training. The model should support: training, validation, testing, saving/loading, prediction, versioning.

### Training

1. Load approved dataset.
2. Validate schema.
3. Split data appropriately.
4. Fit preprocessing only on training data where applicable.
5. Train XGBoost.
6. Evaluate on validation data.
7. Select configuration using predefined criteria.
8. Evaluate final model on held-out test data.
9. Save model and metadata.

For time-dependent prediction tasks, use chronological splits when appropriate.

### Hyperparameters

Document XGBoost hyperparameters used for each experiment: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda. Do not assume default or example values are optimal. Record actual experimental values.

### Evaluation

Do not evaluate using accuracy alone. Depending on the task, evaluate: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, calibration. Keep training, validation, and final test data separate. Never manually enter performance numbers.

### Calibration

If the platform presents probabilities as risk estimates, evaluate whether predicted probabilities are calibrated. Document: calibration method, calibration dataset, calibration metrics, before/after performance. Do not call a probability clinically meaningful merely because it is numerically between 0 and 1.

### Model Inputs

Document the final features used by XGBoost. For every feature record: name, type, unit, definition, preprocessing, source, prediction-time availability. The model must receive the same feature structure used during training.

---

## Prediction

### Pipeline

Input → validation → preprocessing → feature engineering → XGBoost → output validation → risk classification → response. Training and prediction must use compatible preprocessing and feature definitions.

### Risk Classification

Risk labels should be configurable. Example: LOW, MEDIUM, HIGH. Thresholds must be selected based on the actual prediction task and validation results rather than assumed to be medically valid. Keep thresholds centralized in configuration.

---

## Backend

### API Architecture

Use clear, versioned API contracts. Separate concerns for: authentication, data ingestion, validation, predictions, model metadata, health/status. Keep prediction logic out of route handlers where practical.

### Authentication

Define user authentication and pharmacy-level authorization. Users should only access data they are authorized to access. Do not store plaintext passwords. Use established authentication/security practices rather than custom cryptography.

### Data Endpoints

Define endpoints for: uploading/submitting pharmacy data, validating data, retrieving validation results, retrieving processed records when appropriate. Document request/response schemas and errors before implementation.

### Prediction Endpoint

1. Authenticate/authorize the request.
2. Validate input.
3. Run the approved preprocessing/feature pipeline.
4. Load the correct XGBoost model version.
5. Generate the prediction.
6. Return structured results.

### Error Handling

Errors should be: specific, safe, actionable, free of secrets. Separate: validation errors, authentication errors, authorization errors, model errors, server errors. Never expose stack traces or credentials to end users in production.

---

## Frontend

### Design System

**Theme:** Professional pharmacy and healthcare analytics.

**Visual Direction:** Clean, modern, trustworthy, restrained, data-focused.

**Palette:** Professional blue, healthcare teal, white, neutral gray, dark navy/gray text. Risk status: green = low, amber = medium, red = high.

Avoid neon, gaming, crypto, sci-fi, excessive gradients, and excessive animation. Use consistent typography, spacing, buttons, cards, tables, charts, forms, and risk indicators.

### Dashboard

Prioritize: 1. Overall risk, 2. Highest-risk medications, 3. Important trends, 4. Relevant risk factors, 5. Recent predictions. Use clear cards and charts. Mock data must be visibly marked as demo data until connected to real model outputs.

---

## Security

Protect: pharmacy data, user accounts, model artifacts, API credentials, environment secrets. Use least-privilege access and secure defaults.

Never hardcode API keys, passwords, tokens, database credentials, or private certificates. Use environment variables and keep `.env` files out of Git.

---

## Testing

Test each stage independently and the full pipeline end-to-end. Required areas: data, features, model, API, frontend, security-sensitive behavior.

### Testing Stages

- [x] Data tests
- [x] Model tests
- [x] API tests
- [ ] Frontend tests
- [ ] End-to-end tests

---

## Deployment

Deployment should occur only after: core tests pass, secrets are configured securely, environment configuration is documented, model artifacts are versioned, production data handling is reviewed.

---

## Research

Document the research question, prediction target, datasets, methodology, experiments, and limitations. Separate research findings from product demonstrations.

### Research Progress

- [x] Complete literature review
- [x] Run baseline experiments
- [x] Run XGBoost experiments
- [x] Record verified results
- [x] Document limitations

---

## AI Agent Rules

### Before Coding

1. Read this document (`docs.md`).
2. Identify the current development stage.
3. Read only the sections relevant to the requested task.
4. Inspect existing code before creating new code.
5. Reuse existing components and utilities when possible.

### ML Rules

- XGBoost is the primary model.
- Never invent model performance, pharmacy data, or claims of validation.
- Prevent train/test leakage.
- Keep preprocessing reproducible between training and inference.
- Version trained models and their configurations.
- Feature importance is not automatically causal evidence.

### Healthcare Communication

Present the product as a risk-analysis/decision-support prototype. Do not claim diagnosis, guaranteed outcomes, replacement of pharmacists, or clinical validation without evidence.

### UI Rules

Maintain a professional pharmacy/healthcare analytics theme: clean, trustworthy, modern, data-focused, restrained. Avoid gaming, crypto, neon, sci-fi, and excessive animation aesthetics.

### Code Quality

- Prefer small, modular files.
- Avoid unnecessary dependencies.
- Reuse existing utilities before creating duplicates.
- Keep naming consistent.
- Add tests for important behavior.
- Do not rewrite unrelated code.

### Documentation

When implementation changes documented behavior, update this document (`docs.md`).

### Git Discipline

Keep commits focused. Good examples: `setup project structure`, `add data validation pipeline`, `implement xgboost training`, `add prediction endpoint`, `build risk dashboard`.

### Definition of Done

A task is complete when: the intended behavior is implemented; relevant tests pass; no obvious secrets or sensitive data were introduced; documentation is consistent; TODO status is updated when applicable.

---

## Master Plan

### Development Order

1. Project Definition
2. Data
3. Feature Engineering
4. Model
5. Prediction
6. Backend
7. Frontend
8. Security
9. Testing
10. Deployment
11. Research

### Core Architecture

Pharmacy Data → Ingestion → Validation → Cleaning → Feature Engineering → XGBoost → Risk Probability/Score → Risk Classification → Backend API → Frontend Dashboard

### No Fabrication

Never fabricate pharmacy data, model predictions, evaluation metrics, feature importance, calibration results, or research results. Clearly label synthetic/demo values.

### Context Efficiency

Do not read every section for every task. Read the relevant sections only.

---

## TODO

### Stage 00 — Project
- [x] Finalize prediction target
- [x] Finalize technology stack
- [x] Define development milestones

### Stage 01 — Data
- [x] Identify approved data sources
- [x] Define final schema
- [x] Implement ingestion
- [x] Implement validation
- [x] Implement cleaning
- [x] Define storage strategy

### Stage 02 — Features
- [x] Define final feature set
- [x] Implement inventory features
- [x] Implement demand features
- [x] Implement supplier features
- [x] Implement historical features
- [x] Build reproducible feature pipeline

### Stage 03 — Model
- [x] Build XGBoost training pipeline
- [x] Establish baseline models
- [x] Tune/evaluate XGBoost
- [x] Evaluate calibration
- [x] Add explainability
- [x] Version model artifacts

### Stage 04 — Prediction
- [x] Implement prediction pipeline
- [x] Define risk score
- [x] Define validated classification thresholds
- [x] Implement prediction explanations

### Stage 05 — Backend
- [x] Build API
- [x] Build data endpoints
- [x] Build prediction endpoint
- [x] Implement authentication
- [x] Implement error handling

### Stage 06 — Frontend
- [x] Build design system
- [x] Build landing page
- [x] Build dashboard
- [x] Build medications page
- [x] Build medication details
- [x] Build data upload
- [x] Build model page

### Stage 07 — Security
- [x] Review privacy
- [x] Review authentication
- [x] Configure secrets management

### Stage 08 — Testing
- [x] Data tests
- [x] Model tests
- [x] API tests
- [ ] Frontend tests
- [ ] End-to-end tests

### Stage 09 — Deployment
- [x] Configure environments
- [ ] Deploy development version
- [x] Configure production
- [ ] Add monitoring

### Stage 10 — Research
- [x] Complete literature review
- [x] Run baseline experiments
- [x] Run XGBoost experiments
- [x] Record verified results
- [x] Document limitations
