"""Data endpoints for the Pharmacy Risk Prediction Platform."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from .auth import get_current_user
from ..config import settings
from ml.data import DataIngester, ValidationResult

router = APIRouter()
ingester = DataIngester()


def _get_data_dir() -> Path:
    data_dir = Path(settings.DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    rows_validated: int


class SchemaInfo(BaseModel):
    fields: dict[str, Any]


class UploadResponse(BaseModel):
    message: str
    rows: int
    columns: int
    filename: str


@router.post("/validate", response_model=ValidationResponse)
async def validate_data(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Validate uploaded pharmacy data."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result = ingester.validator.validate(df)
        return ValidationResponse(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            rows_validated=result.rows_validated,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/ingest", response_model=UploadResponse)
async def ingest_data(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Ingest, clean, and persist pharmacy data. Trains model automatically."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result = ingester.ingest_dataframe(df)

        if not result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Ingestion failed",
                    "errors": result.validation_result.errors
                    if result.validation_result
                    else [],
                },
            )

        # Persist cleaned data
        data_dir = _get_data_dir()
        cleaned_path = data_dir / "cleaned_data.csv"
        result.data.to_csv(cleaned_path, index=False)

        # Save metadata
        meta = {
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "rows": len(result.data),
            "columns": list(result.data.columns),
        }
        meta_path = data_dir / "data_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Train model on the new data
        training_result = _train_model(result.data)

        return UploadResponse(
            message="Data ingested and model trained successfully",
            rows=len(result.data),
            columns=len(result.data.columns),
            filename=file.filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


def _train_model(df: pd.DataFrame) -> dict:
    """Train the XGBoost model on the given data."""
    from ml.features import FeatureEngineer
    from ml.model import XGBoostTrainer

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ("medication_id", "medication_name", "risk_label")]
    X = df[feature_cols].copy()

    # Feature engineer
    fe = FeatureEngineer()
    X_features = fe.fit_transform(X)

    # Target
    y = df["risk_label"].astype(int)

    # Train
    trainer = XGBoostTrainer()
    trainer.fit(X_features, y)

    # Evaluate
    metrics = trainer.evaluate(X_features, y)

    # Save model
    model_dir = Path(settings.MODEL_PATH)
    trainer.save(model_dir)

    # Save feature engineer metadata
    fe_meta = {
        "feature_names": fe.get_feature_names(),
        "fitted": True,
    }
    fe_path = model_dir / "feature_engineer_meta.json"
    with open(fe_path, "w") as f:
        json.dump(fe_meta, f, indent=2)

    # Save metrics
    metrics_dict = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "roc_auc": metrics.roc_auc,
        "true_positives": metrics.true_positives,
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
    }
    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Generate and store predictions for all medications
    _generate_all_predictions(df, fe, trainer)

    return metrics_dict


def _generate_all_predictions(df: pd.DataFrame, fe, trainer) -> None:
    """Generate predictions for all medications and store them."""
    feature_cols = [c for c in df.columns if c not in ("medication_id", "medication_name", "risk_label")]
    X = df[feature_cols].copy()
    X_features = fe.transform(X)

    probabilities = trainer.predict_proba(X_features)

    # Build predictions dataframe
    predictions_df = df[["medication_id", "medication_name", "medication_category"]].copy()
    predictions_df["risk_score"] = probabilities
    predictions_df["risk_label"] = [
        "high" if p >= settings.RISK_THRESHOLD else "low" for p in probabilities
    ]
    predictions_df["confidence"] = [max(p, 1 - p) for p in probabilities]
    predictions_df["predicted_at"] = datetime.now().isoformat()

    # Sort by risk score descending
    predictions_df = predictions_df.sort_values("risk_score", ascending=False)

    # Save
    data_dir = _get_data_dir()
    pred_path = data_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    # Save stats
    high_count = (predictions_df["risk_label"] == "high").sum()
    low_count = (predictions_df["risk_label"] == "low").sum()
    stats = {
        "total": len(predictions_df),
        "high_risk": int(high_count),
        "low_risk": int(low_count),
        "avg_risk_score": float(predictions_df["risk_score"].mean()),
        "generated_at": datetime.now().isoformat(),
    }
    stats_path = data_dir / "prediction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


@router.get("/schema", response_model=SchemaInfo)
async def get_schema(current_user: dict = Depends(get_current_user)):
    """Get the data schema information."""
    return SchemaInfo(fields=ingester.get_schema_info())


@router.get("/status")
async def get_data_status(current_user: dict = Depends(get_current_user)):
    """Check if data has been uploaded and model is trained."""
    data_dir = _get_data_dir()
    model_dir = Path(settings.MODEL_PATH)

    has_data = (data_dir / "cleaned_data.csv").exists()
    has_model = (model_dir / "xgboost_model.joblib").exists()
    has_predictions = (data_dir / "predictions.csv").exists()

    result = {
        "has_data": has_data,
        "has_model": has_model,
        "has_predictions": has_predictions,
    }

    if has_data:
        meta_path = data_dir / "data_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["data_metadata"] = json.load(f)

    if has_model:
        metrics_path = model_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result["training_metrics"] = json.load(f)

    return result
