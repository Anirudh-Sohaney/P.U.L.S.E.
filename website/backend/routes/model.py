"""Model metadata endpoints for the Pharmacy Risk Prediction Platform."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from .auth import get_current_user
from ..config import settings

router = APIRouter()


@router.get("/info")
async def get_model_info(current_user: dict = Depends(get_current_user)):
    """Get model metadata and information."""
    model_dir = Path(settings.MODEL_PATH)
    metadata_path = model_dir / "model_metadata.json"
    metrics_path = model_dir / "training_metrics.json"

    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="No trained model found")

    try:
        result = {}
        with open(metadata_path) as f:
            result["metadata"] = json.load(f)

        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading model info: {str(e)}"
        )


@router.get("/features")
async def get_feature_importance(current_user: dict = Depends(get_current_user)):
    """Get feature importance from the trained model."""
    model_dir = Path(settings.MODEL_PATH)
    importance_path = model_dir / "feature_importance.csv"

    if not importance_path.exists():
        raise HTTPException(status_code=404, detail="No feature importance data found")

    try:
        df = pd.read_csv(importance_path)
        return {
            "features": df.to_dict(orient="records"),
            "feature_descriptions": {
                "inventory_ratio": "Current inventory level relative to maximum capacity",
                "inventory_deficit": "Inventory level compared to reorder point",
                "inventory_coverage_days": "Number of days inventory will last at current demand",
                "demand_volatility": "Variation in demand over time",
                "demand_per_capacity": "Weekly demand as percentage of storage capacity",
                "supply_risk_score": "Combined supplier reliability and lead time risk",
                "days_of_supply_remaining": "Estimated days until stockout",
                "shortage_frequency": "How often shortages occur (monthly rate)",
                "recency_weighted_shortage": "Historical shortages weighted by how recent they were",
                "demand_supply_mismatch": "Ratio of demand to available supply",
                "reliability_inventory_interaction": "Supplier reliability weighted by inventory level",
                "below_reorder": "Whether inventory is currently below reorder point",
                "low_supplier_reliability": "Whether supplier reliability is below threshold",
                "high_shortage_history": "Whether medication has high historical shortage count",
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading feature importance: {str(e)}"
        )


@router.post("/train")
async def train_model(current_user: dict = Depends(get_current_user)):
    """Retrain the model on the latest uploaded data."""
    data_dir = Path(settings.DATA_PATH)
    cleaned_path = data_dir / "cleaned_data.csv"

    if not cleaned_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No data available. Upload data first via /api/data/ingest.",
        )

    try:
        from datetime import datetime

        import numpy as np
        from ml.features import FeatureEngineer
        from ml.model import XGBoostTrainer

        df = pd.read_csv(cleaned_path)

        # Drop non-feature columns
        feature_cols = [
            c
            for c in df.columns
            if c not in ("medication_id", "medication_name", "risk_label")
        ]
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
        joblib.dump(fe, model_dir / "feature_engineer.joblib")

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

        return {
            "message": "Model retrained successfully",
            "metrics": metrics_dict,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Training error: {str(e)}"
        )
