"""Prediction endpoints for the Pharmacy Risk Prediction Platform."""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .auth import get_current_user
from ..config import settings

router = APIRouter()


class PredictionRequest(BaseModel):
    medication_id: str
    medication_name: str
    medication_category: str
    current_inventory: float = Field(..., ge=0)
    reorder_point: float = Field(..., ge=0)
    max_capacity: float = Field(..., ge=0)
    prescription_volume_30d: float = Field(..., ge=0)
    avg_weekly_demand: float = Field(..., ge=0)
    demand_trend: float = 0.0
    supplier_id: str
    supplier_lead_time_days: float = Field(..., ge=0)
    supplier_reliability_score: float = Field(..., ge=0, le=1)
    historical_shortage_count: int = Field(..., ge=0)
    days_since_last_shortage: Optional[float] = None
    avg_stockout_duration_days: Optional[float] = None


class PredictionResponse(BaseModel):
    risk_score: float
    risk_label: str
    confidence: float
    explanation: dict[str, Any]
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


def _load_prediction_pipeline():
    """Load the legacy risk pipeline, or return a clear unavailable response."""
    model_dir = Path(settings.MODEL_PATH)
    model_path = model_dir / "xgboost_model.joblib"
    feature_path = model_dir / "feature_engineer.joblib"
    if not model_path.exists() or not feature_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Risk model is unavailable. Upload risk data and train the risk model first.",
        )
    try:
        from ml.model import XGBoostTrainer
        from ml.prediction import PredictionPipeline

        return PredictionPipeline(
            model=XGBoostTrainer.load(model_dir),
            feature_engineer=joblib.load(feature_path),
            threshold=settings.RISK_THRESHOLD,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Risk model could not be loaded: {exc}") from exc


def _response(result: Any) -> PredictionResponse:
    return PredictionResponse(
        risk_score=result.risk_score,
        risk_label=result.risk_label,
        confidence=result.confidence,
        explanation=result.explanation,
        model_version=settings.MODEL_VERSION,
    )


def _get_data_dir() -> Path:
    data_dir = Path(settings.DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@router.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest, current_user: dict = Depends(get_current_user)
):
    """Return one legacy inventory-risk prediction from the trained risk model."""
    return _response(_load_prediction_pipeline().predict_single(request.model_dump()))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_risk_batch(
    request: BatchPredictionRequest, current_user: dict = Depends(get_current_user)
):
    """Return legacy inventory-risk predictions for a batch of records."""
    results = _load_prediction_pipeline().predict(
        pd.DataFrame([record.model_dump() for record in request.records])
    )
    return BatchPredictionResponse(predictions=[_response(result) for result in results], count=len(results))


@router.get("/all")
async def get_all_predictions(current_user: dict = Depends(get_current_user)):
    """Get all stored predictions for the dashboard and medications page."""
    data_dir = _get_data_dir()
    pred_path = data_dir / "predictions.csv"

    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No predictions available. Upload data and train model first.",
        )

    try:
        df = pd.read_csv(pred_path)
        return {
            "predictions": df.to_dict(orient="records"),
            "count": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")


@router.get("/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get aggregated dashboard data from predictions."""
    data_dir = _get_data_dir()
    pred_path = data_dir / "predictions.csv"
    stats_path = data_dir / "prediction_stats.json"

    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No data available. Upload data and train model first.",
        )

    try:
        df = pd.read_csv(pred_path)

        # Load stats
        stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

        # Top 10 highest risk medications
        top_risk = df.head(10).to_dict(orient="records")

        # Category breakdown
        category_stats = (
            df.groupby("medication_category")
            .agg(
                count=("medication_id", "count"),
                avg_risk=("risk_score", "mean"),
                high_risk=("risk_label", lambda x: (x == "high").sum()),
            )
            .reset_index()
        )
        category_stats["avg_risk"] = category_stats["avg_risk"].round(3)

        return {
            "stats": {
                "total_medications": stats.get("total", len(df)),
                "high_risk": stats.get("high_risk", int((df["risk_label"] == "high").sum())),
                "low_risk": stats.get("low_risk", int((df["risk_label"] == "low").sum())),
                "avg_risk_score": round(float(df["risk_score"].mean()), 3),
            },
            "top_risk_medications": top_risk,
            "category_breakdown": category_stats.to_dict(orient="records"),
            "generated_at": stats.get("generated_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dashboard data: {str(e)}")


@router.get("/threshold")
async def get_threshold(current_user: dict = Depends(get_current_user)):
    """Get current risk classification threshold."""
    return {"threshold": settings.RISK_THRESHOLD}


@router.put("/threshold")
async def update_threshold(
    threshold: float,
    current_user: dict = Depends(get_current_user),
):
    """Update risk classification threshold."""
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    settings.RISK_THRESHOLD = threshold

    # Re-generate predictions with new threshold if data exists
    data_dir = _get_data_dir()
    pred_path = data_dir / "predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        df["risk_label"] = [
            "high" if p >= threshold else "low" for p in df["risk_score"]
        ]
        df.to_csv(pred_path, index=False)

        # Update stats
        high_count = (df["risk_label"] == "high").sum()
        low_count = (df["risk_label"] == "low").sum()
        from datetime import datetime
        stats = {
            "total": len(df),
            "high_risk": int(high_count),
            "low_risk": int(low_count),
            "avg_risk_score": float(df["risk_score"].mean()),
            "generated_at": datetime.now().isoformat(),
        }
        stats_path = data_dir / "prediction_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    return {"threshold": threshold, "message": "Threshold updated"}
