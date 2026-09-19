"""Locked synthetic demand-forecast demo endpoints."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ml.demand_forecast import DemandForecaster

from .auth import get_current_user
from ..config import settings


router = APIRouter()


def _data_dir() -> Path:
    path = Path(settings.DATA_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _forecast_paths() -> tuple[Path, Path, Path, Path]:
    data_dir = _data_dir()
    return (
        data_dir / "demand_history.csv",
        data_dir / "demand_forecaster.joblib",
        data_dir / "demand_forecasts.csv",
        data_dir / "demand_forecast_metrics.json",
    )


def _synthetic_sales_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "synthetic_pharmacy_data" / "arkansas_clinic_daily_pharmacy_sales.csv"


def ensure_locked_synthetic_demo() -> dict:
    """Create the fixed demo model only when its local artifact is absent."""
    history_path, model_path, forecast_path, metrics_path = _forecast_paths()
    if all(path.exists() for path in (history_path, model_path, forecast_path, metrics_path)):
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("mode") == "locked_synthetic_demo":
            return existing

    source = _synthetic_sales_path()
    if not source.exists():
        raise RuntimeError(f"Locked synthetic source is unavailable: {source}")
    history = pd.read_csv(source, usecols=["date", "drug_name", "units_sold"])
    forecaster = DemandForecaster(use_model_signals=True)
    metrics = forecaster.fit(history)
    forecasts = forecaster.forecast(14)
    history.to_csv(history_path, index=False)
    forecasts.to_csv(forecast_path, index=False)
    joblib.dump(forecaster, model_path)
    payload = {
        **metrics,
        "horizon_days": 14,
        "trained_at": datetime.now().isoformat(),
        "forecast_rows": len(forecasts),
        "use_model_signals": True,
        "mode": "locked_synthetic_demo",
        "dataset": "Arkansas synthetic daily pharmacy sales",
        "source_rows": len(history),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@router.post("/validate")
@router.post("/ingest")
async def demand_uploads_disabled(current_user: dict = Depends(get_current_user)):
    """Keep the demonstration dataset and its trained model immutable."""
    raise HTTPException(
        status_code=403,
        detail="This demonstration is locked to the synthetic dataset; data upload and retraining are disabled.",
    )


@router.get("/status")
async def demand_status(current_user: dict = Depends(get_current_user)):
    metrics = ensure_locked_synthetic_demo()
    history_path, model_path, forecast_path, metrics_path = _forecast_paths()
    result = {
        "mode": "locked_synthetic_demo",
        "dataset": "Arkansas synthetic daily pharmacy sales",
        "uploads_enabled": False,
        "has_history": history_path.exists(),
        "has_model": model_path.exists(),
        "has_forecasts": forecast_path.exists(),
    }
    result["metrics"] = metrics
    return result


@router.get("/forecasts")
async def get_demand_forecasts(
    drug_name: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    ensure_locked_synthetic_demo()
    _, _, forecast_path, metrics_path = _forecast_paths()
    if not forecast_path.exists():
        raise HTTPException(status_code=404, detail="The locked synthetic forecast is unavailable.")
    forecasts = pd.read_csv(forecast_path)
    if drug_name:
        forecasts = forecasts[forecasts["drug_name"].str.casefold() == drug_name.casefold()]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {"forecasts": forecasts.to_dict(orient="records"), "count": len(forecasts), "metrics": metrics}
