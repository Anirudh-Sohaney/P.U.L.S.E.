"""Health check routes."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import signal_store

router = APIRouter()


@router.get("/health")
async def health_check():
    """API-process liveness; source-worker status is in /api/v1/signals/freshness."""
    return {"status": "healthy", "role": "api"}


@router.get("/ready")
async def readiness():
    coverage = signal_store.freshness()
    seed = signal_store.catalog_seed_status()
    ready = (not seed["missing_ids"]
             and coverage["definitions"] >= seed["expected"]
             and coverage["historical_catalog_records"] >= 4097)
    return JSONResponse(status_code=200 if ready else 503,
                        content={"ready": ready, "signal_definitions": coverage["definitions"],
                                 "seed_definitions_expected": seed["expected"],
                                 "seed_definitions_present": seed["present"],
                                 "missing_seed_definition_ids": seed["missing_ids"],
                                 "historical_catalog_records": coverage["historical_catalog_records"],
                                 "latest_observation_date": coverage["latest_observation_date"],
                                 "latest_refresh_run": coverage["latest_refresh_run"]})
