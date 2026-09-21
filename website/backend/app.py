"""FastAPI application for the Pharmacy Risk Prediction Platform."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import auth, data, demand, health, model, predictions, purchasing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    model_dir = Path(settings.MODEL_PATH)
    if model_dir.exists():
        app.state.model_loaded = True
    else:
        app.state.model_loaded = False
    yield
    # Shutdown


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Pharmacy Risk Prediction Platform",
        description="API for predicting pharmacy/pharmaceutical risks using XGBoost",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(data.router, prefix="/api/data", tags=["Data"])
    app.include_router(demand.router, prefix="/api/demand", tags=["Demand Forecast"])
    app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
    app.include_router(model.router, prefix="/api/model", tags=["Model"])
    app.include_router(purchasing.router, prefix="/api/purchasing", tags=["Purchasing Dashboard"])

    return app
