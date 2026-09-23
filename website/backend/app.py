"""FastAPI application for the Pharmacy Risk Prediction Platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import auth, demand, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    yield
    # Shutdown


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Pharmacy Demand Planner",
        description="Account-scoped demand forecasts and medication inventory planning",
        version="0.2.0",
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
    app.include_router(demand.router, prefix="/api/demand", tags=["Demand Forecast"])

    return app
