"""Main entry point for the Pharmacy Risk Prediction Platform backend."""

import uvicorn

from .app import create_app
from .config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
