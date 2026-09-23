"""Configuration settings for the backend."""

from pathlib import Path
import secrets

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3141", "http://localhost:5173"]

    # Model
    MODEL_PATH: str = "./ml/models"
    MODEL_VERSION: str = "v1"

    # Data storage
    DATA_PATH: str = str(Path(__file__).resolve().parents[1] / "data")

    # Prediction
    RISK_THRESHOLD: float = 0.5

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
if not settings.SECRET_KEY:
    secret_file = Path(settings.DATA_PATH) / ".jwt_secret"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if not secret_file.exists():
        secret_file.write_text(secrets.token_urlsafe(48), encoding="utf-8")
        secret_file.chmod(0o600)
    settings.SECRET_KEY = secret_file.read_text(encoding="utf-8").strip()
