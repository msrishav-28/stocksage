"""Application configuration — Pydantic v2 settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # CORS — comma-separated list of allowed frontend origins
    CORS_ORIGINS: str = "http://localhost:3000"

    # Rate limiting — applied per client IP across all routes
    RATE_LIMIT: str = "120/minute"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a clean list of origin strings."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/stocksage"
    # When true, ORM tables are created on startup instead of requiring Alembic.
    # Convenient for local dev; production should run `alembic upgrade head`.
    DB_AUTO_CREATE: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_PRICE: int = 300        # 5 min for live prices
    CACHE_TTL_SENTIMENT: int = 1800   # 30 min for sentiment
    CACHE_TTL_MACRO: int = 86400      # 24h for macro data

    # External APIs
    NEWS_API_KEY: str = ""
    FRED_API_KEY: str = ""
    ALPHAVANTAGE_KEY: str = ""

    # ML
    TFT_CHECKPOINT_PATH: str = "models/tft_checkpoint.ckpt"
    FINBERT_MODEL: str = "ProsusAI/finbert"
    DEVICE: str = "cpu"               # "cuda" on Modal

    # Modal
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
