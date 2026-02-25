from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/stocksage"

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

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
