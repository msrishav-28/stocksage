"""Shared test fixtures for StockSage backend tests.

NOTE: event_loop fixture removed — pytest-asyncio >= 0.21 deprecates it and
     it conflicts with asyncio_mode = "auto" already set in pyproject.toml.
     Do NOT add it back.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock


@pytest.fixture
def mock_settings():
    """Returns a mock Settings object with test-safe values."""
    from backend.config import Settings
    return Settings(
        APP_ENV="test",
        DEBUG=False,
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test",
        REDIS_URL="redis://localhost:6379/1",
        NEWS_API_KEY="test_news_key",
        FRED_API_KEY="test_fred_key",
        TFT_CHECKPOINT_PATH="models/test.ckpt",
        DEVICE="cpu",
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    Returns a 252-row OHLCV DataFrame with a proper DatetimeIndex.
    Long enough for all rolling-window indicators (max window = EMA 200).
    """
    np.random.seed(42)
    n = 300  # Use 300 rows to ensure EMA-200 has coverage after dropna
    dates  = pd.bdate_range("2023-01-01", periods=n)
    base   = 150.0
    rets   = np.random.normal(0.001, 0.02, n)
    prices = base * (1 + rets).cumprod()

    df = pd.DataFrame(
        {
            "open":   prices * np.exp(np.random.normal(0, 0.003, n)),
            "high":   prices * np.exp(abs(np.random.normal(0.008, 0.004, n))),
            "low":    prices * np.exp(-abs(np.random.normal(0.008, 0.004, n))),
            "close":  prices,
            "volume": np.random.randint(1_000_000, 50_000_000, n).astype(float),
        },
        index=dates,
    )
    # Ensure high >= close >= low (ohlcv validity invariant)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    df["ticker"] = "TEST"
    return df


@pytest.fixture
def sample_headlines() -> list:
    """Returns sample financial headlines for FinBERT testing."""
    return [
        "Apple reports record quarterly revenue of $123.9 billion",
        "Tesla stock drops 5% after disappointing delivery numbers",
        "Federal Reserve holds interest rates steady amid inflation concerns",
        "Microsoft announces new AI partnership, stock rises 3%",
        "Oil prices surge as OPEC cuts production targets",
    ]


@pytest.fixture
def mock_redis():
    """Returns a mock Redis client that simulates cache miss on all keys."""
    mock = AsyncMock()
    mock.get      = AsyncMock(return_value=None)
    mock.setex    = AsyncMock()
    mock.delete   = AsyncMock()
    mock.ping     = AsyncMock()
    mock.aclose   = AsyncMock()
    return mock
