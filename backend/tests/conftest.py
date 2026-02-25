"""Shared test fixtures for StockSage backend tests."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Use asyncio event loop for all async tests
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Returns a mock Settings object."""
    from backend.config import Settings
    return Settings(
        APP_ENV="test",
        DEBUG=True,
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test",
        REDIS_URL="redis://localhost:6379/1",
        NEWS_API_KEY="test_key",
        FRED_API_KEY="test_key",
        TFT_CHECKPOINT_PATH="models/test.ckpt",
        DEVICE="cpu",
    )


@pytest.fixture
def sample_ohlcv_df():
    """Returns a sample OHLCV DataFrame for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
        "high": prices * (1 + abs(np.random.normal(0.01, 0.005, len(dates)))),
        "low": prices * (1 - abs(np.random.normal(0.01, 0.005, len(dates)))),
        "close": prices,
        "volume": np.random.randint(1_000_000, 50_000_000, len(dates)),
    }, index=dates)

    df["ticker"] = "TEST"
    return df


@pytest.fixture
def sample_headlines():
    """Returns sample financial headlines for testing."""
    return [
        "Apple reports record quarterly revenue of $123.9 billion",
        "Tesla stock drops 5% after disappointing delivery numbers",
        "Federal Reserve holds interest rates steady amid inflation concerns",
        "Microsoft announces new AI partnership, stock rises 3%",
        "Oil prices surge as OPEC cuts production targets",
    ]


@pytest.fixture
def mock_redis():
    """Returns a mock Redis client."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock()
    mock.delete = AsyncMock()
    mock.ping = AsyncMock()
    return mock
