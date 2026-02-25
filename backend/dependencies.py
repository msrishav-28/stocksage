"""Shared dependency injection for the StockSage backend."""

from fastapi import Depends
from backend.config import Settings, get_settings
from backend.db.session import get_session
from backend.cache.redis_client import get_redis


async def get_db():
    """Yields an async DB session."""
    async for session in get_session():
        yield session


def get_config() -> Settings:
    """Returns the cached application settings."""
    return get_settings()
