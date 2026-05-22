"""Redis async cache client — wraps redis-py 5.x with graceful degradation."""

import redis.asyncio as redis
import orjson
from typing import Any, Optional
from loguru import logger
from backend.config import get_settings

_redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Attempt to connect to Redis; disable caching if unavailable."""
    global _redis_client
    settings = get_settings()
    try:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False,  # orjson works with bytes
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        await _redis_client.ping()
        logger.success("Redis connected.")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        _redis_client = None


async def close_redis():
    """Close the Redis connection pool — uses aclose() for redis-py 5.x."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.aclose()
        except AttributeError:
            # Fallback for older redis-py versions
            await _redis_client.close()
        _redis_client = None


def get_redis() -> Optional[redis.Redis]:
    return _redis_client


async def get_cached(key: str) -> Optional[Any]:
    """Return cached value (deserialized from JSON) or None on miss/error."""
    client = get_redis()
    if client is None:
        return None
    try:
        data = await client.get(key)
        if data:
            logger.debug(f"Cache HIT: {key}")
            return orjson.loads(data)
        logger.debug(f"Cache MISS: {key}")
    except Exception as e:
        logger.warning(f"Cache read error for '{key}': {e}")
    return None


async def set_cache(key: str, value: Any, ttl: int = 300):
    """Serialize value to JSON and store in Redis with TTL."""
    client = get_redis()
    if client is None:
        return
    try:
        await client.setex(key, ttl, orjson.dumps(value, option=orjson.OPT_NON_STR_KEYS))
        logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
    except Exception as e:
        logger.warning(f"Cache write error for '{key}': {e}")


async def invalidate(key: str):
    """Delete a key from the cache."""
    client = get_redis()
    if client is None:
        return
    try:
        await client.delete(key)
        logger.info(f"Cache INVALIDATED: {key}")
    except Exception as e:
        logger.warning(f"Cache invalidate error for '{key}': {e}")
