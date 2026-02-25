import redis.asyncio as redis
import orjson
from typing import Any, Optional
from loguru import logger
from backend.config import get_settings

_redis_client: Optional[redis.Redis] = None


async def init_redis():
    global _redis_client
    settings = get_settings()
    try:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        await _redis_client.ping()
        logger.success("Redis connected.")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        _redis_client = None


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()


def get_redis() -> Optional[redis.Redis]:
    return _redis_client


async def get_cached(key: str) -> Optional[Any]:
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
        logger.warning(f"Cache read error: {e}")
    return None


async def set_cache(key: str, value: Any, ttl: int = 300):
    client = get_redis()
    if client is None:
        return
    try:
        await client.setex(key, ttl, orjson.dumps(value))
        logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


async def invalidate(key: str):
    client = get_redis()
    if client is None:
        return
    try:
        await client.delete(key)
        logger.info(f"Cache INVALIDATED: {key}")
    except Exception as e:
        logger.warning(f"Cache invalidate error: {e}")
