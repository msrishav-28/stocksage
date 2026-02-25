"""Sentiment router — FinBERT scored news for a ticker."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.ml.sentiment_agent import SentimentAgent
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.get("/{ticker}")
async def get_sentiment(ticker: str, hours: int = 48):
    cache_key = f"sentiment:{ticker}:{hours}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        agent = SentimentAgent()
        result = await agent.analyze(ticker, news_window_hours=hours)
        result["ticker"] = ticker
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

    await set_cache(cache_key, result, ttl=1800)
    return result
