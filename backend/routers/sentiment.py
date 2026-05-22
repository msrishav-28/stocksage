"""Sentiment router — FinBERT-scored news for a ticker."""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from backend.schemas.sentiment import SentimentResponse
from backend.ml.sentiment_agent import SentimentAgent
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.get("/{ticker}", response_model=SentimentResponse)
async def get_sentiment(ticker: str, hours: int = Query(48, ge=1, le=168)):
    ticker = ticker.upper().strip()
    cache_key = f"sentiment:{ticker}:{hours}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        agent_result = await SentimentAgent().run({"ticker": ticker, "hours": hours})
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Sentiment pipeline error")

    meta = agent_result.metadata or {}
    result = {
        "ticker": ticker,
        "composite_score": meta.get("composite_score", 0.0),
        "label": meta.get("label", "neutral"),
        "bullish_count": meta.get("bullish_count", 0),
        "bearish_count": meta.get("bearish_count", 0),
        "neutral_count": meta.get("neutral_count", 0),
        "total_articles": meta.get("total_articles", 0),
        "sentiment_momentum": meta.get("sentiment_momentum", 0.0),
        "headlines": meta.get("headlines", []),
    }

    await set_cache(cache_key, result, ttl=1800)
    return result
