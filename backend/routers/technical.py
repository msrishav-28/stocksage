"""Technical router — all indicators + confluence score for a ticker."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv
from backend.indicators import compute_all_indicators, compute_confluence_score
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.get("/{ticker}")
async def get_technical(ticker: str, period: str = "1y"):
    cache_key = f"technical:{ticker}:{period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        df = fetch_ohlcv(ticker, period=period)
        df = compute_all_indicators(df)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {e}")

    confluence = compute_confluence_score(df)
    latest = df.iloc[-1]

    # Extract all indicator values from latest row
    indicator_values = {}
    indicator_cols = [c for c in df.columns if c not in ["ticker", "open", "high", "low", "close", "volume"]]
    for col in indicator_cols:
        val = latest.get(col)
        if val is not None:
            try:
                indicator_values[col] = round(float(val), 4)
            except (TypeError, ValueError):
                indicator_values[col] = str(val)

    result = {
        "ticker": ticker,
        "period": period,
        "confluence": confluence,
        "indicators": indicator_values,
        "price": {
            "open": round(float(latest["open"]), 2),
            "high": round(float(latest["high"]), 2),
            "low": round(float(latest["low"]), 2),
            "close": round(float(latest["close"]), 2),
            "volume": int(latest["volume"]),
        },
    }

    await set_cache(cache_key, result, ttl=300)
    return result
