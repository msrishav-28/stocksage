"""Technical router — all indicators + confluence score for a ticker."""

import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv
from backend.indicators import compute_all_indicators, compute_confluence_score
from backend.cache.redis_client import get_cached, set_cache
from backend.schemas.technical import TechnicalResponse

router = APIRouter()


def _build_history(raw_df, max_points: int = 250) -> list:
    """Downsample a raw OHLCV frame into a compact price series for charting."""
    step = max(1, len(raw_df) // max_points)
    sampled = raw_df.iloc[::step]
    history = []
    for idx, row in sampled.iterrows():
        try:
            date = idx.strftime("%Y-%m-%d")
        except AttributeError:
            date = str(idx)
        history.append({
            "date": date,
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]) if row["volume"] == row["volume"] else 0,
        })
    return history


def _compute_technical(ticker: str, period: str) -> dict:
    """Synchronous fetch + indicator computation — runs in a worker thread."""
    raw = fetch_ohlcv(ticker, period=period)
    df = compute_all_indicators(raw)
    if df.empty:
        raise ValueError(f"Not enough data to compute indicators for {ticker}")

    confluence = compute_confluence_score(df)
    latest = df.iloc[-1]

    skip = {"ticker", "open", "high", "low", "close", "volume"}
    indicator_values = {}
    for col in (c for c in df.columns if c not in skip):
        val = latest.get(col)
        if val is None:
            continue
        try:
            indicator_values[col] = round(float(val), 4)
        except (TypeError, ValueError):
            indicator_values[col] = str(val)

    return {
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
        "history": _build_history(raw),
    }


@router.get("/{ticker}", response_model=TechnicalResponse)
async def get_technical(ticker: str, period: str = "1y"):
    ticker = ticker.upper().strip()
    cache_key = f"technical:{ticker}:{period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        result = await asyncio.to_thread(_compute_technical, ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Technical analysis failed")

    await set_cache(cache_key, result, ttl=300)
    return result
