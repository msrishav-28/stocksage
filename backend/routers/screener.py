"""Screener router — filter a stock universe by multiple criteria."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Query
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv, fetch_ticker_info
from backend.indicators import compute_all_indicators, compute_confluence_score
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()

# Default screener universe (S&P 500 sample).
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
    "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN", "AVGO", "COST", "PEP",
    "KO", "WMT", "T", "VZ", "CSCO", "ORCL", "IBM", "BA", "GE", "MMM",
]

# Indicators need a long lookback (EMA-200) and compute_all_indicators drops
# incomplete rolling windows — fetch a full year so rows survive the dropna.
_SCREENER_PERIOD = "1y"
_MIN_ROWS = 60


def _screen_one(
    ticker: str,
    *,
    min_price: Optional[float],
    max_price: Optional[float],
    min_rsi: Optional[float],
    max_rsi: Optional[float],
    sector: Optional[str],
    trend: Optional[str],
) -> Optional[dict]:
    """
    Synchronous per-ticker screening. Runs in a worker thread so the event
    loop is never blocked. Returns a result dict, or None when filtered out.
    """
    try:
        df = fetch_ohlcv(ticker, period=_SCREENER_PERIOD)
        if df.empty or len(df) < _MIN_ROWS:
            return None

        df = compute_all_indicators(df)
        if df.empty:
            return None

        latest = df.iloc[-1]
        price = float(latest["close"])
        if min_price is not None and price < min_price:
            return None
        if max_price is not None and price > max_price:
            return None

        rsi_val = float(latest.get("rsi_14", 50))
        if min_rsi is not None and rsi_val < min_rsi:
            return None
        if max_rsi is not None and rsi_val > max_rsi:
            return None

        info = {}
        if sector is not None:
            info = fetch_ticker_info(ticker)
            if (info.get("sector") or "").lower() != sector.lower():
                return None

        confluence = compute_confluence_score(df)
        direction = confluence["direction"]
        if trend is not None and direction != trend:
            return None

        return {
            "ticker": ticker,
            "name": info.get("name", ticker),
            "price": round(price, 2),
            "change_pct": round(float(latest.get("daily_return", 0)) * 100, 2),
            "rsi": round(rsi_val, 2),
            "sector": info.get("sector"),
            "signal": direction,
        }
    except Exception as e:
        logger.warning(f"Screener skip {ticker}: {e}")
        return None


@router.get("/")
async def screen_stocks(
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rsi: Optional[float] = Query(None, ge=0, le=100),
    max_rsi: Optional[float] = Query(None, ge=0, le=100),
    sector: Optional[str] = Query(None),
    trend: Optional[str] = Query(None, pattern="^(bullish|bearish|neutral)$"),
    limit: int = Query(20, ge=1, le=50),
):
    cache_key = f"screener:{min_price}:{max_price}:{min_rsi}:{max_rsi}:{sector}:{trend}:{limit}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    filters_applied = {
        k: v for k, v in {
            "min_price": min_price, "max_price": max_price,
            "min_rsi": min_rsi, "max_rsi": max_rsi,
            "sector": sector, "trend": trend,
        }.items() if v is not None
    }

    # Screen every ticker concurrently in worker threads — keeps the event
    # loop free and turns a multi-minute serial scan into a parallel one.
    screened = await asyncio.gather(*[
        asyncio.to_thread(
            _screen_one, ticker,
            min_price=min_price, max_price=max_price,
            min_rsi=min_rsi, max_rsi=max_rsi,
            sector=sector, trend=trend,
        )
        for ticker in DEFAULT_TICKERS
    ])

    results = [r for r in screened if r is not None][:limit]
    response = {
        "results": results,
        "total": len(results),
        "filters_applied": filters_applied,
    }

    await set_cache(cache_key, response, ttl=900)
    return response
