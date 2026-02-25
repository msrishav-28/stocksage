"""Screener router — filter stocks by multiple criteria."""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from typing import Optional

from backend.data.price_fetcher import fetch_ohlcv, fetch_ticker_info
from backend.indicators import compute_all_indicators, compute_confluence_score
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()

# Default screener universe (S&P 500 sample)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
    "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN", "AVGO", "COST", "PEP",
    "KO", "WMT", "T", "VZ", "CSCO", "ORCL", "IBM", "BA", "GE", "MMM",
]


@router.get("/")
async def screen_stocks(
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rsi: Optional[float] = Query(None),
    max_rsi: Optional[float] = Query(None),
    sector: Optional[str] = Query(None),
    trend: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
):
    cache_key = f"screener:{min_price}:{max_price}:{min_rsi}:{max_rsi}:{sector}:{trend}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    results = []
    filters_applied = {}

    if min_price is not None: filters_applied["min_price"] = min_price
    if max_price is not None: filters_applied["max_price"] = max_price
    if min_rsi is not None: filters_applied["min_rsi"] = min_rsi
    if max_rsi is not None: filters_applied["max_rsi"] = max_rsi
    if sector is not None: filters_applied["sector"] = sector
    if trend is not None: filters_applied["trend"] = trend

    for ticker in DEFAULT_TICKERS:
        try:
            df = fetch_ohlcv(ticker, period="3mo")
            if df.empty or len(df) < 20:
                continue

            df = compute_all_indicators(df)
            latest = df.iloc[-1]
            price = float(latest["close"])

            # Apply filters
            if min_price is not None and price < min_price: continue
            if max_price is not None and price > max_price: continue

            rsi_val = float(latest.get("rsi_14", 50))
            if min_rsi is not None and rsi_val < min_rsi: continue
            if max_rsi is not None and rsi_val > max_rsi: continue

            # Get info for sector filtering
            info = {}
            if sector is not None:
                try:
                    info = fetch_ticker_info(ticker)
                    if info.get("sector", "").lower() != sector.lower():
                        continue
                except Exception:
                    continue

            # Get direction signal
            confluence = compute_confluence_score(df)
            direction = confluence["direction"]
            if trend is not None and direction != trend:
                continue

            change_pct = float(latest.get("daily_return", 0)) * 100

            results.append({
                "ticker": ticker,
                "name": info.get("name", ticker),
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "rsi": round(rsi_val, 2),
                "sector": info.get("sector"),
                "signal": direction,
            })

            if len(results) >= limit:
                break

        except Exception as e:
            logger.warning(f"Screener skip {ticker}: {e}")
            continue

    response = {
        "results": results,
        "total": len(results),
        "filters_applied": filters_applied,
    }

    await set_cache(cache_key, response, ttl=900)
    return response
