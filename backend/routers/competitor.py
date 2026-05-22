"""Competitor router — peer comparison with key ratios."""

import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv, fetch_ticker_info
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()

# Industry peer mapping.
INDUSTRY_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "CRM", "ADBE", "INTC", "AMD"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "DIS", "BKNG"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
}


def _fetch_peer(peer: str) -> dict | None:
    """Synchronous per-peer fetch — runs in a worker thread."""
    try:
        peer_info = fetch_ticker_info(peer)
        peer_df = fetch_ohlcv(peer, period="3mo")
        if peer_df.empty:
            return None
        current_price = float(peer_df["close"].iloc[-1])
        pct_3mo = float((peer_df["close"].iloc[-1] / peer_df["close"].iloc[0] - 1) * 100)
        return {
            "ticker": peer,
            "name": peer_info.get("name", peer),
            "price": round(current_price, 2),
            "change_3mo_pct": round(pct_3mo, 2),
            "market_cap": peer_info.get("market_cap"),
            "pe_ratio": peer_info.get("pe_ratio"),
            "beta": peer_info.get("beta"),
            "sector": peer_info.get("sector"),
        }
    except Exception as e:
        logger.warning(f"Could not fetch peer {peer}: {e}")
        return None


@router.get("/{ticker}")
async def get_competitors(ticker: str):
    ticker = ticker.upper().strip()
    cache_key = f"competitor:{ticker}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        info = await asyncio.to_thread(fetch_ticker_info, ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Competitor info fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider error")

    sector = info.get("sector", "Technology")
    peers = INDUSTRY_PEERS.get(sector, INDUSTRY_PEERS["Technology"])
    peers = [p for p in peers if p != ticker][:5]

    # Fetch all peers concurrently in worker threads.
    fetched = await asyncio.gather(*[asyncio.to_thread(_fetch_peer, p) for p in peers])
    comparisons = [c for c in fetched if c is not None]

    result = {
        "ticker": ticker,
        "sector": sector,
        "company": info,
        "peers": comparisons,
    }

    await set_cache(cache_key, result, ttl=3600)
    return result
