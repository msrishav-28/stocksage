"""Competitor router — peer comparison with key ratios."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv, fetch_ticker_info
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()

# Industry peer mapping
INDUSTRY_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "CRM", "ADBE", "INTC", "AMD"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "DIS", "BKNG"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
}


@router.get("/{ticker}")
async def get_competitors(ticker: str):
    cache_key = f"competitor:{ticker}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        # Get ticker info to determine sector
        info = fetch_ticker_info(ticker)
        sector = info.get("sector", "Technology")

        # Find peers in same sector
        peers = INDUSTRY_PEERS.get(sector, INDUSTRY_PEERS["Technology"])
        peers = [p for p in peers if p != ticker][:5]

        # Fetch comparison data
        comparisons = []
        for peer in peers:
            try:
                peer_info = fetch_ticker_info(peer)
                peer_df = fetch_ohlcv(peer, period="3mo")
                if not peer_df.empty:
                    current_price = float(peer_df["close"].iloc[-1])
                    pct_3mo = float((peer_df["close"].iloc[-1] / peer_df["close"].iloc[0] - 1) * 100)
                    comparisons.append({
                        "ticker": peer,
                        "name": peer_info.get("name", peer),
                        "price": round(current_price, 2),
                        "change_3mo_pct": round(pct_3mo, 2),
                        "market_cap": peer_info.get("market_cap"),
                        "pe_ratio": peer_info.get("pe_ratio"),
                        "beta": peer_info.get("beta"),
                        "sector": peer_info.get("sector"),
                    })
            except Exception as e:
                logger.warning(f"Could not fetch peer {peer}: {e}")

        result = {
            "ticker": ticker,
            "sector": sector,
            "company": info,
            "peers": comparisons,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    await set_cache(cache_key, result, ttl=3600)
    return result
