"""Macro router — FRED data and sector analysis."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.data.macro_fetcher import fetch_macro_snapshot, compute_macro_score, SECTOR_ETFS
from backend.data.price_fetcher import fetch_ohlcv
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.get("/snapshot")
async def get_macro_snapshot():
    cache_key = "macro:snapshot"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        snapshot = fetch_macro_snapshot()
        score = compute_macro_score(snapshot)
        result = {
            "snapshot": snapshot,
            "macro_score": score,
        }
    except Exception as e:
        logger.error(f"Macro snapshot failed: {e}")
        raise HTTPException(status_code=500, detail=f"Macro snapshot failed: {e}")

    await set_cache(cache_key, result, ttl=86400)
    return result


@router.get("/sector/{sector}")
async def get_sector_analysis(sector: str):
    cache_key = f"macro:sector:{sector}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    etf = SECTOR_ETFS.get(sector)
    if not etf:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sector: {sector}. Available: {list(SECTOR_ETFS.keys())}"
        )

    try:
        # Get sector ETF performance
        df = fetch_ohlcv(etf, period="6mo")
        if df.empty:
            raise ValueError(f"No data for sector ETF {etf}")

        current = float(df["close"].iloc[-1])
        pct_1mo = float((df["close"].iloc[-1] / df["close"].iloc[-22] - 1) * 100) if len(df) >= 22 else 0
        pct_3mo = float((df["close"].iloc[-1] / df["close"].iloc[-66] - 1) * 100) if len(df) >= 66 else 0

        # Get macro score for sector
        snapshot = fetch_macro_snapshot()
        macro_score = compute_macro_score(snapshot, sector=sector)

        result = {
            "sector": sector,
            "etf": etf,
            "price": round(current, 2),
            "return_1mo_pct": round(pct_1mo, 2),
            "return_3mo_pct": round(pct_3mo, 2),
            "macro_score": macro_score,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Sector analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    await set_cache(cache_key, result, ttl=3600)
    return result
