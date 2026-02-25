"""Macro analysis agent — uses FRED data for macro-level directional signals."""

from loguru import logger
from backend.data.macro_fetcher import fetch_macro_snapshot, compute_macro_score
from backend.data.price_fetcher import fetch_ticker_info


class MacroAgent:
    """
    Pulls latest FRED macro indicators (Fed rate, CPI, VIX, sector ETF momentum)
    and computes a macro risk-adjusted directional signal.
    """

    async def analyze(self, ticker: str) -> dict:
        logger.info(f"MacroAgent analyzing macro environment for {ticker}")

        # Get sector for the ticker
        try:
            info = fetch_ticker_info(ticker)
            sector = info.get("sector", "Technology")
        except Exception:
            sector = "Technology"

        # Get macro snapshot
        snapshot = fetch_macro_snapshot()

        # Compute macro score
        result = compute_macro_score(snapshot, sector=sector)

        return {
            "direction": result["direction"],
            "raw_score": result["raw_score"],
            "confidence": result["confidence"],
            "reasons": result["reasons"],
            "sector": sector,
            "snapshot": result["snapshot"],
        }
