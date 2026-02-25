"""Technical analysis agent — uses indicator confluence for directional signals."""

import pandas as pd
from loguru import logger
from backend.indicators import compute_all_indicators, compute_confluence_score


class TechnicalAgent:
    """
    Analyses 20+ technical indicators and returns a directional signal.
    Uses confluence scoring: counts how many indicators align.
    """

    async def analyze(self, ticker: str, df: pd.DataFrame) -> dict:
        logger.info(f"TechnicalAgent analyzing {ticker}")

        # Ensure indicators are computed
        if "rsi_14" not in df.columns:
            df = compute_all_indicators(df)

        confluence = compute_confluence_score(df)

        # Extract key indicator values for metadata
        latest = df.iloc[-1]
        key_indicators = {
            "rsi_14": round(float(latest.get("rsi_14", 50)), 2),
            "macd_hist": round(float(latest.get("macd_hist", 0)), 4),
            "adx_14": round(float(latest.get("adx_14", 0)), 2),
            "bb_pct": round(float(latest.get("bb_pct", 0.5)), 4),
            "volume_ratio": round(float(latest.get("volume_ratio", 1)), 2),
            "ema_9": round(float(latest.get("ema_9", 0)), 2),
            "ema_21": round(float(latest.get("ema_21", 0)), 2),
            "daily_return": round(float(latest.get("daily_return", 0)), 4),
        }

        return {
            **confluence,
            "key_indicators": key_indicators,
        }
