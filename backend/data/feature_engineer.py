"""Feature engineering pipeline — merges OHLCV + indicators + time features."""

import pandas as pd
import numpy as np
from loguru import logger
from backend.indicators import compute_all_indicators


def build_feature_df(df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and produces a fully-featured DataFrame
    ready for TFT or technical analysis.

    Adds:
      - All 20+ technical indicators via compute_all_indicators()
      - Calendar features (day_of_week, day_of_month, month, quarter)
      - Placeholder time_idx for TFT
      - Static columns: ticker, sector, market_cap_tier
    """
    logger.info(f"Building feature DataFrame for {ticker} ({len(df)} rows)")

    # Compute technical indicators
    df = compute_all_indicators(df)

    # Calendar / time features
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Is-earnings-week & holiday proximity stubs (need external data to populate)
    df["is_earnings_week"] = 0
    df["is_holiday_proximity"] = 0

    # Time index for TFT
    df["time_idx"] = np.arange(len(df))

    # Static categoricals (will be enriched via ticker info later)
    df["ticker"] = ticker
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "market_cap_tier" not in df.columns:
        df["market_cap_tier"] = "mid"

    # Static reals
    if "avg_daily_volume_30d" not in df.columns:
        df["avg_daily_volume_30d"] = df["volume"].rolling(30, min_periods=1).mean()
    if "beta" not in df.columns:
        df["beta"] = 1.0

    # Sentiment stubs (will be filled by sentiment agent)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    if "sentiment_volume" not in df.columns:
        df["sentiment_volume"] = 0.0

    df = df.dropna()
    logger.success(f"Feature DF ready: {len(df)} rows × {len(df.columns)} columns")
    return df
