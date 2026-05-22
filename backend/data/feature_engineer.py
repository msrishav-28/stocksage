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

    Handles both DatetimeIndex and integer-indexed DataFrames gracefully.
    """
    logger.info(f"Building feature DataFrame for {ticker} ({len(df)} rows)")

    # Compute technical indicators
    df = compute_all_indicators(df)

    # Calendar / time features — guard against non-DatetimeIndex (e.g., test fixtures)
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"]  = df.index.dayofweek.astype(float)
        df["day_of_month"] = df.index.day.astype(float)
        df["month"]        = df.index.month.astype(float)
        df["quarter"]      = df.index.quarter.astype(float)
    else:
        # Fallback: try to parse index as datetime; if not possible use zeros
        try:
            dt_index = pd.to_datetime(df.index)
            df["day_of_week"]  = dt_index.dayofweek.astype(float)
            df["day_of_month"] = dt_index.day.astype(float)
            df["month"]        = dt_index.month.astype(float)
            df["quarter"]      = dt_index.quarter.astype(float)
        except Exception:
            df["day_of_week"]  = 0.0
            df["day_of_month"] = 1.0
            df["month"]        = 1.0
            df["quarter"]      = 1.0

    # Earnings-week & holiday proximity stubs (require external calendars)
    df["is_earnings_week"]    = 0.0
    df["is_holiday_proximity"] = 0.0

    # Sequential time index for TFT (must be integer, monotonically increasing)
    df["time_idx"] = np.arange(len(df), dtype=int)

    # Static categoricals (enriched via ticker info later)
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

    # Sentiment stubs (populated by the sentiment agent in ensemble flow)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    if "sentiment_volume" not in df.columns:
        df["sentiment_volume"] = 0.0

    # Ensure close_return (TFT target) is present
    if "close_return" not in df.columns:
        df["close_return"] = df["close"].pct_change()

    df = df.dropna()
    logger.success(f"Feature DF ready: {len(df)} rows × {len(df.columns)} columns")
    return df
