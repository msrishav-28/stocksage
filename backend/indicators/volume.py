"""Volume indicators — individual computation functions."""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if ta: return ta.obv(close, volume)
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume * direction).cumsum()


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series) -> pd.Series:
    if ta: return ta.vwap(high, low, close, volume)
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, length: int = 20) -> pd.Series:
    if ta: return ta.cmf(high, low, close, volume, length=length)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum()


def compute_ad(high: pd.Series, low: pd.Series, close: pd.Series,
               volume: pd.Series) -> pd.Series:
    if ta: return ta.ad(high, low, close, volume)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    return mfv.cumsum()
