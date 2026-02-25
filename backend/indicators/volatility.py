"""Volatility indicators — individual computation functions."""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def compute_bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    if ta: return ta.bbands(close, length=length, std=std)
    sma = close.rolling(length).mean()
    rolling_std = close.rolling(length).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    width = (upper - lower) / sma * 100
    pct = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "upper": upper, "middle": sma, "lower": lower,
        "width": width, "pct": pct,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 14) -> pd.Series:
    if ta: return ta.atr(high, low, close, length=length)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def compute_keltner(high: pd.Series, low: pd.Series, close: pd.Series,
                    length: int = 20, scalar: float = 2.0) -> pd.DataFrame:
    if ta: return ta.kc(high, low, close, length=length, scalar=scalar)
    ema = close.ewm(span=length, adjust=False).mean()
    atr = compute_atr(high, low, close, length=length)
    upper = ema + scalar * atr
    lower = ema - scalar * atr
    return pd.DataFrame({"upper": upper, "lower": lower})
