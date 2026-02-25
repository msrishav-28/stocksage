"""Momentum indicators — individual computation functions."""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if ta: return ta.rsi(close, length=length)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if ta: return ta.macd(close, fast=fast, slow=slow, signal=signal)
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k: int = 14, d: int = 3) -> pd.DataFrame:
    if ta: return ta.stoch(high, low, close, k=k, d=d)
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest)
    stoch_d = stoch_k.rolling(d).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    if ta: return ta.cci(high, low, close, length=length)
    tp = (high + low + close) / 3
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, length: int = 14) -> pd.Series:
    if ta: return ta.mfi(high, low, close, volume, length=length)
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(length).sum()
    mr = pos_mf / neg_mf
    return 100 - (100 / (1 + mr))


def compute_roc(close: pd.Series, length: int = 10) -> pd.Series:
    if ta: return ta.roc(close, length=length)
    return ((close - close.shift(length)) / close.shift(length)) * 100


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                       length: int = 14) -> pd.Series:
    if ta: return ta.willr(high, low, close, length=length)
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    return -100 * (highest - close) / (highest - lowest)
