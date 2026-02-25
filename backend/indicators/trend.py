"""Trend indicators — individual computation functions."""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def compute_ema(close: pd.Series, length: int = 20) -> pd.Series:
    if ta: return ta.ema(close, length=length)
    return close.ewm(span=length, adjust=False).mean()


def compute_sma(close: pd.Series, length: int = 20) -> pd.Series:
    if ta: return ta.sma(close, length=length)
    return close.rolling(window=length).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 14) -> pd.DataFrame:
    if ta: return ta.adx(high, low, close, length=length)
    # Simplified ADX computation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()

    dm_plus = (high - high.shift(1)).where(
        (high - high.shift(1)) > (low.shift(1) - low), 0
    ).clip(lower=0)
    dm_minus = (low.shift(1) - low).where(
        (low.shift(1) - low) > (high - high.shift(1)), 0
    ).clip(lower=0)

    di_plus = 100 * dm_plus.rolling(length).mean() / atr
    di_minus = 100 * dm_minus.rolling(length).mean() / atr
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(length).mean()

    return pd.DataFrame({"ADX": adx, "DMP": di_plus, "DMN": di_minus})


def compute_aroon(high: pd.Series, low: pd.Series, length: int = 25) -> pd.DataFrame:
    if ta: return ta.aroon(high, low, length=length)
    aroon_up = high.rolling(length + 1).apply(
        lambda x: x.argmax() / length * 100, raw=True
    )
    aroon_down = low.rolling(length + 1).apply(
        lambda x: x.argmin() / length * 100, raw=True
    )
    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down})
