"""
Technical Indicators Engine — computes 20+ indicators on OHLCV data.

Uses pandas-ta for standard indicators and PyWavelets for wavelet decomposition.
"""

import pandas as pd
import numpy as np
from loguru import logger
from backend.indicators.momentum import (
    compute_rsi,
    compute_macd,
    compute_stochastic,
    compute_cci,
    compute_mfi,
    compute_roc,
    compute_williams_r,
)
from backend.indicators.trend import compute_ema, compute_sma, compute_adx, compute_aroon
from backend.indicators.volatility import compute_bbands, compute_atr, compute_keltner
from backend.indicators.volume import compute_obv, compute_vwap, compute_cmf, compute_ad

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    logger.warning("pandas-ta not installed. Technical indicators will be limited.")

try:
    import pywt
except ImportError:
    pywt = None
    logger.warning("PyWavelets not installed. Wavelet features disabled.")


def _safe_get(series: pd.Series, col: str, default=0):
    """
    Safely retrieve a value from a pandas Series by label.
    pandas Series.get() does not exist — this is the correct replacement.
    """
    try:
        val = series[col]
        if pd.isna(val):
            return default
        return val
    except (KeyError, TypeError):
        return default


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: appends all technical indicators to OHLCV df.
    Input df must have columns: open, high, low, close, volume.
    All indicator assignment uses np.nan for missing values (never Python None).
    """
    df = df.copy()

    if ta is None:
        logger.warning("pandas-ta not available, using fallback indicator implementations.")
        df["rsi_14"] = compute_rsi(df["close"], 14)
        df["rsi_28"] = compute_rsi(df["close"], 28)

        macd = compute_macd(df["close"])
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]

        stoch = compute_stochastic(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
        df["cci_20"] = compute_cci(df["high"], df["low"], df["close"], 20)
        df["mfi_14"] = compute_mfi(df["high"], df["low"], df["close"], df["volume"], 14)
        df["roc_10"] = compute_roc(df["close"], 10)
        df["williams_r"] = compute_williams_r(df["high"], df["low"], df["close"], 14)

        df["ema_9"] = compute_ema(df["close"], 9)
        df["ema_21"] = compute_ema(df["close"], 21)
        df["ema_50"] = compute_ema(df["close"], 50)
        df["ema_200"] = compute_ema(df["close"], 200)
        df["sma_20"] = compute_sma(df["close"], 20)
        df["sma_50"] = compute_sma(df["close"], 50)

        adx = compute_adx(df["high"], df["low"], df["close"], 14)
        df["adx_14"] = adx.iloc[:, 0]
        df["dmp_14"] = adx.iloc[:, 1]
        df["dmn_14"] = adx.iloc[:, 2]

        aroon = compute_aroon(df["high"], df["low"], 25)
        df["aroon_up"] = aroon.iloc[:, 0]
        df["aroon_down"] = aroon.iloc[:, 1]

        bb = compute_bbands(df["close"], 20, 2)
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_middle"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
        df["bb_width"] = bb.iloc[:, 3]
        df["bb_pct"] = bb.iloc[:, 4]

        df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
        kc = compute_keltner(df["high"], df["low"], df["close"], 20, 2)
        df["kc_upper"] = kc.iloc[:, 0]
        df["kc_lower"] = kc.iloc[:, 1]

        df["obv"] = compute_obv(df["close"], df["volume"])
        df["vwap"] = compute_vwap(df["high"], df["low"], df["close"], df["volume"])
        df["cmf_20"] = compute_cmf(df["high"], df["low"], df["close"], df["volume"], 20)
        df["ad_line"] = compute_ad(df["high"], df["low"], df["close"], df["volume"])
        df["volume_sma20"] = compute_sma(df["volume"], 20)
        df["volume_ratio"] = df["volume"] / df["volume_sma20"]

        if pywt is not None and len(df) > 16:
            df["close_wavelet"] = _wavelet_smooth(df["close"].values)
        else:
            df["close_wavelet"] = df["close"]

        df = _compute_basic_derived(df)
        return df.dropna()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"]     = ta.rsi(df["close"], length=14)
    df["rsi_28"]     = ta.rsi(df["close"], length=28)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_k"] = df["stoch_d"] = np.nan

    df["cci_20"]     = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["mfi_14"]     = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["roc_10"]     = ta.roc(df["close"], length=10)
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema_9"]   = ta.ema(df["close"], length=9)
    df["ema_21"]  = ta.ema(df["close"], length=21)
    df["ema_50"]  = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["sma_20"]  = ta.sma(df["close"], length=20)
    df["sma_50"]  = ta.sma(df["close"], length=50)

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and not adx.empty:
        df["adx_14"] = adx.iloc[:, 0]
        df["dmp_14"] = adx.iloc[:, 1]
        df["dmn_14"] = adx.iloc[:, 2]
    else:
        df["adx_14"] = df["dmp_14"] = df["dmn_14"] = np.nan

    aroon = ta.aroon(df["high"], df["low"], length=25)
    if aroon is not None and not aroon.empty:
        df["aroon_up"]   = aroon.iloc[:, 0]
        df["aroon_down"] = aroon.iloc[:, 1]
    else:
        df["aroon_up"] = df["aroon_down"] = np.nan

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df["bb_upper"]  = bb.iloc[:, 0]
        df["bb_middle"] = bb.iloc[:, 1]
        df["bb_lower"]  = bb.iloc[:, 2]
        # Width and pct columns vary by pandas-ta version — guard with shape check
        df["bb_width"]  = bb.iloc[:, 3] if bb.shape[1] > 3 else (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"]    = bb.iloc[:, 4] if bb.shape[1] > 4 else (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    else:
        df["bb_upper"] = df["bb_middle"] = df["bb_lower"] = np.nan
        df["bb_width"] = df["bb_pct"] = np.nan

    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    kc = ta.kc(df["high"], df["low"], df["close"])
    if kc is not None and not kc.empty:
        df["kc_upper"] = kc.iloc[:, 0]
        df["kc_lower"] = kc.iloc[:, 1] if kc.shape[1] > 1 else kc.iloc[:, 0] - ta.atr(df["high"], df["low"], df["close"], length=20) * 2
    else:
        df["kc_upper"] = df["kc_lower"] = np.nan

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]         = ta.obv(df["close"], df["volume"])
    df["vwap"]        = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["cmf_20"]      = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
    df["ad_line"]     = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    df["volume_sma20"] = ta.sma(df["volume"], length=20)
    # Protect against zero division on volume_sma20
    df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0, np.nan)

    # ── Wavelet Decomposition (noise-filtered signal) ─────────────────────────
    if pywt is not None and len(df) > 16:
        df["close_wavelet"] = _wavelet_smooth(df["close"].values)
    else:
        df["close_wavelet"] = df["close"]

    # ── Price-derived features ────────────────────────────────────────────────
    df = _compute_basic_derived(df)

    return df.dropna()


def _compute_basic_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-derived features and pure-Python indicator fallbacks.
    These run regardless of whether pandas-ta is installed, ensuring all
    columns referenced by strategies and confluence scoring always exist.
    """
    df["daily_return"]   = df["close"].pct_change()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["hl_pct"]         = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
    df["close_return"]   = df["close"].pct_change()  # TFT target column

    # ── Pure-Python fallbacks for key indicator columns ───────────────────────
    # These only run when pandas-ta is absent; they're overwritten by the
    # pandas-ta block above when ta IS available.

    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_28"] = df["rsi_14"]   # approximate

    if "macd" not in df.columns:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

    if "ema_9" not in df.columns:
        df["ema_9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"]= df["close"].ewm(span=200, adjust=False).mean()

    if "sma_20" not in df.columns:
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

    if "bb_upper" not in df.columns:
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_upper"]  = sma + 2 * std
        df["bb_middle"] = sma
        df["bb_lower"]  = sma - 2 * std
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / sma
        df["bb_pct"]    = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    if "atr_14" not in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        df["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    if "adx_14" not in df.columns:
        # Simplified ADX
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        atr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        dm_p  = (df["high"] - df["high"].shift(1)).clip(lower=0)
        dm_n  = (df["low"].shift(1) - df["low"]).clip(lower=0)
        di_p  = 100 * dm_p.rolling(14).mean() / atr.replace(0, np.nan)
        di_n  = 100 * dm_n.rolling(14).mean() / atr.replace(0, np.nan)
        dx    = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
        df["adx_14"] = dx.rolling(14).mean()
        df["dmp_14"] = di_p
        df["dmn_14"] = di_n

    if "stoch_k" not in df.columns:
        lowest  = df["low"].rolling(14).min()
        highest = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    if "obv" not in df.columns:
        direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df["obv"] = (df["volume"] * direction).cumsum()

    if "vwap" not in df.columns:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    if "cmf_20" not in df.columns:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"]).replace(0, np.nan)
        mfv = mfm * df["volume"]
        df["cmf_20"] = mfv.rolling(20).sum() / df["volume"].rolling(20).sum()

    if "mfi_14" not in df.columns:
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        mf  = tp * df["volume"]
        pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mr  = pos / neg.replace(0, np.nan)
        df["mfi_14"] = 100 - (100 / (1 + mr))

    if "volume_sma20" not in df.columns:
        df["volume_sma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0, np.nan)

    if "aroon_up" not in df.columns:
        df["aroon_up"]   = df["high"].rolling(26).apply(lambda x: x.argmax() / 25 * 100, raw=True)
        df["aroon_down"] = df["low"].rolling(26).apply(lambda x: x.argmin() / 25 * 100, raw=True)

    if "ad_line" not in df.columns:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"]).replace(0, np.nan)
        df["ad_line"] = (mfm * df["volume"]).cumsum()

    if "close_wavelet" not in df.columns:
        df["close_wavelet"] = df["close"]

    if "roc_10" not in df.columns:
        df["roc_10"]     = df["close"].pct_change(10) * 100
        df["williams_r"] = -100 * (df["high"].rolling(14).max() - df["close"]) / \
                           (df["high"].rolling(14).max() - df["low"].rolling(14).min()).replace(0, np.nan)
        df["cci_20"]     = (df["close"] - df["close"].rolling(20).mean()) / \
                           (0.015 * df["close"].rolling(20).std())
        df["kc_upper"]   = df["ema_21"] + 2 * df["atr_14"]
        df["kc_lower"]   = df["ema_21"] - 2 * df["atr_14"]

    return df


def _wavelet_smooth(prices: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Applies discrete wavelet transform to remove high-frequency noise.
    Returns the low-frequency approximation coefficients reconstructed to original length.
    """
    if len(prices) < 2 ** (level + 1):
        return prices
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # Zero out detail coefficients (noise), keep approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    # Align length (waverec can return N+1 elements)
    return smoothed[:len(prices)]


def compute_confluence_score(df: pd.DataFrame) -> dict:
    """
    Counts indicator alignment for technical direction signal.
    Returns score from -1.0 (all bearish) to +1.0 (all bullish).
    Uses _safe_get() to avoid pandas Series AttributeError.
    """
    latest = df.iloc[-1]
    signals = []

    # RSI
    rsi = _safe_get(latest, "rsi_14", 50)
    if rsi > 55:   signals.append(1)
    elif rsi < 45: signals.append(-1)
    else:          signals.append(0)

    # MACD histogram direction
    macd_val  = _safe_get(latest, "macd", 0)
    macd_sig  = _safe_get(latest, "macd_signal", 0)
    macd_hist = _safe_get(latest, "macd_hist", 0)
    if macd_val > macd_sig and macd_hist > 0:   signals.append(1)
    elif macd_val < macd_sig and macd_hist < 0: signals.append(-1)
    else:                                        signals.append(0)

    # EMA alignment (9 > 21 > 50 = bullish trend)
    ema9  = _safe_get(latest, "ema_9", 0)
    ema21 = _safe_get(latest, "ema_21", 0)
    ema50 = _safe_get(latest, "ema_50", 0)
    if ema9 > 0 and ema21 > 0 and ema50 > 0:
        if ema9 > ema21 > ema50:   signals.append(1)
        elif ema9 < ema21 < ema50: signals.append(-1)
        else:                      signals.append(0)
    else:
        signals.append(0)

    # Bollinger Band position
    close_val = _safe_get(latest, "close", 0)
    bb_mid    = _safe_get(latest, "bb_middle", close_val)
    if bb_mid and bb_mid != 0:
        if close_val > bb_mid:   signals.append(1)
        elif close_val < bb_mid: signals.append(-1)
        else:                    signals.append(0)
    else:
        signals.append(0)

    # ADX trend strength + direction
    adx = _safe_get(latest, "adx_14", 0)
    dmp = _safe_get(latest, "dmp_14", 0)
    dmn = _safe_get(latest, "dmn_14", 0)
    if adx > 25:
        signals.append(1 if dmp > dmn else -1)
    else:
        signals.append(0)

    # Volume confirmation
    vol_ratio  = _safe_get(latest, "volume_ratio", 1)
    daily_ret  = _safe_get(latest, "daily_return", 0)
    if vol_ratio > 1.5 and daily_ret > 0:   signals.append(1)
    elif vol_ratio > 1.5 and daily_ret < 0: signals.append(-1)
    else:                                    signals.append(0)

    # MFI
    mfi = _safe_get(latest, "mfi_14", 50)
    if mfi > 60:   signals.append(1)
    elif mfi < 40: signals.append(-1)
    else:          signals.append(0)

    # CMF
    cmf = _safe_get(latest, "cmf_20", 0)
    if cmf > 0.1:   signals.append(1)
    elif cmf < -0.1: signals.append(-1)
    else:            signals.append(0)

    score     = float(np.mean(signals))
    direction = "bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral")

    return {
        "direction":        direction,
        "raw_score":        round(score, 4),
        "confluence_score": round(abs(score), 4),
        "signal_count":     len(signals),
        "bullish_signals":  signals.count(1),
        "bearish_signals":  signals.count(-1),
        "neutral_signals":  signals.count(0),
    }
