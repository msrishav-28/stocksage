"""
Technical Indicators Engine — computes 20+ indicators on OHLCV data.

Uses pandas-ta for standard indicators and PyWavelets for wavelet decomposition.
"""

import pandas as pd
import numpy as np
from typing import Optional
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


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: appends all technical indicators to OHLCV df.
    Input df must have columns: open, high, low, close, volume
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
    macd             = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]       = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]  = macd.iloc[:, 2]
    stoch            = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None:
        df["stoch_k"]    = stoch.iloc[:, 0]
        df["stoch_d"]    = stoch.iloc[:, 1]
    df["cci_20"]     = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["mfi_14"]     = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["roc_10"]     = ta.roc(df["close"], length=10)
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema_9"]      = ta.ema(df["close"], length=9)
    df["ema_21"]     = ta.ema(df["close"], length=21)
    df["ema_50"]     = ta.ema(df["close"], length=50)
    df["ema_200"]    = ta.ema(df["close"], length=200)
    df["sma_20"]     = ta.sma(df["close"], length=20)
    df["sma_50"]     = ta.sma(df["close"], length=50)
    adx              = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df["adx_14"]     = adx.iloc[:, 0]
        df["dmp_14"]     = adx.iloc[:, 1]
        df["dmn_14"]     = adx.iloc[:, 2]
    aroon            = ta.aroon(df["high"], df["low"], length=25)
    if aroon is not None:
        df["aroon_up"]   = aroon.iloc[:, 0]
        df["aroon_down"] = aroon.iloc[:, 1]

    # ── Volatility ────────────────────────────────────────────────────────────
    bb               = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_upper"]   = bb.iloc[:, 0]
        df["bb_middle"]  = bb.iloc[:, 1]
        df["bb_lower"]   = bb.iloc[:, 2]
        df["bb_width"]   = bb.iloc[:, 3] if bb.shape[1] > 3 else None
        df["bb_pct"]     = bb.iloc[:, 4] if bb.shape[1] > 4 else None
    df["atr_14"]     = ta.atr(df["high"], df["low"], df["close"], length=14)
    kc               = ta.kc(df["high"], df["low"], df["close"])
    if kc is not None:
        df["kc_upper"]   = kc.iloc[:, 0]
        df["kc_lower"]   = kc.iloc[:, 1] if kc.shape[1] > 1 else None

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]        = ta.obv(df["close"], df["volume"])
    df["vwap"]       = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["cmf_20"]     = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
    df["ad_line"]    = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    df["volume_sma20"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]

    # ── Wavelet Decomposition (noise-filtered signal) ─────────────────────────
    if pywt is not None and len(df) > 16:
        df["close_wavelet"] = _wavelet_smooth(df["close"].values)
    else:
        df["close_wavelet"] = df["close"]

    # ── Price-derived features ────────────────────────────────────────────────
    df = _compute_basic_derived(df)

    return df.dropna()


def _compute_basic_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-derived features that don't need pandas-ta."""
    df["daily_return"]   = df["close"].pct_change()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["hl_pct"]         = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
    df["close_return"]   = df["close"].pct_change()  # TFT target
    return df


def _wavelet_smooth(prices: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Applies discrete wavelet transform to remove high-frequency noise.
    Returns the low-frequency approximation coefficients reconstructed to original length.
    """
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # Zero out detail coefficients (noise), keep approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    # Align length
    return smoothed[:len(prices)]


def compute_confluence_score(df: pd.DataFrame) -> dict:
    """
    Counts indicator alignment for technical direction signal.
    Returns score from -1.0 (all bearish) to +1.0 (all bullish).
    """
    latest = df.iloc[-1]
    signals = []

    # RSI
    rsi = latest.get("rsi_14", 50)
    if rsi > 55: signals.append(1)
    elif rsi < 45: signals.append(-1)
    else: signals.append(0)

    # MACD
    macd_val = latest.get("macd", 0)
    macd_sig = latest.get("macd_signal", 0)
    macd_hist = latest.get("macd_hist", 0)
    if macd_val > macd_sig and macd_hist > 0:
        signals.append(1)
    elif macd_val < macd_sig and macd_hist < 0:
        signals.append(-1)
    else: signals.append(0)

    # EMA alignment
    ema9 = latest.get("ema_9", 0)
    ema21 = latest.get("ema_21", 0)
    ema50 = latest.get("ema_50", 0)
    if ema9 > ema21 > ema50: signals.append(1)
    elif ema9 < ema21 < ema50: signals.append(-1)
    else: signals.append(0)

    # Bollinger
    close_val = latest.get("close", 0)
    bb_mid = latest.get("bb_middle", 0)
    if close_val > bb_mid: signals.append(1)
    elif close_val < bb_mid: signals.append(-1)
    else: signals.append(0)

    # ADX trend strength
    adx = latest.get("adx_14", 0)
    dmp = latest.get("dmp_14", 0)
    dmn = latest.get("dmn_14", 0)
    if adx > 25:
        signals.append(1 if dmp > dmn else -1)
    else: signals.append(0)

    # Volume confirmation
    vol_ratio = latest.get("volume_ratio", 1)
    daily_ret = latest.get("daily_return", 0)
    if vol_ratio > 1.5 and daily_ret > 0: signals.append(1)
    elif vol_ratio > 1.5 and daily_ret < 0: signals.append(-1)
    else: signals.append(0)

    # MFI
    mfi = latest.get("mfi_14", 50)
    if mfi > 60: signals.append(1)
    elif mfi < 40: signals.append(-1)
    else: signals.append(0)

    # CMF
    cmf = latest.get("cmf_20", 0)
    if cmf > 0.1: signals.append(1)
    elif cmf < -0.1: signals.append(-1)
    else: signals.append(0)

    score = float(np.mean(signals))
    direction = "bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral")

    return {
        "direction": direction,
        "raw_score": round(score, 4),
        "confluence_score": round(abs(score), 4),
        "signal_count": len(signals),
        "bullish_signals": signals.count(1),
        "bearish_signals": signals.count(-1),
        "neutral_signals": signals.count(0),
    }
