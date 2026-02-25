"""Predefined backtesting strategies."""

import pandas as pd
from loguru import logger


def get_strategy_signals(df: pd.DataFrame, strategy_name: str) -> tuple:
    """
    Returns (entries, exits) boolean Series for the chosen strategy.
    """
    strategies = {
        "rsi_macd": _rsi_macd_strategy,
        "ema_crossover": _ema_crossover_strategy,
        "bollinger_breakout": _bollinger_breakout_strategy,
        "mean_reversion": _mean_reversion_strategy,
        "momentum": _momentum_strategy,
    }

    strategy_fn = strategies.get(strategy_name)
    if strategy_fn is None:
        available = list(strategies.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    logger.info(f"Generating signals for strategy: {strategy_name}")
    return strategy_fn(df)


def _rsi_macd_strategy(df: pd.DataFrame) -> tuple:
    """Classic RSI + MACD crossover."""
    entries = (df["rsi_14"] < 35) & (df["macd"] > df["macd_signal"])
    exits   = (df["rsi_14"] > 65) | (df["macd"] < df["macd_signal"])
    return entries.astype(bool), exits.astype(bool)


def _ema_crossover_strategy(df: pd.DataFrame) -> tuple:
    """EMA 9/21 crossover."""
    entries = (df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
    exits   = (df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)


def _bollinger_breakout_strategy(df: pd.DataFrame) -> tuple:
    """Buy on lower band touch, sell on upper band touch."""
    entries = df["close"] <= df["bb_lower"]
    exits   = df["close"] >= df["bb_upper"]
    return entries.astype(bool), exits.astype(bool)


def _mean_reversion_strategy(df: pd.DataFrame) -> tuple:
    """RSI extremes with volume confirmation."""
    entries = (df["rsi_14"] < 30) & (df["volume_ratio"] > 1.2)
    exits   = (df["rsi_14"] > 55) | (df["close"] >= df["sma_20"])
    return entries.astype(bool), exits.astype(bool)


def _momentum_strategy(df: pd.DataFrame) -> tuple:
    """Strong trend following with ADX confirmation."""
    entries = (df["adx_14"] > 25) & (df["dmp_14"] > df["dmn_14"]) & (df["rsi_14"] > 50)
    exits   = (df["adx_14"] < 20) | (df["dmp_14"] < df["dmn_14"]) | (df["rsi_14"] > 75)
    return entries.astype(bool), exits.astype(bool)
