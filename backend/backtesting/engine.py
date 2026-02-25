"""Backtesting engine — vectorbt wrapper for running strategy backtests."""

import pandas as pd
import numpy as np
from loguru import logger

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not installed. Backtesting disabled.")


def run_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    ticker: str,
    initial_cash: float = 10_000.0,
    fees: float = 0.001,       # 0.1% per trade
    slippage: float = 0.001,   # 0.1% slippage
) -> dict:
    """
    Runs a vectorbt backtest given entry/exit boolean signals.
    Returns comprehensive performance metrics.
    """
    if not VBT_AVAILABLE:
        raise RuntimeError("vectorbt is not installed.")

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
    )

    stats = portfolio.stats()
    benchmark_returns = close.pct_change().fillna(0)

    sharpe = float(stats.get("Sharpe Ratio", 0))
    max_dd = float(stats.get("Max Drawdown [%]", 0))
    total_return = float(stats.get("Total Return [%]", 0))
    win_rate = float(stats.get("Win Rate [%]", 0))

    bm_total = float(benchmark_returns.add(1).prod() - 1) * 100

    equity_curve = portfolio.value().rename("portfolio").to_frame()
    equity_curve["benchmark"] = initial_cash * benchmark_returns.add(1).cumprod()

    logger.success(
        f"Backtest {ticker}: return={total_return:.1f}% "
        f"sharpe={sharpe:.2f} max_dd={max_dd:.1f}%"
    )

    return {
        "ticker": ticker,
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(bm_total, 2),
        "alpha_pct": round(total_return - bm_total, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": int(stats.get("Total Trades", 0)),
        "avg_trade_duration": str(stats.get("Avg Winning Trade Duration", "N/A")),
        "equity_curve": equity_curve.reset_index().to_dict(orient="records"),
        "initial_cash": initial_cash,
        "final_value": round(float(portfolio.final_value()), 2),
    }


def signal_from_tft(
    df: pd.DataFrame,
    confidence_threshold: float = 0.65,
) -> tuple:
    """
    Generates entry/exit signals from TFT point forecast.
    Entry: predicted next-day return > 0 AND confidence > threshold
    Exit: predicted return < 0 OR RSI > 75
    """
    entries = (df["tft_forecast_return"] > 0) & (df["tft_confidence"] > confidence_threshold)
    exits = (df["tft_forecast_return"] < 0) | (df["rsi_14"] > 75)
    return entries.astype(bool), exits.astype(bool)


def signal_from_rsi_macd(df: pd.DataFrame) -> tuple:
    """Classic RSI + MACD crossover strategy for benchmarking."""
    entries = (df["rsi_14"] < 35) & (df["macd"] > df["macd_signal"])
    exits   = (df["rsi_14"] > 65) | (df["macd"] < df["macd_signal"])
    return entries.astype(bool), exits.astype(bool)
