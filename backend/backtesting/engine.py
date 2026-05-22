"""Backtesting engine — vectorbt wrapper for running strategy backtests."""

import pandas as pd
from loguru import logger

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not installed. Backtesting disabled.")


def _serialize_equity_curve(equity_df: pd.DataFrame) -> list:
    """
    Convert the equity curve DataFrame to a JSON-serializable list of dicts.
    Converts pandas.Timestamp index entries to ISO-8601 strings — orjson
    cannot serialize Timestamp objects and FastAPI will throw TypeError.
    """
    records = equity_df.reset_index()
    # Rename datetime index column (could be 'index', 'Date', etc.)
    if records.columns[0] not in ("date", "timestamp"):
        records = records.rename(columns={records.columns[0]: "date"})
    # Convert any Timestamp / datetime column to ISO string
    for col in records.columns:
        if pd.api.types.is_datetime64_any_dtype(records[col]):
            records[col] = records[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif records[col].dtype == object:
            try:
                records[col] = pd.to_datetime(records[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                pass
    return records.to_dict(orient="records")


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
    Returns comprehensive performance metrics with JSON-serializable equity curve.
    """
    if not VBT_AVAILABLE:
        raise RuntimeError("vectorbt is not installed. Install with: pip install vectorbt")

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

    sharpe       = float(stats.get("Sharpe Ratio", 0) or 0)
    max_dd       = float(stats.get("Max Drawdown [%]", 0) or 0)
    total_return = float(stats.get("Total Return [%]", 0) or 0)
    win_rate     = float(stats.get("Win Rate [%]", 0) or 0)
    total_trades = int(stats.get("Total Trades", 0) or 0)

    bm_total = float(benchmark_returns.add(1).prod() - 1) * 100

    equity_df = portfolio.value().rename("portfolio").to_frame()
    equity_df["benchmark"] = initial_cash * benchmark_returns.add(1).cumprod()

    # Serialize equity curve — converts Timestamps to ISO strings to avoid JSON TypeError
    equity_curve_serializable = _serialize_equity_curve(equity_df)

    logger.success(
        f"Backtest {ticker}: return={total_return:.1f}% "
        f"sharpe={sharpe:.2f} max_dd={max_dd:.1f}%"
    )

    avg_duration = stats.get("Avg Winning Trade Duration")
    avg_duration_str = str(avg_duration) if avg_duration is not None else "N/A"

    return {
        "ticker":                ticker,
        "total_return_pct":      round(total_return, 2),
        "benchmark_return_pct":  round(bm_total, 2),
        "alpha_pct":             round(total_return - bm_total, 2),
        "sharpe_ratio":          round(sharpe, 3),
        "max_drawdown_pct":      round(max_dd, 2),
        "win_rate_pct":          round(win_rate, 2),
        "total_trades":          total_trades,
        "avg_trade_duration":    avg_duration_str,
        "equity_curve":          equity_curve_serializable,
        "initial_cash":          initial_cash,
        "final_value":           round(float(portfolio.final_value()), 2),
    }
