"""Pydantic schemas for backtesting endpoints."""

from pydantic import BaseModel
from typing import Optional, List


class BacktestRequest(BaseModel):
    ticker: str
    strategy: str = "rsi_macd"
    period: str = "2y"
    initial_cash: float = 10_000.0
    fees: float = 0.001
    slippage: float = 0.001


class BacktestResponse(BaseModel):
    ticker: str
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    avg_trade_duration: str
    equity_curve: List[dict]
    initial_cash: float
    final_value: float
