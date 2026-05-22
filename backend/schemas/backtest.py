"""Pydantic schemas for backtesting endpoints."""

from pydantic import BaseModel, Field, field_validator
from typing import List


class BacktestRequest(BaseModel):
    ticker: str
    strategy: str = "rsi_macd"
    period: str = "2y"
    initial_cash: float = Field(default=10_000.0, gt=0)
    fees: float = Field(default=0.001, ge=0, le=0.1)
    slippage: float = Field(default=0.001, ge=0, le=0.1)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = (v or "").upper().strip()
        if not v:
            raise ValueError("ticker must not be empty")
        return v


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
