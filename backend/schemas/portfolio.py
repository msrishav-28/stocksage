"""Pydantic schemas for portfolio analysis endpoints."""

from pydantic import BaseModel, field_validator
from typing import List


class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: List[float] = []
    period: str = "1y"

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("At least 2 tickers required")
        if len(v) > 20:
            raise ValueError("Maximum 20 tickers allowed")
        return [t.upper().strip() for t in v]


class StockMetric(BaseModel):
    ticker: str
    annualized_return: float
    annualized_volatility: float
    sharpe: float


class PortfolioResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    portfolio_return_pct: float
    portfolio_volatility_pct: float
    sharpe_ratio: float
    stock_metrics: List[StockMetric]


class CorrelationResponse(BaseModel):
    tickers: List[str]
    correlation_matrix: dict
