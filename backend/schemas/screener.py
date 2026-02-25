"""Pydantic schemas for screener endpoints."""

from pydantic import BaseModel
from typing import Optional, List


class ScreenerFilters(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_pe: Optional[float] = None
    max_pe: Optional[float] = None
    min_dividend_yield: Optional[float] = None
    sector: Optional[str] = None
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    min_volume: Optional[int] = None
    trend: Optional[str] = None  # "bullish" | "bearish" | "neutral"
    min_beta: Optional[float] = None
    max_beta: Optional[float] = None
    above_sma_50: Optional[bool] = None


class ScreenerResult(BaseModel):
    ticker: str
    name: str
    price: float
    change_pct: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    rsi: Optional[float] = None
    sector: Optional[str] = None
    signal: Optional[str] = None


class ScreenerResponse(BaseModel):
    results: List[ScreenerResult]
    total: int
    filters_applied: dict
