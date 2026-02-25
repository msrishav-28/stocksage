"""Pydantic schemas for prediction endpoints."""

from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    ticker: str
    period: str = "2y"


class PredictResponse(BaseModel):
    ticker: str
    final_signal: str
    confidence: float
    weighted_score: float
    agent_signals: dict
    tft_forecast: Optional[dict] = None
    risk_score: float
    explanation: str
