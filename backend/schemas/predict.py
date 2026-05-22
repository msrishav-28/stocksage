"""Pydantic schemas for prediction endpoints."""

from pydantic import BaseModel, field_validator
from typing import Optional


class PredictRequest(BaseModel):
    ticker: str
    period: str = "2y"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = (v or "").upper().strip()
        if not v:
            raise ValueError("ticker must not be empty")
        return v


class PredictResponse(BaseModel):
    ticker: str
    final_signal: str
    confidence: float
    weighted_score: float
    risk_score: float
    agent_signals: dict
    tft_forecast: Optional[dict] = None
    explanation: str
    thesis: str
    # Guardrail / observability metadata
    guardrail_flags: list = []
    guardrail_applied: bool = False
    override_reason: Optional[str] = None
    trace: Optional[dict] = None
