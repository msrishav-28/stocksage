"""Pydantic schemas for sentiment endpoints."""

from pydantic import BaseModel
from typing import List, Optional


class HeadlineScore(BaseModel):
    headline: str
    label: str
    score: float
    probabilities: Optional[dict] = None


class SentimentResponse(BaseModel):
    ticker: str
    composite_score: float
    label: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_articles: int
    sentiment_momentum: float
    headlines: List[HeadlineScore] = []
