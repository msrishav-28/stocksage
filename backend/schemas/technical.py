"""Pydantic schemas for technical analysis endpoints."""

from pydantic import BaseModel


class TechnicalRequest(BaseModel):
    ticker: str
    period: str = "1y"


class TechnicalResponse(BaseModel):
    ticker: str
    period: str
    confluence: dict
    indicators: dict
    price: dict
    history: list = []
