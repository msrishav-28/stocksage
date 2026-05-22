"""Pydantic schemas for macro economy endpoints."""

from pydantic import BaseModel


class MacroSnapshotResponse(BaseModel):
    snapshot: dict
    macro_score: dict


class SectorAnalysisResponse(BaseModel):
    sector: str
    etf: str
    price: float
    return_1mo_pct: float
    return_3mo_pct: float
    macro_score: dict
