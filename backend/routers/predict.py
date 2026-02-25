"""Prediction router — full ensemble prediction + TFT forecast."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.predict import PredictRequest, PredictResponse
from backend.ml.ensemble import ensemble_predict
from backend.data.price_fetcher import fetch_ohlcv
from backend.data.feature_engineer import build_feature_df
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.post("/", response_model=PredictResponse)
async def predict(req: PredictRequest):
    cache_key = f"predict:{req.ticker}:{req.period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        df = fetch_ohlcv(req.ticker, period=req.period)
        df = build_feature_df(df, ticker=req.ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Data fetch error for {req.ticker}: {e}")
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    try:
        result = await ensemble_predict(req.ticker, df)
    except Exception as e:
        logger.error(f"Ensemble predict failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    await set_cache(cache_key, result, ttl=300)
    return result


@router.get("/{ticker}/history")
async def get_prediction_history(ticker: str, limit: int = 20):
    """Retrieve historical predictions for a ticker from the database."""
    from backend.db.session import get_session
    from backend.db.models import Prediction
    from sqlalchemy import select

    async for session in get_session():
        if session is None:
            raise HTTPException(status_code=503, detail="Database not available")

        try:
            stmt = (
                select(Prediction)
                .where(Prediction.ticker == ticker.upper())
                .order_by(Prediction.predicted_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            predictions = result.scalars().all()

            return {
                "ticker": ticker.upper(),
                "count": len(predictions),
                "predictions": [
                    {
                        "id": p.id,
                        "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
                        "final_signal": p.final_signal,
                        "confidence": p.confidence,
                        "weighted_score": p.weighted_score,
                        "risk_score": p.risk_score,
                        "tft_point_d1": p.tft_point_d1,
                        "tft_point_d5": p.tft_point_d5,
                        "tft_point_d10": p.tft_point_d10,
                    }
                    for p in predictions
                ],
            }
        except Exception as e:
            logger.error(f"Failed to fetch prediction history: {e}")
            raise HTTPException(status_code=500, detail="Database query failed")
