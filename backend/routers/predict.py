"""Prediction router — orchestrated ensemble prediction with DB persistence."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.predict import PredictRequest, PredictResponse
from backend.ml.ensemble import ensemble_predict
from backend.ml.guardrails import InputGuardrail
from backend.data.price_fetcher import fetch_ohlcv
from backend.data.feature_engineer import build_feature_df
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()

_input_guardrail = InputGuardrail()


async def _persist_prediction(result: dict) -> None:
    """
    Write a completed prediction to the predictions table.
    Silently skips if the database is unavailable — a DB failure must never
    invalidate an otherwise-good prediction.
    """
    from backend.db.session import get_session
    from backend.db.models import Prediction

    try:
        tft = result.get("tft_forecast") or {}
        point_forecasts = tft.get("point_forecasts", []) or []

        record = Prediction(
            ticker=result["ticker"],
            final_signal=result["final_signal"],
            confidence=result["confidence"],
            weighted_score=result["weighted_score"],
            risk_score=result["risk_score"],
            thesis=result.get("thesis"),
            tft_point_d1=point_forecasts[0] if len(point_forecasts) > 0 else None,
            tft_point_d5=point_forecasts[4] if len(point_forecasts) > 4 else None,
            tft_point_d10=point_forecasts[9] if len(point_forecasts) > 9 else None,
            agent_technical=result["agent_signals"].get("technical"),
            agent_sentiment=result["agent_signals"].get("sentiment"),
            agent_macro=result["agent_signals"].get("macro"),
            tft_quantile_bands=tft.get("quantile_bands"),
        )

        async for session in get_session():
            if session is None:
                return  # DB unavailable — degrade silently
            session.add(record)
            await session.commit()
            logger.info(f"Prediction persisted for {result['ticker']} ({result['final_signal']})")
            return
    except Exception as e:
        logger.warning(f"Failed to persist prediction: {e}")


@router.post("/", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run the orchestrated ensemble prediction for a ticker."""
    ticker = req.ticker  # already normalised by the schema validator

    # Input guardrail — reject malformed/unsupported tickers before any fetch.
    gr = _input_guardrail.validate(ticker, period=req.period)
    if not gr.passed:
        raise HTTPException(status_code=422, detail=gr.message)

    cache_key = f"predict:{ticker}:{req.period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    # Fetch price data and build the feature frame.
    try:
        df = fetch_ohlcv(ticker, period=req.period)
        df = build_feature_df(df, ticker=ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Data fetch error for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider error")

    # Run the orchestrated ensemble.
    try:
        result = await ensemble_predict(ticker, df)
    except Exception as e:
        logger.error(f"Ensemble predict failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    await _persist_prediction(result)
    await set_cache(cache_key, result, ttl=300)
    return result


@router.get("/{ticker}/history")
async def get_prediction_history(ticker: str, limit: int = 20):
    """Retrieve historical predictions for a ticker from the database."""
    from backend.db.session import get_session
    from backend.db.models import Prediction
    from sqlalchemy import select

    ticker = ticker.upper().strip()
    limit = max(1, min(limit, 100))

    async for session in get_session():
        if session is None:
            raise HTTPException(status_code=503, detail="Database not available")

        try:
            stmt = (
                select(Prediction)
                .where(Prediction.ticker == ticker)
                .order_by(Prediction.predicted_at.desc())
                .limit(limit)
            )
            query = await session.execute(stmt)
            predictions = query.scalars().all()

            return {
                "ticker": ticker,
                "count": len(predictions),
                "predictions": [
                    {
                        "id": p.id,
                        "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
                        "final_signal": p.final_signal,
                        "confidence": p.confidence,
                        "weighted_score": p.weighted_score,
                        "risk_score": p.risk_score,
                        "thesis": p.thesis,
                        "tft_point_d1": p.tft_point_d1,
                        "tft_point_d5": p.tft_point_d5,
                        "tft_point_d10": p.tft_point_d10,
                    }
                    for p in predictions
                ],
            }
        except Exception as e:
            logger.error(f"Prediction history query failed for {ticker}: {e}")
            raise HTTPException(status_code=500, detail="Database query failed")

    raise HTTPException(status_code=503, detail="Database not available")
