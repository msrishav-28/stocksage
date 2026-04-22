"""Unified analyze endpoint for frontend compatibility."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from backend.data.price_fetcher import fetch_ohlcv, fetch_ticker_info
from backend.data.feature_engineer import build_feature_df
from backend.indicators import compute_all_indicators
from backend.ml.ensemble import ensemble_predict
from backend.routers.competitor import INDUSTRY_PEERS

router = APIRouter()


class AnalyzeRequest(BaseModel):
    company_name: str = ""
    ticker: str = ""
    period: str = "1y"


def _risk_level(score: float) -> str:
    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def _recommendation(signal: str, confidence: float) -> dict:
    mapping = {
        "BUY": ("BUY", "Bullish setup from multi-agent analysis."),
        "SELL": ("SELL/AVOID", "Bearish setup from multi-agent analysis."),
        "HOLD": ("HOLD", "Mixed signals; waiting for confirmation."),
    }
    action, summary = mapping.get(signal, ("HOLD", "No strong edge detected."))
    return {
        "action": action,
        "summary": summary,
        "details": f"Model confidence is {confidence:.1f}%.",
    }


def _sentiment_fields(sentiment_payload: dict) -> tuple[str, float, list[str]]:
    label = (sentiment_payload.get("label") or "neutral").lower()
    mapped = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}
    sentiment = mapped.get(label, "NEUTRAL")
    confidence = abs(float(sentiment_payload.get("composite_score", 0.0))) * 100
    reasons = [
        f"{h.get('label', 'neutral').title()}: {h.get('headline', '')}"
        for h in sentiment_payload.get("headlines", [])[:3]
        if h.get("headline")
    ]
    if not reasons:
        reasons = ["No recent news signal available."]
    return sentiment, round(confidence, 1), reasons


def _peer_payload(ticker: str, sector: str) -> list[dict]:
    peers = [p for p in INDUSTRY_PEERS.get(sector, INDUSTRY_PEERS["Technology"]) if p != ticker][:3]
    result = []
    for peer in peers:
        try:
            peer_info = fetch_ticker_info(peer)
            peer_df = fetch_ohlcv(peer, period="3mo")
            if peer_df.empty:
                continue
            recent = peer_df.tail(60)
            result.append(
                {
                    "name": peer_info.get("name", peer),
                    "ticker": peer,
                    "stock_price": round(float(recent["close"].iloc[-1]), 2),
                    "stock_prices": [round(float(v), 2) for v in recent["close"].tolist()],
                    "time_labels": [idx.strftime("%Y-%m-%d") for idx in recent.index],
                }
            )
        except Exception as e:
            logger.warning(f"Peer fetch failed for {peer}: {e}")
    return result


@router.post("")
async def analyze(req: AnalyzeRequest):
    ticker = (req.ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        df = fetch_ohlcv(ticker, period=req.period)
        feature_df = build_feature_df(df, ticker=ticker)
        technical_df = compute_all_indicators(df)
        prediction = await ensemble_predict(ticker, feature_df)
        info = fetch_ticker_info(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Analyze failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

    recent = df.tail(60)
    latest = technical_df.iloc[-1]
    current_price = float(recent["close"].iloc[-1])
    previous_close = float(recent["close"].iloc[-2]) if len(recent) > 1 else current_price
    price_change = current_price - previous_close
    change_percent = (price_change / previous_close * 100) if previous_close else 0.0

    tft = prediction.get("tft_forecast")
    if tft and tft.get("point_forecasts"):
        predicted_price = round(current_price * (1 + float(tft["point_forecasts"][0])), 2)
    else:
        predicted_price = round(current_price, 2)

    sentiment_payload = prediction.get("agent_signals", {}).get("sentiment", {})
    sentiment, sentiment_conf, sentiment_factors = _sentiment_fields(sentiment_payload)
    risk_score = float(prediction.get("risk_score", 5.0))
    recommendation = _recommendation(prediction.get("final_signal", "HOLD"), float(prediction.get("confidence", 50.0)))

    week_high = float(technical_df["high"].tail(252).max())
    week_low = float(technical_df["low"].tail(252).min())
    position_52w = ((current_price - week_low) / (week_high - week_low) * 100) if week_high > week_low else 50.0

    return {
        "success": True,
        "ticker": ticker,
        "company_name": req.company_name or info.get("long_name") or info.get("name") or ticker,
        "description": info.get("description") or f"{info.get('name', ticker)} ({info.get('industry', 'Unknown industry')})",
        "current_price": round(current_price, 2),
        "price_change": round(price_change, 2),
        "change_percent": round(change_percent, 2),
        "predicted_price": predicted_price,
        "prediction_confidence": round(float(prediction.get("confidence", 50.0)), 1),
        "stock_prices": [round(float(v), 2) for v in recent["close"].tolist()],
        "time_labels": [idx.strftime("%Y-%m-%d") for idx in recent.index],
        "volumes": [int(v) for v in recent["volume"].tolist()],
        "top_competitors": _peer_payload(ticker, info.get("sector", "Technology")),
        "ai_analysis": {
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_conf,
            "sentiment_factors": sentiment_factors,
            "sentiment_score": round(float(sentiment_payload.get("composite_score", 0.0)), 4),
            "risk_level": _risk_level(risk_score),
            "risk_score": round(risk_score, 2),
            "risk_factors": [f"Ensemble risk score: {round(risk_score, 2)}/10"],
            "technical_indicators": {
                "rsi": round(float(latest.get("rsi_14", 50.0)), 2),
                "ma_20": round(float(latest.get("sma_20", current_price)), 2),
                "ma_50": round(float(latest.get("sma_50", current_price)), 2),
                "position_52w": round(float(position_52w), 2),
            },
            "eli5_explanation": prediction.get("explanation", ""),
            "recommendation": recommendation,
            "market_cap": info.get("market_cap"),
            "pe_ratio": info.get("pe_ratio"),
            "dividend_yield": info.get("dividend_yield"),
            "week_52_high": round(week_high, 2),
            "week_52_low": round(week_low, 2),
        },
    }
