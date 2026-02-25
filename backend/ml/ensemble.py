"""Multi-agent ensemble — weighted voting across technical, sentiment, and macro agents."""

import asyncio
from typing import Optional
from loguru import logger
from dataclasses import dataclass
import numpy as np


@dataclass
class AgentSignal:
    agent_name: str
    direction: str          # "bullish" | "bearish" | "neutral"
    confidence: float       # 0.0 - 1.0
    weight: float           # agent's ensemble weight
    raw_score: float        # -1.0 to +1.0
    metadata: dict


async def run_technical_agent(ticker: str, df) -> AgentSignal:
    """
    Analyses 20+ technical indicators and returns a directional signal.
    Uses confluence scoring: counts how many indicators align.
    """
    from backend.ml.technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    result = await agent.analyze(ticker, df)
    return AgentSignal(
        agent_name="technical",
        direction=result["direction"],
        confidence=result["confluence_score"],
        weight=0.35,
        raw_score=result["raw_score"],
        metadata=result,
    )


async def run_sentiment_agent(ticker: str, news_window_hours: int = 48) -> AgentSignal:
    """
    Fetches latest news headlines for ticker via NewsAPI,
    scores with FinBERT, returns aggregated sentiment signal.
    """
    from backend.ml.sentiment_agent import SentimentAgent
    agent = SentimentAgent()
    result = await agent.analyze(ticker, news_window_hours)
    return AgentSignal(
        agent_name="sentiment",
        direction=result["label"],
        confidence=abs(result["composite_score"]),
        weight=0.35,
        raw_score=result["composite_score"],
        metadata=result,
    )


async def run_macro_agent(ticker: str) -> AgentSignal:
    """
    Pulls latest FRED macro indicators and computes a macro directional signal.
    """
    from backend.ml.macro_agent import MacroAgent
    agent = MacroAgent()
    result = await agent.analyze(ticker)
    return AgentSignal(
        agent_name="macro",
        direction=result["direction"],
        confidence=result["confidence"],
        weight=0.30,
        raw_score=result["raw_score"],
        metadata=result,
    )


async def ensemble_predict(ticker: str, df) -> dict:
    """
    Runs all 3 agents in parallel, aggregates with weighted voting,
    then optionally feeds combined signal into TFT for final forecast.

    Returns:
        {
          "ticker": str,
          "final_signal": "BUY" | "HOLD" | "SELL",
          "confidence": float (0-100),
          "agent_signals": {technical, sentiment, macro},
          "tft_forecast": dict or null,
          "risk_score": float (0-10),
          "explanation": str
        }
    """
    logger.info(f"Running ensemble prediction for {ticker}")

    # Run agents in parallel
    tech_signal, sent_signal, macro_signal = await asyncio.gather(
        run_technical_agent(ticker, df),
        run_sentiment_agent(ticker),
        run_macro_agent(ticker),
    )

    agents = [tech_signal, sent_signal, macro_signal]

    # Weighted average score
    total_weight = sum(a.weight for a in agents)
    weighted_score = sum(a.raw_score * a.weight for a in agents) / total_weight

    # Confidence = weighted average of individual confidences
    weighted_confidence = sum(a.confidence * a.weight for a in agents) / total_weight

    # Signal decision
    if weighted_score > 0.20:
        final_signal = "BUY"
    elif weighted_score < -0.20:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # Risk score: inverse of confidence, scaled by volatility proxy (std of scores)
    score_std = float(np.std([a.raw_score for a in agents]))
    risk_score = round(min(10.0, (1 - weighted_confidence) * 10 + score_std * 5), 2)

    # TFT price forecast (optional — only if model checkpoint exists)
    tft_result = None
    try:
        from backend.ml.tft_model import TFTPredictor
        from backend.config import get_settings
        import os
        settings = get_settings()
        if os.path.exists(settings.TFT_CHECKPOINT_PATH):
            predictor = TFTPredictor.get_instance(settings.TFT_CHECKPOINT_PATH)
            tft_result = predictor.predict(df, ticker)
    except Exception as e:
        logger.warning(f"TFT forecast unavailable: {e}")

    # Generate explanation
    explanation = _generate_explanation(
        ticker, final_signal, weighted_score,
        tech_signal, sent_signal, macro_signal, tft_result
    )

    return {
        "ticker": ticker,
        "final_signal": final_signal,
        "confidence": round(weighted_confidence * 100, 1),
        "weighted_score": round(weighted_score, 4),
        "agent_signals": {
            "technical": tech_signal.metadata,
            "sentiment": sent_signal.metadata,
            "macro": macro_signal.metadata,
        },
        "tft_forecast": tft_result,
        "risk_score": risk_score,
        "explanation": explanation,
    }


def _generate_explanation(
    ticker, signal, score,
    tech: AgentSignal, sent: AgentSignal, macro: AgentSignal,
    tft: Optional[dict],
) -> str:
    parts = [
        f"{ticker} shows a {signal} signal (score: {score:+.2f}).",
        f"Technical analysis is {tech.direction} (confluence: {tech.confidence:.0%}).",
        f"News sentiment is {sent.direction} with composite score {sent.raw_score:+.2f}.",
        f"Macro environment is {macro.direction}.",
    ]

    if tft and tft.get("point_forecasts"):
        point = tft["point_forecasts"][0]
        direction = "upside" if point > 0 else "downside"
        parts.append(f"TFT projects {abs(point*100):.1f}% {direction} in next trading session.")
    else:
        parts.append("TFT forecast unavailable — using agent signals only.")

    return " ".join(parts)
