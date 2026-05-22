"""
Ensemble entry point.

This module is a thin, stable facade over the agent orchestrator. The
multi-agent voting, memory, telemetry, and guardrail logic all live in
``backend.ml.orchestrator``; ``ensemble_predict`` is kept as the public API
that routers and tests call.
"""

from __future__ import annotations

from loguru import logger

from backend.ml.orchestrator import get_orchestrator


async def ensemble_predict(ticker: str, df, news_window_hours: int = 48) -> dict:
    """
    Run the full orchestrated prediction pipeline for a ticker.

    Delegates to :class:`backend.ml.orchestrator.Orchestrator`, which runs the
    technical / sentiment / macro agents in parallel, synthesises an
    accuracy-weighted signal, applies guardrails, and records telemetry.

    Returns a dict with keys:
        ticker, final_signal, confidence, weighted_score, risk_score,
        agent_signals, tft_forecast, explanation, thesis,
        guardrail_flags, guardrail_applied, trace
    """
    logger.info(f"ensemble_predict: {ticker}")
    return await get_orchestrator().run(ticker, df, news_window_hours=news_window_hours)
