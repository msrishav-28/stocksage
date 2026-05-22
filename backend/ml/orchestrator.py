"""
Prediction orchestrator — the DAG that ties the StockSage agent system together.

Flow:
    1. trace          : open a TraceContext for observability
    2. memory recall  : pull episodic history + semantic priors for the ticker
    3. agents         : run technical / sentiment / macro agents in parallel
    4. synthesis      : accuracy-weighted vote, memory-adjusted, VIX-dampened
    5. TFT forecast   : optional multi-horizon price projection
    6. output guard   : enforce confidence / risk overrides
    7. memory write   : append the result to episodic memory

The orchestrator never raises for a recoverable failure — individual agent
crashes degrade to neutral signals so a partial answer is always returned.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import numpy as np
from loguru import logger

from backend.ml.base_agent import AgentResult
from backend.ml.technical_agent import TechnicalAgent
from backend.ml.sentiment_agent import SentimentAgent
from backend.ml.macro_agent import MacroAgent
from backend.ml.telemetry import TraceContext
from backend.ml.memory import WorkingMemory, EpisodicMemory, SemanticMemory
from backend.ml.guardrails import OutputGuardrail


# Base ensemble weights before accuracy modulation.
_BASE_WEIGHTS = {"technical": 0.35, "sentiment": 0.35, "macro": 0.30}

# Score thresholds for the final BUY / HOLD / SELL decision.
_BUY_THRESHOLD = 0.15
_SELL_THRESHOLD = -0.15

# How much weight the historical episodic prior carries in the blended score.
_PRIOR_BLEND = 0.15


class Orchestrator:
    """Coordinates the agent ensemble for a single prediction request."""

    def __init__(self) -> None:
        self.episodic = EpisodicMemory()
        self.output_guardrail = OutputGuardrail(min_confidence=0.30)

    async def run(self, ticker: str, df, news_window_hours: int = 48) -> dict:
        ticker = ticker.upper().strip()
        logger.info(f"Orchestrator: starting prediction for {ticker}")

        trace = TraceContext(ticker=ticker)
        working = WorkingMemory()

        # ── 1. Memory recall ──────────────────────────────────────────────────
        history = await self.episodic.retrieve(ticker, n=10)
        prior_bias = self.episodic.compute_prior_bias(history)
        accuracy_weight = self.episodic.compute_accuracy_weight(history)
        working.set("prior_bias", prior_bias)
        working.set("history_count", len(history))

        # ── 2. Run agents in parallel ─────────────────────────────────────────
        sector = self._infer_sector(df)
        tech_result, sent_result, macro_result = await asyncio.gather(
            TechnicalAgent().run({"ticker": ticker, "df": df}, trace),
            SentimentAgent().run({"ticker": ticker, "hours": news_window_hours}, trace),
            MacroAgent().run({"ticker": ticker, "sector": sector}, trace),
        )
        results = {"technical": tech_result, "sentiment": sent_result, "macro": macro_result}
        for name, r in results.items():
            working.set(f"agent:{name}", r)

        # ── 3. Synthesis ──────────────────────────────────────────────────────
        synthesis_span = trace.new_span("synthesizer").start()
        synthesis = self._synthesize(results, prior_bias, accuracy_weight)
        synthesis_span.finish(output={
            "signal": synthesis["final_signal"],
            "score": synthesis["weighted_score"],
        })

        # ── 4. TFT forecast (optional) ────────────────────────────────────────
        tft_forecast = self._tft_forecast(ticker, df)

        # ── 5. Assemble result ────────────────────────────────────────────────
        agent_signals = {name: self._agent_metadata(r) for name, r in results.items()}

        result = {
            "ticker": ticker,
            "final_signal": synthesis["final_signal"],
            "confidence": synthesis["confidence"],
            "weighted_score": synthesis["weighted_score"],
            "risk_score": synthesis["risk_score"],
            "agent_signals": agent_signals,
            "tft_forecast": tft_forecast,
            # explanation/thesis are filled in after guardrails so the narrative
            # always reflects the FINAL (possibly overridden) signal.
            "explanation": "",
            "thesis": "",
        }

        # ── 6. Output guardrail ───────────────────────────────────────────────
        gr = self.output_guardrail.validate(result)
        if not gr.passed:
            logger.error(f"Orchestrator: output guardrail rejected {ticker}: {gr.message}")
        result = self.output_guardrail.apply(result, gr)

        # ── 7. Narrative — built from the final, post-guardrail signal ────────
        final_signal = result["final_signal"]
        override_reason = result.get("override_reason")
        result["explanation"] = self._build_explanation(
            ticker, synthesis, final_signal, results, tft_forecast
        )
        result["thesis"] = self._build_thesis(
            ticker, synthesis, final_signal, results, tft_forecast, working, override_reason
        )

        # ── 8. Episodic memory write ──────────────────────────────────────────
        await self.episodic.store(ticker, {
            "final_signal": result["final_signal"],
            "confidence": result["confidence"],
            "weighted_score": result["weighted_score"],
            "risk_score": result["risk_score"],
        })

        # ── 9. Trace export ───────────────────────────────────────────────────
        trace.metadata["history_count"] = len(history)
        trace.metadata["prior_bias"] = prior_bias
        result["trace"] = trace.export()
        logger.info(
            f"Orchestrator: {ticker} -> {result['final_signal']} "
            f"({result['confidence']:.0f}% conf, {trace.total_duration_ms:.0f}ms)"
        )
        return result

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, results: dict, prior_bias: float, accuracy_weight: float) -> dict:
        """Accuracy-weighted vote, blended with episodic prior and VIX-dampened."""
        # Effective weight = base weight * the agent's historical accuracy prior.
        eff_weights = {
            name: _BASE_WEIGHTS[name] * SemanticMemory.get_agent_accuracy_prior(name)
            for name in results
        }
        total_w = sum(eff_weights.values()) or 1.0

        weighted_score = sum(results[n].raw_score * eff_weights[n] for n in results) / total_w
        weighted_conf = sum(results[n].confidence * eff_weights[n] for n in results) / total_w

        # Blend in the historical episodic prior bias.
        blended_score = (1 - _PRIOR_BLEND) * weighted_score + _PRIOR_BLEND * prior_bias

        # Dampen confidence by market-volatility reliability (VIX regime).
        # When VIX data is unavailable we do NOT penalise — absence of data is
        # not the same as an unreliable (high-volatility) regime.
        vix = self._extract_vix(results.get("macro"))
        vix_reliability = (
            SemanticMemory.get_vix_reliability(vix)["reliability"] if vix is not None else 1.0
        )
        confidence_frac = max(0.0, min(1.0, weighted_conf * vix_reliability * accuracy_weight))

        if blended_score > _BUY_THRESHOLD:
            final_signal = "BUY"
        elif blended_score < _SELL_THRESHOLD:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        # Risk: low confidence + agent disagreement + volatility regime.
        score_std = float(np.std([results[n].raw_score for n in results]))
        risk_score = round(
            min(10.0, (1 - confidence_frac) * 7 + score_std * 5 + (1 - vix_reliability) * 3),
            2,
        )

        return {
            "final_signal": final_signal,
            "confidence": round(confidence_frac * 100, 1),
            "weighted_score": round(blended_score, 4),
            "risk_score": risk_score,
            "vix_reliability": round(vix_reliability, 2),
        }

    @staticmethod
    def _extract_vix(macro_result: Optional[AgentResult]) -> Optional[float]:
        if macro_result is None:
            return None
        snapshot = (macro_result.metadata or {}).get("snapshot", {})
        vix = snapshot.get("vix") if isinstance(snapshot, dict) else None
        try:
            return float(vix) if vix is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _agent_metadata(result: AgentResult) -> dict:
        """Flatten an AgentResult into the agent_signals payload, surfacing errors."""
        meta = dict(result.metadata) if result.metadata else {}
        meta.setdefault("direction", result.direction)
        meta.setdefault("raw_score", result.raw_score)
        meta.setdefault("confidence", result.confidence)
        if result.error:
            meta["error"] = result.error
        meta["turns_used"] = result.turns_used
        return meta

    @staticmethod
    def _infer_sector(df) -> str:
        try:
            if df is not None and "sector" in df.columns and len(df) > 0:
                return str(df["sector"].iloc[-1])
        except Exception:
            pass
        return "Unknown"

    # ── TFT ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _tft_forecast(ticker: str, df) -> Optional[dict]:
        try:
            from backend.config import get_settings
            settings = get_settings()

            # Check the checkpoint exists BEFORE importing tft_model — that
            # import pulls in torch (~seconds), which is pure waste when there
            # is no trained model to serve.
            if not os.path.exists(settings.TFT_CHECKPOINT_PATH):
                logger.debug("TFT checkpoint not found — skipping forecast.")
                return None

            from backend.ml.tft_model import TFTPredictor, TFT_AVAILABLE
            if not TFT_AVAILABLE:
                return None
            predictor = TFTPredictor.get_instance(settings.TFT_CHECKPOINT_PATH)
            return predictor.predict(df, ticker)
        except Exception as e:
            logger.warning(f"TFT forecast unavailable for {ticker}: {e}")
            return None

    # ── Narrative ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_explanation(ticker: str, synthesis: dict, final_signal: str,
                           results: dict, tft: Optional[dict]) -> str:
        tech, sent, macro = results["technical"], results["sentiment"], results["macro"]
        parts = [
            f"{ticker}: {final_signal} signal "
            f"(score {synthesis['weighted_score']:+.2f}, {synthesis['confidence']:.0f}% confidence).",
            f"Technical {tech.direction}, news sentiment {sent.direction}, macro {macro.direction}.",
        ]
        if tft and tft.get("point_forecasts"):
            point = tft["point_forecasts"][0]
            parts.append(f"TFT projects {abs(point * 100):.1f}% "
                         f"{'upside' if point > 0 else 'downside'} next session.")
        return " ".join(parts)

    @staticmethod
    def _build_thesis(ticker: str, synthesis: dict, final_signal: str, results: dict,
                      tft: Optional[dict], working: WorkingMemory,
                      override_reason: Optional[str] = None) -> str:
        """A longer multi-sentence narrative for the user-facing report."""
        tech, sent, macro = results["technical"], results["sentiment"], results["macro"]
        signal = final_signal

        lines = [
            f"StockSage rates {ticker} a {signal} with "
            f"{synthesis['confidence']:.0f}% confidence and a risk score of "
            f"{synthesis['risk_score']}/10."
        ]

        # Technical
        tech_meta = tech.metadata or {}
        if "error" in tech_meta:
            lines.append("Technical analysis was unavailable for this run.")
        else:
            lines.append(
                f"Technically the trend reads {tech.direction}: "
                f"{tech_meta.get('bullish_signals', 0)} bullish vs "
                f"{tech_meta.get('bearish_signals', 0)} bearish indicators aligned."
            )

        # Sentiment
        sent_meta = sent.metadata or {}
        total = sent_meta.get("total_articles", 0)
        if total:
            lines.append(
                f"News sentiment is {sent.direction} across {total} recent articles "
                f"({sent_meta.get('bullish_count', 0)} positive, "
                f"{sent_meta.get('bearish_count', 0)} negative)."
            )
        else:
            lines.append("No recent news coverage was found to score sentiment.")

        # Macro
        macro_meta = macro.metadata or {}
        reasons = macro_meta.get("reasons") or []
        if reasons:
            lines.append(f"Macro backdrop is {macro.direction}: {reasons[0].lower()}.")
        else:
            lines.append(f"Macro backdrop is {macro.direction}.")

        # Memory context
        history_count = working.get("history_count", 0)
        if history_count:
            prior = working.get("prior_bias", 0.0)
            tilt = "bullish" if prior > 0.1 else ("bearish" if prior < -0.1 else "balanced")
            lines.append(
                f"Across the last {history_count} StockSage runs the prior signal "
                f"history has been {tilt}."
            )

        # TFT
        if tft and tft.get("point_forecasts"):
            point = tft["point_forecasts"][0]
            lines.append(
                f"The Temporal Fusion Transformer projects roughly "
                f"{abs(point * 100):.1f}% {'upside' if point > 0 else 'downside'} "
                f"over the next session."
            )

        # Guardrail override note — explains why the headline signal was adjusted.
        if override_reason:
            lines.append(f"Risk note: {override_reason}.")

        return " ".join(lines)


# Module-level singleton — agents and memory are cheap to keep warm.
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
