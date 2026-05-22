"""Technical analysis agent — indicator confluence via the ReAct tool loop."""

from __future__ import annotations

from loguru import logger

from backend.ml.base_agent import BaseAgent, AgentStep, AgentResult


class TechnicalAgent(BaseAgent):
    """
    Analyses 20+ technical indicators and returns a directional signal.

    Uses the ``compute_confluence`` tool, which counts how many indicators
    align bullish vs bearish. The input DataFrame is expected to already carry
    indicator columns (the orchestrator passes a feature-engineered frame);
    ``compute_confluence_score`` degrades gracefully if any are missing.
    """

    max_turns = 2

    @property
    def name(self) -> str:
        return "technical"

    @property
    def tool_names(self) -> list[str]:
        return ["compute_confluence"]

    def _initial_thought(self, context: dict) -> str:
        return f"Assess technical indicator confluence for {context.get('ticker', '?')}."

    async def _decide_action(self, context: dict, steps: list[AgentStep]):
        # One tool call: compute the confluence score, then finalise.
        if not any(s.action == "compute_confluence" for s in steps):
            return "compute_confluence", {"df": context["df"]}
        return None, {}

    async def _interpret_observations(self, context: dict, steps: list[AgentStep]) -> AgentResult:
        confluence = self._get_observation(steps, "compute_confluence")

        if not isinstance(confluence, dict) or "raw_score" not in confluence:
            error = confluence.get("error") if isinstance(confluence, dict) else "no confluence result"
            logger.warning(f"TechnicalAgent: confluence unavailable ({error})")
            return AgentResult(
                agent_name=self.name,
                direction="neutral",
                confidence=0.25,
                raw_score=0.0,
                metadata={"error": str(error)},
            )

        df = context.get("df")
        key_indicators = {}
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            for col in ("rsi_14", "macd_hist", "adx_14", "bb_pct", "volume_ratio", "daily_return"):
                try:
                    val = latest[col]
                    key_indicators[col] = round(float(val), 4) if val == val else None
                except (KeyError, TypeError, ValueError):
                    key_indicators[col] = None

        metadata = {**confluence, "key_indicators": key_indicators}
        return AgentResult(
            agent_name=self.name,
            direction=confluence["direction"],
            confidence=float(confluence.get("confluence_score", 0.0)),
            raw_score=float(confluence["raw_score"]),
            metadata=metadata,
        )
