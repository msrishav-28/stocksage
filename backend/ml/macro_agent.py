"""Macro analysis agent — derives a macro-environment signal from FRED data."""

from __future__ import annotations

from loguru import logger

from backend.ml.base_agent import BaseAgent, AgentStep, AgentResult


class MacroAgent(BaseAgent):
    """
    Pulls the latest FRED macro snapshot (Fed rate, yield curve, VIX, ...) and
    converts it into a directional macro signal via the ``fetch_macro`` tool.
    """

    max_turns = 2

    @property
    def name(self) -> str:
        return "macro"

    @property
    def tool_names(self) -> list[str]:
        return ["fetch_macro"]

    def _initial_thought(self, context: dict) -> str:
        return f"Evaluate the macro environment for {context.get('ticker', '?')}."

    async def _decide_action(self, context: dict, steps: list[AgentStep]):
        if not any(s.action == "fetch_macro" for s in steps):
            return "fetch_macro", {"sector": context.get("sector", "") or ""}
        return None, {}

    async def _interpret_observations(self, context: dict, steps: list[AgentStep]) -> AgentResult:
        macro_obs = self._get_observation(steps, "fetch_macro")

        score = macro_obs.get("score") if isinstance(macro_obs, dict) else None
        if not isinstance(score, dict) or "raw_score" not in score:
            error = macro_obs.get("error") if isinstance(macro_obs, dict) else "no macro result"
            logger.warning(f"MacroAgent: macro data unavailable ({error})")
            return AgentResult(
                agent_name=self.name,
                direction="neutral",
                confidence=0.3,
                raw_score=0.0,
                metadata={"error": str(error)},
            )

        return AgentResult(
            agent_name=self.name,
            direction=score["direction"],
            confidence=float(score.get("confidence", 0.3)),
            raw_score=float(score["raw_score"]),
            metadata={
                "direction": score["direction"],
                "raw_score": score["raw_score"],
                "confidence": score.get("confidence", 0.3),
                "reasons": score.get("reasons", []),
                "sector": context.get("sector", "Unknown") or "Unknown",
                "snapshot": score.get("snapshot", {}),
            },
        )
