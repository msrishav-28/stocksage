"""
Base ReAct agent class.

All StockSage agents inherit from BaseAgent, which implements the
observe → reason → act → observe loop with automatic telemetry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger

from backend.ml.tools import ToolRegistry, ToolResult
from backend.ml.telemetry import TraceContext, AgentSpan


@dataclass
class AgentStep:
    """One turn of the ReAct loop."""
    turn: int
    thought: str              # agent's internal reasoning
    action: Optional[str]     # tool name called (None = final answer)
    action_input: dict        # tool kwargs
    observation: Any          # tool result or None


@dataclass
class AgentResult:
    """Structured output from a BaseAgent.run() call."""
    agent_name: str
    direction: str            # "bullish" | "bearish" | "neutral"
    confidence: float         # 0.0 – 1.0
    raw_score: float          # -1.0 to +1.0
    steps: list[AgentStep]    = field(default_factory=list)
    metadata: dict            = field(default_factory=dict)
    error: Optional[str]      = None
    turns_used: int           = 0


class BaseAgent(ABC):
    """
    Abstract base class implementing the ReAct loop.

    Subclasses must implement:
        - name: str property
        - tools: list[str] property  (names of tools this agent uses)
        - _initial_thought(context) → str
        - _decide_action(context, steps) → (action, action_input) | (None, {})
        - _interpret_observations(context, steps) → AgentResult
    """

    max_turns: int = 3
    registry: ToolRegistry = None

    def __init__(self):
        self.registry = ToolRegistry()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def tool_names(self) -> list[str]:
        """Override to declare which tools this agent can use."""
        return []

    async def run(
        self,
        context: dict,
        trace_ctx: Optional[TraceContext] = None,
    ) -> AgentResult:
        """
        Drive the ReAct loop.  Returns AgentResult.
        Never raises — on unexpected error returns a safe neutral AgentResult.
        """
        steps: list[AgentStep] = []

        span = AgentSpan(agent=self.name, inputs=context)
        if trace_ctx:
            trace_ctx.add_span(span)

        try:
            span.start()
            result = await self._react_loop(context, steps)
            result.steps = steps
            result.turns_used = len(steps)
            span.finish(output={"direction": result.direction, "raw_score": result.raw_score})
            return result

        except Exception as e:
            logger.error(f"{self.name} crashed: {type(e).__name__}: {e}")
            span.finish(error=str(e))
            return AgentResult(
                agent_name=self.name,
                direction="neutral",
                confidence=0.2,
                raw_score=0.0,
                error=f"{type(e).__name__}: {e}",
            )

    async def _react_loop(self, context: dict, steps: list[AgentStep]) -> AgentResult:
        """Core ReAct loop — up to max_turns turns before forcing final answer."""
        for turn in range(1, self.max_turns + 1):
            thought = self._initial_thought(context) if turn == 1 else self._continuation_thought(context, steps)

            action, action_input = await self._decide_action(context, steps)

            if action is None:
                # Agent decided it has enough info — emit final answer
                step = AgentStep(turn=turn, thought=thought, action=None, action_input={}, observation=None)
                steps.append(step)
                logger.debug(f"{self.name} reached final answer after {turn} turn(s)")
                break

            # Execute tool call
            tool_result: ToolResult = await self.registry.call(action, **action_input)
            observation = tool_result.output if tool_result.success else {"error": tool_result.error}

            step = AgentStep(
                turn=turn,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)
            logger.debug(
                f"{self.name} | turn {turn} | action={action} | "
                f"success={tool_result.success} | {tool_result.duration_ms:.0f}ms"
            )

        return await self._interpret_observations(context, steps)

    @abstractmethod
    def _initial_thought(self, context: dict) -> str:
        """Return first-turn reasoning string for this agent."""
        ...

    def _continuation_thought(self, context: dict, steps: list[AgentStep]) -> str:
        """Override for multi-turn reasoning. Default: simple continuation."""
        return f"Continuing analysis with {len(steps)} observations so far."

    @abstractmethod
    async def _decide_action(
        self,
        context: dict,
        steps: list[AgentStep],
    ) -> tuple[Optional[str], dict]:
        """
        Decide next action.
        Return (tool_name, kwargs) to call a tool, or (None, {}) for final answer.
        """
        ...

    @abstractmethod
    async def _interpret_observations(
        self,
        context: dict,
        steps: list[AgentStep],
    ) -> AgentResult:
        """
        After all steps, synthesise observations into AgentResult.
        This is the agent's "final answer" logic.
        """
        ...

    def _get_observation(self, steps: list[AgentStep], action: str) -> Any:
        """Helper: find the latest observation for a given tool action."""
        for step in reversed(steps):
            if step.action == action:
                return step.observation
        return None
