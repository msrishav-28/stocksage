"""
Structured observability for StockSage agents.

Provides AgentSpan (context manager per agent) and TraceContext (per-prediction).
No external OTel dependency — pure Python dicts with structured loguru output.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AgentSpan:
    """
    Represents one agent's execution within a prediction trace.
    Use as context manager or call start()/finish() manually.
    """
    agent: str
    inputs: dict = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output: Optional[dict] = None
    error: Optional[str] = None

    def start(self) -> "AgentSpan":
        self.start_time = time.perf_counter()
        return self

    def finish(self, output: Optional[dict] = None, error: Optional[str] = None) -> "AgentSpan":
        self.end_time = time.perf_counter()
        self.output = output
        self.error = error
        return self

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time) * 1000, 1)
        return 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "span_id":    self.span_id,
            "agent":      self.agent,
            "duration_ms": self.duration_ms,
            "succeeded":  self.succeeded,
            "output":     self.output,
            "error":      self.error,
        }

    def __enter__(self) -> "AgentSpan":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.finish(error=str(exc_val) if exc_val else None)
        return False  # don't suppress exceptions


@dataclass
class TraceContext:
    """
    Per-prediction execution trace.
    Collects spans from all agents, the critic, and the synthesizer.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spans: list[AgentSpan] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_span(self, span: AgentSpan) -> None:
        self.spans.append(span)

    def new_span(self, agent: str, inputs: dict = None) -> AgentSpan:
        span = AgentSpan(agent=agent, inputs=inputs or {})
        self.spans.append(span)
        return span

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    @property
    def all_succeeded(self) -> bool:
        return all(s.succeeded for s in self.spans)

    @property
    def failed_agents(self) -> list[str]:
        return [s.agent for s in self.spans if not s.succeeded]

    def export(self) -> dict:
        """Export as JSON-serializable dict for DB storage and API response."""
        return {
            "trace_id":         self.trace_id,
            "ticker":           self.ticker,
            "started_at":       self.started_at,
            "total_duration_ms": self.total_duration_ms,
            "all_succeeded":    self.all_succeeded,
            "failed_agents":    self.failed_agents,
            "span_count":       len(self.spans),
            "spans":            [s.to_dict() for s in self.spans],
            "metadata":         self.metadata,
        }
