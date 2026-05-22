"""
Tool-use framework for StockSage agents.

Agents declare tools as typed callables. The ToolRegistry manages
registration and safe invocation, enabling ReAct-loop tool calling.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from loguru import logger


@dataclass
class ToolSchema:
    """JSON-Schema-like description of a tool's parameters."""
    name: str
    description: str
    parameters: dict  # JSON Schema object


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Tool:
    """
    A single agent tool.  fn may be sync or async.
    schema describes the callable's inputs for prompt injection.
    """
    name: str
    description: str
    fn: Callable
    schema: ToolSchema = field(default=None)

    def __post_init__(self):
        if self.schema is None:
            self.schema = ToolSchema(
                name=self.name,
                description=self.description,
                parameters={"type": "object", "properties": {}, "required": []},
            )


class ToolRegistry:
    """
    Singleton registry — agents look up tools by name and call them safely.
    Supports both sync and async callables.
    """
    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, Tool]

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def call(self, name: str, **kwargs) -> ToolResult:
        """
        Calls a registered tool by name.  Handles sync and async fns.
        Never raises — on error returns ToolResult(success=False).
        """
        import time
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' not registered. Available: {self.list_tools()}",
            )

        t0 = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(tool.fn):
                result = await tool.fn(**kwargs)
            else:
                # Run sync function in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.fn(**kwargs))

            duration_ms = (time.perf_counter() - t0) * 1000
            logger.debug(f"Tool '{name}' completed in {duration_ms:.1f}ms")
            return ToolResult(tool_name=name, success=True, output=result, duration_ms=duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.warning(f"Tool '{name}' failed: {type(e).__name__}: {e}")
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration_ms,
            )


# ── Built-in tool registrations ───────────────────────────────────────────────

def _register_builtin_tools():
    """Register all built-in StockSage tools on startup."""
    registry = ToolRegistry()

    # ── Price data ────────────────────
    def _fetch_ohlcv(ticker: str, period: str = "1y") -> Any:
        from backend.data.price_fetcher import fetch_ohlcv
        return fetch_ohlcv(ticker, period=period)

    registry.register(Tool(
        name="fetch_ohlcv",
        description="Fetch OHLCV price data for a ticker from yfinance",
        fn=_fetch_ohlcv,
        schema=ToolSchema(
            name="fetch_ohlcv",
            description="Fetch OHLCV price data",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {"type": "string", "default": "1y"},
                },
                "required": ["ticker"],
            },
        ),
    ))

    # ── Indicators ────────────────────
    def _compute_indicators(df: Any) -> Any:
        from backend.indicators import compute_all_indicators
        return compute_all_indicators(df)

    registry.register(Tool(
        name="compute_indicators",
        description="Compute 20+ technical indicators on an OHLCV DataFrame",
        fn=_compute_indicators,
        schema=ToolSchema(
            name="compute_indicators",
            description="Compute technical indicators",
            parameters={"type": "object", "properties": {"df": {"type": "object"}}, "required": ["df"]},
        ),
    ))

    # ── News ──────────────────────────
    async def _fetch_news(ticker: str, hours: int = 48) -> list:
        from backend.data.news_fetcher import fetch_news
        return await fetch_news(ticker, hours=hours)

    registry.register(Tool(
        name="fetch_news",
        description="Fetch recent news headlines for a ticker from NewsAPI",
        fn=_fetch_news,
        schema=ToolSchema(
            name="fetch_news",
            description="Fetch news headlines",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "hours": {"type": "integer", "default": 48},
                },
                "required": ["ticker"],
            },
        ),
    ))

    # ── FinBERT ───────────────────────
    def _finbert_score(headlines: list) -> list:
        from backend.ml.finbert_sentiment import score_batch
        return score_batch(headlines)

    registry.register(Tool(
        name="finbert_score",
        description="Score a list of headlines with FinBERT sentiment model",
        fn=_finbert_score,
        schema=ToolSchema(
            name="finbert_score",
            description="FinBERT sentiment scoring",
            parameters={
                "type": "object",
                "properties": {"headlines": {"type": "array", "items": {"type": "string"}}},
                "required": ["headlines"],
            },
        ),
    ))

    # ── Macro ─────────────────────────
    def _fetch_macro(sector: str = "") -> dict:
        from backend.data.macro_fetcher import fetch_macro_snapshot, compute_macro_score
        snap = fetch_macro_snapshot()
        score = compute_macro_score(snap, sector=sector)
        return {"snapshot": snap, "score": score}

    registry.register(Tool(
        name="fetch_macro",
        description="Fetch FRED macro data and compute macro environment score",
        fn=_fetch_macro,
        schema=ToolSchema(
            name="fetch_macro",
            description="Fetch macro data",
            parameters={
                "type": "object",
                "properties": {"sector": {"type": "string", "default": ""}},
                "required": [],
            },
        ),
    ))

    # ── Confluence score ──────────────
    def _compute_confluence(df: Any) -> dict:
        from backend.indicators import compute_confluence_score
        return compute_confluence_score(df)

    registry.register(Tool(
        name="compute_confluence",
        description="Compute indicator confluence directional score",
        fn=_compute_confluence,
        schema=ToolSchema(
            name="compute_confluence",
            description="Compute confluence score",
            parameters={"type": "object", "properties": {"df": {"type": "object"}}, "required": ["df"]},
        ),
    ))

    logger.info(f"ToolRegistry: {len(registry.list_tools())} tools registered: {registry.list_tools()}")


# Auto-register on module import
_register_builtin_tools()
