"""
Three-tier memory system for StockSage agents.

- WorkingMemory  : per-request in-process scratchpad (k/v store)
- EpisodicMemory : Redis-backed, last N signals per ticker
- SemanticMemory : static market priors and accuracy statistics
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from loguru import logger


# ── Working Memory ────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Short-lived, per-request scratchpad.
    Created fresh for each OrchestratorDAG.run() call.
    """
    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def snapshot(self) -> dict:
        return dict(self._store)


# ── Episodic Memory ───────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Redis-backed episodic memory.
    Stores last N prediction signals per ticker as a capped list.
    Key pattern: episodic:{ticker}
    """

    def __init__(self, max_entries: int = 30, ttl_seconds: int = 60 * 60 * 24 * 90):
        self.max_entries = max_entries
        self.ttl = ttl_seconds

    def _key(self, ticker: str) -> str:
        return f"episodic:{ticker.upper()}"

    async def store(self, ticker: str, record: dict) -> None:
        """
        Append a prediction record to the ticker's episodic memory.
        Trims to max_entries. Silently skips if Redis is unavailable.
        """
        try:
            from backend.cache.redis_client import get_redis
            client = get_redis()
            if client is None:
                return

            key = self._key(ticker)
            serialized = json.dumps({**record, "stored_at": datetime.now(timezone.utc).isoformat()})
            await client.rpush(key, serialized)
            await client.ltrim(key, -self.max_entries, -1)
            await client.expire(key, self.ttl)
            logger.debug(f"Episodic memory stored for {ticker}")
        except Exception as e:
            logger.warning(f"Episodic memory write failed for {ticker}: {e}")

    async def retrieve(self, ticker: str, n: int = 10) -> list[dict]:
        """
        Retrieve the last n episodic records for ticker.
        Returns [] when Redis is unavailable or no history exists.
        """
        try:
            from backend.cache.redis_client import get_redis
            client = get_redis()
            if client is None:
                return []

            key = self._key(ticker)
            raw_items = await client.lrange(key, -n, -1)
            records = [json.loads(item) for item in raw_items]
            logger.debug(f"Episodic memory retrieved {len(records)} records for {ticker}")
            return records
        except Exception as e:
            logger.warning(f"Episodic memory read failed for {ticker}: {e}")
            return []

    def compute_prior_bias(self, records: list[dict]) -> float:
        """
        Compute a directional prior bias from historical signals.
        Returns float in [-1, +1]: positive = historically bullish bias.
        """
        if not records:
            return 0.0
        score_map = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
        scores = [score_map.get(r.get("final_signal", "HOLD"), 0.0) for r in records]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def compute_accuracy_weight(self, records: list[dict]) -> float:
        """
        If outcomes are tracked, compute accuracy-weighted confidence multiplier.
        Returns 1.0 when no outcome data is available.
        """
        outcomes = [r for r in records if r.get("outcome") in ("correct", "incorrect")]
        if not outcomes:
            return 1.0
        correct = sum(1 for r in outcomes if r.get("outcome") == "correct")
        accuracy = correct / len(outcomes)
        # Map accuracy [0.0, 1.0] to weight [0.6, 1.4]
        return round(0.6 + accuracy * 0.8, 3)


# ── Semantic Memory ───────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Static market priors and known analytical facts.
    Think of this as the "textbook knowledge" the agent system was born with.
    """

    # Sector beta priors (higher = more sensitive to macro)
    SECTOR_BETA: dict[str, float] = {
        "Technology":       1.3,
        "Financial Services": 1.1,
        "Consumer Cyclical": 1.2,
        "Energy":           1.0,
        "Healthcare":       0.8,
        "Consumer Defensive": 0.6,
        "Utilities":        0.5,
        "Real Estate":      0.9,
        "Industrials":      1.0,
        "Basic Materials":  1.1,
        "Communication Services": 1.1,
        "Unknown":          1.0,
    }

    # Known macro regime signals (from FRED monitoring)
    MACRO_REGIMES = {
        "expansion":    {"description": "GDP positive, unemployment declining, rates moderate"},
        "peak":         {"description": "High inflation, rates rising, yield curve flattening"},
        "contraction":  {"description": "GDP negative, unemployment rising, credit tightening"},
        "trough":       {"description": "Low rates, QE active, early recovery signals"},
    }

    # Historical average signal accuracy by agent type (seeded priors)
    AGENT_BASE_ACCURACY: dict[str, float] = {
        "technical":  0.61,   # Historical accuracy of pure-technical signals
        "sentiment":  0.55,   # Sentiment signals are noisier
        "macro":      0.58,   # Macro signals have longer horizon accuracy
    }

    # Signal reliability by volatility regime
    VIX_SIGNAL_RELIABILITY: dict = {
        "low":    {"vix_range": (0, 15),  "reliability": 0.75, "note": "Trending market, signals reliable"},
        "medium": {"vix_range": (15, 25), "reliability": 0.62, "note": "Normal volatility"},
        "high":   {"vix_range": (25, 40), "reliability": 0.45, "note": "High vol, wider signal uncertainty"},
        "crisis": {"vix_range": (40, 100),"reliability": 0.30, "note": "Crisis vol, signals break down"},
    }

    @classmethod
    def get_sector_beta(cls, sector: str) -> float:
        return cls.SECTOR_BETA.get(sector, 1.0)

    @classmethod
    def get_vix_reliability(cls, vix: float) -> dict:
        for regime, data in cls.VIX_SIGNAL_RELIABILITY.items():
            lo, hi = data["vix_range"]
            if lo <= vix < hi:
                return data
        return cls.VIX_SIGNAL_RELIABILITY["high"]

    @classmethod
    def get_agent_accuracy_prior(cls, agent_name: str) -> float:
        return cls.AGENT_BASE_ACCURACY.get(agent_name, 0.58)
