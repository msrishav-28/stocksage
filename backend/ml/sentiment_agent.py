"""Sentiment analysis agent — fetches news and scores it with FinBERT.

Runs a two-step tool loop: ``fetch_news`` then ``finbert_score``. If FinBERT
is unavailable the agent degrades to a neutral signal rather than failing.
"""

from __future__ import annotations

from loguru import logger

from backend.ml.base_agent import BaseAgent, AgentStep, AgentResult

_LABEL_TO_DIRECTION = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}


def _neutral_aggregate(headline_count: int = 0) -> dict:
    return {
        "composite_score": 0.0,
        "label": "neutral",
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": headline_count,
        "total_articles": headline_count,
        "sentiment_momentum": 0.0,
    }


class SentimentAgent(BaseAgent):
    """Fetches recent news headlines, scores them with FinBERT, and aggregates."""

    max_turns = 3

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def tool_names(self) -> list[str]:
        return ["fetch_news", "finbert_score"]

    def _initial_thought(self, context: dict) -> str:
        return f"Gauge news sentiment for {context.get('ticker', '?')}."

    async def _decide_action(self, context: dict, steps: list[AgentStep]):
        # Step 1: fetch news.
        if not any(s.action == "fetch_news" for s in steps):
            return "fetch_news", {
                "ticker": context["ticker"],
                "hours": context.get("hours", 48),
            }

        # Step 2: score the headlines (only if we actually got some).
        news_obs = self._get_observation(steps, "fetch_news")
        headlines = [a["title"] for a in news_obs if a.get("title")] if isinstance(news_obs, list) else []
        if headlines and not any(s.action == "finbert_score" for s in steps):
            return "finbert_score", {"headlines": headlines}

        return None, {}

    async def _interpret_observations(self, context: dict, steps: list[AgentStep]) -> AgentResult:
        from backend.ml.finbert_sentiment import aggregate_sentiment

        news_obs = self._get_observation(steps, "fetch_news")
        articles = news_obs if isinstance(news_obs, list) else []
        headlines = [a["title"] for a in articles if a.get("title")]

        if not headlines:
            logger.info(f"SentimentAgent: no headlines for {context.get('ticker')}")
            return AgentResult(
                agent_name=self.name,
                direction="neutral",
                confidence=0.3,
                raw_score=0.0,
                metadata={**_neutral_aggregate(), "headlines": []},
            )

        finbert_obs = self._get_observation(steps, "finbert_score")
        if isinstance(finbert_obs, list):
            scored = finbert_obs
        else:
            # FinBERT unavailable — treat every headline as neutral.
            reason = finbert_obs.get("error") if isinstance(finbert_obs, dict) else "finbert unavailable"
            logger.warning(f"SentimentAgent: FinBERT unavailable ({reason}); neutral fallback.")
            scored = [
                {"headline": h, "label": "neutral", "score": 0.0, "probabilities": None}
                for h in headlines
            ]

        aggregated = aggregate_sentiment(scored)
        composite = float(aggregated["composite_score"])
        return AgentResult(
            agent_name=self.name,
            direction=_LABEL_TO_DIRECTION.get(aggregated["label"], "neutral"),
            confidence=min(abs(composite), 1.0),
            raw_score=composite,
            metadata={**aggregated, "headlines": scored[:10]},
        )
