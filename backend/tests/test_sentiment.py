"""Tests for the sentiment pipeline.

The SentimentAgent is a BaseAgent that calls tools via the ToolRegistry.
The tools import their dependencies at call time, so mocks target the
*definition site* (backend.data.news_fetcher / backend.ml.finbert_sentiment).
"""

import pytest
from unittest.mock import patch, AsyncMock


class TestSentimentAgent:
    """Tests for the sentiment agent's tool-driven analysis."""

    @pytest.mark.asyncio
    async def test_sentiment_no_articles(self):
        """No news -> neutral signal."""
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]):
            from backend.ml.sentiment_agent import SentimentAgent
            result = await SentimentAgent().run({"ticker": "NONEXIST", "hours": 48})

        assert result.direction == "neutral"
        assert result.raw_score == 0.0
        assert result.metadata["total_articles"] == 0
        assert isinstance(result.metadata["headlines"], list)

    @pytest.mark.asyncio
    async def test_sentiment_with_scored_headlines(self):
        """Articles + FinBERT scores -> aggregated sentiment."""
        articles = [
            {"title": "Stock surges on great earnings"},
            {"title": "Analysts raise price target sharply"},
        ]
        scored = [
            {"headline": articles[0]["title"], "label": "positive", "score": 0.8,
             "probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}},
            {"headline": articles[1]["title"], "label": "positive", "score": 0.6,
             "probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}},
        ]
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=articles), \
             patch("backend.ml.finbert_sentiment.score_batch", return_value=scored):
            from backend.ml.sentiment_agent import SentimentAgent
            result = await SentimentAgent().run({"ticker": "TEST", "hours": 48})

        assert result.direction == "bullish"
        assert result.raw_score > 0.15
        assert result.metadata["total_articles"] == 2
        assert result.metadata["bullish_count"] == 2

    @pytest.mark.asyncio
    async def test_sentiment_finbert_failure_falls_back(self):
        """FinBERT failure -> graceful neutral fallback, agent still succeeds."""
        articles = [{"title": "Some market headline"}, {"title": "Another headline"}]
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=articles), \
             patch("backend.ml.finbert_sentiment.score_batch", side_effect=RuntimeError("FinBERT down")):
            from backend.ml.sentiment_agent import SentimentAgent
            result = await SentimentAgent().run({"ticker": "TEST", "hours": 48})

        # Degrades to neutral but does not crash.
        assert result.direction == "neutral"
        assert result.error is None
        assert result.metadata["total_articles"] == 2

    @pytest.mark.asyncio
    async def test_sentiment_returns_correct_structure(self):
        """Agent metadata must carry all aggregation keys."""
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]):
            from backend.ml.sentiment_agent import SentimentAgent
            result = await SentimentAgent().run({"ticker": "AAPL", "hours": 48})

        required = {
            "composite_score", "label", "bullish_count", "bearish_count",
            "neutral_count", "total_articles", "sentiment_momentum", "headlines",
        }
        assert required.issubset(result.metadata.keys()), \
            f"Missing keys: {required - result.metadata.keys()}"


class TestAggregation:
    """Tests for sentiment aggregation (pure function — no mocking needed)."""

    def test_empty_headlines(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        result = aggregate_sentiment([])
        assert result["composite_score"] == 0.0
        assert result["total_articles"] == 0
        assert result["label"] == "neutral"

    def test_aggregate_structure(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        scored = [
            {"score": 0.5, "label": "positive"},
            {"score": -0.3, "label": "negative"},
            {"score": 0.1, "label": "neutral"},
        ]
        result = aggregate_sentiment(scored)
        assert result["bullish_count"] == 1
        assert result["bearish_count"] == 1
        assert result["neutral_count"] == 1
        assert result["total_articles"] == 3

    def test_strongly_positive_headlines(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        result = aggregate_sentiment([{"score": 0.8, "label": "positive"}] * 5)
        assert result["label"] == "positive"
        assert result["composite_score"] > 0.15

    def test_strongly_negative_headlines(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        result = aggregate_sentiment([{"score": -0.8, "label": "negative"}] * 5)
        assert result["label"] == "negative"
        assert result["composite_score"] < -0.15
