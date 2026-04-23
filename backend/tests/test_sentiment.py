"""Tests for the sentiment pipeline."""

import pytest
from unittest.mock import patch, AsyncMock


class TestSentimentAgent:
    """Tests for the sentiment agent."""

    @pytest.mark.asyncio
    async def test_sentiment_no_articles(self):
        """Should return neutral when no articles found."""
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]):
            from backend.ml.sentiment_agent import SentimentAgent
            agent = SentimentAgent()
            result = await agent.analyze("NONEXIST")

        assert result["label"] == "neutral"
        assert result["composite_score"] == 0.0
        assert result["total_articles"] == 0

    @pytest.mark.asyncio
    async def test_sentiment_with_articles(self):
        """Should process articles and return sentiment."""
        mock_articles = [
            {"title": "Stock surges on great earnings", "description": "", "url": "", "published_at": "", "source": ""},
            {"title": "Market crashes amid fears", "description": "", "url": "", "published_at": "", "source": ""},
        ]

        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=mock_articles):
            from backend.ml.sentiment_agent import SentimentAgent
            agent = SentimentAgent()
            result = await agent.analyze("TEST")

        assert "composite_score" in result
        assert "label" in result
        assert result["total_articles"] >= 0


class TestAggregation:
    """Tests for sentiment aggregation."""

    def test_empty_headlines(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        result = aggregate_sentiment([])
        assert result["composite_score"] == 0.0
        assert result["total_articles"] == 0

    def test_aggregate_structure(self):
        from backend.ml.finbert_sentiment import aggregate_sentiment
        mock_scored = [
            {"score": 0.5, "label": "positive"},
            {"score": -0.3, "label": "negative"},
            {"score": 0.1, "label": "neutral"},
        ]
        result = aggregate_sentiment(mock_scored)
        assert "composite_score" in result
        assert "bullish_count" in result
        assert result["bullish_count"] == 1
        assert result["bearish_count"] == 1
        assert result["neutral_count"] == 1
