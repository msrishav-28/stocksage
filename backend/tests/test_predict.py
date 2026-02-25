"""Tests for the prediction pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestEnsemble:
    """Tests for the multi-agent ensemble system."""

    @pytest.mark.asyncio
    async def test_ensemble_predict_returns_expected_keys(self, sample_ohlcv_df):
        """Ensemble predict should return all expected response keys."""
        from backend.data.feature_engineer import build_feature_df

        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        with patch("backend.ml.sentiment_agent.fetch_news", new_callable=AsyncMock, return_value=[]), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}), \
             patch("backend.data.price_fetcher.fetch_ticker_info", return_value={"sector": "Technology"}):

            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        assert "ticker" in result
        assert result["ticker"] == "TEST"
        assert "final_signal" in result
        assert result["final_signal"] in ("BUY", "HOLD", "SELL")
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 100
        assert "agent_signals" in result
        assert "risk_score" in result
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_ensemble_agent_signals_structure(self, sample_ohlcv_df):
        """Agent signals should have technical, sentiment, and macro keys."""
        from backend.data.feature_engineer import build_feature_df

        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        with patch("backend.ml.sentiment_agent.fetch_news", new_callable=AsyncMock, return_value=[]), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}), \
             patch("backend.data.price_fetcher.fetch_ticker_info", return_value={"sector": "Technology"}):

            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        signals = result["agent_signals"]
        assert "technical" in signals
        assert "sentiment" in signals
        assert "macro" in signals


class TestPredictRequest:
    """Tests for predict schema validation."""

    def test_default_period(self):
        from backend.schemas.predict import PredictRequest
        req = PredictRequest(ticker="AAPL")
        assert req.period == "2y"

    def test_custom_period(self):
        from backend.schemas.predict import PredictRequest
        req = PredictRequest(ticker="MSFT", period="1y")
        assert req.period == "1y"
