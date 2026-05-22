"""Tests for the orchestrated prediction pipeline."""

import pytest
from unittest.mock import patch, AsyncMock


class TestEnsemble:
    """Tests for the multi-agent orchestrator (via ensemble_predict)."""

    @pytest.mark.asyncio
    async def test_ensemble_predict_returns_expected_keys(self, sample_ohlcv_df):
        """ensemble_predict should return all expected response keys."""
        from backend.data.feature_engineer import build_feature_df
        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        # Tools import their deps at call time — patch the definition sites.
        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}):
            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        assert result["ticker"] == "TEST"
        assert result["final_signal"] in ("BUY", "HOLD", "SELL")
        assert 0 <= result["confidence"] <= 100
        assert -1.0 <= result["weighted_score"] <= 1.0
        assert 0.0 <= result["risk_score"] <= 10.0
        for key in ("agent_signals", "explanation", "thesis", "trace", "guardrail_flags"):
            assert key in result, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_ensemble_agent_signals_structure(self, sample_ohlcv_df):
        """Agent signals should include technical, sentiment, and macro."""
        from backend.data.feature_engineer import build_feature_df
        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}):
            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        signals = result["agent_signals"]
        assert {"technical", "sentiment", "macro"}.issubset(signals.keys())

    @pytest.mark.asyncio
    async def test_ensemble_survives_agent_failure(self, sample_ohlcv_df):
        """A failing news provider must not break the ensemble."""
        from backend.data.feature_engineer import build_feature_df
        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        with patch("backend.data.news_fetcher.fetch_news",
                   new_callable=AsyncMock, side_effect=RuntimeError("NewsAPI down")), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}):
            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        # The ensemble survives: the sentiment agent degrades gracefully to neutral.
        assert result["final_signal"] in ("BUY", "HOLD", "SELL")
        assert result["agent_signals"]["sentiment"]["direction"] == "neutral"

    @pytest.mark.asyncio
    async def test_ensemble_trace_records_all_agents(self, sample_ohlcv_df):
        """The telemetry trace should record every agent span."""
        from backend.data.feature_engineer import build_feature_df
        df = build_feature_df(sample_ohlcv_df, ticker="TEST")

        with patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]), \
             patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}):
            from backend.ml.ensemble import ensemble_predict
            result = await ensemble_predict("TEST", df)

        trace = result["trace"]
        agents_traced = {s["agent"] for s in trace["spans"]}
        assert {"technical", "sentiment", "macro"}.issubset(agents_traced)


class TestPredictRequest:
    """Tests for predict schema validation."""

    def test_default_period(self):
        from backend.schemas.predict import PredictRequest
        req = PredictRequest(ticker="AAPL")
        assert req.period == "2y"

    def test_ticker_is_normalized(self):
        from backend.schemas.predict import PredictRequest
        req = PredictRequest(ticker="  msft ")
        assert req.ticker == "MSFT"

    def test_empty_ticker_rejected(self):
        from backend.schemas.predict import PredictRequest
        with pytest.raises(ValueError):
            PredictRequest(ticker="   ")
