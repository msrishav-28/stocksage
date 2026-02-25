"""Tests for the technical indicators engine."""

import pytest
import pandas as pd
import numpy as np


class TestIndicators:
    """Tests for compute_all_indicators."""

    def test_indicators_output_columns(self, sample_ohlcv_df):
        """Should add indicator columns to the DataFrame."""
        from backend.indicators import compute_all_indicators
        result = compute_all_indicators(sample_ohlcv_df)

        # Should have more columns than input
        assert len(result.columns) > len(sample_ohlcv_df.columns)

        # Should have key indicators
        expected = ["daily_return", "log_return", "close_return"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_indicators_no_nans(self, sample_ohlcv_df):
        """After dropna, result should have no NaN values."""
        from backend.indicators import compute_all_indicators
        result = compute_all_indicators(sample_ohlcv_df)
        assert result.isna().sum().sum() == 0

    def test_indicators_row_count(self, sample_ohlcv_df):
        """Should have fewer rows after dropna (initial rows have incomplete rolling windows)."""
        from backend.indicators import compute_all_indicators
        result = compute_all_indicators(sample_ohlcv_df)
        assert len(result) < len(sample_ohlcv_df)
        assert len(result) > 0


class TestConfluence:
    """Tests for confluence scoring."""

    def test_confluence_structure(self, sample_ohlcv_df):
        """Confluence score should have expected keys."""
        from backend.indicators import compute_all_indicators, compute_confluence_score
        df = compute_all_indicators(sample_ohlcv_df)
        result = compute_confluence_score(df)

        assert "direction" in result
        assert result["direction"] in ("bullish", "bearish", "neutral")
        assert "raw_score" in result
        assert -1.0 <= result["raw_score"] <= 1.0
        assert "confluence_score" in result
        assert 0.0 <= result["confluence_score"] <= 1.0
        assert "signal_count" in result
        assert result["signal_count"] > 0

    def test_confluence_signal_counts(self, sample_ohlcv_df):
        """Signal counts should sum to total."""
        from backend.indicators import compute_all_indicators, compute_confluence_score
        df = compute_all_indicators(sample_ohlcv_df)
        result = compute_confluence_score(df)

        total = result["bullish_signals"] + result["bearish_signals"] + result["neutral_signals"]
        assert total == result["signal_count"]


class TestFeatureEngineer:
    """Tests for the feature engineering pipeline."""

    def test_feature_df_structure(self, sample_ohlcv_df):
        """Feature DataFrame should have time, calendar, and static features."""
        from backend.data.feature_engineer import build_feature_df
        result = build_feature_df(sample_ohlcv_df, ticker="TEST")

        assert "time_idx" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "TEST"

    def test_feature_df_no_nans(self, sample_ohlcv_df):
        """Feature DF should have no NaN values."""
        from backend.data.feature_engineer import build_feature_df
        result = build_feature_df(sample_ohlcv_df, ticker="TEST")
        assert result.isna().sum().sum() == 0
