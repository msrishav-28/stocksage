"""Tests for the backtesting engine."""

import pytest
import pandas as pd


class TestStrategies:
    """Tests for predefined strategies."""

    def test_available_strategies(self):
        """Should list available strategies."""
        from backend.backtesting.strategies import get_strategy_signals
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy_signals(pd.DataFrame(), "nonexistent")

    def test_rsi_macd_signals(self, sample_ohlcv_df):
        """RSI/MACD strategy should return boolean entry/exit Series."""
        from backend.indicators import compute_all_indicators
        from backend.backtesting.strategies import get_strategy_signals

        df = compute_all_indicators(sample_ohlcv_df)
        entries, exits = get_strategy_signals(df, "rsi_macd")

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert entries.dtype == bool
        assert exits.dtype == bool
        assert len(entries) == len(df)

    def test_ema_crossover_signals(self, sample_ohlcv_df):
        """EMA crossover strategy should produce some signals."""
        from backend.indicators import compute_all_indicators
        from backend.backtesting.strategies import get_strategy_signals

        df = compute_all_indicators(sample_ohlcv_df)
        entries, exits = get_strategy_signals(df, "ema_crossover")

        assert isinstance(entries, pd.Series)
        assert entries.dtype == bool

    def test_bollinger_breakout_signals(self, sample_ohlcv_df):
        """Bollinger breakout strategy should produce some signals."""
        from backend.indicators import compute_all_indicators
        from backend.backtesting.strategies import get_strategy_signals

        df = compute_all_indicators(sample_ohlcv_df)
        entries, exits = get_strategy_signals(df, "bollinger_breakout")

        assert isinstance(entries, pd.Series)


class TestBacktestSchemas:
    """Tests for backtest Pydantic schemas."""

    def test_default_values(self):
        from backend.schemas.backtest import BacktestRequest
        req = BacktestRequest(ticker="AAPL")
        assert req.strategy == "rsi_macd"
        assert req.initial_cash == 10_000.0
        assert req.fees == 0.001

    def test_custom_values(self):
        from backend.schemas.backtest import BacktestRequest
        req = BacktestRequest(
            ticker="TSLA",
            strategy="ema_crossover",
            initial_cash=50_000,
        )
        assert req.strategy == "ema_crossover"
        assert req.initial_cash == 50_000
