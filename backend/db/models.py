"""ORM models for TimescaleDB tables."""

from sqlalchemy import Column, Integer, BigInteger, String, Float, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class PriceData(Base):
    __tablename__ = "price_data"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String, primary_key=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    headline = Column(Text, nullable=False)
    source = Column(String)
    published_at = Column(DateTime(timezone=True), nullable=False)
    finbert_label = Column(String)
    finbert_score = Column(Float)
    url = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    final_signal = Column(String)
    confidence = Column(Float)
    weighted_score = Column(Float)
    risk_score = Column(Float)
    thesis = Column(Text)
    tft_point_d1 = Column(Float)
    tft_point_d5 = Column(Float)
    tft_point_d10 = Column(Float)
    agent_technical = Column(JSON)
    agent_sentiment = Column(JSON)
    agent_macro = Column(JSON)
    tft_quantile_bands = Column(JSON)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    run_at = Column(DateTime(timezone=True), server_default=func.now())
    total_return_pct = Column(Float)
    benchmark_return_pct = Column(Float)
    alpha_pct = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown_pct = Column(Float)
    win_rate_pct = Column(Float)
    total_trades = Column(Integer)
    initial_cash = Column(Float)
    final_value = Column(Float)
    equity_curve = Column(JSON)


class MacroSnapshot(Base):
    __tablename__ = "macro_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    captured_at = Column(DateTime(timezone=True), server_default=func.now())
    fed_funds_rate = Column(Float)
    cpi_yoy = Column(Float)
    unemployment = Column(Float)
    gdp_growth = Column(Float)
    yield_curve = Column(Float)
    vix = Column(Float)
    macro_score = Column(Float)
    macro_direction = Column(String)


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    ticker = Column(String)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
