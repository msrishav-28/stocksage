"""initial schema

Creates the StockSage 2.0 core tables. When the target database has the
TimescaleDB extension available, `price_data` is converted to a hypertable;
on plain PostgreSQL the migration still succeeds as a regular table.

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── price_data ────────────────────────────────────────────────────────────
    op.create_table(
        "price_data",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.Column("volume", sa.BigInteger()),
        sa.PrimaryKeyConstraint("time", "ticker"),
    )
    op.create_index("ix_price_data_ticker_time", "price_data", ["ticker", sa.text("time DESC")])

    # ── news_sentiment ────────────────────────────────────────────────────────
    op.create_table(
        "news_sentiment",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("source", sa.String()),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finbert_label", sa.String()),
        sa.Column("finbert_score", sa.Float()),
        sa.Column("url", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_news_sentiment_ticker", "news_sentiment", ["ticker"])
    op.create_index(
        "ix_news_sentiment_ticker_published",
        "news_sentiment",
        ["ticker", sa.text("published_at DESC")],
    )

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("final_signal", sa.String()),
        sa.Column("confidence", sa.Float()),
        sa.Column("weighted_score", sa.Float()),
        sa.Column("risk_score", sa.Float()),
        sa.Column("tft_point_d1", sa.Float()),
        sa.Column("tft_point_d5", sa.Float()),
        sa.Column("tft_point_d10", sa.Float()),
        sa.Column("agent_technical", sa.JSON()),
        sa.Column("agent_sentiment", sa.JSON()),
        sa.Column("agent_macro", sa.JSON()),
        sa.Column("tft_quantile_bands", sa.JSON()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_ticker", "predictions", ["ticker"])
    op.create_index(
        "ix_predictions_ticker_predicted",
        "predictions",
        ["ticker", sa.text("predicted_at DESC")],
    )

    # ── backtest_results ──────────────────────────────────────────────────────
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("total_return_pct", sa.Float()),
        sa.Column("benchmark_return_pct", sa.Float()),
        sa.Column("alpha_pct", sa.Float()),
        sa.Column("sharpe_ratio", sa.Float()),
        sa.Column("max_drawdown_pct", sa.Float()),
        sa.Column("win_rate_pct", sa.Float()),
        sa.Column("total_trades", sa.Integer()),
        sa.Column("initial_cash", sa.Float()),
        sa.Column("final_value", sa.Float()),
        sa.Column("equity_curve", sa.JSON()),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── macro_snapshots ───────────────────────────────────────────────────────
    op.create_table(
        "macro_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("fed_funds_rate", sa.Float()),
        sa.Column("cpi_yoy", sa.Float()),
        sa.Column("unemployment", sa.Float()),
        sa.Column("gdp_growth", sa.Float()),
        sa.Column("yield_curve", sa.Float()),
        sa.Column("vix", sa.Float()),
        sa.Column("macro_score", sa.Float()),
        sa.Column("macro_direction", sa.String()),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── watchlist ─────────────────────────────────────────────────────────────
    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String()),
        sa.Column("ticker", sa.String()),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("notes", sa.Text()),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── TimescaleDB hypertable (optional — skipped on plain PostgreSQL) ────────
    conn = op.get_bind()
    timescale_available = conn.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
    ).first()
    if timescale_available:
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        op.execute(
            "SELECT create_hypertable('price_data', 'time', "
            "if_not_exists => TRUE, migrate_data => TRUE)"
        )


def downgrade() -> None:
    op.drop_table("watchlist")
    op.drop_table("macro_snapshots")
    op.drop_table("backtest_results")
    op.drop_index("ix_predictions_ticker_predicted", table_name="predictions")
    op.drop_index("ix_predictions_ticker", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_news_sentiment_ticker_published", table_name="news_sentiment")
    op.drop_index("ix_news_sentiment_ticker", table_name="news_sentiment")
    op.drop_table("news_sentiment")
    op.drop_index("ix_price_data_ticker_time", table_name="price_data")
    op.drop_table("price_data")
