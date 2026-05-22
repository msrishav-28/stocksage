"""
Seed the database with historical OHLCV price data.

Usage:
    python scripts/seed_db.py --tickers AAPL MSFT GOOGL --period 2y
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


def _rows_from_df(df, ticker: str) -> list[dict]:
    """Convert an OHLCV DataFrame into price_data row dicts."""
    rows = []
    for ts, row in df.iterrows():
        rows.append({
            "time": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "ticker": ticker,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]) if row["volume"] == row["volume"] else 0,
        })
    return rows


async def seed(tickers: list[str], period: str = "2y") -> None:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from backend.db.session import init_db, get_engine, get_session
    from backend.db.models import Base, PriceData
    from backend.data.price_fetcher import fetch_ohlcv

    await init_db()
    engine = get_engine()
    if engine is None:
        logger.error("Database not available. Cannot seed.")
        return

    # Ensure tables exist (idempotent — safe alongside Alembic).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("Tables ensured.")

    total_inserted = 0
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period)
            rows = _rows_from_df(df, ticker)
            if not rows:
                logger.warning(f"{ticker}: no rows to insert")
                continue

            # Upsert: skip rows already present (PK = time + ticker).
            stmt = pg_insert(PriceData).values(rows).on_conflict_do_nothing(
                index_elements=["time", "ticker"]
            )
            async for session in get_session():
                if session is None:
                    break
                await session.execute(stmt)
                await session.commit()
                break

            total_inserted += len(rows)
            logger.success(f"{ticker}: seeded {len(rows)} rows")
        except Exception as e:
            logger.warning(f"Failed to seed {ticker}: {e}")

    logger.success(f"Seeding complete — {total_inserted} rows across {len(tickers)} tickers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed database with historical price data")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    asyncio.run(seed(args.tickers, args.period))


if __name__ == "__main__":
    main()
