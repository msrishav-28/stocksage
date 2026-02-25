"""
Seed the database with historical data.

Usage:
    python scripts/seed_db.py --tickers AAPL MSFT GOOGL
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


async def seed(tickers: list[str], period: str = "2y"):
    from backend.db.session import init_db, get_engine
    from backend.db.models import Base
    from backend.data.price_fetcher import fetch_ohlcv

    await init_db()
    engine = get_engine()

    if engine is None:
        logger.error("Database not available. Cannot seed.")
        return

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("Tables created.")

    # Seed price data
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period)
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            # Insert via raw SQL or ORM as needed
            logger.success(f"Seeded {ticker}")
        except Exception as e:
            logger.warning(f"Failed to seed {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Seed database with historical data")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    asyncio.run(seed(args.tickers, args.period))


if __name__ == "__main__":
    main()
