"""
Train the Temporal Fusion Transformer on historical stock data.

Usage:
    python scripts/train_tft.py --tickers AAPL MSFT GOOGL --period 2y
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "V"],
        help="Tickers to train on",
    )
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--checkpoint-dir", default="models/", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    import pandas as pd
    from backend.data.price_fetcher import fetch_ohlcv
    from backend.data.feature_engineer import build_feature_df
    from backend.ml.tft_model import build_timeseries_dataset, train_tft

    # Fetch and build features for all tickers
    all_dfs = []
    for ticker in args.tickers:
        try:
            logger.info(f"Fetching {ticker}...")
            df = fetch_ohlcv(ticker, period=args.period)
            df = build_feature_df(df, ticker=ticker)
            all_dfs.append(df)
            logger.success(f"{ticker}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")

    if not all_dfs:
        logger.error("No data fetched. Exiting.")
        return

    # Combine
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} cols")

    # Reset time_idx per ticker group
    combined["time_idx"] = combined.groupby("ticker").cumcount()

    # Split
    cutoff = int(combined["time_idx"].max() * 0.8)
    training_dataset = build_timeseries_dataset(combined, training=True, training_cutoff=cutoff)
    validation_dataset = build_timeseries_dataset(combined, training=False, training_cutoff=cutoff)

    # Train
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model = train_tft(training_dataset, validation_dataset, checkpoint_dir=args.checkpoint_dir)

    logger.success(f"Model saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
