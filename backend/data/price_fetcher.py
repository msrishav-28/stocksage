import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ohlcv(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetches OHLCV data with retry logic.
    period: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'
    interval: '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'
    """
    logger.info(f"Fetching {ticker} OHLCV | period={period} interval={interval}")
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval, end=end)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["ticker"] = ticker

    logger.success(f"Fetched {len(df)} rows for {ticker}")
    return df


def fetch_multi_ticker(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetches OHLCV for multiple tickers in one yfinance call."""
    raw = yf.download(
        tickers=" ".join(tickers),
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )
    result = {}
    for t in tickers:
        try:
            df = raw[t].dropna()
            df.columns = [c.lower() for c in df.columns]
            df["ticker"] = t
            result[t] = df
        except Exception as e:
            logger.warning(f"Could not fetch {t}: {e}")
    return result


def fetch_ticker_info(ticker: str) -> dict:
    """Returns fundamental info: sector, market cap, P/E, beta, etc."""
    tkr = yf.Ticker(ticker)
    info = tkr.info
    return {
        "ticker": ticker,
        "name": info.get("shortName", ""),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "beta": info.get("beta", 1.0),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "earnings_date": info.get("earningsTimestamp"),
    }
