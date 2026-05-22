"""Portfolio router — multi-stock portfolio risk analysis and correlation."""

import asyncio

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
import numpy as np
import pandas as pd

from backend.schemas.portfolio import PortfolioRequest, PortfolioResponse, CorrelationResponse
from backend.data.price_fetcher import fetch_multi_ticker
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


@router.post("/analyze", response_model=PortfolioResponse)
async def analyze_portfolio(req: PortfolioRequest):
    """Run portfolio-level risk metrics: annualized return, volatility, Sharpe, correlation."""
    cache_key = f"portfolio:analyze:{':'.join(sorted(req.tickers))}:{req.period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    # Default to equal weights if none provided
    n = len(req.tickers)
    weights = (
        req.weights if req.weights and len(req.weights) == n
        else [1.0 / n] * n
    )

    try:
        data = await asyncio.to_thread(fetch_multi_ticker, req.tickers, req.period)
    except Exception as e:
        logger.error(f"Portfolio data fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider error")

    # Build returns matrix
    returns_dict: dict[str, pd.Series] = {}
    for ticker in req.tickers:
        if ticker in data and not data[ticker].empty:
            returns_dict[ticker] = data[ticker]["close"].pct_change().dropna()

    if len(returns_dict) < 2:
        raise HTTPException(status_code=404, detail="Not enough price data for analysis")

    returns_df  = pd.DataFrame(returns_dict).dropna()
    mean_returns = returns_df.mean() * 252          # annualise
    cov_matrix   = returns_df.cov() * 252

    # Normalise weights to the tickers we actually got data for
    available = list(returns_dict.keys())
    w = np.array(weights[: len(available)], dtype=float)
    w = w / w.sum()

    portfolio_return = float(np.dot(w, mean_returns[available]))
    portfolio_vol    = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.loc[available, available], w))))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

    stock_metrics = []
    for i, ticker in enumerate(available):
        ret = float(mean_returns[ticker])
        vol = float(returns_df[ticker].std() * np.sqrt(252))
        stock_metrics.append({
            "ticker":                ticker,
            "annualized_return":     round(ret * 100, 2),
            "annualized_volatility": round(vol * 100, 2),
            "sharpe":                round(ret / vol, 3) if vol > 0 else 0.0,
        })

    result = {
        "tickers":                  available,
        "weights":                  w.tolist(),
        "portfolio_return_pct":     round(portfolio_return * 100, 2),
        "portfolio_volatility_pct": round(portfolio_vol * 100, 2),
        "sharpe_ratio":             round(sharpe, 3),
        "stock_metrics":            stock_metrics,
    }

    await set_cache(cache_key, result, ttl=600)
    return result


@router.get("/correlation", response_model=CorrelationResponse)
async def get_correlation(
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    period: str = Query("1y"),
):
    """Compute pairwise return correlation matrix for a list of tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tickers")

    cache_key = f"portfolio:corr:{':'.join(sorted(ticker_list))}:{period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        data = await asyncio.to_thread(fetch_multi_ticker, ticker_list, period)
    except Exception as e:
        logger.error(f"Correlation data fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider error")

    returns_dict: dict[str, pd.Series] = {}
    for ticker in ticker_list:
        if ticker in data and not data[ticker].empty:
            returns_dict[ticker] = data[ticker]["close"].pct_change().dropna()

    if len(returns_dict) < 2:
        raise HTTPException(status_code=404, detail="Not enough data to compute correlation")

    returns_df   = pd.DataFrame(returns_dict).dropna()
    corr_matrix  = returns_df.corr()

    result = {
        "tickers":            list(returns_dict.keys()),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
    }

    await set_cache(cache_key, result, ttl=600)
    return result
