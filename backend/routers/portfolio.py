"""Portfolio router — multi-stock portfolio risk analysis and correlation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
import numpy as np

from backend.data.price_fetcher import fetch_ohlcv, fetch_multi_ticker
from backend.cache.redis_client import get_cached, set_cache

router = APIRouter()


class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: List[float] = []
    period: str = "1y"


@router.post("/analyze")
async def analyze_portfolio(req: PortfolioRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tickers")
    if len(req.tickers) > 20:
        raise HTTPException(status_code=400, detail="Max 20 tickers")

    cache_key = f"portfolio:analyze:{':'.join(sorted(req.tickers))}:{req.period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    # Default to equal weights
    weights = req.weights if req.weights and len(req.weights) == len(req.tickers) else [1/len(req.tickers)] * len(req.tickers)

    try:
        data = fetch_multi_ticker(req.tickers, period=req.period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    # Build returns matrix
    returns_dict = {}
    for ticker in req.tickers:
        if ticker in data and not data[ticker].empty:
            returns_dict[ticker] = data[ticker]["close"].pct_change().dropna()

    if len(returns_dict) < 2:
        raise HTTPException(status_code=404, detail="Not enough data for analysis")

    import pandas as pd
    returns_df = pd.DataFrame(returns_dict).dropna()

    # Portfolio metrics
    mean_returns = returns_df.mean() * 252  # Annualized
    cov_matrix = returns_df.cov() * 252

    w = np.array(weights[:len(returns_dict)])
    w = w / w.sum()  # Normalize

    portfolio_return = float(np.dot(w, mean_returns))
    portfolio_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

    # Individual stock metrics
    stock_metrics = []
    for ticker in returns_dict:
        ret = float(mean_returns[ticker])
        vol = float(returns_df[ticker].std() * np.sqrt(252))
        stock_metrics.append({
            "ticker": ticker,
            "annualized_return": round(ret * 100, 2),
            "annualized_volatility": round(vol * 100, 2),
            "sharpe": round(ret / vol, 3) if vol > 0 else 0,
        })

    result = {
        "tickers": list(returns_dict.keys()),
        "weights": w.tolist(),
        "portfolio_return_pct": round(portfolio_return * 100, 2),
        "portfolio_volatility_pct": round(portfolio_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "stock_metrics": stock_metrics,
    }

    await set_cache(cache_key, result, ttl=600)
    return result


@router.get("/correlation")
async def get_correlation(tickers: str, period: str = "1y"):
    """tickers: comma-separated list of ticker symbols."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    if len(ticker_list) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tickers")

    cache_key = f"portfolio:corr:{':'.join(sorted(ticker_list))}:{period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        data = fetch_multi_ticker(ticker_list, period=period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    import pandas as pd
    returns_dict = {}
    for ticker in ticker_list:
        if ticker in data and not data[ticker].empty:
            returns_dict[ticker] = data[ticker]["close"].pct_change().dropna()

    if len(returns_dict) < 2:
        raise HTTPException(status_code=404, detail="Not enough data")

    returns_df = pd.DataFrame(returns_dict).dropna()
    corr_matrix = returns_df.corr()

    result = {
        "tickers": list(returns_dict.keys()),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
    }

    await set_cache(cache_key, result, ttl=600)
    return result
