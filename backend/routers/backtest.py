"""Backtest router — run vectorbt backtests on chosen strategies."""

import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.backtest import BacktestRequest, BacktestResponse
from backend.data.price_fetcher import fetch_ohlcv
from backend.indicators import compute_all_indicators
from backend.backtesting.strategies import get_strategy_signals, STRATEGY_NAMES

router = APIRouter()


def _run_backtest_sync(req: BacktestRequest) -> dict:
    """Synchronous fetch + indicators + backtest — runs in a worker thread."""
    from backend.backtesting.engine import run_backtest

    df = fetch_ohlcv(req.ticker, period=req.period)
    df = compute_all_indicators(df)
    if df.empty:
        raise ValueError(f"Not enough data to backtest {req.ticker}")

    entries, exits = get_strategy_signals(df, req.strategy)
    return run_backtest(
        close=df["close"],
        entries=entries,
        exits=exits,
        ticker=req.ticker,
        initial_cash=req.initial_cash,
        fees=req.fees,
        slippage=req.slippage,
    )


@router.post("/", response_model=BacktestResponse)
async def run_backtest_endpoint(req: BacktestRequest):
    # Validate the strategy up front so a bad name is a clean 400.
    if req.strategy not in STRATEGY_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: {sorted(STRATEGY_NAMES)}",
        )

    try:
        result = await asyncio.to_thread(_run_backtest_sync, req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # vectorbt not installed — backtesting engine unavailable.
        logger.error(f"Backtest engine unavailable: {e}")
        raise HTTPException(status_code=501, detail="Backtesting engine not available")
    except Exception as e:
        logger.error(f"Backtest failed for {req.ticker}: {e}")
        raise HTTPException(status_code=500, detail="Backtest failed")

    return result
