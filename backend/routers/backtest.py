"""Backtest router — run vectorbt backtests on chosen strategies."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.backtest import BacktestRequest, BacktestResponse
from backend.data.price_fetcher import fetch_ohlcv
from backend.indicators import compute_all_indicators

router = APIRouter()


@router.post("/", response_model=BacktestResponse)
async def run_backtest_endpoint(req: BacktestRequest):
    try:
        df = fetch_ohlcv(req.ticker, period=req.period)
        df = compute_all_indicators(df)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Data fetch for backtest failed: {e}")
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    try:
        from backend.backtesting.engine import run_backtest
        from backend.backtesting.strategies import get_strategy_signals

        entries, exits = get_strategy_signals(df, req.strategy)
        result = run_backtest(
            close=df["close"],
            entries=entries,
            exits=exits,
            ticker=req.ticker,
            initial_cash=req.initial_cash,
            fees=req.fees,
            slippage=req.slippage,
        )
    except ImportError:
        logger.error("vectorbt not installed. Backtesting unavailable.")
        raise HTTPException(status_code=501, detail="Backtesting engine not available")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    return result
