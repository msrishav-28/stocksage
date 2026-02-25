from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from backend.routers import predict, sentiment, technical, backtest, screener, competitor, macro, portfolio
from backend.cache.redis_client import init_redis, close_redis
from backend.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting StockSage backend...")
    await init_db()
    await init_redis()
    yield
    await close_redis()
    logger.info("StockSage backend shut down.")


app = FastAPI(
    title="StockSage API",
    description="AI-powered stock analysis and prediction platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://stocksage.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,    prefix="/api/predict",    tags=["Prediction"])
app.include_router(sentiment.router,  prefix="/api/sentiment",  tags=["Sentiment"])
app.include_router(technical.router,  prefix="/api/technical",  tags=["Technical"])
app.include_router(backtest.router,   prefix="/api/backtest",   tags=["Backtesting"])
app.include_router(screener.router,   prefix="/api/screener",   tags=["Screener"])
app.include_router(competitor.router, prefix="/api/competitor", tags=["Competitor"])
app.include_router(macro.router,      prefix="/api/macro",      tags=["Macro"])
app.include_router(portfolio.router,  prefix="/api/portfolio",  tags=["Portfolio"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}
