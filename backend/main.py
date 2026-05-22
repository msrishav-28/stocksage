from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from backend.config import get_settings
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


settings = get_settings()

app = FastAPI(
    title="StockSage API",
    description="AI-powered stock analysis and prediction platform",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS (env-driven origins) ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Rate limiting (per client IP) ─────────────────────────────────────────────
# Guarded import keeps the app runnable even if slowapi is not installed,
# consistent with how the codebase treats other optional infrastructure.
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info(f"Rate limiting enabled: {settings.RATE_LIMIT} per IP")
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled.")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router,    prefix="/api/predict",    tags=["Prediction"])
app.include_router(sentiment.router,  prefix="/api/sentiment",  tags=["Sentiment"])
app.include_router(technical.router,  prefix="/api/technical",  tags=["Technical"])
app.include_router(backtest.router,   prefix="/api/backtest",   tags=["Backtesting"])
app.include_router(screener.router,   prefix="/api/screener",   tags=["Screener"])
app.include_router(competitor.router, prefix="/api/competitor", tags=["Competitor"])
app.include_router(macro.router,      prefix="/api/macro",      tags=["Macro"])
app.include_router(portfolio.router,  prefix="/api/portfolio",  tags=["Portfolio"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": "2.0.0"}
