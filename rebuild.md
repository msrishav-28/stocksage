# StockSage 2.0 — Full Backend Rebuild Plan
> Target Standard: 2026–2027 Production-Grade AI Stock Intelligence Platform  
> Scope: Backend, ML Engine, Data Pipeline, API, Deployment (Frontend excluded)

---

## Table of Contents

1. [Current State vs Target State](#1-current-state-vs-target-state)
2. [New Project Structure](#2-new-project-structure)
3. [New Requirements](#3-new-requirements)
4. [Backend Architecture — FastAPI Migration](#4-backend-architecture--fastapi-migration)
5. [ML Engine — Temporal Fusion Transformer](#5-ml-engine--temporal-fusion-transformer)
6. [Sentiment Pipeline — FinBERT](#6-sentiment-pipeline--finbert)
7. [Multi-Agent Prediction System](#7-multi-agent-prediction-system)
8. [Data Pipeline](#8-data-pipeline)
9. [Technical Indicators Engine](#9-technical-indicators-engine)
10. [Backtesting Engine](#10-backtesting-engine)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Database Schema — TimescaleDB](#12-database-schema--timescaledb)
13. [Caching Layer — Redis](#13-caching-layer--redis)
14. [Model Serving — Modal.com](#14-model-serving--modalcom)
15. [Environment & Config](#15-environment--config)
16. [Deployment Architecture](#16-deployment-architecture)
17. [Git Hygiene Fixes](#17-git-hygiene-fixes)
18. [Sprint Roadmap](#18-sprint-roadmap)

---

## 1. Current State vs Target State

| Dimension | Current (v1) | Target (v2) |
|---|---|---|
| Framework | Flask 2.1 | FastAPI 0.115+ (async) |
| ML Model | sklearn `.pkl` (703 bytes) | Temporal Fusion Transformer (TFT) |
| Sentiment | None | FinBERT + NewsAPI pipeline |
| Prediction style | Single-point regression | Multi-horizon with confidence intervals |
| Data source | yfinance batch pull | yfinance + FRED + NewsAPI + Options data |
| Architecture | Monolithic `app.py` (31KB) | Modular Blueprints / Routers |
| Serving | Flask dev server | FastAPI + Uvicorn + Modal GPU inference |
| Database | None (in-memory) | TimescaleDB (time-series optimized) |
| Cache | None | Redis (price cache, sentiment cache) |
| Backtesting | None | vectorbt engine |
| Tests | None | pytest + httpx async test suite |
| Indicators | ~5 basic | 20+ including Wavelet, ATR, OBV, ADX |
| Agents | 1 monolith | 3 specialized agents + ensemble |

---

## 2. New Project Structure

```
stocksage/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Pydantic settings
│   ├── dependencies.py            # Shared DI (DB, Redis, HTTP clients)
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py             # /api/predict
│   │   ├── sentiment.py           # /api/sentiment
│   │   ├── technical.py           # /api/technical
│   │   ├── backtest.py            # /api/backtest
│   │   ├── screener.py            # /api/screener
│   │   ├── competitor.py          # /api/competitor
│   │   ├── macro.py               # /api/macro
│   │   └── portfolio.py           # /api/portfolio
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── tft_model.py           # TFT training + inference
│   │   ├── finbert_sentiment.py   # FinBERT news scoring
│   │   ├── technical_agent.py     # Technical analysis agent
│   │   ├── sentiment_agent.py     # Sentiment agent
│   │   ├── macro_agent.py         # Macro data agent
│   │   └── ensemble.py            # Multi-agent ensemble layer
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── price_fetcher.py       # yfinance OHLCV fetcher
│   │   ├── news_fetcher.py        # NewsAPI / eodhd fetcher
│   │   ├── macro_fetcher.py       # FRED API fetcher
│   │   ├── options_fetcher.py     # Options flow data
│   │   └── feature_engineer.py   # Feature pipeline
│   │
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── momentum.py            # RSI, MACD, Stochastic, MFI, ROC
│   │   ├── trend.py               # EMA, SMA, ADX, Aroon, Ichimoku
│   │   ├── volatility.py          # Bollinger, ATR, Keltner, VIX corr
│   │   ├── volume.py              # OBV, VWAP, CMF, AD Line
│   │   └── wavelet.py             # Wavelet decomposition
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py              # vectorbt wrapper
│   │   └── strategies.py          # Predefined strategies
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py             # SQLAlchemy async engine
│   │   ├── models.py              # ORM models
│   │   └── migrations/            # Alembic migrations
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_client.py        # Redis cache helpers
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── predict.py             # Pydantic I/O schemas
│   │   ├── sentiment.py
│   │   ├── backtest.py
│   │   └── screener.py
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_predict.py
│       ├── test_sentiment.py
│       ├── test_technical.py
│       └── test_backtest.py
│
├── models/                        # Serialized TFT checkpoints (Git LFS)
│   ├── tft_checkpoint.ckpt
│   └── scaler.pkl
│
├── scripts/
│   ├── train_tft.py               # Standalone training script
│   ├── seed_db.py                 # DB seeding
│   └── modal_deploy.py            # Modal.com deployment
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 3. New Requirements

Replace `requirements.txt` with `pyproject.toml` using `uv` or `pip`. Below is the full flat `requirements.txt` equivalent:

```txt
# ── Web Framework ──────────────────────────────────────────────
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
httpx==0.27.0

# ── Data Validation ────────────────────────────────────────────
pydantic==2.8.2
pydantic-settings==2.4.0

# ── Database ───────────────────────────────────────────────────
sqlalchemy[asyncio]==2.0.35
asyncpg==0.29.0
alembic==1.13.2

# ── Cache ──────────────────────────────────────────────────────
redis[asyncio]==5.0.8
hiredis==3.0.0

# ── Data Fetching ──────────────────────────────────────────────
yfinance==0.2.43
pandas==2.2.2
numpy==1.26.4
pandas-market-calendars==4.4.1
fredapi==0.5.2
newsapi-python==0.2.7
requests==2.32.3

# ── Technical Analysis ─────────────────────────────────────────
pandas-ta==0.3.14b0
ta==0.11.0
PyWavelets==1.7.0

# ── Machine Learning ───────────────────────────────────────────
torch==2.4.0
pytorch-lightning==2.4.0
pytorch-forecasting==1.1.1
scikit-learn==1.5.1
scipy==1.14.1
joblib==1.4.2

# ── NLP / Sentiment ────────────────────────────────────────────
transformers==4.44.0
huggingface-hub==0.24.5
sentencepiece==0.2.0
accelerate==0.33.0

# ── Backtesting ────────────────────────────────────────────────
vectorbt==0.26.2

# ── Utilities ──────────────────────────────────────────────────
python-dotenv==1.0.1
loguru==0.7.2
tenacity==9.0.0
orjson==3.10.7
pytz==2024.1

# ── Dev & Testing ──────────────────────────────────────────────
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-cov==5.0.0
ruff==0.5.7
```

---

## 4. Backend Architecture — FastAPI Migration

### `backend/main.py`

```python
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
```

### `backend/config.py`

```python
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/stocksage"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_PRICE: int = 300        # 5 min for live prices
    CACHE_TTL_SENTIMENT: int = 1800   # 30 min for sentiment
    CACHE_TTL_MACRO: int = 86400      # 24h for macro data

    # External APIs
    NEWS_API_KEY: str = ""
    FRED_API_KEY: str = ""
    ALPHAVANTAGE_KEY: str = ""

    # ML
    TFT_CHECKPOINT_PATH: str = "models/tft_checkpoint.ckpt"
    FINBERT_MODEL: str = "ProsusAI/finbert"
    DEVICE: str = "cpu"               # "cuda" on Modal

    # Modal
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## 5. ML Engine — Temporal Fusion Transformer

The TFT replaces the existing `best_model.pkl`. It is a multi-horizon, attention-based architecture that handles static covariates (sector, market cap tier), time-varying known inputs (earnings dates, holidays), and time-varying unknown inputs (OHLCV, technical indicators, sentiment scores).

### `backend/ml/tft_model.py`

```python
import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ENCODER_LENGTH = 60      # 60 trading days lookback
MAX_PREDICTION_LENGTH = 10   # 10-day forecast horizon
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3

TIME_VARYING_UNKNOWN = [
    "close", "open", "high", "low", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_pct",
    "atr_14", "obv", "vwap", "adx_14",
    "stoch_k", "stoch_d", "cci_20", "mfi_14",
    "sentiment_score", "sentiment_volume",
]

TIME_VARYING_KNOWN = [
    "day_of_week", "day_of_month", "month", "quarter",
    "is_earnings_week", "is_holiday_proximity",
]

STATIC_CATEGORICALS = ["ticker", "sector", "market_cap_tier"]
STATIC_REALS = ["avg_daily_volume_30d", "beta"]


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_timeseries_dataset(
    df: pd.DataFrame,
    training: bool = True,
    training_cutoff: Optional[int] = None,
) -> TimeSeriesDataSet:
    """
    df must have columns: time_idx, ticker, sector, market_cap_tier,
    all TIME_VARYING_UNKNOWN, TIME_VARYING_KNOWN, STATIC_REALS columns,
    and target column 'close_next' (next day close, normalized).
    """
    if training_cutoff is None:
        training_cutoff = int(df["time_idx"].max() * 0.8)

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff] if training else df,
        time_idx="time_idx",
        target="close_return",           # predict % return, not raw price
        group_ids=["ticker"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=TIME_VARYING_KNOWN,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN,
        target_normalizer=GroupNormalizer(
            groups=["ticker"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=64,
        output_size=7,              # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )
    logger.info(f"TFT model has {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ── Trainer ───────────────────────────────────────────────────────────────────

def train_tft(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    checkpoint_dir: str = "models/",
) -> TemporalFusionTransformer:
    from torch.utils.data import DataLoader

    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=4
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=4
    )

    model = build_tft_model(training_dataset)

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, mode="min"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tft_checkpoint",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.success(f"Training complete. Best val_loss: {early_stop.best_score:.4f}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

class TFTPredictor:
    _instance = None

    def __init__(self, checkpoint_path: str):
        self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        logger.info(f"TFT loaded from {checkpoint_path}")

    @classmethod
    def get_instance(cls, checkpoint_path: str) -> "TFTPredictor":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def predict(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> dict:
        """
        Returns a dict with:
          - point_forecasts: list of 10 median predicted returns
          - quantile_bands: {q02, q10, q25, q50, q75, q90, q98} each with 10 values
          - attention_weights: top feature importance from TFT interpreter
        """
        dataset = build_timeseries_dataset(df, training=False)
        loader = dataset.to_dataloader(train=False, batch_size=1)

        with torch.no_grad():
            raw_predictions = self.model.predict(
                loader,
                mode="quantiles",
                return_index=True,
                return_decoder_lengths=True,
            )

        predictions = raw_predictions.output.squeeze().numpy()
        interpretation = self.model.interpret_output(
            raw_predictions, reduction="sum"
        )

        quantile_labels = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]

        return {
            "ticker": ticker,
            "horizon_days": MAX_PREDICTION_LENGTH,
            "point_forecasts": predictions[:, 3].tolist(),   # median = q50 index 3
            "quantile_bands": {
                f"q{int(q*100):02d}": predictions[:, i].tolist()
                for i, q in enumerate(quantile_labels)
            },
            "attention_weights": {
                "encoder_variables": interpretation["encoder_variables"]
                    .numpy().tolist(),
                "decoder_variables": interpretation["decoder_variables"]
                    .numpy().tolist(),
            },
        }
```

---

## 6. Sentiment Pipeline — FinBERT

### `backend/ml/finbert_sentiment.py`

```python
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from typing import List
from loguru import logger
import asyncio
from functools import lru_cache


FINBERT_MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def load_finbert():
    logger.info("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    logger.success("FinBERT loaded.")
    return tokenizer, model


def score_headline(text: str) -> dict:
    """
    Returns:
        {
          "label": "positive" | "negative" | "neutral",
          "score": float (-1.0 to +1.0),  # positive_prob - negative_prob
          "probabilities": {"positive": float, "negative": float, "neutral": float}
        }
    """
    tokenizer, model = load_finbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze().numpy()

    prob_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}
    sentiment_score = float(prob_dict["positive"] - prob_dict["negative"])

    return {
        "label": LABELS[int(np.argmax(probs))],
        "score": round(sentiment_score, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
    }


def score_batch(headlines: List[str], batch_size: int = 32) -> List[dict]:
    """
    Scores a list of headlines in batches for efficiency.
    Returns list of dicts from score_headline.
    """
    tokenizer, model = load_finbert()
    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1).numpy()

        for j, prob_row in enumerate(probs):
            prob_dict = {label: float(p) for label, p in zip(LABELS, prob_row)}
            score = float(prob_dict["positive"] - prob_dict["negative"])
            results.append({
                "headline": batch[j],
                "label": LABELS[int(np.argmax(prob_row))],
                "score": round(score, 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            })

    return results


def aggregate_sentiment(scored_headlines: List[dict]) -> dict:
    """
    Produces a single aggregated sentiment signal from a list of scored headlines.
    Uses volume-weighted average (more headlines = stronger signal).
    Returns:
        {
          "composite_score": float (-1 to +1),
          "label": str,
          "bullish_count": int,
          "bearish_count": int,
          "neutral_count": int,
          "total_articles": int,
          "sentiment_momentum": float  # change vs prev window
        }
    """
    if not scored_headlines:
        return {
            "composite_score": 0.0,
            "label": "neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "total_articles": 0,
            "sentiment_momentum": 0.0,
        }

    scores = [h["score"] for h in scored_headlines]
    composite = float(np.mean(scores))
    labels = [h["label"] for h in scored_headlines]

    return {
        "composite_score": round(composite, 4),
        "label": "positive" if composite > 0.15 else ("negative" if composite < -0.15 else "neutral"),
        "bullish_count": labels.count("positive"),
        "bearish_count": labels.count("negative"),
        "neutral_count": labels.count("neutral"),
        "total_articles": len(scored_headlines),
        "sentiment_momentum": round(float(np.std(scores)), 4),
    }
```

---

## 7. Multi-Agent Prediction System

### `backend/ml/ensemble.py`

```python
import asyncio
from typing import Optional
from loguru import logger
from dataclasses import dataclass
import numpy as np


@dataclass
class AgentSignal:
    agent_name: str
    direction: str          # "bullish" | "bearish" | "neutral"
    confidence: float       # 0.0 - 1.0
    weight: float           # agent's ensemble weight
    raw_score: float        # -1.0 to +1.0
    metadata: dict


async def run_technical_agent(ticker: str, df) -> AgentSignal:
    """
    Analyses 20+ technical indicators and returns a directional signal.
    Uses confluence scoring: counts how many indicators align.
    """
    from backend.ml.technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    result = await agent.analyze(ticker, df)
    return AgentSignal(
        agent_name="technical",
        direction=result["direction"],
        confidence=result["confluence_score"],
        weight=0.35,
        raw_score=result["raw_score"],
        metadata=result,
    )


async def run_sentiment_agent(ticker: str, news_window_hours: int = 48) -> AgentSignal:
    """
    Fetches latest news headlines for ticker via NewsAPI,
    scores with FinBERT, returns aggregated sentiment signal.
    """
    from backend.ml.sentiment_agent import SentimentAgent
    agent = SentimentAgent()
    result = await agent.analyze(ticker, news_window_hours)
    return AgentSignal(
        agent_name="sentiment",
        direction=result["label"],
        confidence=abs(result["composite_score"]),
        weight=0.35,
        raw_score=result["composite_score"],
        metadata=result,
    )


async def run_macro_agent(ticker: str) -> AgentSignal:
    """
    Pulls latest FRED macro indicators (Fed rate, CPI, VIX, sector ETF momentum)
    and computes a macro risk-adjusted directional signal.
    """
    from backend.ml.macro_agent import MacroAgent
    agent = MacroAgent()
    result = await agent.analyze(ticker)
    return AgentSignal(
        agent_name="macro",
        direction=result["direction"],
        confidence=result["confidence"],
        weight=0.30,
        raw_score=result["raw_score"],
        metadata=result,
    )


async def ensemble_predict(ticker: str, df) -> dict:
    """
    Runs all 3 agents in parallel, aggregates with weighted voting,
    then feeds combined signal score into TFT for final price forecast.
    
    Returns:
        {
          "ticker": str,
          "final_signal": "BUY" | "HOLD" | "SELL",
          "confidence": float (0-100),
          "agent_signals": {technical, sentiment, macro},
          "tft_forecast": {point_forecasts, quantile_bands, attention_weights},
          "risk_score": float (0-10),
          "explanation": str
        }
    """
    logger.info(f"Running ensemble prediction for {ticker}")

    # Run agents in parallel
    tech_signal, sent_signal, macro_signal = await asyncio.gather(
        run_technical_agent(ticker, df),
        run_sentiment_agent(ticker),
        run_macro_agent(ticker),
    )

    agents = [tech_signal, sent_signal, macro_signal]

    # Weighted average score
    total_weight = sum(a.weight for a in agents)
    weighted_score = sum(a.raw_score * a.weight for a in agents) / total_weight

    # Confidence = weighted average of individual confidences
    weighted_confidence = sum(a.confidence * a.weight for a in agents) / total_weight

    # Signal decision
    if weighted_score > 0.20:
        final_signal = "BUY"
    elif weighted_score < -0.20:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # Risk score: inverse of confidence, scaled by volatility proxy (std of scores)
    score_std = float(np.std([a.raw_score for a in agents]))
    risk_score = round(min(10.0, (1 - weighted_confidence) * 10 + score_std * 5), 2)

    # TFT price forecast
    from backend.ml.tft_model import TFTPredictor
    from backend.config import get_settings
    settings = get_settings()
    predictor = TFTPredictor.get_instance(settings.TFT_CHECKPOINT_PATH)
    tft_result = predictor.predict(df, ticker)

    # Add sentiment score as auxiliary feature for the explanation
    explanation = _generate_explanation(
        ticker, final_signal, weighted_score,
        tech_signal, sent_signal, macro_signal, tft_result
    )

    return {
        "ticker": ticker,
        "final_signal": final_signal,
        "confidence": round(weighted_confidence * 100, 1),
        "weighted_score": round(weighted_score, 4),
        "agent_signals": {
            "technical": tech_signal.metadata,
            "sentiment": sent_signal.metadata,
            "macro": macro_signal.metadata,
        },
        "tft_forecast": tft_result,
        "risk_score": risk_score,
        "explanation": explanation,
    }


def _generate_explanation(
    ticker, signal, score,
    tech: AgentSignal, sent: AgentSignal, macro: AgentSignal,
    tft: dict,
) -> str:
    point = tft["point_forecasts"]
    direction = "upside" if point > 0 else "downside"
    return (
        f"{ticker} shows a {signal} signal (score: {score:+.2f}). "
        f"Technical analysis is {tech.direction} (confluence: {tech.confidence:.0%}). "
        f"News sentiment is {sent.direction} with composite score {sent.raw_score:+.2f}. "
        f"Macro environment is {macro.direction}. "
        f"TFT projects {abs(point*100):.1f}% {direction} in next trading session."
    )
```

---

## 8. Data Pipeline

### `backend/data/price_fetcher.py`

```python
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
```

### `backend/data/news_fetcher.py`

```python
import httpx
from datetime import datetime, timedelta
from typing import List
from loguru import logger
from backend.config import get_settings


async def fetch_news(ticker: str, hours: int = 48) -> List[dict]:
    """
    Fetches recent news headlines for a ticker using NewsAPI.
    Returns list of {title, description, url, publishedAt, source}.
    """
    settings = get_settings()
    from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": settings.NEWS_API_KEY,
        "pageSize": 50,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://newsapi.org/v2/everything",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    articles = data.get("articles", [])
    logger.info(f"Fetched {len(articles)} articles for {ticker}")

    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")
    ]
```

### `backend/data/macro_fetcher.py`

```python
from fredapi import Fred
from backend.config import get_settings
from loguru import logger
import pandas as pd


MACRO_SERIES = {
    "fed_funds_rate":   "FEDFUNDS",
    "cpi_yoy":          "CPIAUCSL",
    "unemployment":     "UNRATE",
    "gdp_growth":       "A191RL1Q225SBEA",
    "10y_treasury":     "DGS10",
    "2y_treasury":      "DGS2",
    "yield_curve":      "T10Y2Y",
    "vix":              "VIXCLS",
    "consumer_confidence": "UMCSENT",
}

SECTOR_ETFS = {
    "Technology":       "XLK",
    "Healthcare":       "XLV",
    "Financials":       "XLF",
    "Energy":           "XLE",
    "ConsumerDisc":     "XLY",
    "Industrials":      "XLI",
    "Materials":        "XLB",
    "Utilities":        "XLU",
    "RealEstate":       "XLRE",
    "CommunicationSvcs": "XLC",
}


def fetch_macro_snapshot() -> dict:
    """Returns the latest value for each macro indicator."""
    settings = get_settings()
    fred = Fred(api_key=settings.FRED_API_KEY)
    snapshot = {}

    for name, series_id in MACRO_SERIES.items():
        try:
            series = fred.get_series(series_id).dropna()
            snapshot[name] = float(series.iloc[-1])
        except Exception as e:
            logger.warning(f"FRED fetch failed for {name}: {e}")
            snapshot[name] = None

    logger.info(f"Macro snapshot: {snapshot}")
    return snapshot


def compute_macro_score(snapshot: dict, sector: str = "Technology") -> dict:
    """
    Converts raw macro data into a directional macro signal.
    Simple rule-based scoring (can be upgraded to a classifier later).
    """
    score = 0.0
    reasons = []

    if snapshot.get("yield_curve") is not None:
        if snapshot["yield_curve"] > 0.5:
            score += 0.2
            reasons.append("Yield curve positive (non-inverted)")
        elif snapshot["yield_curve"] < -0.3:
            score -= 0.3
            reasons.append("Inverted yield curve (recession risk)")

    if snapshot.get("vix") is not None:
        if snapshot["vix"] < 18:
            score += 0.15
            reasons.append("Low VIX (low fear)")
        elif snapshot["vix"] > 30:
            score -= 0.25
            reasons.append("High VIX (market fear elevated)")

    if snapshot.get("fed_funds_rate") is not None:
        if snapshot["fed_funds_rate"] < 3.0:
            score += 0.1
            reasons.append("Accommodative Fed policy")
        elif snapshot["fed_funds_rate"] > 5.0:
            score -= 0.1
            reasons.append("Restrictive Fed policy")

    direction = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")

    return {
        "direction": direction,
        "raw_score": round(score, 4),
        "confidence": min(abs(score) + 0.3, 1.0),
        "reasons": reasons,
        "snapshot": snapshot,
    }
```

---

## 9. Technical Indicators Engine

### `backend/indicators/__init__.py`

```python
import pandas as pd
import pandas_ta as ta
import pywt
import numpy as np
from typing import Optional


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: appends all technical indicators to OHLCV df.
    Input df must have columns: open, high, low, close, volume
    """
    df = df.copy()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"]     = ta.rsi(df["close"], length=14)
    df["rsi_28"]     = ta.rsi(df["close"], length=28)
    macd             = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]       = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]  = macd["MACDh_12_26_9"]
    stoch            = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"]    = stoch["STOCHk_14_3_3"]
    df["stoch_d"]    = stoch["STOCHd_14_3_3"]
    df["cci_20"]     = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["mfi_14"]     = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["roc_10"]     = ta.roc(df["close"], length=10)
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema_9"]      = ta.ema(df["close"], length=9)
    df["ema_21"]     = ta.ema(df["close"], length=21)
    df["ema_50"]     = ta.ema(df["close"], length=50)
    df["ema_200"]    = ta.ema(df["close"], length=200)
    df["sma_20"]     = ta.sma(df["close"], length=20)
    df["sma_50"]     = ta.sma(df["close"], length=50)
    adx              = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"]     = adx["ADX_14"]
    df["dmp_14"]     = adx["DMP_14"]
    df["dmn_14"]     = adx["DMN_14"]
    aroon            = ta.aroon(df["high"], df["low"], length=25)
    df["aroon_up"]   = aroon["AROONU_25"]
    df["aroon_down"] = aroon["AROOND_25"]

    # ── Volatility ────────────────────────────────────────────────────────────
    bb               = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"]   = bb["BBU_20_2.0"]
    df["bb_middle"]  = bb["BBM_20_2.0"]
    df["bb_lower"]   = bb["BBL_20_2.0"]
    df["bb_pct"]     = bb["BBP_20_2.0"]
    df["bb_width"]   = bb["BBB_20_2.0"]
    df["atr_14"]     = ta.atr(df["high"], df["low"], df["close"], length=14)
    kc               = ta.kc(df["high"], df["low"], df["close"])
    df["kc_upper"]   = kc["KCUe_20_2"]
    df["kc_lower"]   = kc["KCLe_20_2"]

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]        = ta.obv(df["close"], df["volume"])
    df["vwap"]       = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["cmf_20"]     = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
    df["ad_line"]    = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    df["volume_sma20"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]

    # ── Wavelet Decomposition (noise-filtered signal) ─────────────────────────
    df["close_wavelet"] = _wavelet_smooth(df["close"].values)

    # ── Price-derived features ────────────────────────────────────────────────
    df["daily_return"]   = df["close"].pct_change()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["hl_pct"]         = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
    df["close_return"]   = df["close"].pct_change()  # TFT target

    return df.dropna()


def _wavelet_smooth(prices: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Applies discrete wavelet transform to remove high-frequency noise.
    Returns the low-frequency approximation coefficients reconstructed to original length.
    """
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # Zero out detail coefficients (noise), keep approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    # Align length
    return smoothed[:len(prices)]


def compute_confluence_score(df: pd.DataFrame) -> dict:
    """
    Counts indicator alignment for technical direction signal.
    Returns score from -1.0 (all bearish) to +1.0 (all bullish).
    """
    latest = df.iloc[-1]
    signals = []

    # RSI
    if latest["rsi_14"] > 55: signals.append(1)
    elif latest["rsi_14"] < 45: signals.append(-1)
    else: signals.append(0)

    # MACD
    if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
        signals.append(1)
    elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
        signals.append(-1)
    else: signals.append(0)

    # EMA alignment
    if latest["ema_9"] > latest["ema_21"] > latest["ema_50"]: signals.append(1)
    elif latest["ema_9"] < latest["ema_21"] < latest["ema_50"]: signals.append(-1)
    else: signals.append(0)

    # Bollinger
    if latest["close"] > latest["bb_middle"]: signals.append(1)
    elif latest["close"] < latest["bb_middle"]: signals.append(-1)
    else: signals.append(0)

    # ADX trend strength
    if latest["adx_14"] > 25:
        signals.append(1 if latest["dmp_14"] > latest["dmn_14"] else -1)
    else: signals.append(0)

    # Volume confirmation
    if latest["volume_ratio"] > 1.5 and latest["daily_return"] > 0: signals.append(1)
    elif latest["volume_ratio"] > 1.5 and latest["daily_return"] < 0: signals.append(-1)
    else: signals.append(0)

    # MFI
    if latest["mfi_14"] > 60: signals.append(1)
    elif latest["mfi_14"] < 40: signals.append(-1)
    else: signals.append(0)

    # CMF
    if latest["cmf_20"] > 0.1: signals.append(1)
    elif latest["cmf_20"] < -0.1: signals.append(-1)
    else: signals.append(0)

    score = float(np.mean(signals))
    direction = "bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral")

    return {
        "direction": direction,
        "raw_score": round(score, 4),
        "confluence_score": round(abs(score), 4),
        "signal_count": len(signals),
        "bullish_signals": signals.count(1),
        "bearish_signals": signals.count(-1),
        "neutral_signals": signals.count(0),
    }
```

---

## 10. Backtesting Engine

### `backend/backtesting/engine.py`

```python
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Callable, Optional
from loguru import logger


def run_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    ticker: str,
    initial_cash: float = 10_000.0,
    fees: float = 0.001,       # 0.1% per trade
    slippage: float = 0.001,   # 0.1% slippage
) -> dict:
    """
    Runs a vectorbt backtest given entry/exit boolean signals.
    Returns comprehensive performance metrics.
    """
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
    )

    stats = portfolio.stats()
    returns = portfolio.returns()
    benchmark_returns = close.pct_change().fillna(0)

    # Sharpe ratio (annualised)
    sharpe = float(stats.get("Sharpe Ratio", 0))

    # Max drawdown
    max_dd = float(stats.get("Max Drawdown [%]", 0))

    # Total return
    total_return = float(stats.get("Total Return [%]", 0))

    # Win rate
    win_rate = float(stats.get("Win Rate [%]", 0))

    # vs Benchmark
    bm_total = float(benchmark_returns.add(1).prod() - 1) * 100

    equity_curve = portfolio.value().rename("portfolio").to_frame()
    equity_curve["benchmark"] = initial_cash * benchmark_returns.add(1).cumprod()

    logger.success(
        f"Backtest {ticker}: return={total_return:.1f}% "
        f"sharpe={sharpe:.2f} max_dd={max_dd:.1f}%"
    )

    return {
        "ticker": ticker,
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(bm_total, 2),
        "alpha_pct": round(total_return - bm_total, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": int(stats.get("Total Trades", 0)),
        "avg_trade_duration": str(stats.get("Avg Winning Trade Duration", "N/A")),
        "equity_curve": equity_curve.reset_index().to_dict(orient="records"),
        "initial_cash": initial_cash,
        "final_value": round(float(portfolio.final_value()), 2),
    }


def signal_from_tft(
    df: pd.DataFrame,
    confidence_threshold: float = 0.65,
) -> tuple[pd.Series, pd.Series]:
    """
    Generates entry/exit signals from TFT point forecast.
    Entry: predicted next-day return > 0 AND confidence > threshold
    Exit: predicted return < 0 OR RSI > 75
    """
    entries = (df["tft_forecast_return"] > 0) & (df["tft_confidence"] > confidence_threshold)
    exits = (df["tft_forecast_return"] < 0) | (df["rsi_14"] > 75)
    return entries.astype(bool), exits.astype(bool)


def signal_from_rsi_macd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Classic RSI + MACD crossover strategy for benchmarking."""
    entries = (df["rsi_14"] < 35) & (df["macd"] > df["macd_signal"])
    exits   = (df["rsi_14"] > 65) | (df["macd"] < df["macd_signal"])
    return entries.astype(bool), exits.astype(bool)
```

---

## 11. API Endpoints Reference

### `backend/routers/predict.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from backend.ml.ensemble import ensemble_predict
from backend.data.price_fetcher import fetch_ohlcv
from backend.data.feature_engineer import build_feature_df
from backend.cache.redis_client import get_cached, set_cache
from backend.config import get_settings

router = APIRouter()


class PredictRequest(BaseModel):
    ticker: str
    period: str = "2y"


class PredictResponse(BaseModel):
    ticker: str
    final_signal: str
    confidence: float
    weighted_score: float
    agent_signals: dict
    tft_forecast: dict
    risk_score: float
    explanation: str


@router.post("/", response_model=PredictResponse)
async def predict(req: PredictRequest):
    cache_key = f"predict:{req.ticker}:{req.period}"
    cached = await get_cached(cache_key)
    if cached:
        return cached

    try:
        df = fetch_ohlcv(req.ticker, period=req.period)
        df = build_feature_df(df)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    try:
        result = await ensemble_predict(req.ticker, df)
    except Exception as e:
        logger.error(f"Ensemble predict failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    await set_cache(cache_key, result, ttl=300)
    return result
```

### Full Endpoint Map

| Method | Path | Description | Cache TTL |
|---|---|---|---|
| `POST` | `/api/predict/` | Full ensemble prediction + TFT forecast | 5 min |
| `GET` | `/api/predict/{ticker}/history` | Historical predictions stored in DB | — |
| `GET` | `/api/sentiment/{ticker}` | FinBERT scored news for ticker | 30 min |
| `GET` | `/api/technical/{ticker}` | All 20+ indicators + confluence score | 5 min |
| `POST` | `/api/backtest/` | Run backtest on chosen strategy | No cache |
| `GET` | `/api/screener/` | Filter stocks by 15+ metric criteria | 15 min |
| `GET` | `/api/competitor/{ticker}` | Peer comparison with key ratios | 1 hour |
| `GET` | `/api/macro/snapshot` | Latest FRED macro data + score | 24 hours |
| `GET` | `/api/macro/sector/{sector}` | Sector ETF momentum + macro score | 1 hour |
| `POST` | `/api/portfolio/analyze` | Multi-stock portfolio risk analysis | 10 min |
| `GET` | `/api/portfolio/correlation` | Correlation matrix for tickers | 10 min |
| `GET` | `/health` | Health check | No cache |

---

## 12. Database Schema — TimescaleDB

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV price data (hypertable for time-series efficiency)
CREATE TABLE price_data (
    time         TIMESTAMPTZ NOT NULL,
    ticker       TEXT        NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       BIGINT,
    PRIMARY KEY (time, ticker)
);
SELECT create_hypertable('price_data', 'time');
CREATE INDEX ON price_data (ticker, time DESC);

-- News articles and FinBERT scores
CREATE TABLE news_sentiment (
    id           SERIAL PRIMARY KEY,
    ticker       TEXT        NOT NULL,
    headline     TEXT        NOT NULL,
    source       TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    finbert_label TEXT,
    finbert_score DOUBLE PRECISION,
    url          TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON news_sentiment (ticker, published_at DESC);

-- TFT prediction log
CREATE TABLE predictions (
    id                SERIAL PRIMARY KEY,
    ticker            TEXT           NOT NULL,
    predicted_at      TIMESTAMPTZ    DEFAULT NOW(),
    final_signal      TEXT,
    confidence        DOUBLE PRECISION,
    weighted_score    DOUBLE PRECISION,
    risk_score        DOUBLE PRECISION,
    tft_point_d1      DOUBLE PRECISION,
    tft_point_d5      DOUBLE PRECISION,
    tft_point_d10     DOUBLE PRECISION,
    agent_technical   JSONB,
    agent_sentiment   JSONB,
    agent_macro       JSONB,
    tft_quantile_bands JSONB
);
CREATE INDEX ON predictions (ticker, predicted_at DESC);

-- Backtest results
CREATE TABLE backtest_results (
    id                SERIAL PRIMARY KEY,
    ticker            TEXT           NOT NULL,
    strategy          TEXT           NOT NULL,
    run_at            TIMESTAMPTZ    DEFAULT NOW(),
    total_return_pct  DOUBLE PRECISION,
    benchmark_return_pct DOUBLE PRECISION,
    alpha_pct         DOUBLE PRECISION,
    sharpe_ratio      DOUBLE PRECISION,
    max_drawdown_pct  DOUBLE PRECISION,
    win_rate_pct      DOUBLE PRECISION,
    total_trades      INTEGER,
    initial_cash      DOUBLE PRECISION,
    final_value       DOUBLE PRECISION,
    equity_curve      JSONB
);

-- Macro snapshot log
CREATE TABLE macro_snapshots (
    id               SERIAL PRIMARY KEY,
    captured_at      TIMESTAMPTZ DEFAULT NOW(),
    fed_funds_rate   DOUBLE PRECISION,
    cpi_yoy          DOUBLE PRECISION,
    unemployment     DOUBLE PRECISION,
    gdp_growth       DOUBLE PRECISION,
    yield_curve      DOUBLE PRECISION,
    vix              DOUBLE PRECISION,
    macro_score      DOUBLE PRECISION,
    macro_direction  TEXT
);
SELECT create_hypertable('macro_snapshots', 'captured_at');

-- Screener watchlist (user-defined)
CREATE TABLE watchlist (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT,
    ticker     TEXT,
    added_at   TIMESTAMPTZ DEFAULT NOW(),
    notes      TEXT
);
```

---

## 13. Caching Layer — Redis

### `backend/cache/redis_client.py`

```python
import redis.asyncio as redis
import orjson
from typing import Any, Optional
from loguru import logger
from backend.config import get_settings

_redis_client: Optional[redis.Redis] = None


async def init_redis():
    global _redis_client
    settings = get_settings()
    _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
    await _redis_client.ping()
    logger.success("Redis connected.")


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()


def get_redis() -> redis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis not initialised. Call init_redis() first.")
    return _redis_client


async def get_cached(key: str) -> Optional[Any]:
    client = get_redis()
    data = await client.get(key)
    if data:
        logger.debug(f"Cache HIT: {key}")
        return orjson.loads(data)
    logger.debug(f"Cache MISS: {key}")
    return None


async def set_cache(key: str, value: Any, ttl: int = 300):
    client = get_redis()
    await client.setex(key, ttl, orjson.dumps(value))
    logger.debug(f"Cache SET: {key} (TTL={ttl}s)")


async def invalidate(key: str):
    client = get_redis()
    await client.delete(key)
    logger.info(f"Cache INVALIDATED: {key}")
```

---

## 14. Model Serving — Modal.com

### `scripts/modal_deploy.py`

```python
import modal

# Define Modal image with all ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.4.0",
        "pytorch-lightning==2.4.0",
        "pytorch-forecasting==1.1.1",
        "transformers==4.44.0",
        "huggingface-hub==0.24.5",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "pandas-ta==0.3.14b0",
        "PyWavelets==1.7.0",
        "yfinance==0.2.43",
        "fredapi==0.5.2",
    ])
)

# Mount model files from local
model_volume = modal.Volume.from_name("stocksage-models", create_if_missing=True)

app = modal.App("stocksage-ml", image=image)


@app.function(
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=120,
    retries=2,
)
def predict_tft(ticker: str, df_json: str) -> dict:
    """
    GPU-accelerated TFT inference on Modal.
    Called remotely from FastAPI backend via modal.Function.lookup().
    """
    import pandas as pd
    from backend.ml.tft_model import TFTPredictor

    df = pd.read_json(df_json)
    predictor = TFTPredictor.get_instance("/models/tft_checkpoint.ckpt")
    return predictor.predict(df, ticker)


@app.function(
    gpu=None,                   # FinBERT runs fine on CPU
    volumes={"/models": model_volume},
    timeout=60,
)
def score_sentiment_batch(headlines: list[str]) -> list[dict]:
    """
    Batch FinBERT sentiment scoring on Modal.
    """
    from backend.ml.finbert_sentiment import score_batch
    return score_batch(headlines)


@app.local_entrypoint()
def main():
    # Test call
    result = predict_tft.remote("AAPL", "{}")
    print(result)
```

---

## 15. Environment & Config

### `.env.example`

```env
# App
APP_ENV=development
DEBUG=true

# Database (TimescaleDB / PostgreSQL)
DATABASE_URL=postgresql+asyncpg://stocksage:password@localhost:5432/stocksage

# Redis
REDIS_URL=redis://localhost:6379/0

# External APIs
NEWS_API_KEY=your_newsapi_key_here           # https://newsapi.org
FRED_API_KEY=your_fred_key_here              # https://fred.stlouisfed.org/docs/api/
ALPHAVANTAGE_KEY=your_alphavantage_key_here  # https://www.alphavantage.co (optional)

# ML
TFT_CHECKPOINT_PATH=models/tft_checkpoint.ckpt
FINBERT_MODEL=ProsusAI/finbert
DEVICE=cpu

# Modal
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

---

## 16. Deployment Architecture

```
┌───────────────────────────────────────────────────────┐
│              React Frontend (Vercel)                  │
│         TradingView Lightweight Charts                 │
└───────────────────────┬───────────────────────────────┘
                        │  HTTPS / WebSocket
┌───────────────────────▼───────────────────────────────┐
│           FastAPI Backend (Railway)                    │
│   Uvicorn workers + async routers                     │
│   Routers: predict / sentiment / technical /          │
│            backtest / screener / macro / portfolio     │
└──────────┬────────────────────────┬───────────────────┘
           │                        │
┌──────────▼──────┐      ┌──────────▼──────────────────┐
│  TimescaleDB    │      │     Redis Cache               │
│  (Railway)      │      │     (Upstash free tier)       │
│  price_data     │      │     TTL: 5min–24h             │
│  predictions    │      └─────────────────────────────-┘
│  news_sentiment │
│  backtest_results│     ┌─────────────────────────────-┐
└─────────────────┘      │  Modal.com (Serverless GPU)  │
                         │  TFT inference (T4 GPU)      │
                         │  FinBERT batch scoring       │
                         │  Pay-per-call (no idle cost) │
                         └─────────────────────────────-┘
```

### `docker-compose.yml` (local dev)

```yaml
version: "3.9"

services:
  db:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_USER: stocksage
      POSTGRES_PASSWORD: password
      POSTGRES_DB: stocksage
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models

volumes:
  pgdata:
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 17. Git Hygiene Fixes

Run these **immediately** before any new development:

```bash
# 1. Remove tracked node_modules (17k lines committed — must be fixed)
git rm -r --cached frontend/node_modules
echo "frontend/node_modules/" >> .gitignore
git commit -m "fix: remove node_modules from tracking"

# 2. Set up Git LFS for model files
git lfs install
git lfs track "*.pkl"
git lfs track "*.ckpt"
git lfs track "*.pt"
git lfs track "*.pth"
echo "*.pkl filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
echo "*.ckpt filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
git commit -m "chore: track model files with Git LFS"

# 3. Remove old Flask files from root
git rm app.py model.py best_model.pkl scaler.pkl AIAnalysis.css
git commit -m "refactor: remove legacy Flask files, migrating to backend/"

# 4. Enable issues in GitHub repo settings (do via GitHub UI)
# Settings → Features → Issues ✓

# 5. Add .gitignore entries
cat >> .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env

# ML models
*.pkl
*.ckpt
*.pt
*.pth
models/

# Node
node_modules/
dist/
build/
.next/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
EOF

git add .gitignore
git commit -m "chore: update .gitignore"
```

---

## 18. Sprint Roadmap

### Week 1 — Foundation
- [ ] Run Git hygiene fixes (Section 17)
- [ ] Set up `docker-compose.yml` with TimescaleDB + Redis locally
- [ ] Create `backend/` folder structure (Section 2)
- [ ] Write `main.py`, `config.py`, `dependencies.py`
- [ ] Implement `price_fetcher.py` + `news_fetcher.py` + `macro_fetcher.py`
- [ ] Implement all technical indicators in `backend/indicators/`
- [ ] Port old Flask routes to FastAPI routers (technical, competitor)
- [ ] Write `test_technical.py` with pytest

### Week 2 — ML Engine
- [ ] Implement `finbert_sentiment.py` + test on 20 sample headlines
- [ ] Implement `sentiment_agent.py` + `sentiment.py` router
- [ ] Build `feature_engineer.py` that merges OHLCV + indicators + sentiment
- [ ] Train TFT model via `scripts/train_tft.py` on 10 tickers (2 years data)
- [ ] Implement `tft_model.py` inference path
- [ ] Implement `macro_agent.py` + `technical_agent.py`
- [ ] Wire up `ensemble.py` end-to-end
- [ ] Write `test_predict.py`

### Week 3 — Power Features
- [ ] Implement `backtesting/engine.py` + `backtest.py` router
- [ ] Implement `screener.py` router with 15+ filters
- [ ] Deploy TFT + FinBERT to Modal.com (Section 14)
- [ ] Set up Redis caching on all expensive routes (Section 13)
- [ ] Run database migrations with Alembic
- [ ] Write `test_backtest.py`, `test_sentiment.py`

### Week 4 — Deployment & Polish
- [ ] Deploy backend to Railway with TimescaleDB + Upstash Redis
- [ ] Set up Alembic auto-migrations in CI
- [ ] Add Loguru structured logging + Sentry error tracking
- [ ] Add rate limiting with `slowapi`
- [ ] Write API documentation (FastAPI auto-generates `/docs`)
- [ ] Load test with `locust` (target: 100 req/s on `/api/predict/`)
- [ ] Pin all dependency versions for reproducibility
