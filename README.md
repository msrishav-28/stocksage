# StockSage 2.0 — AI-Powered Stock Analysis & Prediction

StockSage is an AI stock-intelligence platform. It analyses a ticker through a
multi-agent ensemble — technical indicators, news sentiment, and the macro
environment — and produces an explainable BUY / HOLD / SELL signal with a
confidence score, risk score, and a written investment thesis.

## Architecture

```
                ┌──────────────────────────────────────────┐
   HTTP  ──────▶│  FastAPI  (backend/main.py)               │
                │  8 routers · CORS · per-IP rate limiting  │
                └───────────────┬──────────────────────────-┘
                                │
                 ┌──────────────▼───────────────┐
                 │  Orchestrator                 │
                 │  (backend/ml/orchestrator.py) │
                 │  trace ▸ memory ▸ agents ▸    │
                 │  synthesis ▸ guardrails       │
                 └───┬───────────┬───────────┬──-┘
                     │           │           │
            ┌────────▼──┐ ┌──────▼─────┐ ┌───▼────────┐
            │ Technical │ │ Sentiment  │ │   Macro    │   ← BaseAgent
            │  agent    │ │  agent     │ │   agent    │     (ReAct + tools)
            └───────────┘ └────────────┘ └────────────┘
                     │           │           │
            indicators   FinBERT + NewsAPI   FRED
                                │
                 TFT forecaster (optional, GPU via Modal)

   Postgres / TimescaleDB  ·  Redis cache  (both degrade gracefully)
```

### Key components

| Area | Implementation |
|---|---|
| API | FastAPI, 8 routers under `/api/*`, env-driven CORS, slowapi rate limiting |
| Agents | `TechnicalAgent`, `SentimentAgent`, `MacroAgent` — `BaseAgent` ReAct subclasses calling tools via a `ToolRegistry` |
| Orchestration | Accuracy-weighted vote + episodic-memory prior + VIX-reliability dampening, with input/output guardrails and per-request tracing |
| Forecasting | Temporal Fusion Transformer (`pytorch-forecasting`) — optional, loaded only when a checkpoint is present |
| Sentiment | FinBERT (`ProsusAI/finbert`) scoring of NewsAPI headlines |
| Indicators | 20+ technical indicators (`pandas-ta`, with pure-Python fallbacks) |
| Backtesting | `vectorbt` engine with 5 predefined strategies |
| Data | yfinance (prices), NewsAPI (news), FRED (macro) |
| Storage | SQLAlchemy async + PostgreSQL/TimescaleDB; Redis response cache |

## Tech stack

Python 3.11 · FastAPI · SQLAlchemy (async) · Alembic · Redis · PyTorch /
pytorch-forecasting · Transformers · pandas-ta · vectorbt · loguru · pytest.

## Project structure

```
backend/
  main.py            FastAPI app + middleware
  config.py          Pydantic settings
  routers/           API endpoints (predict, sentiment, technical, ...)
  ml/                agents, orchestrator, tools, memory, guardrails, telemetry, TFT
  data/              price / news / macro fetchers + feature engineering
  indicators/        technical indicator engine
  backtesting/       vectorbt engine + strategies
  db/                async session, ORM models, Alembic migrations
  cache/             Redis client
  schemas/           Pydantic request/response models
  tests/             pytest suite
scripts/             train_tft.py, seed_db.py, modal_deploy.py
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt       # + test/lint tooling

# 2. Configure environment
cp .env.example .env                      # then fill in API keys

# 3. Start infrastructure (Postgres/TimescaleDB + Redis)
docker compose up -d db redis

# 4. Apply database migrations
alembic upgrade head
#    (or set DB_AUTO_CREATE=true in .env for local dev)
```

`NEWS_API_KEY` (newsapi.org) and `FRED_API_KEY` (fred.stlouisfed.org) are
optional — without them the sentiment and macro agents return neutral signals
and the rest of the system still works.

## Running

```bash
# Local
uvicorn backend.main:app --reload

# Full stack via Docker
docker compose up
```

Interactive API docs: `http://localhost:8000/docs`

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Health check |
| `POST` | `/api/predict/` | Full orchestrated ensemble prediction |
| `GET`  | `/api/predict/{ticker}/history` | Stored prediction history |
| `GET`  | `/api/sentiment/{ticker}` | FinBERT-scored news sentiment |
| `GET`  | `/api/technical/{ticker}` | 20+ indicators + confluence score |
| `POST` | `/api/backtest/` | Run a strategy backtest |
| `GET`  | `/api/screener/` | Screen the universe by criteria |
| `GET`  | `/api/competitor/{ticker}` | Peer comparison |
| `GET`  | `/api/macro/snapshot` | FRED macro snapshot + score |
| `GET`  | `/api/macro/sector/{sector}` | Sector ETF analysis |
| `POST` | `/api/portfolio/analyze` | Portfolio risk metrics |
| `GET`  | `/api/portfolio/correlation` | Correlation matrix |

## Testing

```bash
pytest                 # full suite
ruff check backend/    # lint
```

## Model training

The TFT forecaster is optional. To train a checkpoint:

```bash
python scripts/train_tft.py --tickers AAPL MSFT GOOGL --period 2y
```

This writes `models/tft_checkpoint.ckpt` and `models/tft_dataset_params.pkl`.
Inference loads both; without a checkpoint the ensemble runs on agent signals
alone.

## Frontend

`frontend/` is a React single-page app (Create React App) that consumes the v2
API. It presents the prediction signal, the written thesis, the per-agent
breakdown, an interactive technical price chart, and peer comparison.

```bash
cd frontend
npm install
cp .env.example .env       # optional — only needed for a non-local API URL
npm start                  # dev server → http://localhost:3000
npm run build              # production bundle → frontend/build
```

The dev server proxies `/api` and `/health` to `http://127.0.0.1:8000` (see
`proxy` in `package.json`), so just run the backend alongside it. For a
production build, set `REACT_APP_API_URL` to the deployed API origin.

Stack: React 18 · MUI 5 · Chart.js 3 · axios. See `frontend/README.md` for
more detail.

## Running the full stack

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — frontend
cd frontend && npm start
```

Then open `http://localhost:3000`.
