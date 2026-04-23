# StockSage

StockSage is an AI-powered stock analysis platform with:
- **FastAPI backend** (`backend/`)
- **React frontend** (`frontend/`)
- Multi-agent prediction (technical + sentiment + macro), technical indicators, backtesting, screener, and portfolio analytics.

## Architecture

- **Backend**: FastAPI + Pydantic + SQLAlchemy (async) + Redis cache
- **Frontend**: React (CRA) + MUI + Axios + Chart.js
- **Data providers**: Yahoo Finance (prices), NewsAPI (news), FRED (macro)
- **Optional ML**: FinBERT + TFT forecast (checkpoint-based)

## Repository Layout

```text
backend/     FastAPI app, routers, ML agents, indicators, tests
frontend/    React app
scripts/     Training/seed/deploy helpers
```

## Required Environment Variables

Copy `.env.example` to `.env` and set values:

| Variable | Required | Purpose |
|---|---|---|
| APP_ENV | yes | Runtime environment (`development`, `production`) |
| DEBUG | yes | Debug logging toggle |
| DATABASE_URL | yes | Async PostgreSQL URL |
| REDIS_URL | yes | Redis connection URL |
| NEWS_API_KEY | optional | News sentiment source |
| FRED_API_KEY | optional | Macro data source |
| ALPHAVANTAGE_KEY | optional | Reserved external provider key |
| TFT_CHECKPOINT_PATH | optional | TFT checkpoint file path |
| FINBERT_MODEL | optional | HuggingFace model id |
| DEVICE | optional | `cpu` or `cuda` |
| MODAL_TOKEN_ID | optional | Modal auth token id |
| MODAL_TOKEN_SECRET | optional | Modal auth token secret |

## Local Development

### Backend

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend dev server proxies API requests to `http://127.0.0.1:8000`.

## Core API Endpoints

- `POST /api/analyze`
- `POST /api/predict`
- `GET /api/sentiment/{ticker}`
- `GET /api/technical/{ticker}`
- `POST /api/backtest`
- `GET /api/screener`
- `GET /api/competitor/{ticker}`
- `GET /api/macro/snapshot`
- `POST /api/portfolio/analyze`

## Validation Commands

```bash
# Backend tests
python -m pytest -q

# Frontend lint/build/test
cd frontend
npx eslint src --ext .js,.jsx
npm run build
CI=true npm test -- --watch=false --passWithNoTests
```

