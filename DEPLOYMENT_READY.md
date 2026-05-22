# Deployment Readiness

## Architecture

- **Frontend:** React (Create React App) single-page app — `frontend/`
- **Backend:** FastAPI 0.115 + Uvicorn — `backend/`
- **Database:** PostgreSQL / TimescaleDB via SQLAlchemy async (degrades gracefully if absent)
- **Cache:** Redis (degrades gracefully if absent)
- **Auth:** none — no user-identity layer in the current scope

## Environment variables

| Variable | Purpose | Notes |
|---|---|---|
| `APP_ENV` | Runtime mode | `production` in prod |
| `DEBUG` | Debug logging / SQL echo | `false` in prod |
| `DATABASE_URL` | PostgreSQL/TimescaleDB connection (asyncpg) | required for persistence |
| `DB_AUTO_CREATE` | Create tables on startup instead of Alembic | `false` in prod |
| `REDIS_URL` | Redis connection | optional — caching only |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | set to the deployed frontend URL |
| `RATE_LIMIT` | Per-IP rate limit (slowapi syntax) | e.g. `120/minute` |
| `NEWS_API_KEY` | News sentiment ingestion | optional — neutral fallback without it |
| `FRED_API_KEY` | Macro data ingestion | optional — neutral fallback without it |
| `TFT_CHECKPOINT_PATH` | Trained TFT checkpoint | optional — forecast omitted without it |
| `FINBERT_MODEL` | HuggingFace model id for sentiment | defaults to `ProsusAI/finbert` |
| `DEVICE` | Inference device (`cpu`/`cuda`) | — |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Modal GPU inference | optional |

## Deployment steps

1. Provision PostgreSQL/TimescaleDB and Redis.
2. Configure the environment variables above.
3. Apply database migrations: `alembic upgrade head`.
4. Build the backend image from `Dockerfile`.
5. Deploy the backend: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
6. Build the frontend (`cd frontend && npm install && npm run build`) with
   `REACT_APP_API_URL` pointing at the backend; serve `frontend/build`.

## Post-deployment checklist

- [ ] DB and Redis connectivity confirmed in startup logs
- [ ] `GET /health` returns `{"status":"ok","version":"2.0.0"}`
- [ ] `POST /api/predict/` returns a signal for a known ticker (e.g. `AAPL`)
- [ ] `GET /api/technical/{ticker}` and `/api/competitor/{ticker}` respond
- [ ] Frontend loads and completes a prediction end-to-end
- [ ] `CORS_ORIGINS` includes the deployed frontend origin
- [ ] If using the TFT forecaster, the checkpoint exists at `TFT_CHECKPOINT_PATH`

## Known limitations / future work

- External providers (NewsAPI, FRED) are needed for full-fidelity signals;
  without them the sentiment and macro agents return neutral fallbacks.
- No user-auth / identity layer is implemented.
- The frontend is on a Create React App baseline (a Vite migration is optional
  future work).
- The TFT forecaster is optional and must be trained before it contributes.
