# Deployment Readiness Report

## Status: READY FOR DEPLOYMENT

## Architecture
- Frontend: React (react-scripts 5)
- Backend: FastAPI 0.115 + Uvicorn
- Database: PostgreSQL/TimescaleDB via SQLAlchemy async
- Auth: No user-auth system implemented in current product scope

## Required Environment Variables

| Variable | Purpose | Source |
|---|---|---|
| APP_ENV | Runtime mode and environment behavior | Set by deploy platform (`production`) |
| DEBUG | Debug-level logging toggle | Set by deploy platform (`false` in prod) |
| DATABASE_URL | Primary DB connection string | PostgreSQL/Timescale provider |
| REDIS_URL | Cache connection string | Redis provider |
| NEWS_API_KEY | News sentiment ingestion | https://newsapi.org |
| FRED_API_KEY | Macroeconomic data ingestion | https://fred.stlouisfed.org |
| ALPHAVANTAGE_KEY | Optional external market API key | https://www.alphavantage.co |
| TFT_CHECKPOINT_PATH | Optional TFT model checkpoint path | Artifact storage / mounted volume |
| FINBERT_MODEL | HuggingFace model id for sentiment | HuggingFace model registry |
| DEVICE | Inference device (`cpu`/`cuda`) | Deployment runtime config |
| MODAL_TOKEN_ID | Modal API token id for remote inference | Modal account settings |
| MODAL_TOKEN_SECRET | Modal API token secret | Modal account settings |

## Deployment Steps
1. Provision PostgreSQL/TimescaleDB and Redis.
2. Configure all required environment variables.
3. Build backend image from `Dockerfile`.
4. Deploy backend (`uvicorn backend.main:app --host 0.0.0.0 --port 8000`).
5. Build and deploy frontend from `frontend/` with API proxy/base URL pointing to backend.
6. Verify `/health` and key API endpoints (`/api/analyze`, `/api/predict`, `/api/technical/{ticker}`).

## Post-Deployment Checklist
- [ ] Validate DB and Redis connectivity in logs
- [ ] Verify `GET /health` returns `{"status":"ok","version":"2.0.0"}`
- [ ] Verify frontend submit flow works end-to-end against `/api/analyze`
- [ ] Confirm external API keys are present (NewsAPI/FRED) for full-signal mode
- [ ] If using TFT, ensure checkpoint exists at `TFT_CHECKPOINT_PATH`

## Known Limitations / Future Work
- External market/news providers are required for full real-time functionality.
- No user-auth/identity layer is currently implemented.
- Frontend currently uses a CRA baseline (consider Vite/Next migration later).

## Issues Fixed in This Session
- Unified frontend API flow to FastAPI (`/api/analyze`) instead of legacy Flask route.
- Added compatibility analyze endpoint in FastAPI for frontend payload shape.
- Resolved Python dependency install blockers by removing incompatible `pandas-ta` pin.
- Added robust fallback indicator computation when `pandas-ta` is unavailable.
- Fixed sentiment test/runtime fallback behavior for offline/no-model scenarios.
- Fixed failing tests due incorrect patch targets in sentiment/predict tests.
- Improved retry error behavior in price fetcher to return meaningful 404s instead of 500s.
- Updated README to reflect actual architecture, setup, env vars, and validation commands.

