# StockSage Frontend

React single-page app for the StockSage 2.0 API. Enter a ticker and the UI
runs a prediction, then shows the signal, confidence, risk, written thesis,
the three-agent breakdown, an interactive price chart, and peer comparison.

## Stack

- React 18 (Create React App)
- MUI 5 — components and dark theme
- Chart.js 3 + react-chartjs-2 — price & comparison charts
- axios — API client
- react-hot-toast — notifications

## Getting started

```bash
npm install
cp .env.example .env       # optional
npm start                  # http://localhost:3000
```

The dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`
(`proxy` in `package.json`), so run the FastAPI backend alongside it:

```bash
# from the repo root
uvicorn backend.main:app --reload
```

## Build

```bash
npm run build              # outputs to frontend/build
```

For a non-local API, set `REACT_APP_API_URL` (e.g. in `.env`) to the API
origin before building — CRA inlines `REACT_APP_*` variables at build time.

## Structure

```
src/
  api.js              axios client + endpoint helpers
  theme.js            MUI dark theme + signal/direction colours
  chartSetup.js       one-time Chart.js registration
  App.jsx             layout, search, tab routing
  components/
    StockForm.jsx       ticker search bar
    SignalHero.jsx      signal · confidence · risk headline
    ThesisCard.jsx      written thesis + guardrail flags
    AgentBreakdown.jsx  technical / sentiment / macro agent cards
    TechnicalPanel.jsx  price chart + indicators (/api/technical)
    CompetitorPanel.jsx peer comparison (/api/competitor)
```

## API endpoints used

| Helper | Endpoint |
|---|---|
| `getPrediction` | `POST /api/predict/` |
| `getTechnical`  | `GET /api/technical/{ticker}` |
| `getCompetitors`| `GET /api/competitor/{ticker}` |
