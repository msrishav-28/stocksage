"""HTTP-level tests for the FastAPI routers.

Uses FastAPI's TestClient. The data layer (yfinance / NewsAPI / FRED) is mocked
so the suite is hermetic and offline-safe.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Module-scoped client so the app lifespan runs only once."""
    from backend.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0.0"


def test_predict_rejects_overlong_ticker(client):
    """The input guardrail should reject malformed tickers with 422."""
    resp = client.post("/api/predict/", json={"ticker": "TOOLONGNAME"})
    assert resp.status_code == 422


def test_predict_rejects_bad_characters(client):
    resp = client.post("/api/predict/", json={"ticker": "A@B"})
    assert resp.status_code == 422


def test_predict_endpoint(client, sample_ohlcv_df):
    """A full prediction round-trip with the data layer mocked."""
    with patch("backend.routers.predict.fetch_ohlcv", return_value=sample_ohlcv_df), \
         patch("backend.data.news_fetcher.fetch_news", new_callable=AsyncMock, return_value=[]), \
         patch("backend.data.macro_fetcher.fetch_macro_snapshot", return_value={}):
        resp = client.post("/api/predict/", json={"ticker": "TEST", "period": "1y"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "TEST"
    assert body["final_signal"] in ("BUY", "HOLD", "SELL")
    assert 0 <= body["confidence"] <= 100
    assert body["thesis"]
    assert {"technical", "sentiment", "macro"}.issubset(body["agent_signals"].keys())


def test_technical_endpoint(client, sample_ohlcv_df):
    with patch("backend.routers.technical.fetch_ohlcv", return_value=sample_ohlcv_df):
        resp = client.get("/api/technical/TEST", params={"period": "1y"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "TEST"
    assert "confluence" in body
    assert body["confluence"]["direction"] in ("bullish", "bearish", "neutral")


def test_backtest_rejects_unknown_strategy(client):
    resp = client.post("/api/backtest/", json={"ticker": "AAPL", "strategy": "does_not_exist"})
    assert resp.status_code == 400
