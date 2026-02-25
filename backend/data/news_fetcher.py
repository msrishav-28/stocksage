import httpx
from datetime import datetime, timedelta
from typing import List
from loguru import logger
from backend.config import get_settings


async def fetch_news(ticker: str, hours: int = 48) -> List[dict]:
    """
    Fetches recent news headlines for a ticker using NewsAPI.
    Returns list of {title, description, url, published_at, source}.
    """
    settings = get_settings()
    if not settings.NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set. Returning empty news list.")
        return []

    from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": settings.NEWS_API_KEY,
        "pageSize": 50,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.error(f"NewsAPI fetch failed for {ticker}: {e}")
        return []

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
