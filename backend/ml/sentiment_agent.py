"""Sentiment analysis agent — fetches news and scores with FinBERT."""

from loguru import logger


class SentimentAgent:
    """
    Fetches latest news headlines for ticker via NewsAPI,
    scores with FinBERT, returns aggregated sentiment signal.
    """

    async def analyze(self, ticker: str, news_window_hours: int = 48) -> dict:
        logger.info(f"SentimentAgent analyzing {ticker} (window={news_window_hours}h)")

        # Fetch news
        from backend.data.news_fetcher import fetch_news
        articles = await fetch_news(ticker, hours=news_window_hours)

        if not articles:
            logger.warning(f"No articles found for {ticker}")
            return {
                "composite_score": 0.0,
                "label": "neutral",
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "total_articles": 0,
                "sentiment_momentum": 0.0,
                "headlines": [],
            }

        # Score headlines with FinBERT
        headlines = [a["title"] for a in articles if a.get("title")]

        try:
            from backend.ml.finbert_sentiment import score_batch, aggregate_sentiment
            scored = score_batch(headlines)
            aggregated = aggregate_sentiment(scored)
        except Exception:
            # FinBERT not available — use rule-based fallback
            logger.warning("FinBERT unavailable. Using neutral fallback.")
            aggregated = {
                "composite_score": 0.0,
                "label": "neutral",
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": len(headlines),
                "total_articles": len(headlines),
                "sentiment_momentum": 0.0,
            }
            scored = [{"headline": h, "label": "neutral", "score": 0.0} for h in headlines]

        return {
            **aggregated,
            "headlines": scored[:10],  # top 10 scored headlines
        }
