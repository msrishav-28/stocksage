"""FinBERT sentiment scoring for financial news headlines.

torch / transformers are imported lazily inside the scoring functions so that
importing this module (e.g. for the pure-Python ``aggregate_sentiment``) does
not pay the multi-second cost of loading torch.
"""

import numpy as np
from typing import List
from loguru import logger
from functools import lru_cache

FINBERT_MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def load_finbert():
    """Load the FinBERT tokenizer + model once (cached). Raises if unavailable."""
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
    except ImportError as e:
        raise RuntimeError(f"transformers not installed: {e}")

    logger.info("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    logger.success("FinBERT loaded.")
    return tokenizer, model


def score_headline(text: str) -> dict:
    """
    Score a single headline.

    Returns:
        {
          "label": "positive" | "negative" | "neutral",
          "score": float (-1.0 to +1.0),  # positive_prob - negative_prob
          "probabilities": {"positive": float, "negative": float, "neutral": float}
        }
    """
    import torch
    from torch.nn.functional import softmax

    tokenizer, model = load_finbert()

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
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
    """Score a list of headlines in batches for efficiency."""
    import torch
    from torch.nn.functional import softmax

    tokenizer, model = load_finbert()
    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
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
    Aggregate a list of scored headlines into a single sentiment signal.
    Pure function — no torch dependency.
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
