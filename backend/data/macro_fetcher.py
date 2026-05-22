from loguru import logger
from backend.config import get_settings


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
    if not settings.FRED_API_KEY:
        logger.warning("FRED_API_KEY not set. Returning empty snapshot.")
        return {name: None for name in MACRO_SERIES}

    try:
        from fredapi import Fred
        fred = Fred(api_key=settings.FRED_API_KEY)
    except ImportError:
        logger.warning("fredapi not installed. Returning empty snapshot.")
        return {name: None for name in MACRO_SERIES}

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
