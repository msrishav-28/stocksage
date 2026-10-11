"""Causal features and explicitly delayed next-session labels."""

import numpy as np
import pandas as pd

FEATURES = ["return_1", "return_5", "return_20", "volatility_20", "trend_20",
            "intraday", "range", "relative_volume"]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Input must have passed data.validate. Labels never enter FEATURES."""
    frames = []
    for symbol, g in bars.groupby("symbol", sort=True):
        g = g.sort_values("date").copy()
        close = g["close"]
        for lag in (1, 5, 20):
            g[f"return_{lag}"] = close.pct_change(lag, fill_method=None)
        g["volatility_20"] = g["return_1"].rolling(20).std(ddof=1)
        g["trend_20"] = close / close.rolling(20).mean() - 1
        g["intraday"] = close / g["open"] - 1
        g["range"] = (g["high"] - g["low"]) / close
        mean_volume = g["volume"].rolling(20).mean()
        g["relative_volume"] = g["volume"] / mean_volume.where(mean_volume > 0, 1)
        g["target"] = g["intraday"].shift(-1)
        g["label_end"] = g["date"].shift(-1)
        g["symbol"] = symbol
        frames.append(g[["date", "symbol", *FEATURES, "target", "label_end"]].iloc[20:])
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if result.empty or not np.isfinite(result[FEATURES].to_numpy()).all():
        raise ValueError("Insufficient or invalid causal features")
    return result
