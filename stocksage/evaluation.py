"""Purged chronological experiments and auditable numerical diagnostics."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .models import MODEL_NAMES, predict


@dataclass(frozen=True)
class Config:
    min_train: int = 252
    test_size: int = 63
    embargo: int = 5
    seed: int = 28
    bootstrap_samples: int = 1000
    block_size: int = 10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.min_train < 60 or self.test_size < 1 or self.embargo < 1:
            raise ValueError("Require min_train >=60, test_size >=1, embargo >=1")
        if not 0 <= self.seed < 2**32 or not 100 <= self.bootstrap_samples <= 10_000 or self.block_size < 1:
            raise ValueError("Invalid seed or bootstrap budget")


def walk_forward(features: pd.DataFrame, config: Config):
    """Train labels must end strictly before the embargo boundary.

    No future observations ever enter training. A post-test exclusion is thus
    vacuous within a fold; the explicit embargo is a pre-test session gap.
    Each date belongs to exactly one test block, including the last partial one.
    """
    labelled = features.dropna(subset=["target", "label_end"])
    dates = pd.DatetimeIndex(sorted(labelled["date"].unique()))
    first = config.min_train + config.embargo + 1
    if len(dates) <= first:
        raise ValueError(f"Need >{first} labelled feature sessions; got {len(dates)}")
    for fold, start in enumerate(range(first, len(dates), config.test_size)):
        end = min(start + config.test_size, len(dates))
        boundary = dates[start - config.embargo]
        train = labelled[(labelled["date"] < boundary) & (labelled["label_end"] < boundary)]
        test = labelled[labelled["date"].isin(dates[start:end])]
        if train["date"].nunique() < config.min_train:
            raise ValueError("Insufficient training history after purging")
        if not (train["label_end"].max() < boundary <= test["date"].min()):
            raise ValueError("Purging invariant violated")
        yield fold, train, test, boundary


def predictive_metrics(p: pd.DataFrame) -> dict:
    y, point = p["target"].to_numpy(), p["prediction"].to_numpy()
    low, high = p["q10"].to_numpy(), p["q90"].to_numpy()
    pinball = lambda q, value: float(np.maximum(q * (y - value), (q - 1) * (y - value)).mean())
    return {
        "n": len(p), "sessions": int(p["date"].nunique()),
        "mae": float(np.abs(y - point).mean()),
        "rmse": float(np.sqrt(np.mean((y - point)**2))),
        "direction_accuracy": float((np.sign(y) == np.sign(point)).mean()),
        "prediction_tie_rate": float((point == 0).mean()),
        "coverage_80": float(((y >= low) & (y <= high)).mean()),
        "mean_interval_width": float((high - low).mean()),
        "pinball_q10": pinball(.1, low), "pinball_q90": pinball(.9, high),
    }


def economic_metrics(p: pd.DataFrame, cost_bps: float, always_long: bool = False) -> dict:
    """Daily reset long/cash, fixed opening-notional costs on BOTH sides.

    This is a linear cost approximation, not exchange-level fill simulation.
    Each symbol gets equal capital; inactive symbols retain their cash allocation.
    """
    if not math.isfinite(cost_bps) or not 0 <= cost_bps <= 100:
        raise ValueError("Per-side cost must be finite and in [0,100] bps")
    if p.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate prediction in economic calculation")
    if p.groupby("date")["symbol"].nunique().nunique() != 1:
        raise ValueError("Unequal daily universe in economic calculation")
    round_trip = 2 * cost_bps / 10_000
    active = np.ones(len(p), dtype=bool) if always_long else p["prediction"].to_numpy() > round_trip
    frame = p[["date"]].copy()
    frame["net"] = active * (p["target"].to_numpy() - round_trip)
    if (frame["net"] <= -1).any() or not np.isfinite(frame["net"]).all():
        raise ValueError("Invalid economic returns after costs")
    daily = frame.groupby("date", sort=True)["net"].mean().to_numpy()
    equity = np.concatenate([[1.0], np.cumprod(1 + daily)])
    vol = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    return {
        "cost_bps_per_side": float(cost_bps), "sessions": len(daily),
        "active_symbol_sessions": int(active.sum()),
        "active_fraction": float(active.mean()),
        "net_compound_return": float(equity[-1] - 1),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "annualized_sharpe_zero_rf": float(np.mean(daily) / vol * np.sqrt(252)) if vol > 0 else None,
        "mean_daily_net_return": float(np.mean(daily)),
    }


def paired_mae_interval(p: pd.DataFrame, config: Config) -> dict:
    """Moving-block bootstrap of session-mean paired errors vs zero return."""
    frame = p[["date"]].copy()
    frame["delta"] = np.abs(p["target"].to_numpy() - p["prediction"].to_numpy()) - np.abs(p["target"].to_numpy())
    delta = frame.groupby("date", sort=True)["delta"].mean().to_numpy()
    n, block = len(delta), min(config.block_size, len(delta))
    rng = np.random.default_rng(config.seed)
    means = np.empty(config.bootstrap_samples)
    for i in range(config.bootstrap_samples):
        starts = rng.integers(0, n - block + 1, math.ceil(n / block))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = delta[idx].mean()
    return {"mean_delta": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "block_sessions": block, "replicates": config.bootstrap_samples,
            "interpretation": "Negative favors model; descriptive dependence-aware interval, not a discovery claim"}


def evaluate(features: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict, list]:
    predictions, folds = [], []
    for fold, train, test, boundary in walk_forward(features, config):
        folds.append({
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "train_sessions": int(train["date"].nunique()),
            "test_sessions": int(test["date"].nunique()),
            "train_start": str(train["date"].min().date()),
            "train_end": str(train["date"].max().date()),
            "last_train_label_end": str(train["label_end"].max().date()),
            "embargo_boundary": str(boundary.date()),
            "test_start": str(test["date"].min().date()),
            "test_end": str(test["date"].max().date()),
            "last_test_label_end": str(test["label_end"].max().date()),
        })
        for model in MODEL_NAMES:
            p = test[["date", "symbol", "label_end", "target"]].copy()
            p[["prediction", "q10", "q90"]] = predict(train, test, model, config.seed)
            p["model"], p["fold"] = model, fold
            predictions.append(p)
    p = pd.concat(predictions, ignore_index=True)
    report = {"config": asdict(config), "models": {}, "always_long_intraday": {}}
    for name, group in p.groupby("model", sort=True):
        report["models"][name] = {
            "overall": predictive_metrics(group),
            "by_fold": {str(k): predictive_metrics(g) for k, g in group.groupby("fold")},
            "by_symbol": {str(k): predictive_metrics(g) for k, g in group.groupby("symbol")},
            "paired_mae_vs_zero": paired_mae_interval(group, config),
            "long_cash": {str(c): economic_metrics(group, c) for c in (0, 5, 10, 25)},
        }
    baseline = p[p["model"] == "zero"]
    report["always_long_intraday"] = {str(c): economic_metrics(baseline, c, True) for c in (0, 5, 10, 25)}
    return p, report, folds


def forecast_latest(features: pd.DataFrame, config: Config) -> dict:
    """Refit on mature past labels with the same embargo, return all comparators."""
    dates = pd.DatetimeIndex(sorted(features["date"].unique()))
    if len(dates) <= config.min_train + config.embargo + 1:
        raise ValueError("Insufficient history for latest forecast")
    asof, boundary = dates[-1], dates[-1 - config.embargo]
    train = features[(features["date"] < boundary) & (features["label_end"] < boundary)].dropna(subset=["target"])
    test = features[features["date"] == asof]
    if train["date"].nunique() < config.min_train:
        raise ValueError("Insufficient mature labels for latest forecast")
    forecasts = []
    for name in MODEL_NAMES:
        values = predict(train, test, name, config.seed)
        for symbol, (point, low, high) in zip(test["symbol"], values):
            forecasts.append({"symbol": symbol, "model": name, "prediction": float(point),
                              "q10": float(low), "q90": float(high)})
    return {"asof_session_close": str(asof.date()), "train_label_end_max": str(train["label_end"].max().date()),
            "embargo_boundary": str(boundary.date()), "target": "next available session open-to-close simple return",
            "future_session_date": None, "forecasts": forecasts,
            "warning": "Historical snapshot forecast, not a live quote or a calibrated investment recommendation"}
