"""Bounded immutable datasets with explicit provenance; no silent repairs."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
MAX_ROWS = 100_000
MAX_BYTES = 32 * 1024 * 1024
SAMPLE_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "6a88de1c9a95f0019566d3b568eb26444feda686/finance-charts-apple.csv"
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a complete, date-sorted common calendar. Never fill missing bars."""
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"CSV columns must be exactly {COLUMNS}")
    if not 22 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Expected 22..{MAX_ROWS} total rows")
    f = frame.copy()
    if f.isna().any().any():
        raise ValueError("Missing values are not permitted")
    dates = f["date"].astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Dates must be ISO YYYY-MM-DD session dates")
    f["date"] = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if (f["date"] > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()).any():
        raise ValueError("Future bars are not observations")
    if not f["symbol"].map(lambda s: isinstance(s, str) and bool(re.fullmatch(r"[A-Z0-9.^-]{1,15}", s))).all():
        raise ValueError("Symbols must be nonempty uppercase identifiers")
    if not 1 <= f["symbol"].nunique() <= 20:
        raise ValueError("Maximum 20 symbols")
    if f.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol/session")
    reference = None
    for _, g in f.groupby("symbol", sort=True):
        d = g["date"].reset_index(drop=True)
        if not d.is_monotonic_increasing:
            raise ValueError("Each symbol must be sorted by session")
        if reference is not None and not d.equals(reference):
            raise ValueError("Symbols must have identical session calendars; no silent intersection")
        reference = d
    for col in COLUMNS[2:]:
        f[col] = pd.to_numeric(f[col], errors="raise").astype(float)
    if not np.isfinite(f[COLUMNS[2:]].to_numpy()).all():
        raise ValueError("Non-finite OHLCV")
    if (f[["open", "high", "low", "close"]] <= 0).any().any() or (f["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((f["high"] < f[["open", "close", "low"]].max(axis=1)) |
            (f["low"] > f[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    return f.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_dataset(csv_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    if csv_path.stat().st_size > MAX_BYTES:
        raise ValueError("Input exceeds 32 MiB budget")
    raw = csv_path.read_bytes()
    meta = json.loads(manifest_path.read_text())
    for key in ("source", "license", "retrieved_at", "limitations"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"Manifest requires nonempty {key}")
    stamp = datetime.fromisoformat(meta["retrieved_at"])
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError("retrieved_at must be a timezone-aware past timestamp")
    if meta.get("kind") not in ("historical_sample", "market", "synthetic"):
        raise ValueError("Manifest kind must be market, historical_sample or synthetic")
    if meta.get("adjustment") not in ("raw", "split_adjusted"):
        raise ValueError("Only raw or split_adjusted OHLCV is supported")
    if type(meta.get("point_in_time")) is not bool:
        raise ValueError("Manifest must explicitly declare point_in_time")
    if meta.get("sha256") != digest(raw):
        raise ValueError("Dataset SHA256 does not match manifest")
    f = validate(pd.read_csv(BytesIO(raw), dtype={"date": str, "symbol": str}))
    return f, meta


def save_dataset(frame: pd.DataFrame, out: Path, metadata: dict) -> tuple[Path, Path]:
    f = validate(frame)
    out.mkdir(parents=True, exist_ok=False)
    csv_path, meta_path = out / "prices.csv", out / "manifest.json"
    f.to_csv(csv_path, index=False, date_format="%Y-%m-%d", float_format="%.12g")
    meta = {**metadata, "sha256": digest(csv_path.read_bytes()),
            "retrieved_at": datetime.now(timezone.utc).isoformat()}
    write_json(meta_path, meta)
    return csv_path, meta_path


def synthetic(days: int = 630, symbols: int = 3, seed: int = 28) -> pd.DataFrame:
    """Correlated random walk fixture, deliberately without a predictive signal."""
    if not 300 <= days <= 1500 or not 1 <= symbols <= 20:
        raise ValueError("Synthetic budget: 300..1500 sessions, 1..20 symbols")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=days).strftime("%Y-%m-%d")
    market = rng.normal(0, .008, days)
    frames = []
    for i in range(symbols):
        intraday = market + rng.normal(0, .01, days)
        gaps = rng.normal(0, .004, days)
        close = (80 + i * 20) * np.exp(np.cumsum(gaps + intraday))
        opening = close / np.exp(intraday)
        spread = rng.uniform(.001, .015, days)
        frames.append(pd.DataFrame({
            "date": dates, "symbol": f"SYN{i}", "open": opening,
            "high": np.maximum(opening, close) * (1 + spread),
            "low": np.minimum(opening, close) * (1 - spread), "close": close,
            "volume": rng.integers(100_000, 2_000_000, days).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def fetch_sample(out: Path) -> tuple[Path, Path]:
    """Download a fixed educational archive, not live or PIT market data."""
    with urlopen(SAMPLE_URL, timeout=30) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Provider response exceeds budget")
    f = pd.read_csv(BytesIO(raw))
    f = f.rename(columns={"Date": "date", **{f"AAPL.{c.title()}": c for c in COLUMNS[2:]}})
    f["symbol"] = "AAPL"
    return save_dataset(f[COLUMNS], out, {
        "source": SAMPLE_URL, "upstream_sha256": digest(raw),
        "kind": "historical_sample", "adjustment": "raw", "point_in_time": False,
        "license": "Dataset redistribution rights unverified; download for local evaluation only",
        "limitations": "Old single-survivor AAPL illustration, not a vintage archive or modern universe; no dividends",
    })
