"""Re-run fixed experiments and optionally compare with committed evidence.

Usage: python scripts/reproduce.py --out runs/replay --verify evidence
Use --offline for synthetic integrity evidence only; no download is attempted.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    subprocess.run([sys.executable, "-m", "stocksage", *map(str, args)], cwd=ROOT, check=True)


def compare_values(left, right, path="root"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Different keys at {path}")
        for key in left:
            compare_values(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise AssertionError(f"Different lengths at {path}")
        for i, (x, y) in enumerate(zip(left, right)):
            compare_values(x, y, f"{path}[{i}]")
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"Numerical mismatch at {path}: {left} vs {right}")
    elif left != right:
        raise AssertionError(f"Mismatch at {path}: {left} vs {right}")


def verify_run(actual: Path, expected: Path):
    a, b = [json.loads((p / "run.json").read_text()) for p in (actual, expected)]
    for key in ("source_sha256", "config", "features", "model_parameters", "versions"):
        compare_values(a[key], b[key], key)
    compare_values(a["data"]["sha256"], b["data"]["sha256"], "input sha256")
    for folder, meta in ((actual, a), (expected, b)):
        for name, expected_hash in meta["artifacts_sha256"].items():
            if sha256((folder / name).read_bytes()).hexdigest() != expected_hash:
                raise AssertionError(f"Corrupt artifact: {folder / name}")
    for name in ("metrics.json", "folds.json", "forecast.json"):
        compare_values(json.loads((actual / name).read_text()), json.loads((expected / name).read_text()), name)
    assert_frame_equal(pd.read_csv(actual / "predictions.csv"), pd.read_csv(expected / "predictions.csv"),
                       check_exact=False, rtol=1e-9, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = ["synthetic"] if args.offline else ["synthetic", "aapl"]
    for name in names:
        data, result = out / "data" / name, out / name
        cli("synthetic" if name == "synthetic" else "fetch-sample", "--out", data)
        cli("evaluate", "--csv", data / "prices.csv", "--manifest", data / "manifest.json", "--out", result)
        if args.verify:
            verify_run(result, args.verify.resolve() / name)
            print(f"VERIFIED {name}: input, source, config, predictions, metrics, folds and latest forecast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
