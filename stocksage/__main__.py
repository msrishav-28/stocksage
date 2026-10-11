"""CLI: explicit data, deterministic experiments, immutable run directories."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tracemalloc

from .data import digest, fetch_sample, load_dataset, save_dataset, synthetic, write_json
from .evaluation import Config, evaluate, forecast_latest
from .features import FEATURES, build_features
from .models import TREE_PARAMS
from .runtime import hardware, provenance


def run(csv_path: Path, manifest: Path, out: Path, config: Config, forecast_only=False) -> dict:
    """Compute before publishing; incomplete artifacts never look like completed runs."""
    if out.exists():
        raise FileExistsError(f"Run destination already exists: {out}")
    start = time.perf_counter()
    bars, metadata = load_dataset(csv_path, manifest)
    features = build_features(bars)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".stocksage-", dir=out.parent))
    tracemalloc.start()
    try:
        if not forecast_only:
            predictions, report, folds = evaluate(features, config)
            predictions.to_csv(temp / "predictions.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g")
            write_json(temp / "metrics.json", report)
            write_json(temp / "folds.json", {"folds": folds})
        latest = forecast_latest(features, config)
        write_json(temp / "forecast.json", latest)
        meta = provenance(Path(__file__).resolve().parents[1])
        meta.update({
            "data": metadata, "config": asdict(config),
            "command": "forecast" if forecast_only else "evaluate", "features": FEATURES,
            "model_parameters": {"ridge_alpha": 10.0, "hist_gradient_boosting": TREE_PARAMS},
            "target": "next-session open-to-close simple return",
            "input_rows": len(bars), "symbols": sorted(bars["symbol"].unique().tolist()),
            "elapsed_seconds": time.perf_counter() - start,
            "python_traced_peak_bytes": tracemalloc.get_traced_memory()[1],
            "memory_note": "Python allocation peak during model work; excludes some native allocations and is not total RSS/VRAM",
        })
        meta["artifacts_sha256"] = {p.name: digest(p.read_bytes()) for p in sorted(temp.iterdir())}
        write_json(temp / "run.json", meta)
        temp.rename(out)
        return meta
    finally:
        tracemalloc.stop()
        if temp.exists():
            shutil.rmtree(temp)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="StockSage: auditable return experiments; no investment recommendations")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect hardware; optionally fail unless CUDA kernel succeeds")
    doctor.add_argument("--require-cuda", action="store_true")
    demo = sub.add_parser("synthetic", help="Generate labeled synthetic integrity fixture")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--days", type=int, default=630)
    demo.add_argument("--symbols", type=int, default=3)
    demo.add_argument("--seed", type=int, default=28)
    sample = sub.add_parser("fetch-sample", help="Download pinned 2015–17 AAPL illustration, not live data")
    sample.add_argument("--out", type=Path, required=True)
    validate = sub.add_parser("validate", help="Validate an explicitly sourced CSV/manifest pair")
    validate.add_argument("--csv", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    for name in ("evaluate", "forecast"):
        p = sub.add_parser(name)
        p.add_argument("--csv", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--min-train", type=int, default=252)
        p.add_argument("--test-size", type=int, default=63)
        p.add_argument("--embargo", type=int, default=5)
        p.add_argument("--seed", type=int, default=28)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = hardware()
            print(json.dumps(result, indent=2))
            return 2 if args.require_cuda and not result["cuda_verified"] else 0
        if args.command == "synthetic":
            csv, _ = save_dataset(synthetic(args.days, args.symbols, args.seed), args.out, {
                "kind": "synthetic", "source": f"stocksage synthetic seed={args.seed}, days={args.days}, symbols={args.symbols}",
                "license": "Generated numerical test fixture; no third-party market observations",
                "adjustment": "raw", "point_in_time": False,
                "limitations": "Synthetic random walk, not market evidence; weekday calendar is artificial",
            })
            print(csv)
        elif args.command == "fetch-sample":
            csv, _ = fetch_sample(args.out)
            print(csv)
        elif args.command == "validate":
            bars, meta = load_dataset(args.csv, args.manifest)
            print(json.dumps({"rows": len(bars), "symbols": sorted(bars.symbol.unique().tolist()),
                              "kind": meta["kind"], "sha256": meta["sha256"]}))
        else:
            config = Config(min_train=args.min_train, test_size=args.test_size, embargo=args.embargo, seed=args.seed)
            result = run(args.csv, args.manifest, args.out, config, args.command == "forecast")
            print(json.dumps({"out": str(args.out), "source_sha256": result["source_sha256"],
                              "kind": result["data"]["kind"], "elapsed_seconds": result["elapsed_seconds"]}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"StockSage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
