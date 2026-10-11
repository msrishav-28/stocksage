# StockSage — auditable stock-return experiments

StockSage is a CPU research CLI that trains and evaluates numerical models for next-session open-to-close equity returns. It replaces the old heuristic agents, invented accuracy priors, dashboard, API and database stack. **Markets are near-efficient. This system has not demonstrated reliable prediction or alpha.**

**Status:** the CPU pipeline is implemented and executed. Lenovo/Blackwell validation, foundation-model comparisons, point-in-time financial text and a contemporary multi-equity experiment are **PENDING**. This is a measured research rebuild, not a validated trading product. The scope and blockers are explicit in [BLOCKERS.md](BLOCKERS.md).

Read the evidence in order:

1. [Research and model/license decisions](research/RESEARCH.md), recorded before implementation.
2. [Verdict for all 89 original files](research/VERDICT.md), including the breaking product changes.
3. [Historical run](evidence/aapl/run.json), [fold boundaries](evidence/aapl/folds.json), [every prediction](evidence/aapl/predictions.csv) and [full metrics](evidence/aapl/metrics.json).
4. [Executed test log](evidence/tests.txt), [validation record](evidence/validation.json) and [hardware diagnostic](evidence/hardware.json).

## Measured result: the nonlinear model loses

The accessible historical smoke benchmark uses 506 AAPL bars from a commit-pinned Plotly archive (2015–2017). After warm-up and purging, it scores 227 forward predictions across four walk-forward folds. Decision sessions start 2016-03-24; the final realized label is 2017-02-16. This is a single surviving equity in an old interval, not modern-market validation.

| Model | MAE (return units) | RMSE (return units) | Empirical 80% band coverage | Long/cash net return, 5 bps/side |
|---|---:|---:|---:|---:|
| zero | 0.006786 | 0.008997 | 94.71% | 0.00% |
| train_mean | 0.006834 | 0.009027 | 94.71% | 0.00% |
| ridge | 0.007232 | 0.009405 | 91.19% | 5.52% |
| hist_gradient_boosting | 0.007490 | 0.010003 | 82.38% | 3.29% |

Zero return wins MAE and RMSE. Gradient boosting's MAE is **10.4% worse** than zero. The paired moving-block bootstrap MAE difference is +0.000704, with descriptive 95% interval [0.000069, 0.001271]; positive favors zero. These intervals do not correct for research selection or establish a market-wide result. No hyperparameters were changed in response to these results.

The always-long **intraday** comparator returned 10.73% over the same observed sessions at 5 bps/side, versus 3.29% for gradient boosting. It is not buy-and-hold: all positions close each session. Returns are diagnostic simulations, not realized trades. All models, 0/5/10/25-bps cost scenarios, drawdowns, descriptive Sharpes, pinball losses, per-fold and per-symbol results are in [metrics.json](evidence/aapl/metrics.json).

Coverage is empirical, not a confidence guarantee. The baseline's 94.71% coverage of a nominal 80% band indicates an overly wide interval on this sample. Ridge uses training-residual bands; boosting uses quantile regressors with deterministic crossing correction. Neither is independently calibrated. Direction accuracy counts sign ties explicitly: zero's 0% score reflects its 100% prediction-tie rate, not a 100% wrong directional forecast.

The separate [synthetic run](evidence/synthetic/metrics.json) contains 1,890 generated rows, three symbols and 351 out-of-sample sessions (1,053 symbol-session predictions per model). It tests mechanics and reproducibility only. Its numbers are **not market evidence**.

## Run it

Python **3.12** is the tested/pinned version. No API key, database, paid service, GPU or model-weight download is required for the CPU path. [requirements.txt](requirements.txt) pins the environment; [validation.json](evidence/validation.json) records a clean-venv install and successful test execution.

```bash
python3.12 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v

# Full reproduction: generated fixture + pinned historical archive.
python scripts/reproduce.py --out runs/replay --verify evidence

# Offline integrity reproduction after dependencies are installed:
python scripts/reproduce.py --out runs/offline --offline --verify evidence
```

Use a new output directory for each run; existing outputs are rejected. The historical download needs access to raw.githubusercontent.com. Failures exit nonzero; there is no fabricated or synthetic market-data fallback. Verification checks input and source hashes, versions, configuration, all predictions, fold boundaries, numerical metrics and latest forecasts. Numeric tolerance is 1e-9 relative / 1e-12 absolute; timestamps and measured runtimes are expected to vary.

Individual commands:

```bash
python -m stocksage fetch-sample --out data/aapl
python -m stocksage validate --csv data/aapl/prices.csv --manifest data/aapl/manifest.json
python -m stocksage evaluate --csv data/aapl/prices.csv --manifest data/aapl/manifest.json --out runs/aapl
python -m stocksage forecast --csv data/aapl/prices.csv --manifest data/aapl/manifest.json --out runs/forecast
python -m stocksage doctor
```

`forecast.json` contains all four model outputs for the next available session after the input's final close. It reports the historical as-of date; it is not a live forecast when the input is old. The future session date stays null rather than guessing an exchange calendar. Models are refitted from matured, embargoed labels; no unsafe pickle/model artifact is needed. The entry point is also available as `stocksage` after installation.

## Your own data

Supply a finalized daily CSV with exactly these columns, in this order:

```text
date,symbol,open,high,low,close,volume
```

Dates are ISO `YYYY-MM-DD` session dates. Each symbol must be sorted and have the same dates. Missing, duplicate, nonfinite, invalid OHLC or future-dated rows fail validation; the loader never fills gaps, invents neutral values or silently drops symbols. Limits are 20 symbols, 100,000 rows and 32 MiB. These are enforced input budgets, not a measured claim about maximum Lenovo capacity. Exchange-session completeness and historical corporate-action accuracy still require a trustworthy source.

Write a JSON manifest beside the CSV. Fill source and rights descriptions with facts about your dataset; `point_in_time` is your provenance declaration, not certification by the program:

```python
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

csv = Path("data/my-universe/prices.csv")
manifest = {
    "kind": "market",
    "source": "YOUR provider, dataset ID and snapshot/vintage",
    "license": "YOUR actual permitted use and redistribution terms",
    "retrieved_at": datetime.now(timezone.utc).isoformat(),
    "adjustment": "raw",  # or split_adjusted, consistently across OHLC
    "point_in_time": False,
    "limitations": "Describe revisions, survivor selection and missing coverage",
    "sha256": sha256(csv.read_bytes()).hexdigest(),
}
csv.with_name("manifest.json").write_text(json.dumps(manifest, indent=2))
```

Run `validate`, then `evaluate` with the same arguments as above. Raw prices can have split discontinuities in past-return features; split-adjusted vendor histories can incorporate later revisions. Neither a manifest nor a hash eliminates point-in-time leakage. No fundamentals, current constituent list, revised macro series or news scores are backfilled into past observations.

## Architecture and experimental contract

| Component | Role | Integrity evidence |
|---|---|---|
| [data.py](stocksage/data.py) | Bounded CSV validation, immutable input hashes, pinned sample download, explicitly synthetic fixture | [Data tests](tests/test_data_features.py) |
| [features.py](stocksage/features.py) | Eight backward-looking OHLCV features, next-session label alignment | Prefix and future-perturbation tests |
| [models.py](stocksage/models.py) | Zero, train mean, ridge and small histogram boosting models | Label-blind prediction and planted-relation tests |
| [evaluation.py](stocksage/evaluation.py) | Purged folds, numerical metrics, paired block intervals, cost diagnostics and latest inference | [Boundary and arithmetic tests](tests/test_evaluation.py) |
| [runtime.py](stocksage/runtime.py), [CLI](stocksage/__main__.py) | Source/version provenance, hardware preflight and complete-run publication | [End-to-end tests](tests/test_cli_runtime.py) |
| [reproduce.py](scripts/reproduce.py) | Recreate experiments and compare all numerical evidence | Independently repeated benchmark in the validation record |

At close t, the label is `close[t+1] / open[t+1] - 1`. Features include returns over 1/5/20 sessions, 20-session volatility and trend, current intraday return/range, and relative volume. A forecast made after close can only enter at the next open. No same-close fills.

All symbols share fold dates. The first fit has 252 usable feature sessions. Each test block spans up to 63 sessions. Training requires `label_end < session[test_start_index - 5]`, implementing label purging plus a five-session pre-test embargo. There are no future training rows, so a conventional post-test embargo is vacuous within a fold. Earlier test outcomes may join a later fit once mature; this is scheduled walk-forward retraining, not a permanently untouched holdout. Scaling happens inside training only. Boosting disables random early stopping. Settings were recorded in the research commit before execution.

The long/cash diagnostic invests equal capital per symbol when forecast return exceeds twice the per-side cost; inactive allocations stay in cash. Each active session pays entry and exit costs on opening notional. This linear approximation omits auction impact, actual proportional exit fees, spread dynamics, latency, taxes and capacity. Positions never overlap overnight. There is no execution connector.

Run directories contain `metrics.json`, `predictions.csv`, `folds.json`, `forecast.json` and `run.json`. The last includes input provenance, package versions, actual source-file hashes, git head, parameters, output hashes and runtime. The [AAPL source run](evidence/aapl/run.json) took 6.63 seconds on this cloud execution host, **not on the Lenovo**. Its Python-traced allocation peak excludes some native allocations and must not be read as total RAM or VRAM.

## License and hardware boundaries

The CPU stack uses NumPy/pandas/scikit-learn and their pinned dependencies. No pretrained weights are bundled. Research candidates have separate code and weight licenses:

- Chronos-2 and TimesFM 2.5 model cards: Apache-2.0.
- Moirai 2.0 small: CC-BY-NC-4.0.
- TimesFM 3.0 downloadable weights: non-commercial, distinct from Cloud service terms.
- TabPFN 2.5/2.6/3/3.5 weights: non-commercial, distinct from Apache repository code.

Primary sources and rationale are in [RESEARCH.md](research/RESEARCH.md). The Plotly archive has no verified dataset redistribution grant here; raw CSV stays outside Git, with a commit-pinned downloader and hashes. The original project asserted MIT in metadata without a LICENSE file; this rebuild does not manufacture a copyright grant. Confirm project and data rights before distributing or commercializing.

The actual [hardware run](evidence/hardware.json) found no torch, NVIDIA device access or nvidia-smi. GPU model implementation stopped at this gate. The CUDA success path in unit tests is explicitly simulated; it is not a GPU benchmark. On the Lenovo, use a separate GPU environment, install `torch>=2.8` from `https://download.pytorch.org/whl/cu128`, and require a real successful diagnostic:

```bash
python -m pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128
python -m stocksage doctor --require-cuda
```

A nonzero exit means stop GPU work. No claim is made that this command has been executed on sm_120. Foundation-model accuracy, peak VRAM and latency, contemporary multi-stock data, PIT text evaluation, and paper trading remain **PENDING**; see [BLOCKERS.md](BLOCKERS.md). The included [CI workflow](.github/workflows/ci.yml) exercises offline CPU integrity only; remote CI status must be read from GitHub, not inferred from local tests.
