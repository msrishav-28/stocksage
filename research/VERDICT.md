# StockSage verdict

Audited base: `1cb980a57e6b13458fceb63d9f65526591dcb046`. Research precedes implementation; see [RESEARCH.md](RESEARCH.md). Every one of the 89 tracked base files is classified below. The frontend lock was inspected structurally (manifest plus 1,364 resolved packages), not as a source-code module. No untracked user files existed in this checkout. Git preserves everything removed.

## Assessment

The application is a heuristic signal dashboard with optional neural components, not an honestly evaluated ML system. The accuracy priors are invented constants. Memory reinforces its own prior outputs without observed outcomes. The TFT validation path is not a holdout, and its signed-return normalization needs redesign. The full-series wavelet feature leaks future observations into historical rows. Backtest timing lacks an executable delay. None of the repository's tests establishes out-of-sample forecast value.

The useful ideas are OHLCV ingestion, pure feature functions, baselines, explicit failures and regression tests. Their current implementations do not meet the research contract. There are **no KEEP-as-is files**: that is deliberate, so no retained legacy code escapes testing. Four operational files are modified for their load-bearing roles; all executable legacy subsystems are purged. Existing tests are not claimed to pass on this host (their environment is absent). They test removed contracts; new tests must validate the replacement's real invariants.

## Product decisions

- Replace the web app with `python -m stocksage`: dataset generation/download, validation, walk-forward evaluation, latest-session numerical inference, and hardware inspection. This makes the entire product auditable from a terminal and JSON/CSV artifacts. It is an intentional breaking API change; no compatibility wrappers.
- Ship CPU numerical ML only. Research candidates and unblock criteria are documentation, not unused model scaffolding. This follows the GPU gate and nothing-dead rule, not a claim that foundations are inferior.
- Use one-day intraday returns and long/cash diagnostic execution to avoid same-close fills, shorting assumptions and overlapping holding-period bookkeeping. Prediction is a numerical experiment, never an investment recommendation.
- Use a bounded CSV panel with source metadata, not Redis/Postgres. No live provider is silently required. Fail malformed/partial data instead of inventing fields.
- Freeze comparisons before results and include all negative outcomes. The accessible AAPL archive is a smoke benchmark; contemporary multi-asset and laptop verification remain explicit blockers.
- Do not add a license grant: repository metadata said MIT, but no license text/owner attribution was supplied. Dependencies and excluded weight restrictions remain documented.

## Per-file decisions

| Base file | Decision | Evidence / justification |
|---|---|---|
| `.env.example` | PURGE | No external service credentials are needed by the CPU research CLI. |
| `.gitignore` | MODIFY | Replace generic service/frontend ignores with actual data, run and Python outputs. |
| `DEPLOYMENT_READY.md` | PURGE | Readiness checklist has no execution evidence; remove the unsupported deployment narrative. |
| `Dockerfile` | PURGE | API container and compiler/Postgres tooling are not used by the local experiment. |
| `README.md` | MODIFY | Replace feature marketing and confidence claims with measured results and exact reproduction. |
| `alembic.ini` | PURGE | No database survives; migration configuration is dead. |
| `backend/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/backtesting/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/backtesting/engine.py` | PURGE | Close-derived signals execute at the same close; benchmark excess return is mislabeled alpha; no ML OOS predictions. |
| `backend/backtesting/strategies.py` | PURGE | Five fixed indicator strategies are outside the predeclared ML experiment. |
| `backend/cache/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/cache/redis_client.py` | PURGE | Redis is unnecessary for bounded immutable local experiment artifacts. |
| `backend/config.py` | PURGE | Settings configure removed API, databases, providers and Modal; some knobs are unused. |
| `backend/data/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/data/feature_engineer.py` | PURGE | Constant earnings/sentiment/beta stubs and TFT-specific columns; replace with tested causal features. |
| `backend/data/macro_fetcher.py` | PURGE | Latest revised levels are not PIT vintages; cpi_yoy names the CPI index level; untrained score rules. |
| `backend/data/news_fetcher.py` | PURGE | Recent headlines cannot support a historical PIT experiment; missing data collapses to empty. |
| `backend/data/price_fetcher.py` | PURGE | Useful OHLCV idea, but no immutable snapshot/provenance and partial batch failures silently drop symbols. |
| `backend/db/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/db/migrations/README.md` | PURGE | Migration or template for the removed service schema; preserve history in Git only. |
| `backend/db/migrations/env.py` | PURGE | Migration or template for the removed service schema; preserve history in Git only. |
| `backend/db/migrations/script.py.mako` | PURGE | Migration or template for the removed service schema; preserve history in Git only. |
| `backend/db/migrations/versions/0001_initial_schema.py` | PURGE | Migration or template for the removed service schema; preserve history in Git only. |
| `backend/db/migrations/versions/0002_prediction_thesis.py` | PURGE | Migration or template for the removed service schema; preserve history in Git only. |
| `backend/db/models.py` | PURGE | Prediction schema persists heuristic confidence without training/data/split provenance. |
| `backend/db/session.py` | PURGE | Async Postgres lifecycle is unused by a file-based reproducible experiment. |
| `backend/indicators/__init__.py` | PURGE | Full-array wavelet smoothing is noncausal; optional packages change feature semantics; confluence is uncalibrated. |
| `backend/main.py` | PURGE | Eight routers and optional infrastructure hide a non-evaluated predictor behind an always-ok health endpoint. |
| `backend/ml/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/ml/base_agent.py` | PURGE | Deterministic scripted tool routing dressed as ReAct; no learned reasoner or measurable advantage. |
| `backend/ml/ensemble.py` | PURGE | Facade solely forwards to the discarded heuristic orchestrator. |
| `backend/ml/finbert_sentiment.py` | PURGE | Headline classification is not return forecasting; no PIT corpus or executed model evaluation. |
| `backend/ml/guardrails.py` | PURGE | Arbitrary risk/confidence overrides cannot calibrate probabilities; percent/fraction interpretation is ambiguous. |
| `backend/ml/macro_agent.py` | PURGE | Wraps untrained macro scoring and presents unavailable data as neutral. |
| `backend/ml/memory.py` | PURGE | Hard-coded 0.61/0.55/0.58 historical accuracies have no supporting artifact; prior predictions feed back as evidence. |
| `backend/ml/orchestrator.py` | PURGE | Hand-set votes and fabricated accuracy priors yield uncalibrated BUY/HOLD/SELL confidence. |
| `backend/ml/sentiment_agent.py` | PURGE | FinBERT failure is converted to neutral headlines with no error on AgentResult. |
| `backend/ml/technical_agent.py` | PURGE | Indicator agreement is presented as confidence without calibration or forecast evaluation. |
| `backend/ml/telemetry.py` | PURGE | Service span system is unnecessary; sum of concurrent spans is not wall-clock elapsed time. |
| `backend/ml/tft_model.py` | PURGE | Validation dataset is rebuilt from all data; no purged folds; softplus target transform is inappropriate for signed returns; inference lacks future decoder rows. |
| `backend/ml/tools.py` | PURGE | Singleton tool registry adds indirection for a few fixed function calls; removed agents are its only clients. |
| `backend/routers/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/routers/backtest.py` | PURGE | Passes unshifted same-close indicator signals into the old backtester. |
| `backend/routers/competitor.py` | PURGE | Hard-coded current peer mapping adds current-snapshot selection bias and no training value. |
| `backend/routers/macro.py` | PURGE | Current macro/sector snapshots do not satisfy PIT historical inputs. |
| `backend/routers/portfolio.py` | PURGE | Separate retrospective portfolio calculator does not evaluate the ML pipeline. |
| `backend/routers/predict.py` | PURGE | Serves and caches unsupported probabilities without a versioned ML experiment. |
| `backend/routers/screener.py` | PURGE | Current survivor list and indicator confluence are not a PIT universe or ML screen. |
| `backend/routers/sentiment.py` | PURGE | Presents missing/failing sentiment as neutral instead of unavailable. |
| `backend/routers/technical.py` | PURGE | Chart/indicator endpoint is unused once the dashboard is removed. |
| `backend/schemas/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/schemas/backtest.py` | PURGE | Response/request schema for the removed backtest endpoint; no surviving consumer. |
| `backend/schemas/macro.py` | PURGE | Response/request schema for the removed macro endpoint; no surviving consumer. |
| `backend/schemas/portfolio.py` | PURGE | Response/request schema for the removed portfolio endpoint; no surviving consumer. |
| `backend/schemas/predict.py` | PURGE | Response/request schema for the removed predict endpoint; no surviving consumer. |
| `backend/schemas/screener.py` | PURGE | Response/request schema for the removed screener endpoint; no surviving consumer. |
| `backend/schemas/sentiment.py` | PURGE | Response/request schema for the removed sentiment endpoint; no surviving consumer. |
| `backend/schemas/technical.py` | PURGE | Response/request schema for the removed technical endpoint; no surviving consumer. |
| `backend/tests/__init__.py` | PURGE | Package marker/re-exports for a removed subsystem; no remaining caller. |
| `backend/tests/conftest.py` | PURGE | Tests of removed conftest behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `backend/tests/test_api.py` | PURGE | Tests of removed api behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `backend/tests/test_backtest.py` | PURGE | Tests of removed backtest behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `backend/tests/test_predict.py` | PURGE | Tests of removed predict behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `backend/tests/test_sentiment.py` | PURGE | Tests of removed sentiment behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `backend/tests/test_technical.py` | PURGE | Tests of removed technical behavior; replace with causal, split, metric and CLI tests, not vacuous API-shape checks. |
| `docker-compose.yml` | PURGE | Postgres/Redis/API fleet consumes budget without serving the experiment. |
| `frontend/.env.example` | PURGE | API-origin configuration: no role in the ML-first CLI or its tests. |
| `frontend/README.md` | PURGE | Dashboard documentation: no role in the ML-first CLI or its tests. |
| `frontend/package-lock.json` | PURGE | 1,364-package web lockfile: no role in the ML-first CLI or its tests. |
| `frontend/package.json` | PURGE | CRA/MUI/chart dependencies: no role in the ML-first CLI or its tests. |
| `frontend/public/index.html` | PURGE | Web shell with references to absent icon/manifest assets: no role in the ML-first CLI or its tests. |
| `frontend/src/App.css` | PURGE | Dashboard styling: no role in the ML-first CLI or its tests. |
| `frontend/src/App.jsx` | PURGE | Dashboard state and tabs: no role in the ML-first CLI or its tests. |
| `frontend/src/api.js` | PURGE | API wrappers, including unused sentiment/health helpers: no role in the ML-first CLI or its tests. |
| `frontend/src/chartSetup.js` | PURGE | Chart registration: no role in the ML-first CLI or its tests. |
| `frontend/src/components/AgentBreakdown.jsx` | PURGE | Heuristic-agent visualization: no role in the ML-first CLI or its tests. |
| `frontend/src/components/CompetitorPanel.jsx` | PURGE | Current-peer chart: no role in the ML-first CLI or its tests. |
| `frontend/src/components/SignalHero.jsx` | PURGE | Prominent unsupported confidence/risk display: no role in the ML-first CLI or its tests. |
| `frontend/src/components/StockForm.jsx` | PURGE | Ticker search form: no role in the ML-first CLI or its tests. |
| `frontend/src/components/TechnicalPanel.jsx` | PURGE | Indicator chart: no role in the ML-first CLI or its tests. |
| `frontend/src/components/ThesisCard.jsx` | PURGE | Templated investment-thesis display: no role in the ML-first CLI or its tests. |
| `frontend/src/index.js` | PURGE | React entry point: no role in the ML-first CLI or its tests. |
| `frontend/src/theme.js` | PURGE | Signal color and MUI theme: no role in the ML-first CLI or its tests. |
| `pyproject.toml` | MODIFY | Preserve packaging role, replace backend discovery and broad dependency graph with tested CLI. |
| `requirements-dev.txt` | PURGE | Unittest needs no pytest plugins; consolidate tested runtime pins in requirements.txt. |
| `requirements.txt` | MODIFY | Remove torch 2.4 incompatible with the stated Blackwell requirement and unused web/NLP/database dependencies. |
| `scripts/modal_deploy.py` | PURGE | T4 cloud/torch 2.4 path is outside the laptop budget and has no executed deployment. |
| `scripts/seed_db.py` | PURGE | Seeds a database that predictions fetch around; unnecessary second source of price data. |
| `scripts/train_tft.py` | PURGE | No date-group purging; validation includes training rows; parsed batch-size/max-epochs are not used. |

## Replacement acceptance criteria

All production modules must be exercised by tests and the CLI. Artifacts must identify source hashes, code hashes, configuration, versions and fold dates. Future-data perturbations must not change earlier features/predictions; labels must mature before the purged/embargoed boundary. Hand calculations must match fees and compounding. README numbers must be generated from committed run artifacts. Documentation is retained only to specify the experiment, audit the purge or explain a real blocker.

## Implemented replacement and evidence

The complete replacement is seven production modules under `stocksage/`, three unittest files, one reproduction script and one CPU workflow. Packaging/README/ignore/dependency files are the four MODIFY decisions. The `evidence/` files are consumed by the README and reproduction verifier: every prediction, numerical metric, split boundary, source/data hash, hardware status and test outcome can be checked. Source changes precede benchmark execution in the Git history.

Implementation details made explicit: `min_train=252` means 252 usable feature sessions after label purging, so the first test begins at feature-session index 258 for embargo=5. Costs are a linear opening-notional approximation on both sides. Ridge intervals use training residuals; they are measured diagnostics, not calibrated confidence. Eight features are used, not a large indicator library.

The first 33-test run caught a provenance bug: Git discovery could attribute an enclosing checkout to a temporary/non-repository source tree. The fix verifies the repository root; all 33 tests then passed, including a fresh-venv run. The historical results did not cause any model/feature/configuration selection change. Both ridge and histogram boosting lose to zero-return MAE on the AAPL smoke benchmark. The final README reports this directly.
