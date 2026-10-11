# StockSage research and decision record

Research date: 2026-10-11. Scope: evidence available now, not predictions about 2027. Upstream pages are mutable; model IDs and paper identifiers below let readers inspect the claims independently. Published benchmark results are third-party results, never StockSage measurements. No model performance was measured before these choices were recorded.

## Decision

Build an offline-first, CPU research CLI for **next-session open-to-close returns**, using information through the preceding session close. Compare zero return, expanding training mean, standardized ridge, and histogram gradient boosting on identical purged walk-forward folds. Predict intervals and measure their coverage; never manufacture a confidence percentage. Keep financial text out of the numerical forecast until timestamped data and incremental out-of-sample evidence exist. Delete the frontend and service fleet.

This is a bounded implementation choice, not a claim that gradient boosting is the best financial predictor. A modern research system needs an honest experiment before it needs more model adapters. No GPU model code is to be added without an operational hardware check. The current execution host has no torch, nvidia-smi, or NVIDIA device nodes; the user's Lenovo is not accessible here. GPU comparisons are PENDING.

## Model landscape and licensing

| Candidate | Evidence and tradeoff | StockSage decision |
|---|---|---|
| Chronos-2 | Amazon's 120M encoder supports multivariate targets, covariates and quantiles; both CPU and GPU inference are documented. Public general forecasting benchmarks do not establish stock-return predictability. Apache-2.0 model card. [S1,S2] | First foundation-model candidate after hardware validation; zero-shot only initially, one model at a time, context 256, batch 1. Not installed or silently mocked. |
| Chronos-Bolt / smaller Chronos-2 variants | Official repository lists 9M Bolt tiny and 28M Chronos-2-small. Smaller alternatives reduce weight memory, but runtime and accuracy still require measurement. [S1] | Possible resource comparison, not another unused adapter. |
| TimesFM 2.5 | 200M checkpoint, Apache-2.0 weights. Forecasting priors learned from heterogeneous data are not finance-specific evidence. [S3] | Second permissive candidate. Fixed checkpoint/revision required before benchmarking. |
| TimesFM 3.0 | Official repository reports an August 2026 release with native multivariate/covariate support. Its downloadable model card says non-commercial, despite Apache code and separate commercial Cloud availability. [S4,S5] | Do not infer weight permission from repository license. Exclude default local stack. |
| Moirai / Moirai-MoE / Moirai 2.0 | Uni2TS documents the progression. Moirai 2.0 switches to decoder/quantile forecasting; its small checkpoint is CC-BY-NC-4.0. [S6,S7] | Research-only candidate; no bundled weights or dependency. |
| TabPFN-2.5 and successors | The 2.5 report describes support for 50K rows/2K features and strong TabArena comparisons. These are generic tabular tasks, not purged return tests. Official live repository also lists 2.6, 3 and 3.5, and explicitly separates Apache code from non-commercial newer weights. [S8,S9] | No default dependency. License restrictions plus unverified memory/runtime outweigh an untested benchmark advantage. |
| Histogram gradient boosting | Earlier controlled tabular benchmarks support trees as serious medium-data baselines, but predate TabPFN-2.5 and cannot settle the newer comparison. sklearn provides a compact CPU implementation. [S10,S11] | Implement a fixed, small challenger. Disable automatic random validation/early stopping. Report losses to baselines. |
| Ridge / zero / train mean | Regularized linear and constant predictors make added nonlinear complexity falsifiable. Zero predicts no intraday move; it is not a trading strategy or a fitted model. | Mandatory comparators on identical samples. Fit scaler and coefficients only inside each training fold. |

**No invented hardware benchmarks:** 120M float32 weights alone are about 480 MB (parameter arithmetic), not a VRAM peak estimate. Activations, KV/intermediate storage, libraries and batches add memory. On the Lenovo cap numerical input at 20 symbols and 100,000 rows, about 10 features; start a substantive experiment with 5 predeclared liquid equities over roughly 10 years (~12,600 rows). Keep only one model resident. Do not use 8 GB VRAM as evidence that all contexts or fine-tuning fit. The CPU core does not need CUDA. For GPU work the user requires torch >=2.8 with cu128 wheels; official previous-version instructions provide that wheel index [S12]. An actual CUDA matrix multiplication is required after installation, not just an import.

## Financial text: useful retrieval, unproven forecasting

FinanceBench supplies evidence-linked financial QA examples and reports substantial errors in the tested 2023 systems [S13]. It is evidence for evaluating retrieval/grounding, not a current model ranking. FinRobot presents a multi-agent equity research workflow [S14]; generated research prose is not proof of forecast skill. A 2026 point-in-time benchmark raises the risk that model knowledge leaks future events into historical financial workflows [S15]; it is a preprint, not a universal guarantee about any provider.

If text is later added, ingest filings from SEC EDGAR [S16] with accession IDs, accepted/publication times, retrieval times, immutable content hashes and exact source spans. Restrict retrieval to documents available by decision time, evaluate retrieval recall and numeric citation support, and abstain when evidence is absent. Treat document content as untrusted data, never agent instructions. Backtest text only with point-in-time model knowledge or label the contamination risk. Measure numerical forecast improvement separately from QA quality. No API key, archived news corpus or independently evaluated text workflow exists here; an LLM agent/RAG subsystem would currently be dead code, so it is excluded.

## Evaluation contract, fixed before results

1. **Target/execution:** at close t, predict close(t+1)/open(t+1)-1. Use only bars through t for predictors. Next open/close appear only in labels. A long/cash diagnostic enters at next open and exits at next close; each active session pays both entry and exit costs. It has no overnight exposure. An always-long intraday comparator is not buy-and-hold.
2. **Features:** backward returns over 1/5/20 sessions, rolling 20-session volatility, close versus rolling mean, same-day open/close return, high/low range, and trailing relative volume. No centered smoothers, global normalization, backward fills, future covariates, fabricated sector/beta or sentiment zeros.
3. **Split:** expanding chronological train, fixed forward test blocks, all symbols grouped by session. Purge every train observation whose label-end reaches the test boundary, with five additional session gap by default. Operational rule: train label_end < session(test_start_index - embargo). This is a conservative pre-test embargo. Conventional post-test embargo is vacuous within a past-only fold because no later training examples are admitted. No shuffled folds or random validation. Earlier test outcomes may enter later training once mature; this is scheduled online retraining, not a permanently untouched holdout. [S17,S18]
4. **Budget:** start with 252 feature sessions in the training window and 63-session test blocks, including the final partial block. Fixed ridge alpha=10; histogram models use 100 iterations, 7 leaves, min leaf=20, learning rate=.05, L2=1, seed=28, early_stopping=False. No search or retrospective parameter change after reading results. Historical mean/quantiles use training labels only. Histogram quantile models at .1 and .9 supply an empirical 80% band; sort bounds around the point estimate and report coverage/width and pinball loss. These are not distribution-free coverage guarantees.
5. **Metrics:** MAE/RMSE in return units, direction accuracy (zero is an explicit tie), interval coverage/width, quantile loss; per-fold and per-symbol results. Paired MAE difference against zero with a moving-block bootstrap over session-averaged losses preserves within-session cross-sectional dependence. Report 95% intervals as descriptive, not proof of significance after model selection.
6. **Economics:** long only when forecast exceeds round-trip cost, otherwise cash; costs 0/5/10/25 basis points per side. Fix threshold to each scenario's cost without optimization. Equal capital per symbol every day, cash for inactive symbols, compound daily portfolio returns, drawdown from an initial equity of 1. Sharpe is a descriptive zero-risk-free annualization, not a forecast or an alpha estimate. No leverage, shorts, overlapping holdings or unconstrained returns aggregation.
7. **Data integrity:** strict duplicate/date/OHLC/finite-value checks, balanced symbol calendars, immutable input and code hashes, no silent symbol drops, bounded input size, input manifest with origin/license/adjustment/point-in-time limitations. CSV import deliberately makes provenance explicit. Provider snapshots can still be retrospectively revised; hashes cannot cure economic leakage.
8. **Reproducibility:** commit config, metrics, fold boundaries, every out-of-sample prediction, environment metadata, and a command that reconstructs the experiment. Separate generated synthetic integrity tests from historical observations. Preserve negative outcomes. Multiple tries inflate discovery risk [S19]; there is no model-selection or alpha claim here.

## Available data and what it can prove

The broad intended universe is PENDING: direct Yahoo chart request returned HTTP 429 and Stooq requests HTTP 404 on this host. A small historical AAPL OHLCV sample in Plotly's public datasets repository is accessible [S20]. Pin it to commit `6a88de1c9a95f0019566d3b568eb26444feda686`; discard its precomputed indicators and use only raw OHLCV. Download locally; do not redistribute the source CSV because a data redistribution license has not been established. Record both download and normalized-input hashes. This old, single surviving equity is only a historical smoke benchmark, not a contemporary multi-asset validation. Publication in 2017 means it is not a contemporaneous vintage archive of the 2015 observations. Foundation models pretrained later must not be scored on it as clean zero-shot financial evidence.

Generated correlated random-walk panels exercise execution and leakage invariants offline. They provide zero evidence about real-market predictive value. Tests must include future perturbation invariance, hand-computed label alignment, no overlapping train labels, invalid data failures, exact economic cost arithmetic and full CLI reproduction. No test count or accuracy is promised until executed.

## Sources (primary sources, inspected 2026-10-11)

- S1: Amazon code/model inventory — https://github.com/amazon-science/chronos-forecasting
- S2: Chronos-2 model card and report — https://huggingface.co/amazon/chronos-2 ; https://arxiv.org/abs/2510.15821
- S3: Google TimesFM-2.5 model card — https://huggingface.co/google/timesfm-2.5-200m-pytorch
- S4: Google TimesFM repository/release record — https://github.com/google-research/timesfm
- S5: Google TimesFM-3.0 model card — https://huggingface.co/google/timesfm-3.0-pytorch
- S6: Salesforce Uni2TS — https://github.com/SalesforceAIResearch/uni2ts
- S7: Moirai-2.0 checkpoint card — https://huggingface.co/Salesforce/moirai-2.0-R-small
- S8: TabPFN-2.5 report, v2 February 2026 — https://arxiv.org/abs/2511.08667
- S9: Prior Labs code and weight licensing — https://github.com/PriorLabs/TabPFN#license
- S10: Grinsztajn et al., NeurIPS 2022 tabular benchmark — https://arxiv.org/abs/2207.08815
- S11: sklearn estimator contract — https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- S12: PyTorch wheel instructions — https://pytorch.org/get-started/previous-versions/
- S13: FinanceBench — https://arxiv.org/abs/2311.11944
- S14: FinRobot — https://arxiv.org/abs/2411.08804
- S15: Look-Ahead-Bench — https://arxiv.org/abs/2601.13770
- S16: SEC EDGAR API — https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- S17: sklearn chronological split/gap contract — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- S18: mlfinlab implementation documentation, purging/embargo — https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html
- S19: Bailey et al., Probability of Backtest Overfitting — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- S20: Plotly historical sample — https://github.com/plotly/datasets/blob/6a88de1c9a95f0019566d3b568eb26444feda686/finance-charts-apple.csv
