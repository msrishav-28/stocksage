# Blockers and boundaries

Observed 2026-10-11, before implementation:

| Work | Observed blocker | Status / next concrete action |
|---|---|---|
| Lenovo / CUDA validation | Execution host is Linux cloud runtime, Python 3.12.14; no nvidia-smi, torch installation or /dev/nvidia* devices. No access to the Lenovo. | PENDING. Stop GPU implementation. On Lenovo install torch >=2.8 from cu128 index and run hardware preflight with a real CUDA operation. CPU work continues. |
| Contemporary broad market experiment | Yahoo chart request returned HTTP 429; Stooq AAPL/MSFT endpoints returned HTTP 404. No licensed point-in-time file attached. | PENDING. Accept user-supplied licensed OHLCV snapshots through validated CSV import. Use pinned historical Plotly AAPL only as a smoke benchmark. |
| Point-in-time news/macro and text-model evaluation | No archived time-stamped corpus, vintages, API keys or verified model cutoff supplied. | PENDING. Exclude text agents rather than invent neutral evidence. |
| Data redistribution | No dataset-specific redistribution permission established for Plotly AAPL sample. | Raw market CSV stays untracked; downloader pins source commit and verifies hashes. |

A run on this host cannot certify Blackwell compatibility, Lenovo memory use, or 2026–27 market usefulness. No speculative GPU adapter is shipped. Repository code previously declared MIT in pyproject without a LICENSE file; this rebuild does not invent a copyright grant. Upstream component licenses are distinct from project/data rights.
