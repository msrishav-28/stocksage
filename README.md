# StockSage — evidence-first rebuild

Research and the per-file verdict are committed in [research/RESEARCH.md](research/RESEARCH.md) and [research/VERDICT.md](research/VERDICT.md).

The CPU CLI and integrity tests replace the former heuristic service/dashboard. Historical benchmark execution and the final results table are in progress on this branch. See [BLOCKERS.md](BLOCKERS.md) for the missing GPU and contemporary-data validation; no claim of completed Lenovo validation is made.

Run `python -m unittest discover -s tests -v` and `python -m stocksage --help` from this checkout. Final setup, experiment commands, metrics and limitations will be published with the measured artifacts.
