# Results

- 2026-05-27: RoRo engine implementation plan executed end-to-end.
- All numeric kernels covered by pytest unit + hypothesis property tests.
- Reproducibility invariant enforced: two runs over same fingerprint produce byte-identical CSVs.
- Golden integration test pinned at tests/golden/2024-Q1/.
- Full suite green: pytest, mypy --strict, ruff check.
- Engine produces beta_series.csv, regimes.csv, correlation.csv, external_validation.csv, alerts.csv, tripwire.csv, snapshot.json per run.
- Acceptance gates (G1–G6) wired in backtest harness, evaluated via `roro backtest --assert-gates`.
