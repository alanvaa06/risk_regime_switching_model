# Results

- 2026-05-27: RoRo engine implementation plan executed end-to-end.
- All numeric kernels covered by pytest unit + hypothesis property tests.
- Reproducibility invariant enforced: two runs over same fingerprint produce byte-identical CSVs.
- Golden integration test pinned at tests/golden/2024-Q1/.
- Full suite green: pytest, mypy --strict, ruff check.
- Engine produces beta_series.csv, regimes.csv, correlation.csv, external_validation.csv, alerts.csv, tripwire.csv, snapshot.json per run.
- Acceptance gates (G1–G6) wired in backtest harness, evaluated via `roro backtest --assert-gates`.
- 2026-06-03 Task 4: roro/regime_hmm.py created — _fit_params uses model.param_names (not Series index) since statsmodels 0.14.x returns plain ndarray; 2 tests green, mypy strict clean, ruff clean.
- 2026-06-03 T12: _evaluate_gates parametrized over explicit labels/transitions; run_backtest scores both percentile + HMM through shared G1–G6 gates; writes acceptance_report_hmm.json + acceptance_compare.json when HMM enabled; fast+slow tests green, mypy+ruff clean.
- 2026-06-03 T14: HMM cadence bench: monthly(21) vs daily(1) refit label agreement = 0.9966 on a 501-day 3-regime series → monthly cadence validated as ~equivalent to daily (slow bench runtime ~7m47s: tests/test_regime_hmm_cadence.py).
