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
- 2026-06-03 T15: real-data HMM-vs-percentile backtest (2008–2026, quarterly refit, outputs/hmm_eval/). Neither passes all 6 gates. G3 events: percentile 7/8 vs HMM 6/8 (HMM lost 2010 Greek). G5 stability: HMM 9 vs percentile 18 transitions/calm-qtr (HMM halves flicker — thesis validated directionally). G4: 0.183 vs 0.161. G1/G2 shared (method-agnostic in v1.1). DECISION (§8): percentile stays production default; HMM ships off-by-default overlay (hmm_enabled=False). HMM regressed G3 + passes no gate → cannot be default.
- 2026-06-03 NOTE: percentile baseline fails 5/6 gates on full 2008–2026 data (G1 0.507, G2 0.682 vs 0.80; G5 18 vs ≤2). Pre-existing calibration gap — "S9 done" = harness built, not gates green on real data. Revisit gate thresholds / params.
