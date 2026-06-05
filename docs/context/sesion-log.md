# Session log

- 2026-05-27: brainstorm → design spec → implementation plan → engine v0.1.0 built (23 tasks, subagent-driven).
- 2026-05-27: viz layer landed — roro.report package + roro report CLI. 3 Plotly figures (vol×ret scatter, β×ret scatter, β time-series with tercile bands) in single self-contained HTML.
- 2026-05-28: viz v2 — 7 segments (added DM_Eq, DM_FI), EM recolored green (#2ca02c), 95% CI ribbons via scipy.linregress, seaborn-like aesthetic, height=700, dropdown clear of legend.
- 2026-05-28: viz v3 — per-segment scatter axis ranges, legend top-left overlay, slider clear of axis title, beta time-series now spans full history, hover behavior locked with a test.
- 2026-05-28: viz v4 — regime bands legible: 21-day hysteresis smoothing collapses daily tercile flicker into broad blocks; opacity raised to 0.28/0.15/0.28. Cosmetic only; engine regimes.csv unchanged.
- 2026-06-03: pivot — defer S8 dashboard. Brainstormed HMM/Markov-switching regime classifier (3-state, β mean+var switching, statsmodels, filtered+point-in-time monthly refit, coexist with percentile). Spec written: docs/superpowers/specs/2026-06-03-roro-hmm-regime-design.md. Branch feat/hmm-regime.
- 2026-06-03: Task 4 — roro/regime_hmm.py + tests/test_regime_hmm.py committed (9130a23). TDD green; discovered statsmodels 0.14.x res.params is ndarray not Series — extraction uses model.param_names.
- 2026-06-03: HMM build complete (T1–T15, subagent-driven, 24 commits, branch feat/hmm-regime). Causality + cadence(0.9966) verified. T15 real-data backtest → percentile stays default, HMM off-by-default overlay (no gate passed, G3 regressed, G5 flicker halved 18→9). Suite green, mypy+ruff clean.
