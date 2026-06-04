# Todo

- [x] S0 — project scaffolding, FRED key wiring
- [x] S1 — IO (Excel + FRED) + validators
- [x] S2 — returns + EWMA vol + cross-sectional regression
- [x] S3 — segmentation (10 cuts)
- [x] S4 — percentile classifier (tercile + quintile + direction)
- [x] S5 — correlation + PC1 panel
- [x] S6 — external + internal validation
- [x] S7 — 1M tripwire
- [x] S9 — backtest harness + acceptance gates
- [ ] S8 — HTML dashboard + JSON API (deferred to viz phase)
- [ ] v1.1 — GFP integration, time-varying mcap weights, composite price wiring through PriceFrame

## v1.1 — HMM regime classification (active)
Spec: `docs/superpowers/specs/2026-06-03-roro-hmm-regime-design.md`

- [ ] H1 — `statsmodels` dep + config knobs (`hmm_enabled` off by default) + `HmmRegimeFrame` contract
- [ ] H2 — HMM core (`fit_core`): MarkovRegression 3-state mean+var, sort-by-mean ordering, filtered probs, convergence/determinism guards
- [ ] H3 — walk-forward engine: expanding-window monthly refit + daily Hamilton filter (causal-by-construction)
- [x] H4 — `classify_hmm` per-segment orchestration + engine wiring + `regimes_hmm.csv`/`hmm_refit_log.csv`/snapshot block
- [x] H5 — alerts reuse (`hmm_bucket_transitions`) + backtest scorer parametrized over label source + compare report
- [ ] H6 — tests: causality/no-lookahead, state-ordering, determinism, cold-start, convergence, synthetic recovery
- [ ] H7 — cadence-invariance bench (`@pytest.mark.slow`) → log result to results.md; decide production default per §8 rule
