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

- [x] H1 — `statsmodels` dep + config knobs (`hmm_enabled` off by default) + `HmmRegimeFrame` contract
- [x] H2 — HMM core (`fit_core`): MarkovRegression 3-state mean+var, sort-by-mean ordering, filtered probs, convergence/determinism guards
- [x] H3 — walk-forward engine: expanding-window monthly refit + daily Hamilton filter (causal-by-construction)
- [x] H4 — `classify_hmm` per-segment orchestration + engine wiring + `regimes_hmm.csv`/`hmm_refit_log.csv`/snapshot block
- [x] H5 — alerts reuse (`hmm_bucket_transitions`) + backtest scorer parametrized over label source + compare report
- [x] H6 — tests: causality/no-lookahead, state-ordering, determinism, cold-start, convergence, synthetic recovery
- [x] H7 — cadence-invariance bench (`@pytest.mark.slow`) → log result to results.md; decide production default per §8 rule

### HMM outcome (2026-06-03)
- Built + verified end-to-end (24 commits). Default `hmm_enabled=False` (overlay).
- §8 decision: **percentile stays production default** — HMM passed no gate on real data, regressed G3 (events 6/8 vs 7/8), but halved G5 flicker (9 vs 18). Evidence: `outputs/hmm_eval/acceptance_compare.json`.
- Follow-up: percentile baseline also fails 5/6 gates on 2008–2026 — gate/param calibration gap to revisit.
