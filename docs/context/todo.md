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

## Gate-Diagnostics (2026-06-09) — DONE
- [x] Harness built (Tasks 1–6): roro/gate_diagnostics.py + `roro gate-diagnostics` CLI subcommand. 188 tests green.
- [x] Task 7: harness run, memo written. Only G5 is a real discriminating failure. Memo: docs/analysis/2026-06-09-gate-diagnostics-memo.md.
- [ ] Follow-up: **Gate recalibration spec** — tier G1/G2/G6 as pipeline-health; fix G3 (drop out-of-range 2008 event); re-set G4/G5 thresholds off a calibration split; possibly adopt frontier-Pareto as G5 bar.

## Statistical Jump Model (2026-06-09) — DONE (feat/jump-model)
Spec: `docs/superpowers/specs/2026-06-09-jm-regime-design.md` · Plan: `docs/superpowers/plans/2026-06-09-jump-model.md`
- [x] J0 — vendored numpy-only JM core (`roro/jump_model.py`): seeded k-means++, forward/Viterbi DP, coordinate descent, empty-cluster reseed, n_init restarts, mean-canonical labels; discrete + continuous (simplex grid, BLAS-free).
- [x] J1 — causal `walk_forward` (`roro/regime_jm.py`): expanding/rolling refit + frozen-scaler forward-DP online inference; no-lookahead (incl. block boundary) + last_good fallback tested. Config `jm_*` (12 fields) + `JmRegimeFrame`.
- [x] J2 — `classify_jm` + engine/io/alerts wiring (`regimes_jm.csv`, `jm_refit_log.csv`, snapshot block); golden/byte-identical determinism (AJ-1 off, AJ-2 on for discrete+CJM).
- [x] J3 — continuous JM soft probabilities (BLAS-free broadcast + 10-dp round).
- [x] J4 — backtest scores JM through G1–G6 → `acceptance_report_jm.json` + 3-way `acceptance_compare.json`.
- [x] J5 — report 3-way Percentile/HMM/JM band toggle + JM state-probability figure (`fig_jm_probs`).
- [x] J6 — end-to-end real-data run (2008–2026): JM regimes render in HTML (Risk-off 1268 / Transitional 1320 / Risk-on 1637 / 315 warmup; non-degenerate). `outputs/jm_only_report.html`.
- Decisions D1–D5 locked (research-backed). Off by default. Promotion to production default deferred to the AJ-7 decision (JM clears G5 outright AND in-range G3 ≥ 7/7) against the recalibrated scorecard.
- [ ] Follow-ups (out of scope): λ re-calibration on standardized features (S-JM3); cadence-invariance bench (AJ-4); sparse multivariate JM + FTIC (S-JM6, where log(P)>0 activates FTIC).
