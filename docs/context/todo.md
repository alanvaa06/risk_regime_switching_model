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

## Regime Attribution (2026-09-05) — DONE (feat/attribution)
Spec: `docs/superpowers/specs/2026-09-05-roro-attribution-design.md` · Plan: `docs/superpowers/plans/2026-09-05-roro-attribution.md` · Memo: `docs/analysis/2026-09-05-attribution-memo.md`

- [x] A1 — numeric kernels (`roro/attribution.py`): `contributions`, `attribute_panel`, `attribute_delta`, `concentration`, `rank_against_prior`, `pc1_loadings`, `find_anchor`
- [x] A2 — orchestrator `compute_attribution` → `AttributionFrame`; engine step 5d (`attribution_enabled`, on by default); `_anchor_labels` fallback chain percentile/hmm/jm → percentile + warning
- [x] A3 — io (5 CSVs: `attribution.csv`, `attribution_delta.csv`, `attribution_rollup.csv`, `concentration.csv`, `attribution_pc1.csv` + optional `attribution_history_global.csv`) + `snapshot.json["attribution"]`; alerts `kind=concentration` (`transition_day` / `fragile`); goldens regenerated (5 new files; `alerts.csv` golden widened by the new kind; 5 pre-existing numeric goldens unchanged); off-switch byte-identity test
- [x] A4 — report: `roro/report/attribution_figs.py`, 5 figures each gated on its own frame; `band_shapes` promoted to public
- [x] A5 — real-data memo (`docs/analysis/2026-09-05-attribution-memo.md`): 2026-05-26 global-cap snapshot reproduced 15/15 to 4dp; concentration stats; D3 verdict; event review (AA-7)

Follow-ups (not started):
- [ ] D3 robust-slope spec — trigger evidence: global cap sign-flip on drop-top-1 = 17.15% of days, top1_share>0.5 = 19.25% (clears the 20% bar by only 0.75pp, provisional); per-cut EM 59.4% / LatAm 49.7% / EM_Eq 42.0% would already fire if the trigger is ever restated per-cut
- [ ] Make `fragile` a persistence-gated alert (n consecutive days) or demote it to a metadata field — as a daily alert it fires on 27.74% of global cap days (14,632 / 15,749 concentration alert rows), which is unactionable
- [x] Run with `attribution_history_global=True` to get the exact jackknife sign test from spec section 6 — done via `configs/attribution-history.yaml`: `top1_series` matches `argmax_i |c_i|` on 100.0% of days, `sign(beta - beta_ex_top1) == sign(c_top1)` on 85.7% of days, row sums equal `beta` to max abs error 3.0e-15 (see memo "Jackknife sign check (spec §6)")
- [ ] Document `fragile_flag`/`pct_ex_top1` as cap-only in the artifact schema docs (`roro/attribution.py:446` — every `eq` row reads `fragile_flag=False`, `pct_ex_top1=NaN` by construction, silently under-reporting sign flips like the 2026-05-26 global eq row)
- [ ] Squash decision for commit `9f7290e` (broken intermediate state: engine called `detect_alerts(attribution=...)` before alerts.py accepted the kwarg, fixed in `a36f4cf`) — squash-merge recommended when merging `feat/attribution`, or accept as-is if history is kept linear
