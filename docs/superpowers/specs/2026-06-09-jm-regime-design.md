# RoRo — Statistical Jump Model Regime Layer

**Date:** 2026-06-09
**Status:** Design approved (autonomous, research-backed). Ready for implementation planning.
**Author:** Alan Vazquez, CFA
**Scope:** Add `roro/regime_jm.py` — a 3-state **statistical jump model (JM)** regime classifier on the cap-weighted cross-sectional slope, mirroring the HMM contract exactly (same per-segment input, `*_RegimeFrame` output shape, causal/no-lookahead guarantee, acceptance-gate scoring). Off by default (`jm_enabled`). Ships with a 3-way report band toggle (Percentile ↔ HMM ↔ JM) + a JM state-probability figure. Implements `docs/prd/JM-PRD.md`; the open decisions D1–D5 are locked below with academic backing from the JM literature.

> **Disambiguation (carry into code comments):** the JM "jump" is a **regime-transition penalty** in an unsupervised clustering objective — *unrelated* to Poisson price-jump SDEs (Merton jump-diffusion). The module must not imply jump-diffusion pricing.

---

## 1. Goal

The HMM overlay (shipped 2026-06-03) halved calm-quarter flicker (G5: 9 vs 18) but regressed event recognition (G3: 6/8 vs percentile 7/8) — a learned transition matrix smears fast regime changes. The JM's thesis: a **fixed jump penalty** `λ` regularizes persistence *structurally* (an L0 cost per transition), so it holds regimes through noise **yet** snaps cleanly on genuine breaks — the G5-without-losing-G3 profile RoRo needs (Shu-Mulvey 2024: JM beats HMM on **both** persistence and crisis capture). The JM ships as an off-by-default overlay; promotion to production default is a separate gated decision (D5 / AJ-7).

The gate-diagnostics work (this branch's predecessor) established that the JM's acceptance is judged on the **in-range G3** (the binary 8/8 bar counted an out-of-range 2008 event) and the **G3↔G5 frontier** — so the JM's real target is to push that frontier out past where the percentile classifier and HMM sit.

---

## 2. Locked decisions (D1–D5, research-backed)

| Fork | Choice | Academic justification |
|---|---|---|
| **D1** Feature | **Univariate** cap-wtd slope (`bf.cap_wtd["beta"]`), K=3 | Clean apples-to-apples A/B/C vs HMM/percentile on the *same* series; isolates "fixed jump penalty vs learned transition matrix." Single-asset downside-risk template — Shu-Mulvey (2024), arXiv:2402.05272. Sparse multivariate JM + FTIC is the S-JM6 fast-follow (Nystrup-Kolm-Lindström 2021). |
| **D2** λ selection | **Fixed, gate-calibrated** (default `50.0`); `jump_penalty` injectable into `walk_forward` | FTIC's complexity coefficient `a_n = log(log N)·log(P)` (Fan-Tang 2013; Cortese-Kolm-Lindström 2024) **collapses to 0 at P=1** (`log 1 = 0`) → cannot discriminate λ in 1-D. Defer FTIC to multivariate v1.1. Calibrate λ by scanning log-spaced [0,100] on a validation split vs the RoRo gate objective (G3 s.t. G5≤baseline) — the diagnostic analog of Shu-Mulvey's validation-Sharpe. λ~50–100 ≈ <1 shift/yr / realistic bear durations, **but must be re-scanned on standardized features** (loss scale differs from raw slope). |
| **D3** Window/cadence | **Expanding + monthly** (`jm_refit_interval_days=21`, `jm_min_history_days=252`); expose `jm_window='rolling'`, `jm_rolling_window_days=2000` | Exact HMM parity → reuses the HMM's monthly≈daily 0.9966 cadence-invariance evidence (AJ-4 ≥0.99). Rolling recipe = Shu-Mulvey (2000-day v1 preprint; 3000-day published — cite each to its version, don't mix). |
| **D4** Dependency | **VENDOR** a ~150–180-line numpy-only deterministic JM in `roro/regime_jm.py`; do NOT adopt `jumpmodels` | `jumpmodels` (PyPI v0.1.1, Apache-2.0) declares **unpinned** hard deps on numpy+pandas+scipy+scikit-learn+matplotlib; RoRo has no sklearn and pins numpy<2.1/pandas<2.3. Its determinism is real but contingent on sklearn's `kmeans_plusplus` RNG stream, which sklearn does **not** guarantee stable across versions → a bump would silently change labels and break the golden fixture. The only sklearn surface used (`kmeans_plusplus`, `check_random_state`) is replaceable with ~20 lines of seeded PCG64. Apache-2.0 permits copying the algorithm with attribution. Keep `jumpmodels` as a **dev-only test oracle** (λ=0 == k-means identity; label agreement up to permutation). |
| **D5** Promotion | **Off-by-default overlay** (`jm_enabled=False`); promotion deferred to an S-JM5 memo gated on **AJ-7** (JM clears G5 outright AND in-range G3 ≥ percentile 7/7) | Same conservatism the HMM overlay got. `jm_enabled=False` ⇒ engine/report/backtest byte-identical to today (AJ-1). |

PRD citation fixes (applied to `docs/prd/JM-PRD.md`): the GIC paper is **Cortese, Kolm & Lindström** (SSRN 4774429 / AStA 2026), not "Cortese 2026" alone and not Nystrup (who authored the *feature-selection* precursor, SSRN 3805831); Shu-Mulvey's 2000-day/~8.4× figures are the **v1 preprint** while 3000-day/44%-vs-141% is the **published** Journal of Asset Management version.

---

## 3. Algorithm (implementation-ready)

### 3.1 Objective (discrete JM, K=3, D=1)
Over causal observations `y_0…y_{T-1}` (standardized cap-wtd slope), centroids `Θ={θ_0,θ_1,θ_2}`, path `S={s_0…s_{T-1}}`:

```
J = Σ_t  ½(y_t − θ_{s_t})²   +   λ · Σ_{t≥1} 1{s_{t-1} ≠ s_t}
```

(Shu-Mulvey 2024 Eq.1; loss `l(x,θ)=½‖x−θ‖²`.) `λ=0` → k-means; `λ→∞` → single state.

### 3.2 Coordinate descent (one restart; pure fn of `(y, seed, λ, K, n_init, max_iter, tol)`)
1. `loss_mx` (T×K): `loss_mx[t,k] = ½(y_t − θ_k)²`.
2. `penalty_mx` (K×K): `λ·(1 − I_K)` — 0 on diagonal, λ off-diagonal.
3. Repeat until **`S` unchanged** (primary criterion — label-path fixed point) OR `iter == max_iter`:
   - **DP/E-step:** `S = viterbi_argmin(loss_mx, penalty_mx)` (§3.3).
   - **M-step:** for each k, `θ_k = mean(y[S==k])`; if empty → **reseed** (§3.5).
   - recompute `loss_mx`, `J`.
   Objective is **monotone non-increasing** each step (Bemporad 2018 Alg.1), so `prev_J − J ≥ 0` always. Use `(prev_J − J) < jm_tol` (absolute, on the standardized-loss scale) only as a *secondary* early-stop after the M-step, never as the primary gate — J can plateau for one iteration then drop, so stopping on `tol` alone risks halting before the label fixed point and makes the winning restart `tol`-sensitive. `S`-unchanged is the deterministic primary stop. Array contract: `y: (T,)`, `Θ: (K,)`, `loss_mx: (T,K)` — keep `y` 1-D (not `(T,1)`) so the CJM loss broadcast (§3.7) is unambiguous.

### 3.3 DP path recurrence (verbatim from jumpmodels/jump.py; O(T·K²), linear in T)
```python
values = np.empty((T, K)); values[0] = loss_mx[0]
for t in range(1, T):
    values[t] = loss_mx[t] + (values[t-1][:, None] + penalty_mx).min(axis=0)
assign = np.empty(T, dtype=int); assign[T-1] = values[T-1].argmin()
for t in range(T-1, 0, -1):
    assign[t-1] = (values[t-1] + penalty_mx[:, assign[t]]).argmin()
```
`np.argmin` returns the **first** minimal index → ties deterministic given bit-identical inputs.

### 3.4 Restarts / k-means++ init
Run `n_init=10` restarts. Seed via `child = np.random.SeedSequence(jm_random_seed).spawn(n_init)`; restart i uses `np.random.Generator(np.random.PCG64(child[i]))`. k-means++: center 0 = `y[rng.integers(T)]`; each next center drawn with prob ∝ D² (min squared dist to chosen centers) via `rng.random()` against cumulative-D²; derive `S^0` by nearest-centroid. Keep the restart with **strict-lowest** final `J` (lowest-index restart wins ties → deterministic).

### 3.5 Empty-cluster rule (jumpmodels has none — mandatory here)
If any state is unused after the DP-step, reseed empties **sequentially, one at a time, recomputing per-point loss after each**: pick the empty state with the lowest index, set its centroid to `y[argmax_t(min squared dist to the currently-occupied centroids)]` (farthest-point, lowest-index tie-break), mark that point claimed, recompute distances, then handle the next empty state. This prevents two simultaneously-empty clusters from grabbing the *same* farthest point (degenerate duplicate centroids). Deterministic, never NaN. (Thin-cut segments whose standardized slope genuinely has <3 regimes may legitimately collapse to fewer effective states — this is the model being honest, surfaced via `thin_cut_flag`, not a bug; a degenerate-window test asserts no NaN and ≥2 distinct states where the data supports them.)

### 3.6 Causal standardization + online inference (the no-lookahead primitive)
Per refit at index `r`: fit the **scaler** (mean/std, ddof=0, std==0→1.0) AND `Θ` on `clean.iloc[:r]` only. For the block `[r, block_end)`: standardize `clean.iloc[:block_end]` with the **frozen** scaler, build `loss_mx`, run the **forward DP pass only** (the `values` recurrence — NO backward `assign` reconstruction over future rows), and read `s_hat_e = values[e].argmin()` for `e ∈ [r, block_end)`. Because `values[e]` depends only on `values[:e]`, this single per-block forward pass equals the online filter at each `e` — the direct causal analog of the HMM Hamilton filter. **Backward reconstruction (§3.3) is offline-only research; it peeks at the future, exactly like HMM smoothing which `regime_hmm.py` forbids.**

### 3.7 Continuous JM (CJM, `jm_continuous=True`) — soft probs for the report figure
Replace hard `s_t` with a probability vector on a **fixed discretized simplex grid** `C = discretize_simplex(K, grid_size=0.05)`: all weight vectors with components in multiples of `grid_size` summing to 1, enumerated in a **fixed lexicographic order over integer compositions** (deterministic grid index → vector map; ~231 candidates at K=3). Per-day CJM loss row `cjm_loss[t,n] = Σ_k C[n,k]·loss_mx[t,k]` computed by **broadcast-and-sum, NOT a BLAS matmul** (`(loss_mx[:,None,:] * C[None,:,:]).sum(axis=2)`) — this keeps D=1 truly BLAS-free so reduction order can't drift across thread counts (see §8). Transition penalty = `λ·‖c_i − c_j‖₁²` between grid vectors (Aydinhan et al. 2024 L1-squared); **same forward DP** over the grid. `values[e].argmin()` (prefix-stable, same recurrence) picks grid vector `c*`; emit soft `(prob_risk_off, prob_transitional, prob_risk_on) = c*` reordered by `perm`, `state/label = argmax(c* reordered)`, `confidence = max(c*)`. Discrete path → one-hot probs with `state == argmax(probs)` (asserted by test) and `confidence = 1.0`. **Round all emitted probabilities to 10 decimals before CSV melt (mandatory for CJM, harmless for discrete).** Behind the `jm_continuous` flag; Shu-Mulvey found no 0/1-allocation edge, so CJM exists here only to feed the stacked-area figure (HMM-parity), not to change the discrete production labels. The no-lookahead property test (AJ-3) runs against both discrete and CJM paths.

### 3.8 `walk_forward` (mirrors `regime_hmm.walk_forward`, regime_hmm.py:114-176)
`r = jm_min_history_days`; while `r < n`: `block_end = min(r+refit, n)`; define the **fit window** `W_r` = `clean.iloc[:r]` (expanding, default) or `clean.iloc[max(0, r−jm_rolling_window_days):r]` (rolling); fit `Θ`+scaler on `W_r` (n_init restarts); if converged → `last_good=(Θ,scaler,perm)`, append refit_date; `used = fit if converged else last_good`. **Online inference for the block:** standardize the **inference window** `I_block` with the frozen scaler and run the forward DP (§3.6) over it, reading `s_hat_e = values[e_local].argmin()` for `e ∈ [r, block_end)`. The inference window matches the fit window's span: expanding → `I_block = clean.iloc[:block_end]` (DP from index 0); rolling → `I_block = clean.iloc[max(0, r−jm_rolling_window_days):block_end]` (DP from the window start, so the block's states are decoded over the same lookback the parameters were fit on). In both cases `values[e]` depends only on rows ≤ e *within* `I_block` (forward recurrence) → causal and prefix-stable, so the AJ-3 no-lookahead test holds for both windows. `r = block_end`. Rows `< jm_min_history_days` (and NaN-beta rows) stay `state/prob_* = NaN`, `cold_start = True` via the same `reindex(full_index)` / `fill_value=True` handling as regime_hmm.py:158-173. Re-canonicalize `perm = argsort(centroid means)` ascending → `(Risk-off, Transitional, Risk-on)` **every refit** (FJ2.2; defeats label-permutation symmetry exactly as regime_hmm.py:85 sorts EM states by mean).

---

## 4. Data contract

### 4.1 `walk_forward` return dict — same keys as the HMM (regime_hmm.py:165-176)
`state` (float; 0/1/2 or NaN), `label` (str; Risk-off/Transitional/Risk-on, `Unknown` for NaN), `prob_risk_off`, `prob_transitional`, `prob_risk_on` (float; NaN where cold/NaN-beta; one-hot for discrete), `confidence` (float; max prob; 1.0 for discrete), `cold_start` (bool; reindex fill_value=True), `refit_dates` (`list[pd.Timestamp]`, converged refits only). Same `_series()` reindex helper.

### 4.2 `JmRegimeFrame` (new in `types.py`, structurally identical to `HmmRegimeFrame`)
`state, label, prob_risk_off, prob_transitional, prob_risk_on, confidence, n_per_segment, thin_cut_flag, cold_start_flag` (all wide DataFrames) + `refit_dates: dict[str, list[pd.Timestamp]]`. Optional `jump_penalty_used: dict[str,float]` per segment (for when λ is auto-selected later; constant in v1).

### 4.3 `classify_jm(bbs, *, cfg, thin_cuts) -> JmRegimeFrame`
Signature-parallel to `classify_hmm` (regime_hmm.py:179). Per segment: operate on `bf.cap_wtd["beta"]`, `n_per_segment = bf.cap_wtd["n"]`, `thin_cut_flag = (cut in thin_cuts)` broadcast. Consumes `cfg.jm_jump_penalty`, `jm_n_states`, `jm_refit_interval_days`, `jm_min_history_days`, `jm_window`, `jm_rolling_window_days`, `jm_continuous`, `jm_random_seed`.

---

## 5. Configuration (extend `EngineConfig`, config.py)

New frozen fields (defaults = HMM parity + determinism). `to_dict()`/`load_config` are generic over `fields()` → **zero change** (new keys auto-allowed + auto-serialized into snapshot.json).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `jm_enabled` | bool | `False` | master switch (mirrors `hmm_enabled`) |
| `jm_jump_penalty` | float | `50.0` | λ; calibrated in S-JM3 (D2) |
| `jm_n_states` | int | `3` | parity with HMM/terciles |
| `jm_refit_interval_days` | int | `21` | monthly refit (HMM parity) |
| `jm_min_history_days` | int | `252` | warmup (HMM parity) |
| `jm_window` | str | `"expanding"` | `"expanding"` (HMM parity) or `"rolling"` (Shu-Mulvey) — D3 |
| `jm_rolling_window_days` | int | `2000` | lookback when `jm_window="rolling"` |
| `jm_continuous` | bool | `False` | CJM soft probs vs discrete one-hot |
| `jm_n_init` | int | `10` | k-means++ restarts (Shu-Mulvey) |
| `jm_max_iter` | int | `30` | coordinate-descent cap (a finite cap, not a convergence gate — see §3.2; a max_iter fit with finite centroids still counts converged) |
| `jm_tol` | float | `1e-8` | secondary early-stop on `prev_J − J` (absolute, standardized scale); FJ5.1 requires it in config so it serializes into snapshot.json |
| `jm_random_seed` | int | `0` | determinism seed (PCG64 / SeedSequence) |

> Note: `jm_continuous` defaults `False` (discrete) in v1 — the discrete JM is the production label source; CJM can be flipped on to populate the probability figure. (PRD §FJ5 had `jm_continuous=True`; we default `False` because Shu-Mulvey found no label edge and the discrete one-hot still renders the stacked-area figure as a clean step-area. Flip to `True` only when soft bands are wanted.)

---

## 6. Integration surface (each = minimal parallel of the HMM; see `docs/superpowers/jm-integration-contract.md`)

- **engine.py:96** — `regime_jm = classify_jm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"})) if cfg.jm_enabled else None`; pass `regime_jm=` to `detect_alerts` and `RunResult`.
- **types.py** — `JmRegimeFrame`; `RunResult.regime_jm: JmRegimeFrame | None = None`; `AlertSet.jm_bucket_transitions` (empty-DF default).
- **alerts.py:12** — `detect_alerts(..., regime_jm=None)` → `jm_bucket_transitions = _bucket_transitions(regime_jm.label)` gated on not-None (reuse the model-agnostic `_bucket_transitions`).
- **io.py:131,259,285,315** — `_write_regime_jm` → `regimes_jm.csv` (cols `date,segment,state,label,p_risk_off,p_transitional,p_risk_on,confidence,cold_start,thin_cut` — identical to `regimes_hmm.csv`), `_write_jm_refit_log` → `jm_refit_log.csv`, `snapshot["regime_jm"]` last-row block. Inside the existing `.tmp/` atomic boundary.
- **backtest.py:107** — parallel block: `_evaluate_gates(result, labels=regime_jm.label, transitions=jm_bucket_transitions)` → `acceptance_report_jm.json`; extend `acceptance_compare.json` to 3-way `{percentile, hmm, jm}`. (G1/G2/G6 are method-shared, per the gate-diagnostics memo.)

---

## 7. Report viz (3-way band toggle + JM probability figure)

`regimes_jm.csv` column contract is identical to `regimes_hmm.csv`.
- **bundle.py:29** — add `seg_jm_label / seg_jm_p_off / seg_jm_p_tr / seg_jm_p_on` (Optional, default None).
- **load.py:150** — parallel `if (run_dir/"regimes_jm.csv").exists():` pivot block → `DataBundle`.
- **figures.py:684** (`beta_band_lookup`) — emit `out[seg] = {"percentile":…, "hmm":…, "jm":…}`; JM bands via `_band_shapes(seg_jm_label[seg], smooth=False)` (raw runs — JM is persistent like the HMM, no hysteresis). Reuse `REGIME_COLORS`.
- **figures.py:862** — clone `regime_probability_area` → `regime_probability_area_jm` (swap `seg_hmm_*`→`seg_jm_*`, title "JM regime probabilities", div_id `fig_jm_probs`).
- **html.py:53** — add `<option value="jm">Jump Model</option>` (third). `apply()` already keys `byMethod[method]` → zero JS-logic change.
- **orchestrate.py:48** — `if bundle.seg_jm_label is not None:` append the JM probability FigureSpec + include `jm` in the lookup.
- Figures appear **only** when the run carries JM output → `jm_enabled=False` keeps HTML byte-identical (AJ-1). Determinism preserved: literal div_ids, no timestamps, ISO-date shape payloads. Tests: `write_jm_csv` fixture + `test_build_report_includes_jm_when_present` + byte-identical HTML regression.

---

## 8. Determinism plan (RoRo's #1 invariant)

- **Seeding:** JM is a pure fn of `(y, jm_random_seed, λ, K, n_init, max_iter, tol)`. `SeedSequence(seed).spawn(n_init)` → `Generator(PCG64(child))` per restart. PCG64 stream is fixed-seed-stable within a numpy major (RoRo pins numpy<2.1 via the uv lockfile). No `np.random` global, no shuffle, no unseeded draws. Set `jm_random_seed` explicitly in `walk_forward`.
- **Convergence/fallback:** mirror the HMM `last_good` (regime_hmm.py:141-156). A restart "converged" iff all-finite centroids + finite J; keep strict-lowest J; if none converge, reuse `last_good`; empty-cluster reseed guarantees finite output; never emit NaN. Wrap fit in `warnings.catch_warnings(); simplefilter("ignore")` (HMM pattern) so `pytest filterwarnings=["error"]` does not trip on convergence/empty-cluster warnings.
- **Tie-breaking / label canonicalization:** `argmin`/`argmax` first-index; strict `<` for best-restart; `perm = argsort(means)` every refit → deterministic regime IDs.
- **Float/BLAS:** the discrete path at D=1 is scalar adds/means (no BLAS) → reduction-order nondeterminism is genuinely absent. The CJM path must NOT introduce a BLAS gemm: compute `cjm_loss` by broadcast-and-sum (§3.7), not `loss_mx @ C.T`, so it stays BLAS-free. As belt-and-suspenders for CJM and any future D>1 work: (1) **round emitted probabilities to 10 decimals before the CSV melt** (mandatory, §3.7), and (2) the JM run harness sets `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` before invoking the engine so any incidental reduction is single-threaded and stable. The CSV writer reuses the `_write_regime_hmm` pattern. **AJ-2 determinism (two runs → byte-identical `regimes_jm.csv`) is asserted for BOTH `jm_continuous=False` and `True`.**

---

## 9. Acceptance criteria

- **AJ-1** `jm_enabled=False` ⇒ engine/report/backtest **byte-identical** to pre-change (regression + golden + reproducibility tests pass).
- **AJ-2** `jm_enabled=True` ⇒ two runs over identical inputs produce **byte-identical** `regimes_jm.csv` (determinism gate). Golden JM fixture under `tests/golden/`.
- **AJ-3** No-lookahead property test: `s_hat_t` invariant to appended future rows.
- **AJ-4** Cadence-invariance: monthly-refit JM labels agree with daily-refit ≥0.99 (mirror HMM 0.9966) → monthly default justified.
- **AJ-5** Full quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest` (`filterwarnings=["error"]`). One test module per code module; property tests for no-lookahead + scale-invariance + λ=0==k-means identity.
- **AJ-6** JM scored through G1–G6 with a 3-way comparison artifact written.
- **AJ-7 (promotion bar, not ship bar):** JM clears **G5 outright** *and* **in-range G3 ≥ percentile (7/7)**. If met → trigger the S-JM5 promotion decision; else ships as documented overlay.

---

## 10. Testing (one module per code module)

- `tests/test_regime_jm.py` — JM core: λ=0==k-means identity; DP correctness on a hand-checkable tiny series; empty-cluster reseed; convergence/last-good fallback; state-ordering by mean; determinism (two fits → identical); scale-invariance; no-lookahead (prefix-stable); cold-start; synthetic two-regime recovery.
- `tests/test_regime_jm_cadence.py` (`@pytest.mark.slow`) — monthly≈daily agreement ≥0.99.
- Engine/io/alerts/backtest: extend existing parallel tests (mirror the HMM test coverage).
- Report: `tests/report/` — JM band toggle + JM probability figure present when `regimes_jm.csv` exists; byte-identical HTML when absent.
- Dev-only oracle (optional, `@pytest.mark.skipif jumpmodels missing`): vendored discrete-JM labels agree with `jumpmodels` up to permutation at a fixed seed; λ=0 k-means identity.

---

## 11. Milestones (TDD task groups for the plan)

| Group | Deliverable |
|---|---|
| **J0** | Vendored JM core (`_kmeanspp_init`, `_viterbi_path`, `_fit_once`, `_coordinate_descent`, empty-cluster, restart selection) + unit/property tests (λ=0 identity, DP, determinism, scale-invariance) |
| **J1** | Causal `walk_forward` (expanding + monthly refit + per-block frozen-scaler forward-DP online inference) + no-lookahead + cold-start tests; `JmRegimeFrame` + config fields |
| **J2** | `classify_jm` per-segment orchestration; engine wiring (`jm_enabled`); io writers (`regimes_jm.csv`, `jm_refit_log.csv`, snapshot block); alerts (`jm_bucket_transitions`); golden fixture + determinism regression |
| **J3** | CJM soft-probability path (simplex grid + same DP), behind `jm_continuous` |
| **J4** | Backtest: score JM through G1–G6 → `acceptance_report_jm.json` + 3-way `acceptance_compare.json`; CLI passthrough |
| **J5** | Report: 3-way band toggle + JM probability figure + byte-identical HTML test |
| **J6** | End-to-end run on real data with `jm_enabled=True` → HTML showing JM regime bands + JM probability figure; verify viz; (λ calibration + cadence bench) |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| JM non-determinism (k-means++/restarts) breaks byte-identical invariant | Seed every RNG via PCG64/SeedSequence; determinism regression + golden fixture; assert in CI (AJ-2) |
| λ~50 prior is on raw features; standardized loss scale differs | Re-scan λ on causally-standardized slope in S-JM3; don't adopt the literature value blindly |
| Empty-cluster / thin-LatAm (N=10) degeneracy emits NaN | Farthest-point reseed + last_good fallback (mandatory); degenerate-window unit test |
| Label-flip across refits (constant-λ permutation symmetry) | `perm=argsort(means)` every refit (FJ2.2); refit-boundary stability test |
| Online vs smoothed confusion leaks lookahead | Production = forward-only last-state; backward reconstruction quarantined to offline; no-lookahead property test is the gate |
| BLAS reduction-order nondeterminism (CJM/D>1) | Dormant at D=1; pin BLAS threads before multivariate; byte-identical CSV assert in CI now |
| JM wins G5 but still misses in-range G3 | Acceptable — still a diagnostic; the sparse multivariate fast-follow (D1/S-JM6) is the next lever |

---

## 13. Out of scope / future

- Sparse **multivariate** JM with per-refit FTIC selection over the feature panel (slope, slope-spread, PC1 share, avg pairwise corr, tripwire) — S-JM6; where `log(P)>0` activates FTIC and feature-selection pays off.
- Predictive layer (RoRo v2) — JM transition structure is a clean substrate, but forecasting stays deferred.
- Wiring JM labels through external/internal validators so G1/G2/G6 discriminate method (currently method-shared, a known v1.1 roadmap item).

---

## 14. References

- Bemporad, Breschi, Piga, Boyd (2018), "Fitting jump models", *Automatica* 96:11-21 — objective, k-means loss, Algorithm 1, DP/Viterbi, HMM-nesting (Prop. 1).
- Nystrup, Lindström, Madsen (2020), *ESWA* 150:113307 — financial JM via temporal clustering + jump penalty.
- Nystrup, Kolm, Lindström (2021), *ESWA* 184:115558 (SSRN 3805831) — sparse JM + feature selection (the FTIC precursor).
- Aydinhan, Kolm, Mulvey, Shu (2024), *Annals of OR*, SSRN 4556048 — Continuous SJM (simplex, L1-squared transition penalty).
- Shu, Mulvey (2024), "Downside Risk Reduction…", *J. Asset Management*, arXiv:2402.05272 **v3** — canonical Eq.1 objective, 10 restarts, 3000-day/biannual, 44% vs 141% turnover. **v1** preprint = 2000-day/~8.4×.
- Cortese, Kolm, Lindström (2024), SSRN 4774429 / AStA 2026 — FTIC `a_n=log(log N)·log(P)`; Fan-Tang (2013) JRSS-B 75(3) origin.
- `jumpmodels` (Yizhan-Oliver-Shu, PyPI v0.1.1, Apache-2.0) — DP recurrence reference; D4 dependency rationale.
- NumPy PCG64/SeedSequence reproducibility guarantee.
- Repo contracts mirrored: `regime_hmm.py` (walk_forward 114-176, mean-sort 85, last_good), `types.py:69`, `config.py:35`, `io.py:259/285`, `report/figures.py:684/862`, `report/html.py:38`, `report/bundle.py:29`+`load.py:150`. Full map: `docs/superpowers/jm-integration-contract.md`.
