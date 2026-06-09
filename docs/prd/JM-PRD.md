# PRD — RoRo Statistical Jump Model Regime Layer (v1.2 feature)

**Author:** drafted for Alan Vazquez, CFA
**Target repo:** [`alanvaa06/risk_regime_switching_model`](https://github.com/alanvaa06/risk_regime_switching_model) (the RoRo engine)
**Status:** Design document — proposes a third regime classifier (Statistical Jump Model) alongside the percentile classifier (production default) and the HMM overlay (`roro/regime_hmm.py`).
**Date:** 2026-06-08
**Drop-in location:** `docs/prd/PRD-jump-model.md`

---

## 0. Context — why this PRD exists

RoRo already ships two regime classifiers on the cap-weighted cross-sectional slope (`β̂`) per segment:

1. **Percentile classifier** (`roro/classify.py`) — production default. Ranks today's slope against its trailing 5Y distribution → tercile band. **No notion of persistence** → the raw daily tercile flickers.
2. **HMM / Markov-switching overlay** (`roro/regime_hmm.py`) — optional, off by default (`hmm_enabled`). 3-state `statsmodels.MarkovRegression` (switching mean + variance), causal (expanding window, monthly refit, Hamilton **filtered** probabilities). On the 2008–2026 backtest it **halved calm-quarter flicker (G5: 9 vs 18 transitions)** but **regressed event recognition (G3: 6/8 vs 7/8)** and cleared no gate outright → shipped as overlay, not replacement.

The existing PRD §15 already names the next step:

> *"Statistical Jump Models (Cortese et al. 2026, FTIC criterion) as alternative regime engine — sparser than HMM, scales to high-dim feature spaces."*

This PRD specifies that engine. The thesis: a **statistical jump model (JM)** should beat the HMM at *exactly the HMM's weakness* — it delivers persistence **without** sacrificing sharp-event detection, because the jump penalty regularizes transitions directly rather than relying on a learned transition matrix that smears fast regime changes. This is the Shu-Yu-Mulvey (2024) finding: JM > HMM on **both** persistence (~5× lower turnover) **and** return capture (catches dot-com, GFC, COVID).

**Wiki anchors** (already compiled into the vault):
- `wiki/Finance/Regime-Aware Asset Allocation (Shu Yu Mulvey 2024).md` — the JM-vs-HMM asset-allocation result and the 4-step framework.
- `wiki/Finance/jumpmodels (Python Library).md` — reference implementation (`JumpModel`, `SparseJumpModel`, `.predict_online()`).
- `wiki/Finance/Generalized Information Criteria for Jump Models (Cortese 2026).md` — FTIC/AIC/BIC for selecting `λ` (and `κ`, `K`).
- `wiki/Finance/Risk-On Risk-Off Regime Classifier (Alan 2026).md` — the design doc this whole repo implements; Phase 7 = "HMM / Markov-switching layer… SJM alternative."

> **Disambiguation (carry into code comments):** the JM "jump" is a **regime-transition penalty** in an unsupervised clustering objective. It is *unrelated* to the Poisson price-jump SDEs of `wiki/Finance/Jump-Diffusion Option Pricing (Merton).md`. Same word, different machinery. Module must not be named anything that implies jump-diffusion pricing.

---

## 1. TL;DR

Add `roro/regime_jm.py` — a **3-state statistical jump model** regime classifier that mirrors the `roro/regime_hmm.py` contract exactly (same per-segment input, same `*_RegimeFrame` output shape, same causal/no-lookahead guarantee, same acceptance-gate scoring). It fits a JM on each segment's cap-weighted slope series, infers regimes causally via a **periodic refit + daily online inference** loop (the JM analog of the Hamilton filter), and orders the three states by fitted mean → **Risk-off / Transitional / Risk-on**.

Off by default (`jm_enabled: false`). When enabled, the engine emits `regimes_jm.csv` + `jm_refit_log.csv`, the report gains a **3-way band toggle (Percentile ↔ HMM ↔ JM)** and a JM state-probability figure (when using the continuous JM), and `roro backtest` writes `acceptance_report_jm.json` + a 3-way `acceptance_compare.json`. The percentile classifier remains the production default until the JM clears the gates the HMM could not.

**Definition of done:** JM passes ≥ G5 outright (persistence) **and** matches or beats the percentile baseline on G3 (event recognition) — the bar the HMM failed. If it does, JM becomes a candidate production default via a documented method-selection decision.

---

## 2. Goals / Non-Goals

### Goals
- **G-1** New module `roro/regime_jm.py` with a `classify_jm(beta, *, cfg, thin_cuts) -> JmRegimeFrame` entry point, signature-parallel to `classify_hmm`.
- **G-2** Causal-by-construction inference: parameters re-estimated on a point-in-time window (rolling 2000-day per Shu-Mulvey, or expanding to match HMM — see §6.3), states inferred by **online inference** (`.predict_online()`), never using data beyond `t`.
- **G-3** **Determinism** — byte-identical CSV output across repeat runs given fixed inputs (RoRo's #1 invariant). The JM's k-means++ init and coordinate-descent restarts **must be seeded**; output must survive the existing reproducibility regression test and a new golden fixture.
- **G-4** 3-state labels ordered by fitted state mean → `("Risk-off", "Transitional", "Risk-on")`, with per-refit re-canonicalization to defeat label-switching (same defense the HMM uses for EM).
- **G-5** Scored through the **existing G1–G6 acceptance gates** (`roro/backtest.py`), with a 3-way percentile-vs-HMM-vs-JM comparison artifact.
- **G-6** Optional, off by default; when off, engine + report + backtest behave bit-identically to today.
- **G-7** Full repo-standard quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest` with `filterwarnings=["error"]`, one test module per new code module, property tests for no-lookahead + scale-invariance.

### Non-Goals
- **No predictive layer.** JM stays diagnostic (consistent with RoRo v1.x). The JM transition structure is a *natural substrate* for the deferred v2 predictive layer, but that is out of scope here.
- **No change to the percentile production default** in this PRD — JM ships as overlay; promotion is a separate, gated decision.
- **No portfolio/Sharpe objective.** Shu-Mulvey select `λ` by maximizing strategy Sharpe; RoRo is diagnostic with no portfolio, so `λ` is selected against RoRo's own gate objective (§6.4), not Sharpe.
- **No multivariate JM in the first cut** (recommended) — ship univariate parity with the HMM first; sparse multivariate JM is a fast-follow (§9, Decision D1).
- **No FX, intraday, or new universe changes.**

---

## 3. Background — what a Statistical Jump Model is (precise spec)

Given standardized features `y_t ∈ R^D` over `t = 0…T-1`, fit `K` states by solving:

```
argmin_{Θ,S}  Σ_t  l(y_t, θ_{s_t})  +  λ · Σ_t  1{s_{t-1} ≠ s_t}
```

- `l(y, θ) = ½‖y − θ‖₂²` (scaled squared ℓ₂ loss)
- `Θ = {θ_k}` — `K` state centroids; `S = {s_t}` — the state path
- **`λ` (jump penalty)** — fixed cost charged on every state transition. The single knob controlling persistence:
  - `λ = 0` → reduces to **k-means** (no temporal structure)
  - `λ → ∞` → collapses to **one state**
- Solved by **coordinate descent** (alternate: fit centroids given path → fit path given centroids via dynamic programming), with **k-means++ restarts** to handle non-convexity → **the restart RNG must be seeded** for determinism.
- **Online inference** (`.predict_online()`): fix `Θ` at last-fitted values, run only the path-optimization stage over `beta[:t]`, take the last state as `ŝ_t`. This is the causal primitive — the direct analog of the HMM's Hamilton filter, and the reason JMs are deployable live (per the jumpmodels wiki note).

**Variants** (in the `jumpmodels` library):
- **Discrete JM** — hard state labels.
- **Continuous JM (CJM)** — state becomes a probability vector → soft per-day probabilities, the analog of the HMM's filtered marginal probabilities (feeds the stacked-area report figure).
- **Sparse JM (SJM)** — adds an ℓ₁ feature-selection penalty `κ`; only relevant for the multivariate fast-follow (§9 D1).

---

## 4. Users / consumers

Identical to the existing PRD §3 (PM/Allocator, Research, Risk, IC). The JM adds **one new diagnostic affordance**: a persistence-regularized regime band that, if it clears the gates, gives the PM a less-flickering daily label than the percentile classifier **without** the HMM's lag on sharp events.

---

## 5. Functional Requirements

### FJ1 — Module `roro/regime_jm.py`
- **FJ1.1** Public entry: `classify_jm(bbs: BetaBySegment, *, cfg: EngineConfig, thin_cuts: frozenset[str]) -> JmRegimeFrame`. Signature mirrors `classify_hmm` (`roro/regime_hmm.py:179`).
- **FJ1.2** Per segment, operate on `bf.cap_wtd["beta"]` (same series the HMM consumes), standardized (z-scored) on a point-in-time basis before fitting (JM loss is scale-sensitive; standardization must itself be causal — fit mean/std on the in-window data only).
- **FJ1.3** A `walk_forward(beta, *, jump_penalty, refit_interval_days, min_history_days, n_states, continuous) -> dict` returning index-aligned `state`, `label`, soft probabilities (`prob_risk_off/_transitional/_risk_on`; degenerate to one-hot for the discrete JM), `confidence`, `cold_start`, `refit_dates` — the **same dict keys `walk_forward` returns in `regime_hmm.py`** so downstream wiring is symmetric.
- **FJ1.4** Drop NaN beta rows before fitting; emit them as `Unknown` (mirror HMM).
- **FJ1.5** Degenerate / non-converged fit → fall back to previous good parameters (mirror HMM `last_good`).

### FJ2 — State ordering & label-switching defense
- **FJ2.1** Order the `K=3` fitted centroids ascending by mean → `perm`; map to `("Risk-off","Transitional","Risk-on")`.
- **FJ2.2** Re-canonicalize ordering **at every refit** (centroid means are unconstrained across refits → must re-sort, exactly as the HMM re-canonicalizes EM state order).

### FJ3 — Causality (no lookahead)
- **FJ3.1** Refit clock: every `jm_refit_interval_days` trading days starting at `jm_min_history_days`; parameters estimated on `beta[:r]` only.
- **FJ3.2** Inference clock: daily online inference with frozen parameters over `beta[:t]`. Within a refit block `[r, r')`, states come from online inference seeded by `params(beta[:r])`.
- **FJ3.3** **No-lookahead property test** (mirror the HMM's): a label at date `t` is unchanged when future data is appended.

### FJ4 — Determinism
- **FJ4.1** Seed every stochastic step (k-means++ init, restart selection) from a fixed config seed `jm_random_seed` (default `0`). Identical inputs → identical labels → byte-identical `regimes_jm.csv`.
- **FJ4.2** Add a determinism regression test (two runs → identical CSV) and a `tests/golden/` JM fixture, matching the existing golden-baseline discipline.

### FJ5 — Configuration (extend `EngineConfig`, `roro/config.py`)
New frozen fields (defaults chosen for HMM parity + determinism):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `jm_enabled` | `bool` | `False` | Master switch (mirrors `hmm_enabled`) |
| `jm_jump_penalty` | `float` | `50.0` | `λ`; calibrated in S-JM3 (§7). See Decision D2 |
| `jm_n_states` | `int` | `3` | Regime count (parity with HMM/percentile terciles) |
| `jm_refit_interval_days` | `int` | `21` | Monthly refit (parity with `hmm_refit_interval_days`) |
| `jm_min_history_days` | `int` | `252` | Warmup (parity with `hmm_min_history_days`) |
| `jm_window` | `str` | `"expanding"` | `"expanding"` (HMM parity) or `"rolling"` (Shu-Mulvey 2000-day) — Decision D3 |
| `jm_rolling_window_days` | `int` | `2000` | Lookback when `jm_window="rolling"` |
| `jm_continuous` | `bool` | `True` | CJM (soft probs for the report) vs discrete JM |
| `jm_random_seed` | `int` | `0` | Determinism seed (FJ4.1) |

- **FJ5.1** All new keys flow through `to_dict()` into `snapshot.json` (the config-capture invariant). No special-casing needed beyond plain scalars.
- **FJ5.2** Unknown-key rejection in `load_config` already covers typos — no change needed.

### FJ6 — Types (extend `roro/types.py`)
- **FJ6.1** Add `JmRegimeFrame` — structurally identical to `HmmRegimeFrame` (`roro/types.py:69`): `state, label, prob_risk_off, prob_transitional, prob_risk_on, confidence, n_per_segment, thin_cut_flag, cold_start_flag, refit_dates`. Optionally add `jump_penalty_used: dict[str,float]` per segment if `λ` is auto-selected (Decision D2).
- **FJ6.2** Extend `RunResult` with `regime_jm: JmRegimeFrame | None = None` (mirrors `regime_hmm`).
- **FJ6.3** Extend `AlertSet` with `jm_bucket_transitions: pd.DataFrame` (mirrors `hmm_bucket_transitions`).

### FJ7 — Engine wiring (`roro/engine.py`)
- **FJ7.1** After the HMM block (`engine.py:95`), add a symmetric block:
  ```python
  regime_jm = (
      classify_jm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
      if cfg.jm_enabled
      else None
  )
  ```
- **FJ7.2** Pass `regime_jm` into `detect_alerts(...)` and into `RunResult`.
- **FJ7.3** `roro/io.py write_run` emits `regimes_jm.csv` + `jm_refit_log.csv` when `regime_jm is not None`; add a `regime_jm` block to `snapshot.json` (mirror the HMM block). Atomic-write path unchanged.

### FJ8 — Alerts (`roro/alerts.py`)
- **FJ8.1** Emit JM bucket-transition rows into `alerts.csv` (same shape as HMM transitions), gated on `regime_jm is not None`.

### FJ9 — Report (`roro/report/`)
- **FJ9.1** Extend the existing "Regime bands: Percentile ↔ HMM" toggle on the segment-β chart to a **3-way Percentile ↔ HMM ↔ JM** band-source selector. JM bands derived from `regimes_jm.csv` labels.
- **FJ9.2** When `jm_continuous=True`, add a **JM state-probability stacked-area figure** (Risk-off/Transitional/Risk-on summing to 1.0), directly parallel to the HMM probability figure — reuse the existing figure spec, swap the data source.
- **FJ9.3** Figures present only when the run carries JM output; reproducibility (deterministic `div_id`s, no timestamps) preserved.

### FJ10 — Backtest (`roro/backtest.py`)
- **FJ10.1** When `jm_enabled`, score JM labels through the **same `_evaluate_gates` G1–G6** path used for percentile/HMM; write `acceptance_report_jm.json`.
- **FJ10.2** Extend `acceptance_compare.json` to a 3-way structure: `{gate: {"percentile": ..., "hmm": ..., "jm": ...}}`.
- **FJ10.3** `--assert-gates` semantics unchanged (still gates the **production** method); JM rows are comparative until promotion.

---

## 6. Methodology decisions (the substantive design)

### 6.1 Input feature — univariate first
Ship the JM on the **same 1-D cap-weighted slope series** the HMM uses. This gives a clean apples-to-apples A/B/C through the gates and isolates "does the jump penalty beat the Markov transition matrix" from "does a richer feature set help." Multivariate/sparse JM is the fast-follow (Decision D1).

### 6.2 State count `K = 3`
Matches terciles and the HMM's 3 states → the three classifiers share a label vocabulary and the report toggle is meaningful.

### 6.3 Window & refit cadence
Two defensible choices:
- **Expanding + monthly refit** — exact HMM parity; cleanest comparison; reuses the HMM's cadence-invariance evidence (monthly ≈ daily, 0.9966 agreement).
- **Rolling 2000-day + 6-month refit** — the literal Shu-Mulvey recipe (one bull/bear cycle lookback, biannual refit).

**Recommendation:** default to **expanding + monthly** for parity, expose `jm_window`/`jm_rolling_window_days` so the Shu-Mulvey recipe is one config flip away and testable in S-JM3. (Decision D3.)

### 6.4 Jump-penalty `λ` selection — RoRo's gate objective, not Sharpe
Shu-Mulvey pick `λ` by maximizing validation-period **Sharpe**; RoRo has no portfolio. Three options:
- **(a) Fixed `λ` from config** — simplest; calibrate once in S-JM3 by scanning `λ` and picking the value that maximizes the RoRo gate objective on the validation split. **Recommended for v1.**
- **(b) Information-criterion auto-select (Cortese-Kolm-Lindström GIC: FTIC/AIC/BIC)** — pick `λ` (and later `κ`,`K`) per refit by the GIC in `wiki/Finance/Generalized Information Criteria for Jump Models (Cortese 2026).md` (Cortese, Kolm & Lindström, SSRN 4774429 / AStA 2026; FTIC = Fan-Tang 2013 criterion, `a_n = log(log N)·log(P)`). FTIC is the documented best performer. **Structurally dormant at P=1** (`log P = 0` zeroes the complexity term) → it cannot discriminate `λ` in the univariate v1; its value is feature-selection in the multivariate fast-follow. Adds a per-refit model-selection loop. **Recommended as v1.1 (multivariate) enhancement, not v1.**
- **(c) Gate-objective CV** — the RoRo-native analog of Shu-Mulvey: pick `λ` maximizing **G3 event recognition subject to G5 ≤ baseline** on the validation period. Most aligned with what RoRo actually optimizes, but couples selection to the gate harness.

**Recommendation:** ship **(a)** with the calibrated default, design `regime_jm.py` so the `jump_penalty` argument can later be produced by **(b)** or **(c)** without changing the `walk_forward` contract. (Decision D2.)

### 6.5 Why JM is expected to beat the HMM here
The HMM regressed **G3 (event recognition)** because a learned transition matrix penalizes *all* transitions probabilistically and smears fast regime changes; it won **G5 (stability)**. The JM's fixed jump penalty regularizes persistence **without** a generative transition model, so (per Shu-Mulvey) it tends to hold regimes through noise **yet** snap cleanly on genuine breaks — the exact G5-without-losing-G3 profile RoRo needs. This is the falsifiable hypothesis of this PRD.

---

## 7. Milestones

| Sprint | Deliverable | Notes |
|---|---|---|
| **S-JM0** | Dependency spike: integrate `jumpmodels` (or vendor — Decision D4); confirm determinism under a fixed seed; resolve version pins vs RoRo's `numpy<2.1` / `pandas<2.3`; add `mypy` override if no stubs | 0.5 wk |
| **S-JM1** | `roro/regime_jm.py` + `JmRegimeFrame` + config fields; univariate discrete JM `walk_forward`; unit + no-lookahead property tests | 1 wk |
| **S-JM2** | Causal CJM soft probabilities; engine wiring (`jm_enabled`), `io` writers, alerts; golden fixture + determinism regression test | 1 wk |
| **S-JM3** | **`λ` calibration** (scan vs gate objective on validation split) + run JM through G1–G6; produce 3-way `acceptance_compare.json`; cadence-invariance bench | 1.5 wks |
| **S-JM4** | Report: 3-way band toggle + JM probability figure; byte-identical HTML test | 1 wk |
| **S-JM5** | Decision memo: does JM clear G5 **and** match/beat percentile on G3? Promote-to-default recommendation or keep-as-overlay | 0.5 wk |
| **(later) S-JM6** | Sparse multivariate JM on the regime feature panel (slope, slope-spread, PC1 share, avg pairwise corr, tripwire) with FTIC selection — Decision D1 | TBD |

---

## 8. Acceptance criteria

- **AJ-1** With `jm_enabled: false`, engine/report/backtest output is **byte-identical** to pre-change (regression + golden tests pass).
- **AJ-2** With `jm_enabled: true`, two runs over identical inputs produce **byte-identical** `regimes_jm.csv` (determinism gate, FJ4).
- **AJ-3** No-lookahead property test passes (label at `t` invariant to appended future data).
- **AJ-4** Cadence-invariance: monthly-refit JM labels agree with daily-refit ≥ 0.99 (mirror the HMM's 0.9966 result) → monthly default justified.
- **AJ-5** Full quality bars green: `mypy --strict`, `ruff`, `pytest` (with `filterwarnings=["error"]` — no unseeded-RNG or convergence warnings leaking through).
- **AJ-6** JM scored through G1–G6 with a 3-way comparison artifact written.
- **AJ-7 (promotion bar, not a ship bar):** JM clears **G5 outright** *and* **G3 ≥ percentile baseline (7/8 events)**. If met → trigger the §S-JM5 promotion decision. If not → ships as overlay, documented like the HMM.

---

## 9. Open Decisions

| ID | Decision | Options | Recommended |
|---|---|---|---|
| **D1** | Feature design | Univariate (cap-wtd slope) **/** multivariate sparse JM on the regime feature panel | **Univariate first** (HMM parity, clean comparison); sparse multivariate as S-JM6 fast-follow — this is JM's real edge over the HMM and the payoff of the SJM/FTIC literature |
| **D2** | `λ` selection | Fixed config **/** FTIC auto-select (Cortese) **/** gate-objective CV | **Fixed (calibrated) for v1**, FTIC for v1.1; keep `walk_forward` agnostic so either drops in |
| **D3** | Window/cadence | Expanding+monthly (HMM parity) **/** rolling-2000d+biannual (Shu-Mulvey) | **Expanding+monthly default**, rolling exposed via config for S-JM3 test |
| **D4** | Dependency | Use `jumpmodels` PyPI lib **/** vendor a ~150-line deterministic JM | **Spike `jumpmodels` first**; vendor only if it breaks determinism, the version pins, or `mypy --strict`. The core algorithm (k-means++ init + DP-Viterbi jump-penalty coordinate descent) is small and self-containable — a vendored fallback de-risks RoRo's minimal-dep + determinism culture |
| **D5** | Promotion | If AJ-7 met, replace percentile as default **/** keep both, JM overlay | Decide in S-JM5 from the backtest; default to **overlay** until a full-window, out-of-sample case is made (same conservatism the HMM got) |

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **JM non-determinism** (k-means++/restarts) breaks RoRo's byte-identical invariant | Fails the core reproducibility gate | Seed every RNG (`jm_random_seed`); determinism regression test + golden fixture; assert in CI |
| `jumpmodels` version conflicts with `numpy<2.1`/`pandas<2.3` or lacks type stubs | Dependency hell / `mypy` failure | S-JM0 spike; `mypy` override for the module; **vendored fallback** (D4) |
| `filterwarnings=["error"]` trips on JM convergence/empty-cluster warnings | Suite fails silently-passing numerics | Wrap fit in scoped `warnings.catch_warnings()` (as the HMM does); explicit convergence check + fallback to last-good params |
| Standardization leaks future info | Lookahead violation | Causal z-score (in-window mean/std only); covered by the no-lookahead property test |
| JM wins G5 but, like the HMM, still misses G3 | No promotion, effort yields only a second overlay | Acceptable outcome — JM still adds a diagnostic; **but** this is the explicit hypothesis under test (§6.5), and the sparse multivariate follow-on (D1) is the next lever if univariate underperforms |
| Thin LatAm cut (N=10) destabilizes JM fits | Noisy LatAm regime | Inherit the `thin_cut` flag; fall back to last-good params on degenerate fits |
| Label-switching across refits | Garbled bands | Re-canonicalize by mean every refit (FJ2.2) |

---

## 11. Out of scope / future

- Predictive layer (RoRo v2) — the JM regime sequence and its empirical transition frequencies are a clean substrate for Beber-style transition-persistence forecasting, but forecasting stays deferred.
- Sparse multivariate JM with FTIC model selection over the full feature panel (D1 / S-JM6) — the highest-upside extension, where JM's feature-selection edge over the HMM is realized.
- Wiring JM labels through the external/internal validators so G1/G2/G6 discriminate the method (currently method-shared in the HMM path too — a known RoRo v1.1 roadmap item).

---

## 12. References

- Shu, Mulvey (2024) — JM vs HMM asset allocation; 4-step framework; `λ`-by-Sharpe selection; persistence + return-capture. **Cite versions distinctly:** "Downside Risk Reduction Using Regime-Switching Signals" (*J. Asset Management*, arXiv:2402.05272 **v3**) = 3000-day window, 44% vs 141% turnover; the **v1** preprint "Regime-Aware Asset Allocation" = 2000-day window, ~8.4× turnover (14 vs 115 shifts). Do not mix the two.
- `jumpmodels` (Yizhan-Oliver-Shu, PyPI v0.1.1, Apache-2.0) — `JumpModel` (discrete + CJM via `cont`), `SparseJumpModel`, `.predict_online()` causal primitive, scikit-learn-style API. **Used as a dev-only test oracle, not a runtime dep (D4 = vendor).**
- Cortese, **Kolm & Lindström** (2024), "Generalized Information Criteria for High-Dimensional Sparse Statistical Jump Models" (SSRN 4774429; AStA Adv. Stat. Anal. 2026) — FTIC/AIC/BIC for `λ`/`κ`/`K` selection (Decision D2). FTIC = Fan-Tang (2013, JRSS-B 75(3)) criterion `a_n=log(log N)·log(P)`. **Not** Nystrup (who authored the feature-selection precursor, SSRN 3805831) and **not** "Cortese 2026" alone.
- `wiki/Finance/Risk-On Risk-Off Regime Classifier (Alan 2026).md` — RoRo design doc; Phase 7 SJM-as-HMM-alternative.
- Repo contracts mirrored: `roro/regime_hmm.py` (walk-forward + filter pattern), `roro/types.py:69` (`HmmRegimeFrame`), `roro/config.py` (`hmm_*` fields), `roro/engine.py:95` (optional-classifier wiring), `roro/backtest.py` (G1–G6 + `acceptance_compare.json`).
- Bemporad et al. (2018) *Automatica* 96:11-21 — original jump models (objective, Algorithm 1, DP/Viterbi, HMM-nesting). Nystrup, Lindström, Madsen (2020, *ESWA* 150) + Nystrup, Kolm, Lindström (2021, *ESWA* 184) — JM in finance, feature selection, online inference. Aydinhan, Kolm, Mulvey, Shu (2024, SSRN 4556048) — Continuous SJM.
