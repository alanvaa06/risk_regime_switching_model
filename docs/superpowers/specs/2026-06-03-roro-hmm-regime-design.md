# RoRo — HMM / Markov-Switching Regime Classification

**Date:** 2026-06-03
**Status:** Design approved through brainstorming. Ready for implementation planning.
**Author:** Alan Vazquez, CFA
**Scope:** Add a 3-state Markov-switching regime classifier on segment β, running in parallel with the existing rolling-percentile classifier. PRD v1.1+ / Phase 7 item ("HMM / Markov-switching on validated β"). S8 dashboard explicitly deferred.

---

## 1. Goal

Upgrade regime classification from threshold-based rolling percentiles to a statistical state model. The percentile classifier (`roro/classify.py`) labels each segment's β by its trailing-window percentile rank (tercile / quintile + slope direction). It flickers day-to-day — viz v4 had to hand-roll hysteresis smoothing for display because "raw daily tercile is too noisy to shade," while the engine's own regime output stayed noisy.

A Hidden Markov Model with a learned transition matrix produces persistent regimes intrinsically (no post-hoc smoothing), plus probabilistic state membership and transition structure. This spec adds that method **alongside** the percentile classifier, compares both through the existing S9 acceptance gates, and adopts HMM as the production default only if the evidence supports it.

---

## 2. Locked decisions

| Fork | Choice | Rationale |
|---|---|---|
| Method | HMM / Markov-switching | Persistent regimes + transition probabilities; the Phase 7 upgrade |
| States | 3 (Risk-off / Transitional / Risk-on) | Drop-in for existing tercile vocabulary; downstream unchanged |
| Causality | Filtered + point-in-time refit | Honest backtest; matches production behaviour, no lookahead |
| Observable | β level, mean + variance switching | Captures risk-off vol clustering; classic Markov-switching setup |
| Coexistence | Parallel to percentile; compare via S9; then pick | Keeps the passing baseline as fallback; evidence-driven cutover |
| Engine | `statsmodels` MarkovRegression | `filtered_marginal_probabilities` is the causal Hamilton filter; `switching_variance` direct |
| Refit cadence | Monthly (`~21` trading days), config-driven, cadence-invariance tested | Daily-responsive output at ~6% the fit cost; equivalence proven, not assumed |

---

## 3. Architecture

New module `roro/regime_hmm.py`, sibling to `classify.py`. One responsibility: β → causal 3-state HMM labels. The point-in-time discipline lives **inside** this module (causal-by-construction within a single engine run), exactly as the percentile classifier's trailing rolling window is causal-by-construction. Consequence: one engine run yields point-in-time-correct HMM labels, and the backtest harness keeps its current single-run-then-score shape ([backtest.py:82](../../../roro/backtest.py)).

Three layers within the module:
1. **`fit_core`** — fit one HMM on a β window, return ordered filtered probabilities (§5).
2. **Walk-forward engine** — expanding-window monthly refit + daily filter, producing causal label series (§6).
3. **`classify_hmm`** — per-segment orchestration mirroring `classify()`'s loop, returning an `HmmRegimeFrame`.

### 3.1 Data contract

New frozen dataclass in `roro/types.py`, parallel to `RegimeFrame` (not a mutation of it):

```python
@dataclass(frozen=True)
class HmmRegimeFrame:
    state: pd.DataFrame            # ordered state id 0/1/2 per segment per day
    label: pd.DataFrame           # "Risk-off"/"Transitional"/"Risk-on" — same vocab as tercile
    prob_risk_off: pd.DataFrame   # filtered P(state) per segment per day
    prob_transitional: pd.DataFrame
    prob_risk_on: pd.DataFrame
    confidence: pd.DataFrame      # max filtered prob (uncertainty flag when low)
    n_per_segment: pd.DataFrame   # reused from BetaFrame
    thin_cut_flag: pd.DataFrame
    cold_start_flag: pd.DataFrame # pre-min-history suppression (mirrors bootstrap_flag)
    refit_dates: dict[str, list[pd.Timestamp]]  # provenance: when params were re-estimated
```

`label` deliberately reuses the exact `{Risk-off, Transitional, Risk-on}` vocabulary so the gate scorer, alert/transition detector, and viz consume it without special-casing.

`RunResult` gains `regime_hmm: HmmRegimeFrame | None = None` — `None` when `hmm_enabled=False`, so existing runs and tests are byte-unchanged.

Rationale for a separate frame rather than extra `RegimeFrame` columns: percentile labels carry no probabilities; HMM carries no percentile/quintile. One frame each, one shared label vocabulary.

---

## 4. Configuration

New fields on `EngineConfig` ([config.py](../../../roro/config.py)), all defaulted so existing YAML and runs are untouched:

```python
hmm_enabled: bool = False           # opt-in; coexist means off by default until proven
hmm_refit_interval_days: int = 21   # ~monthly (trading days)
hmm_min_history_days: int = 252     # cold-start gate; reuses bootstrap_min_days default
hmm_switching_variance: bool = True
hmm_window: str = "expanding"       # vs "rolling" — future knob; expanding for v1.1
```

`to_dict` already serializes plain scalars, so snapshot provenance is free (no enum added).

---

## 5. HMM core (single fit)

Pure function: fit one 3-state HMM on a β window, return ordered filtered probabilities.

**Model:**
```python
MarkovRegression(endog=beta, k_regimes=3, trend="c", switching_variance=hmm_switching_variance)
```
`trend="c"` → regime-dependent mean; `switching_variance=True` → regime-dependent variance. No exog, no AR.

**State ordering (critical — solves label-switching).** EM returns regimes in arbitrary order, and the order changes between refits. After each fit, sort regimes by fitted **mean β** ascending and remap:

| sorted rank | mean β | label |
|---|---|---|
| 0 | lowest | Risk-off |
| 1 | mid | Transitional |
| 2 | highest | Risk-on |

Economic prior: low β (weak vol-slope sensitivity) = de-risking = Risk-off. Every probability column is reindexed to this canonical order before returning. Without this, labels scramble across refit boundaries — the single biggest correctness trap.

**Output:** filtered probabilities only — `res.filtered_marginal_probabilities` (Hamilton filter, causal: P(state_t | β_{0:t})). Never `smoothed_*` (that peeks). `label = argmax`; `confidence = max prob`.

**Argmax always wins** — low `confidence` surfaces a flag but does not override the label to "Transitional". Downstream decides what to do with weak separation.

**Convergence / warnings handling** (`filterwarnings=["error"]` in tests, [pyproject.toml:73](../../../pyproject.toml)):
- Wrap `.fit()` in `warnings.catch_warnings()` that captures rather than raises.
- On non-convergence or degenerate fit (collapsed variance, NaN params): record `fit_failed`; that window's labels become `Unknown` + flagged, and the previous good params carry forward to the daily filter until the next refit succeeds.
- **Determinism:** pin EM start (fixed `start_params`, `search_reps=0`) so re-runs are bit-reproducible — required by the engine fingerprint.

---

## 6. Walk-forward engine + refit cadence

Turns `fit_core` into a causal-by-construction label series.

```
params = None
for t in trading_days:
    if t < min_history:                      # cold start -> Unknown
        emit Unknown; continue
    if refit_due(t):                         # refit clock (monthly)
        params = fit_core(beta[:t])          # EM on data <= t only
    filtered_t = hamilton_filter(beta[:t], params)   # filter clock (daily)
    emit label/probs from filtered_t[-1]
```

Two decoupled clocks:
- **Refit clock** (`refit_due`, monthly): re-estimates params on the expanding window `β[:t]`. Never sees the future.
- **Filter clock** (daily): produces today's regime from today's β against current params.

Regimes flip any day; only the param *definitions* refresh monthly. Both use `β[:t]` exclusively → no lookahead by construction.

**Cost:** one engine run rebuilds the whole causal history. Monthly = ~14yr × 12 × 7 cuts ≈ 1,200 fits (minutes). Daily ≈ 24,500 fits (hours). Same daily-responsive output.

### 6.1 Cadence-invariance test

Dedicated bench (not production wiring), proves monthly ≈ daily:
- Run the walk-forward at `{daily, weekly, monthly, quarterly}` on a fixed β fixture (one liquid cut, e.g. DM_Eq, real history).
- Metrics: (a) **label agreement** — % days monthly label == daily label; (b) **event lag** — Risk-off flip date per cadence vs daily across the 8 known events; (c) **param drift** — L2 distance of fitted means/vars vs daily baseline.
- **Acceptance:** label agreement ≥ ~98%, event-flip lag ≤ 1 day. If it holds, monthly is proven equivalent at ~6% cost. If it fails, the test surfaces the real cadence needed and the default is raised.
- Marked `@pytest.mark.slow`; full sweep runs on demand / pre-merge. Result logged to `docs/context/results.md`.

---

## 7. Integration

**Engine wiring** ([engine.py:84](../../../roro/engine.py)) — one guarded block after percentile classify:
```python
regime_hmm = (
    classify_hmm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    if cfg.hmm_enabled else None
)
```
Threaded into `RunResult(regime_hmm=regime_hmm)`. Everything downstream of percentile is unchanged.

**Alerts** ([alerts.py:25](../../../roro/alerts.py)). `_bucket_transitions(tercile)` is already label-frame-in → call it on `regime_hmm.label`. Extend `detect_alerts` to optionally take `regime_hmm` and emit `hmm_bucket_transitions` as a new `AlertSet` field (defaulted empty). Percentile alerts stay byte-identical when HMM is off.

**Output artifacts** (`io.py` `write_run`) — only when `regime_hmm is not None`:
- `regimes_hmm.csv` — long form: date, segment, state, label, p_risk_off, p_transitional, p_risk_on, confidence, cold_start, thin_cut.
- `hmm_refit_log.csv` — date, segment, fitted means/vars per state, converged flag → param transparency + determinism audit trail.
- `snapshot.json` — `regime_hmm` block per segment (label, 3 probs, confidence), mirroring the existing `global`/`segments` shape.

**Frozen:** `RegimeFrame`, the percentile path, all existing CSVs, all current tests. HMM is purely additive and off-by-default.

---

## 8. Backtest comparison

Run both methods through the same PRD §10 gates; choose the production default from evidence.

The gate functions read a label frame (`result.regime.tercile`) and a transitions frame (`result.alerts.bucket_transitions`); both have HMM equivalents of identical shape/vocabulary. Parametrize the scorer over a label source:

```python
def _evaluate_gates(result, *, labels: pd.DataFrame, transitions: pd.DataFrame) -> dict
```
- Percentile: `labels=result.regime.tercile`, `transitions=result.alerts.bucket_transitions`.
- HMM: `labels=result.regime_hmm.label`, `transitions=result.alerts.hmm_bucket_transitions`.

`run_backtest` runs the scorer twice when `hmm_enabled` and emits:
```
acceptance_report.json        # percentile (unchanged)
acceptance_report_hmm.json    # HMM, same 6 gates
acceptance_compare.json       # side-by-side: gate, percentile metric, hmm metric, delta
```

Gate expectations: **G5 (calm-quarter stability)** is where HMM should win (transition-matrix persistence kills the flicker). **G3 (known-event recognition)** is where filtered-HMM risks losing (filter can lag a sharp flip vs a percentile threshold). The compare report makes the trade numeric.

**Decision rule (written into the spec):** HMM becomes the production default only if it passes all 6 gates **and** strictly improves G5 **without** regressing G3 (8/8 events retained). Otherwise percentile stays default and HMM ships as an overlay signal.

---

## 9. Testing strategy

Mirrors existing `tests/` discipline (pytest + hypothesis, `filterwarnings=error`):

- **State-ordering** — known 3-level β means → assert Risk-off↔lowest-mean; holds again after refit on a permuted slice (label-switching guard).
- **Causality / no-lookahead** (critical) — label at t on `β[:t]`; append future data; recompute; assert label at t unchanged. Property test over random cut points.
- **Determinism** — two identical runs → byte-identical labels/probs/refit-log.
- **Cold start** — history < `min_history_days` → all `Unknown`, `cold_start_flag=True`.
- **Convergence handling** — degenerate fixture (near-constant β) → `Unknown` + `fit_failed`, no crash, no escaping warning.
- **Synthetic recovery** — β generated from a known 3-state HMM → filtered labels recover true states above a hit-rate floor.
- **Gate scorer method-agnostic** — hand-built label frame → same result regardless of source.
- **Cadence-invariance** — §6.1 bench, `@pytest.mark.slow`.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| EM non-determinism breaks fingerprint | Pin start params + `search_reps=0`; determinism test gates merge |
| statsmodels warnings error-out the suite | Catch in `warnings.catch_warnings()`; convert to `fit_failed`, never propagate |
| Label-switching across refits | Canonical sort-by-mean remap every fit (§5) |
| Full-history refit slow per run | Monthly cadence; equivalence proven by §6.1; caching deferred |
| HMM fails gates (esp. G3 event lag) | Coexist design — percentile stays default, no forced cutover |
| Thin cuts (LatAm/EM_FI) unstable fits | `confidence` + `thin_cut_flag` surfaced; no silent bad labels |
| New heavy dep (statsmodels) | Pin version; mypy override mirroring the scipy override ([pyproject.toml:66](../../../pyproject.toml)) |

---

## 11. Non-goals (this spec)

- No viz / HMM overlay (future viz pass).
- No incremental / cached daily production fit (stateless full-recompute for v1.1; optimize later).
- No HMM in the tripwire (short-window mirror stays percentile).
- No 2-state or configurable `n_states` (locked at 3).
- No multivariate emission (β-only; correlation/PC1 fusion is separate future work).
- No S8 dashboard (deferred — the pivot that motivated this work).

---

## 12. New dependency

`statsmodels` (pinned, e.g. `>=0.14,<0.15`) added to `pyproject.toml` dependencies, with a mypy `ignore_missing_imports` override mirroring the existing scipy override.
