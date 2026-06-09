# RoRo — Acceptance-Gate Diagnostics Harness + Findings Memo

**Date:** 2026-06-08
**Status:** Design approved through brainstorming. Ready for implementation planning.
**Author:** Alan Vazquez, CFA
**Scope:** Build a read-only diagnostic harness that recomputes the S9 acceptance gates (G1–G6) over the existing backtest, sweeps thresholds, and emits an evidence base + a findings memo. **Diagnosis only** — no threshold changes, no classifier work. The Statistical Jump Model PRD (`docs/prd/JM-PRD.md`) is parked behind this: a JM bake-off on a mis-specified scorecard is wasted motion.

---

## 0. Why this exists

`todo.md` records that on the 2008–2026 backtest the percentile production default "fails 5/6 gates" and the HMM "passed no gate." Taken at face value that says the regime engine is broken. The artifact on disk (`outputs/hmm_eval/acceptance_compare.json`) says something very different once read carefully:

| Gate | Bar | Percentile | HMM | Real status |
|---|---|---|---|---|
| **G1 vix** | frac ≥ 0.8 | 0.507 | 0.507 *(identical)* | **Shared** — measures β-vs-VIX corr, not the classifier; cannot discriminate method |
| **G2 bbb** | frac ≥ 0.8 | 0.682 | 0.682 *(identical)* | **Shared** — same |
| **G3 events** | 8/8 | 7/8 | 6/8 | **Spec bug** — only miss is `2008_lehman` (2008-10-10), which is *before* `start=2008-12-31`; its ±6-day window has zero rows → can never match. Percentile is 7/7 on in-range events |
| **G4 seg-lift** | ≥ 0.2 | 0.183 | 0.161 | **Borderline** — a hair under the bar |
| **G5 stability** | ≤ 2 | 18 | 9 | **Real & binding** — the one honest method-discriminating failure; HMM genuinely halved it |
| **G6 internal** | ≤ 5 | pass *(vacuous)* | pass | **Vacuous** — `"no DM mapping data"`; composite price unwired (deferred to v1.1), auto-passes |

De-spun headline: **RoRo fails G5 (stability); the rest of the scorecard is mis-specified.** This harness proves each of those claims with reproducible evidence and answers the one question the artifacts cannot: is there *any* feasible point on the G3↔G5 sensitivity-stability tradeoff for the percentile classifier, or are the two gates mutually unsatisfiable?

---

## 1. Goal

Produce a trustworthy, reproducible evidence base — `gate_diagnostics.json` + `g3_g5_frontier.csv` + a markdown memo — that, per gate, classifies the failure as **bug / vacuous / real / borderline**, flags it as **method-shared or method-discriminating**, and reports its **achievable range** under a threshold/persistence sweep. The memo surfaces recalibration options but commits to none — the redesign is a separate, later decision.

---

## 2. Locked decisions

| Fork | Choice | Rationale |
|---|---|---|
| Deliverable | Findings memo (terminal) | Evidence first; recalibration deferred to a follow-up decision |
| Gate parametrization | Option (a) — defaulted kwargs on existing `_gate_*` | Single source of truth, no logic duplication/drift; production output byte-identical |
| Faithfulness guard | Baseline pin must reproduce `acceptance_compare.json` exactly before any sweep | Proves the harness scores the *real* gates, not a re-implementation |
| Persistence knob | Causal `_smooth_regime_hysteresis` (`roro/report/figures.py:485`), sweep confirm-days `n` | Already causal (forward walk, confirms after `n` consecutive obs, no future peek) → honest G3↔G5 frontier |
| Code placement | New read-only `roro/gate_diagnostics.py` + `roro gate-diagnostics` CLI subcommand | Sibling to `backtest.py`; Click group already hosts `run`/`backtest`/`report` |
| Memo posture | Descriptive + non-committal "recalibration options" appendix | User chose diagnose-only; redesign decided in follow-up |
| JM PRD | Parked behind this work | Bake-off on a mis-specified scorecard is wasted motion |

---

## 3. Architecture

A pure, read-only analysis layer parallel to `backtest.py`: one backtested `RunResult` in → evidence artifacts out. No engine, classifier, or production-gate behavior changes; `jm_*` config untouched.

```
RunResult (existing engine/backtest)
        │
        ▼
roro/gate_diagnostics.py   ──► gate_diagnostics.json
   (pure, read-only)        ──► g3_g5_frontier.csv
        │                   ──► memo evidence (cited by the .md)
        ▼
roro gate-diagnostics CLI  (--from-output <dir>  |  --config/--start/--end)
```

### 3.1 Small refactor to `roro/backtest.py` (behavior-preserving)

- The **scalar-threshold gates** (`_gate_external_corr` for G1/G2, `_gate_segmentation_lift` for G4, `_gate_stability` for G5) gain their thresholds as keyword args **defaulting to today's module constants** (`_G5_MAX_CALM_TRANSITIONS`, `_G1_VIX_RHO_MIN`, etc.). `_evaluate_gates` keeps calling with defaults → output unchanged; the existing golden/determinism tests are the regression guard. These defaulted signatures are what the harness sweeps.
- **`_gate_events` / `EVENTS` are NOT changed.** G3 has no scalar threshold; its only knob is the event set + the binary 8/8 rule. Touching it would flip default scoring 7/8→7/7, breaking both the byte-identical-production invariant and the baseline pin against the on-disk artifact (which records 7/8). The out-of-range repair (drop `2008_lehman`, graded `k/7`) is computed **entirely inside `gate_diagnostics.py`** as a diagnostic view — production G3 keeps counting the out-of-range event as a miss, exactly as today.

### 3.2 New module `roro/gate_diagnostics.py`

Pure functions over a `RunResult`. Stages:

1. **Baseline pin** — recompute all six gates at production thresholds; assert byte-equality against the on-disk `acceptance_compare.json` (or fail loudly). Nothing downstream runs until this matches.
2. **Per-gate sweeps**
   - **G1/G2** over `(rho_min, fraction_min)` grids → achievable-fraction surface; also assert percentile == HMM (method-shared proof).
   - **G4** over `(gap, fraction)` grid → achievable-fraction surface.
   - **G5** over the calm-quarter transition ceiling → which bar each method would pass.
3. **G3 repair view** — recompute dropping the out-of-range `2008_lehman`; report graded `k/7` and binary at the in-range set for percentile and HMM.
4. **G3↔G5 frontier (crown jewel)** — apply causal `_smooth_regime_hysteresis` to the percentile labels for `n = 0,1,2,…,N_max` (~40) confirm-days; at each `n` record `(in_range_events_caught, max_calm_quarter_transitions)`. Overlay the HMM single point. → the feasibility curve: does adding causal persistence to percentile recover G5 without sacrificing G3, and where would JM have to land?
5. **Classification** — tag each gate `bug | vacuous | real | borderline` and `shared | discriminating` from the evidence above.

### 3.3 CLI

`roro gate-diagnostics` (Click subcommand in `roro/cli.py`):
- `--from-output <dir>` — score an existing run directory (default path for the on-disk eval).
- `--config/--start/--end/--fred-key` — run a fresh backtest then diagnose.
- `--out <dir>` — artifact destination.

---

## 4. Outputs (deterministic — no timestamps, RoRo invariant)

- **`gate_diagnostics.json`** — per gate: `current_value`, `threshold`, `passes_at_default`, `achievable_range`, `shared_or_discriminating`, `root_cause ∈ {bug, vacuous, real, borderline}`, short note.
- **`g3_g5_frontier.csv`** — columns `confirm_days, events_caught, max_calm_transitions`, plus the HMM reference point.
- **The memo** (`docs/analysis/2026-06-08-gate-diagnostics-memo.md`, written by *running* the harness in the implementation phase, not by this spec):
  - One section per gate: root cause, shared-vs-discriminating, achievable range.
  - The G3↔G5 frontier finding (feasible point: yes/no, and at what persistence).
  - Non-committal appendix — recalibration *options* (gate tiering / threshold candidates / frontier-reframe) as input to the follow-up decision. **No recommendation locked.**

---

## 5. Testing (`tests/test_gate_diagnostics.py`, one module per code module)

- **Baseline pin** reproduces `acceptance_compare.json` byte-for-byte.
- **Sweep monotonicity** — looser threshold never lowers pass count; more confirm-days never increases calm-quarter transitions.
- **Frontier determinism** — repeat run → identical `g3_g5_frontier.csv`.
- **No-lookahead** — the causal smoother's label at `t` is invariant to appended future data.
- **Refactor regression** — `backtest.py` gates with default kwargs produce output identical to pre-change (existing golden test is the guard).

---

## 6. Scope / non-goals

- ❌ **No threshold changes** to production gates. Diagnosis only; recalibration is a separate decision.
- ❌ **No JM work**, no classifier changes, no new regime methods.
- ❌ **No composite-price wiring** — G6 stays documented-vacuous, not fixed here.
- ❌ **No new dependencies.**
- ✅ Full quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest` with `filterwarnings=["error"]`, determinism.
- ✅ Production engine/backtest/report output **byte-identical** to pre-change when the diagnostic is not invoked.

---

## 7. Definition of done

The harness reproduces the on-disk gate artifact exactly, emits `gate_diagnostics.json` + `g3_g5_frontier.csv`, and the memo answers — per gate — *bug, vacuous, real, or borderline; shared or discriminating; and what is achievable* — with the G3↔G5 curve establishing whether a feasible persistence/sensitivity point exists for the percentile classifier (and where a JM would need to land). All quality bars green; production output unchanged.

---

## 8. Downstream (out of scope here)

- **Gate recalibration** — using this evidence: tier pipeline-health (G1/G2/G6) vs method-discriminating (G3/G4/G5); fix G3 scoring; re-set G4/G5 thresholds off a calibration/test split; possibly reframe G3↔G5 as a frontier. *Separate spec.*
- **JM regime layer** (`docs/prd/JM-PRD.md`) — runs against the *recalibrated* scorecard; its AJ-6/AJ-7 rewritten to the new bar. *Resumes after recalibration.*
- **G1/G2 method-specific wiring** — route each classifier through validation so the external-corr gates discriminate method (known v1.1 roadmap item).
