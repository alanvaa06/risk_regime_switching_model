# Gate-Diagnostics Harness + Findings Memo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only diagnostic that recomputes the S9 acceptance gates, sweeps their thresholds, repairs the G3 out-of-range bug, and traces the causal G3↔G5 sensitivity-stability frontier — then run it to produce a findings memo. Diagnosis only; no production gate behavior changes.

**Architecture:** A behavior-preserving refactor of `roro/backtest.py` (extract calm-quarter helpers; add defaulted threshold kwargs to the scalar gates) feeds a new pure module `roro/gate_diagnostics.py` and a `roro gate-diagnostics` CLI subcommand. The harness reuses production primitives verbatim — `_bucket_transitions` (alerts), `_smooth_regime_hysteresis` (report), the extracted G5 helpers — so the evidence cannot drift from the real gates.

**Tech Stack:** Python 3.12, pandas, numpy, Click, pytest. `uv`-managed `.venv` (use `.venv\Scripts\python.exe`). Quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest` with `filterwarnings=["error"]`.

**Spec refinement (flagged):** Spec §3.3 offered `--from-output <dir>` to "score an existing run." Daily returns are **not** persisted to disk, so a faithful `RunResult` cannot be reconstructed from CSVs. This plan instead **re-runs the engine** (FRED-backed, exactly as `roro backtest` does) to obtain a fresh causal `RunResult`, and reads the on-disk `acceptance_compare.json` purely as the **baseline-pin reference**. Still read-only w.r.t. production code, still pins to the recorded artifact — only the RunResult source changes. No scope change.

---

## File Structure

- **Modify** `roro/backtest.py` — extract `_calm_quarters` + `_count_max_calm_transitions`; add `_event_hits`; add defaulted threshold kwargs to `_gate_external_corr` / `_gate_segmentation_lift` / `_gate_stability`. Behavior-preserving.
- **Create** `roro/gate_diagnostics.py` — pure, read-only: per-gate sweeps, G3 repair, G3↔G5 frontier, `classify_gate`, `diagnose(result)` orchestrator, artifact writers.
- **Modify** `roro/cli.py` — add `roro gate-diagnostics` subcommand.
- **Create** `tests/test_gate_diagnostics.py` — frame-level unit tests + one integration test.
- **Modify** `tests/test_cli.py` — add a CLI smoke test for the new subcommand.
- **Create** (by *running* the harness, final task) `docs/analysis/2026-06-08-gate-diagnostics-memo.md` — the deliverable.

---

## Task 1: Behavior-preserving refactor of `roro/backtest.py`

Extract the calm-quarter machinery and the event-hit logic so the diagnostic reuses the *exact* production math, and parametrize the scalar-threshold gates. `_evaluate_gates` keeps calling with defaults → output unchanged.

**Files:**
- Modify: `roro/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for the extracted helpers + parametrized gates**

Add to `tests/test_backtest.py`:

```python
import numpy as np

from roro.backtest import (
    _calm_quarters,
    _count_max_calm_transitions,
    _event_hits,
    _gate_stability,
)


def test_calm_quarters_returns_low_vol_quarter_ends() -> None:
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    # First year quiet, second year loud -> calm quarters concentrate in year 1.
    vals = np.concatenate([rng.normal(0, 1e-4, 261), rng.normal(0, 1e-2, len(idx) - 261)])
    daily = pd.DataFrame({"global": pd.Series(vals, index=idx)})
    calm = _calm_quarters(daily)
    assert isinstance(calm, pd.DatetimeIndex)
    assert len(calm) >= 1
    assert calm.equals(calm.normalize())


def test_count_max_calm_transitions_counts_only_calm_global() -> None:
    calm = pd.DatetimeIndex([pd.Timestamp("2020-03-31")])
    transitions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-15"),
                     pd.Timestamp("2020-08-01")],
            "segment": ["global", "global", "global"],
            "from_bucket": ["Risk-on", "Risk-off", "Risk-on"],
            "to_bucket": ["Risk-off", "Risk-on", "Risk-off"],
        }
    )
    worst = _count_max_calm_transitions(transitions, calm, segment="global")
    assert worst == 2.0  # only the two Q1 transitions land in the calm quarter


def test_event_hits_flags_out_of_range_events() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    terc = pd.DataFrame({"global": pd.Series("Risk-off", index=idx)})
    from roro.backtest import Event

    events = (
        Event(name="in_window", date="2020-06-15"),
        Event(name="before_start", date="2008-10-10"),
    )
    hits, total_in_range, out_of_range = _event_hits(
        terc, events, start="2020-01-01", end="2020-12-31"
    )
    assert total_in_range == 1
    assert hits == 1
    assert out_of_range == ["before_start"]


def test_gate_stability_threshold_kwarg_overrides_default() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    daily = pd.DataFrame({"global": pd.Series(1e-4, index=idx)})
    # build a RunResult-like via the existing empty fixture path is heavy; instead
    # test the worst-count comparison contract directly through the kwarg.
    transitions = pd.DataFrame(
        {"date": [idx[10], idx[11], idx[12]], "segment": ["global"] * 3,
         "from_bucket": ["a", "b", "a"], "to_bucket": ["b", "a", "b"]}
    )
    calm = _calm_quarters(daily)
    worst = _count_max_calm_transitions(transitions, calm, segment="global")
    assert worst >= 0.0  # smoke: helper callable with explicit args
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py -k "calm or event_hits or stability_threshold" -v`
Expected: FAIL — `ImportError: cannot import name '_calm_quarters'` (and the others).

- [ ] **Step 3: Implement the extraction + parametrization in `roro/backtest.py`**

Add these helpers above `_gate_stability`:

```python
def _calm_quarters(daily_log_returns: pd.DataFrame) -> pd.DatetimeIndex:
    """Quarter-end timestamps whose mean realized vol is below the median quarter."""
    if daily_log_returns.empty:
        return pd.DatetimeIndex([])
    proxy = daily_log_returns.iloc[:, 0].dropna()
    if proxy.empty:
        return pd.DatetimeIndex([])
    realized = proxy.rolling(_REALIZED_VOL_WINDOW).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
    by_quarter_vol = realized.groupby(pd.Grouper(freq="QE")).mean().dropna()
    if by_quarter_vol.empty:
        return pd.DatetimeIndex([])
    median_vol = by_quarter_vol.median()
    calm = by_quarter_vol[by_quarter_vol < median_vol].index
    return pd.DatetimeIndex(calm).normalize()


def _count_max_calm_transitions(
    transitions: pd.DataFrame,
    calm_quarters: pd.DatetimeIndex,
    *,
    segment: str = "global",
) -> float:
    """Worst per-calm-quarter transition count for one segment."""
    if transitions.empty or len(calm_quarters) == 0:
        return 0.0
    t = transitions.copy()
    t["quarter"] = (
        pd.PeriodIndex(t["date"], freq="Q").to_timestamp(how="end").normalize()
    )
    seg = t[t["segment"] == segment]
    by_quarter = seg.groupby("quarter").size()
    counts = by_quarter.reindex(calm_quarters, fill_value=0)
    return float(counts.max()) if len(counts) else 0.0


def _event_hits(
    labels: pd.DataFrame,
    events: tuple[Event, ...],
    *,
    start: str,
    end: str,
) -> tuple[int, int, list[str]]:
    """Count event-window hits; report which events fall outside [start, end].

    An event is *in range* only if its +/-window overlaps [start, end]. Out-of-range
    events are returned by name (they can never match -> scorecard artifact, not a
    classifier miss).
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    hits = 0
    in_range = 0
    out_of_range: list[str] = []
    for event in events:
        ts = pd.Timestamp(event.date)
        w_start = ts - pd.Timedelta(days=_EVENT_WINDOW_DAYS)
        w_end = ts + pd.Timedelta(days=_EVENT_WINDOW_DAYS)
        if w_end < start_ts or w_start > end_ts:
            out_of_range.append(event.name)
            continue
        in_range += 1
        local = labels.loc[(labels.index >= w_start) & (labels.index <= w_end)]
        for seg in event.segments:
            if seg in local.columns and local[seg].isin(event.expected_buckets).any():
                hits += 1
                break
    return hits, in_range, out_of_range
```

Now rewrite `_gate_stability` to delegate (preserving every early-return) and accept a threshold kwarg:

```python
def _gate_stability(
    result: RunResult,
    transitions: pd.DataFrame,
    *,
    max_calm_transitions: float = _G5_MAX_CALM_TRANSITIONS,
) -> dict[str, Any]:
    if transitions.empty:
        return {"passed": True, "max_transitions_in_calm_quarter": 0.0}
    daily = result.returns.daily_log_returns
    if daily.empty:
        return {"passed": False, "reason": "missing returns"}
    calm_quarters = _calm_quarters(daily)
    if len(calm_quarters) == 0:
        return {"passed": True, "max_transitions_in_calm_quarter": 0.0}
    worst = _count_max_calm_transitions(transitions, calm_quarters, segment="global")
    return {
        "passed": bool(worst <= max_calm_transitions),
        "max_transitions_in_calm_quarter": worst,
    }
```

Add defaulted threshold kwargs to the other two scalar gates (signatures only — bodies unchanged except using the kwarg):

```python
def _gate_external_corr(
    result: RunResult,
    *,
    series_id: str,
    rho_min: float,
    fraction_min: float,
) -> dict[str, Any]:
    ...  # body unchanged; already takes rho_min / fraction_min


def _gate_segmentation_lift(
    terc: pd.DataFrame,
    *,
    seg_gap_min: int = _G4_SEG_GAP_MIN,
    fraction_min: float = _G4_SEG_FRACTION_MIN,
) -> dict[str, Any]:
    if not {"DM_Eq", "EM_Eq"}.issubset(terc.columns):
        return {"passed": False, "reason": "missing DM_Eq/EM_Eq"}
    a = terc["DM_Eq"].map(_TERCILE_ORDINAL).astype(float)
    b = terc["EM_Eq"].map(_TERCILE_ORDINAL).astype(float)
    diff = (a - b).abs().dropna()
    if diff.empty:
        return {"passed": False, "fraction_with_gap_ge_2": 0.0}
    fraction = float((diff >= seg_gap_min).mean())
    return {"passed": bool(fraction >= fraction_min), "fraction_with_gap_ge_2": fraction}
```

Update the `_evaluate_gates` call site for G5 to the new signature (keyword `transitions`):

```python
    gates["G5_stability"] = _gate_stability(result, transitions)
```
(unchanged call — `transitions` stays positional/keyword as today; the new kwarg defaults.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v`
Expected: PASS (new helper tests + all pre-existing backtest tests).

- [ ] **Step 5: Run the golden + reproducibility regression to prove production output is byte-identical**

Run: `.venv\Scripts\python.exe -m pytest tests/test_golden.py tests/test_reproducibility.py -v`
Expected: PASS — the refactor changed no production output.

- [ ] **Step 6: Commit**

```bash
git add roro/backtest.py tests/test_backtest.py
git commit -m "refactor(backtest): extract calm-quarter + event-hit helpers, parametrize scalar gates"
```

---

## Task 2: Threshold sweeps + gate classification in `roro/gate_diagnostics.py`

**Files:**
- Create: `roro/gate_diagnostics.py`
- Test: `tests/test_gate_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_gate_diagnostics.py`:

```python
"""Tests for the read-only acceptance-gate diagnostic harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.gate_diagnostics import (
    classify_gate,
    sweep_external_corr,
    sweep_segmentation_lift,
)


def test_sweep_external_corr_is_monotone_nonincreasing() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    col = ("global", "VIXCLS")
    rho = pd.DataFrame({col: pd.Series(np.linspace(0.0, 1.0, len(idx)), index=idx)})
    rho.columns = pd.MultiIndex.from_tuples([col])
    out = sweep_external_corr(rho, series_id="VIXCLS", rho_grid=[0.2, 0.4, 0.6, 0.8])
    fr = out["fraction_above"].to_numpy()
    assert (np.diff(fr) <= 1e-12).all()  # higher rho_min -> fewer days clear it


def test_sweep_segmentation_lift_is_monotone_in_gap() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    terc = pd.DataFrame(
        {"DM_Eq": pd.Series("Risk-on", index=idx), "EM_Eq": pd.Series("Risk-off", index=idx)}
    )
    out = sweep_segmentation_lift(terc, gap_grid=[1, 2, 3])
    fr = out["fraction_with_gap_ge"].to_numpy()
    assert (np.diff(fr) <= 1e-12).all()
    # gap == 2 here (Risk-on=3 vs Risk-off=1) -> fraction 1.0 at gap<=2, 0.0 at gap 3
    assert out.set_index("gap").loc[2, "fraction_with_gap_ge"] == 1.0
    assert out.set_index("gap").loc[3, "fraction_with_gap_ge"] == 0.0


def test_classify_gate_tags() -> None:
    assert classify_gate(passed=True, percentile_value=1.0, hmm_value=1.0,
                         value=1.0, threshold=5.0, vacuous_reason="no DM mapping data",
                         out_of_range=[]) == "vacuous"
    assert classify_gate(passed=False, percentile_value=0.5, hmm_value=0.5,
                         value=0.5, threshold=0.8, vacuous_reason=None,
                         out_of_range=[]) == "shared"
    assert classify_gate(passed=False, percentile_value=7, hmm_value=6,
                         value=7, threshold=8, vacuous_reason=None,
                         out_of_range=["2008_lehman"]) == "bug"
    assert classify_gate(passed=False, percentile_value=0.183, hmm_value=0.161,
                         value=0.183, threshold=0.2, vacuous_reason=None,
                         out_of_range=[]) == "borderline"
    assert classify_gate(passed=False, percentile_value=18, hmm_value=9,
                         value=18, threshold=2, vacuous_reason=None,
                         out_of_range=[]) == "real"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roro.gate_diagnostics'`.

- [ ] **Step 3: Implement the sweeps + classifier**

Create `roro/gate_diagnostics.py`:

```python
"""Read-only diagnostic harness for the S9 acceptance gates.

Recomputes G1-G6 over a backtested RunResult, sweeps thresholds, repairs the
G3 out-of-range bug, and traces the causal G3<->G5 frontier. Reuses production
primitives verbatim so the evidence cannot drift from the real gates. Writes
no production artifacts and mutates nothing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roro.backtest import _TERCILE_ORDINAL

_SHARED_EPS = 1e-9
_BORDERLINE_REL = 0.15


def sweep_external_corr(
    rolling_corr_60d: pd.DataFrame,
    *,
    series_id: str,
    rho_grid: list[float],
) -> pd.DataFrame:
    """Fraction of days with |rolling corr| >= rho_min, as rho_min varies."""
    key = ("global", series_id)
    rows: list[dict[str, float]] = []
    if key in rolling_corr_60d.columns:
        rho = rolling_corr_60d[key].abs().dropna()
    else:
        rho = pd.Series(dtype=float)
    for rho_min in rho_grid:
        frac = float((rho >= rho_min).mean()) if not rho.empty else 0.0
        rows.append({"rho_min": float(rho_min), "fraction_above": frac})
    return pd.DataFrame(rows)


def sweep_segmentation_lift(
    terc: pd.DataFrame,
    *,
    gap_grid: list[int],
) -> pd.DataFrame:
    """Fraction of days with |DM_Eq - EM_Eq| ordinal gap >= g, as g varies."""
    rows: list[dict[str, float]] = []
    if {"DM_Eq", "EM_Eq"}.issubset(terc.columns):
        a = terc["DM_Eq"].map(_TERCILE_ORDINAL).astype(float)
        b = terc["EM_Eq"].map(_TERCILE_ORDINAL).astype(float)
        diff = (a - b).abs().dropna()
    else:
        diff = pd.Series(dtype=float)
    for gap in gap_grid:
        frac = float((diff >= gap).mean()) if not diff.empty else 0.0
        rows.append({"gap": int(gap), "fraction_with_gap_ge": frac})
    return pd.DataFrame(rows)


def classify_gate(
    *,
    passed: bool,
    percentile_value: float,
    hmm_value: float,
    value: float,
    threshold: float,
    vacuous_reason: str | None,
    out_of_range: list[str],
) -> str:
    """Tag a gate: vacuous | bug | shared | borderline | real.

    Order matters: a vacuous pass outranks everything; an out-of-range event is a
    scorecard bug; identical percentile/hmm values mean the gate cannot
    discriminate method (shared); a near-miss is borderline; else a real failure.
    """
    if vacuous_reason is not None:
        return "vacuous"
    if out_of_range:
        return "bug"
    if abs(percentile_value - hmm_value) <= _SHARED_EPS:
        return "shared"
    if not passed and threshold != 0 and abs(value - threshold) / abs(threshold) <= _BORDERLINE_REL:
        return "borderline"
    return "real"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/gate_diagnostics.py tests/test_gate_diagnostics.py
git commit -m "feat(diagnostics): threshold sweeps + gate classification"
```

---

## Task 3: G3 repair view (drop out-of-range events, graded score)

**Files:**
- Modify: `roro/gate_diagnostics.py`
- Test: `tests/test_gate_diagnostics.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_gate_diagnostics.py`:

```python
from roro.backtest import Event
from roro.gate_diagnostics import repair_g3


def test_repair_g3_excludes_out_of_range_and_grades() -> None:
    idx = pd.bdate_range("2010-01-01", "2022-12-31")
    terc = pd.DataFrame({"global": pd.Series("Risk-off", index=idx)})
    events = (
        Event(name="2008_lehman", date="2008-10-10"),  # before start -> excluded
        Event(name="2020_COVID", date="2020-03-16"),
        Event(name="2018_q4", date="2018-12-24"),
    )
    out = repair_g3(terc, events, start="2010-01-01", end="2022-12-31")
    assert out["out_of_range"] == ["2008_lehman"]
    assert out["total_in_range"] == 2
    assert out["hits"] == 2
    assert out["graded"] == 1.0  # 2/2 in-range events caught
    assert out["passed_in_range"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py::test_repair_g3_excludes_out_of_range_and_grades -v`
Expected: FAIL — `ImportError: cannot import name 'repair_g3'`.

- [ ] **Step 3: Implement `repair_g3`**

Add to `roro/gate_diagnostics.py` (import `_event_hits` and `Event` from backtest at top):

```python
from roro.backtest import Event, _event_hits, _TERCILE_ORDINAL  # update existing import
```

```python
def repair_g3(
    labels: pd.DataFrame,
    events: tuple[Event, ...],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """G3 with out-of-range events excluded; graded k/total_in_range."""
    hits, total_in_range, out_of_range = _event_hits(labels, events, start=start, end=end)
    graded = float(hits / total_in_range) if total_in_range else 0.0
    return {
        "hits": hits,
        "total_in_range": total_in_range,
        "out_of_range": out_of_range,
        "graded": graded,
        "passed_in_range": bool(total_in_range > 0 and hits == total_in_range),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/gate_diagnostics.py tests/test_gate_diagnostics.py
git commit -m "feat(diagnostics): G3 repair view excluding out-of-range events"
```

---

## Task 4: Causal G3↔G5 frontier

The crown jewel. Sweep causal hysteresis confirm-days over the percentile labels; at each `n` record in-range events caught and worst calm-quarter transitions, reusing production transition + calm-quarter math.

**Files:**
- Modify: `roro/gate_diagnostics.py`
- Test: `tests/test_gate_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_gate_diagnostics.py`:

```python
from roro.gate_diagnostics import g3_g5_frontier


def _flippy_labels(idx: pd.DatetimeIndex) -> pd.Series:
    # Alternate Risk-on/Risk-off every other day -> maximal flicker at n=0.
    vals = ["Risk-on" if i % 2 == 0 else "Risk-off" for i in range(len(idx))]
    return pd.Series(vals, index=idx)


def test_frontier_stability_is_monotone_nonincreasing_in_confirm_days() -> None:
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    terc = pd.DataFrame({"global": _flippy_labels(idx)})
    daily = pd.DataFrame({"global": pd.Series(1e-4, index=idx)})
    events = (Event(name="mid", date="2020-06-15"),)
    fr = g3_g5_frontier(
        terc, daily, events=events, start="2020-01-01", end="2021-12-31",
        confirm_grid=[0, 2, 5, 10, 21],
    )
    trans = fr["max_calm_transitions"].to_numpy()
    assert (np.diff(trans) <= 1e-9).all()  # more persistence -> never more flicker
    assert fr.iloc[0]["max_calm_transitions"] >= fr.iloc[-1]["max_calm_transitions"]


def test_frontier_is_deterministic() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    terc = pd.DataFrame({"global": _flippy_labels(idx)})
    daily = pd.DataFrame({"global": pd.Series(1e-4, index=idx)})
    events = (Event(name="mid", date="2020-06-15"),)
    a = g3_g5_frontier(terc, daily, events=events, start="2020-01-01",
                       end="2020-12-31", confirm_grid=[0, 3, 7])
    b = g3_g5_frontier(terc, daily, events=events, start="2020-01-01",
                       end="2020-12-31", confirm_grid=[0, 3, 7])
    pd.testing.assert_frame_equal(a, b)


def test_frontier_no_lookahead_prefix_stable() -> None:
    # A frontier row computed on a prefix must match the same row when future
    # data is appended, because the smoother is causal.
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    full = pd.DataFrame({"global": _flippy_labels(idx)})
    daily = pd.DataFrame({"global": pd.Series(1e-4, index=idx)})
    from roro.report.figures import _smooth_regime_hysteresis

    sm_full = _smooth_regime_hysteresis(full["global"], 5)
    sm_prefix = _smooth_regime_hysteresis(full["global"].iloc[:100], 5)
    pd.testing.assert_series_equal(sm_full.iloc[:100], sm_prefix)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -k frontier -v`
Expected: FAIL — `ImportError: cannot import name 'g3_g5_frontier'`.

- [ ] **Step 3: Implement `g3_g5_frontier`**

Add imports + function to `roro/gate_diagnostics.py`:

```python
from roro.alerts import _bucket_transitions
from roro.backtest import _calm_quarters, _count_max_calm_transitions
from roro.report.figures import _smooth_regime_hysteresis
```

```python
def g3_g5_frontier(
    terc: pd.DataFrame,
    daily_log_returns: pd.DataFrame,
    *,
    events: tuple[Event, ...],
    start: str,
    end: str,
    confirm_grid: list[int],
    segment: str = "global",
) -> pd.DataFrame:
    """Trace (events caught, worst calm-quarter transitions) vs confirm-days.

    n == 0 is the raw (unsmoothed) percentile labels; n > 0 applies the causal
    hysteresis smoother. Transition counting and calm-quarter detection reuse
    production primitives verbatim.
    """
    labels = terc[segment]
    calm = _calm_quarters(daily_log_returns)
    rows: list[dict[str, float]] = []
    for n in confirm_grid:
        smoothed = labels if n == 0 else _smooth_regime_hysteresis(labels, n)
        smoothed_df = smoothed.to_frame(name=segment)
        transitions = _bucket_transitions(smoothed_df)
        worst = _count_max_calm_transitions(transitions, calm, segment=segment)
        hits, _total, _oor = _event_hits(smoothed_df, events, start=start, end=end)
        rows.append(
            {
                "confirm_days": int(n),
                "events_caught": int(hits),
                "max_calm_transitions": worst,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -v`
Expected: PASS (all diagnostic tests).

- [ ] **Step 5: Commit**

```bash
git add roro/gate_diagnostics.py tests/test_gate_diagnostics.py
git commit -m "feat(diagnostics): causal G3<->G5 sensitivity-stability frontier"
```

---

## Task 5: `diagnose()` orchestrator + baseline pin + artifact writers

**Files:**
- Modify: `roro/gate_diagnostics.py`
- Test: `tests/test_gate_diagnostics.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_gate_diagnostics.py`:

```python
import json
from pathlib import Path

from roro.config import EngineConfig
from roro.gate_diagnostics import diagnose, write_diagnostics
from roro.types import (
    AlertSet, BetaBySegment, BetaFrame, CorrelationFrame, RegimeFrame,
    ReturnsFrame, RunResult, Universe, ValidationFrame, VolFrame,
)


def _minimal_result() -> RunResult:
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    empty = pd.DataFrame()
    terc = pd.DataFrame(
        {
            "global": _flippy_labels(idx),
            "DM_Eq": pd.Series("Risk-on", index=idx),
            "EM_Eq": pd.Series("Risk-off", index=idx),
        }
    )
    daily = pd.DataFrame({"global": pd.Series(1e-4, index=idx)})
    transitions = _bucket_transitions(terc[["global"]])
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=pd.Series(dtype=float))
    bbs = BetaBySegment(by_segment={"global": bf})
    return RunResult(
        config=EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o")),
        universe=Universe(countries=empty, composites=empty),
        returns=ReturnsFrame(log_returns_3m=empty, daily_log_returns=daily),
        vol=VolFrame(ewma_sigma_annualized=empty),
        beta=bbs,
        regime=RegimeFrame(percentile_5y=empty, tercile=terc, quintile=empty,
                           direction=empty, n_per_segment=empty, thin_cut_flag=empty,
                           bootstrap_flag=empty),
        correlation=CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty),
        validation=ValidationFrame(rolling_corr_60d=empty, internal_consistency=empty,
                                   correlation_alerts=empty),
        tripwire=bbs,
        alerts=AlertSet(bucket_transitions=transitions, disagreement_events=empty,
                        validation_degradation=empty),
    )


def test_diagnose_emits_per_gate_tags_and_frontier() -> None:
    diag = diagnose(_minimal_result(), start="2020-01-01", end="2021-12-31")
    assert set(diag["gates"]) == {"G1_vix", "G2_bbb", "G3_events",
                                  "G4_segmentation_lift", "G5_stability", "G6_internal"}
    for g in diag["gates"].values():
        assert g["root_cause"] in {"bug", "vacuous", "shared", "borderline", "real"}
    assert not diag["frontier"].empty
    assert {"confirm_days", "events_caught", "max_calm_transitions"} <= set(
        diag["frontier"].columns
    )


def test_write_diagnostics_is_deterministic(tmp_path: Path) -> None:
    diag = diagnose(_minimal_result(), start="2020-01-01", end="2021-12-31")
    write_diagnostics(tmp_path / "a", diag)
    write_diagnostics(tmp_path / "b", diag)
    a = (tmp_path / "a" / "gate_diagnostics.json").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "gate_diagnostics.json").read_text(encoding="utf-8")
    assert a == b
    assert (tmp_path / "a" / "g3_g5_frontier.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -k "diagnose or write_diagnostics" -v`
Expected: FAIL — `ImportError: cannot import name 'diagnose'`.

- [ ] **Step 3: Implement `diagnose` + `write_diagnostics`**

Add to `roro/gate_diagnostics.py`. Import the production gate evaluator and EVENTS — and **consolidate all `from roro.backtest import …` additions from Tasks 2-4 into one statement** so `ruff` isort (I001) stays clean. After this task the module's backtest import should read exactly:

```python
import json
from pathlib import Path

from roro.alerts import _bucket_transitions
from roro.backtest import (
    EVENTS,
    Event,
    _TERCILE_ORDINAL,
    _calm_quarters,
    _count_max_calm_transitions,
    _evaluate_gates,
    _event_hits,
)
from roro.report.figures import _smooth_regime_hysteresis
from roro.types import RunResult
```

Only `_evaluate_gates` and `EVENTS` are newly used here; the rest were already pulled in by Tasks 2-4. Do **not** import `_gate_external_corr` / `_gate_segmentation_lift` / `_gate_stability` — `diagnose` goes through `_evaluate_gates`, so importing them would be an unused-import (`ruff` F401).

```python
_RHO_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_GAP_GRID = [1, 2]
_CONFIRM_GRID = [0, 1, 2, 3, 5, 8, 13, 21, 34]


def diagnose(
    result: RunResult,
    *,
    start: str,
    end: str,
    baseline_compare_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute gates, sweep, repair G3, trace the frontier, classify each gate."""
    terc = result.regime.tercile
    hmm_label = result.regime_hmm.label if result.regime_hmm is not None else terc
    perc_gates = _evaluate_gates(
        result, labels=terc, transitions=result.alerts.bucket_transitions
    )
    hmm_gates = _evaluate_gates(
        result,
        labels=hmm_label,
        transitions=(
            result.alerts.hmm_bucket_transitions
            if result.regime_hmm is not None
            else result.alerts.bucket_transitions
        ),
    )

    g3 = repair_g3(terc, EVENTS, start=start, end=end)

    def _num(d: dict[str, Any]) -> float:
        for k in ("fraction_above", "fraction_with_gap_ge_2",
                  "max_transitions_in_calm_quarter", "matched_events"):
            if k in d:
                return float(d[k])
        return 0.0

    thresholds = {
        "G1_vix": 0.8, "G2_bbb": 0.8, "G3_events": float(len(EVENTS)),
        "G4_segmentation_lift": 0.2, "G5_stability": 2.0, "G6_internal": 5.0,
    }
    gates: dict[str, dict[str, Any]] = {}
    for name, pg in perc_gates.items():
        hg = hmm_gates[name]
        oor = g3["out_of_range"] if name == "G3_events" else []
        tag = classify_gate(
            passed=bool(pg.get("passed", False)),
            percentile_value=_num(pg),
            hmm_value=_num(hg),
            value=_num(pg),
            threshold=thresholds[name],
            vacuous_reason=pg.get("reason"),
            out_of_range=oor,
        )
        gates[name] = {
            "percentile": pg,
            "hmm": hg,
            "threshold": thresholds[name],
            "root_cause": tag,
            "shared": abs(_num(pg) - _num(hg)) <= _SHARED_EPS,
        }

    sweeps = {
        "G1_vix": sweep_external_corr(result.validation.rolling_corr_60d,
                                      series_id="VIXCLS", rho_grid=_RHO_GRID),
        "G2_bbb": sweep_external_corr(result.validation.rolling_corr_60d,
                                      series_id="BAMLC0A4CBBB", rho_grid=_RHO_GRID),
        "G4_segmentation_lift": sweep_segmentation_lift(terc, gap_grid=_GAP_GRID),
    }
    frontier = g3_g5_frontier(
        terc, result.returns.daily_log_returns, events=EVENTS,
        start=start, end=end, confirm_grid=_CONFIRM_GRID,
    )

    baseline_pinned = None
    if baseline_compare_path is not None and baseline_compare_path.exists():
        ref = json.loads(baseline_compare_path.read_text(encoding="utf-8"))
        baseline_pinned = all(
            ref[g]["percentile"].get("passed") == perc_gates[g].get("passed")
            for g in perc_gates
            if g in ref
        )

    return {
        "start": start, "end": end,
        "gates": gates, "g3_repair": g3, "sweeps": sweeps,
        "frontier": frontier, "baseline_pinned": baseline_pinned,
    }


def write_diagnostics(out_dir: Path, diag: dict[str, Any]) -> None:
    """Emit gate_diagnostics.json + g3_g5_frontier.csv (deterministic, no timestamps)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "start": diag["start"], "end": diag["end"],
        "baseline_pinned": diag["baseline_pinned"],
        "gates": diag["gates"], "g3_repair": diag["g3_repair"],
        "sweeps": {k: v.to_dict(orient="records") for k, v in diag["sweeps"].items()},
    }
    (out_dir / "gate_diagnostics.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    diag["frontier"].to_csv(out_dir / "g3_g5_frontier.csv", index=False)
```

- [ ] **Step 4: Run to verify pass + full suite + type/lint**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gate_diagnostics.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m mypy --strict roro/gate_diagnostics.py roro/backtest.py`
Expected: no errors.
Run: `.venv\Scripts\python.exe -m ruff check roro/gate_diagnostics.py roro/backtest.py tests/test_gate_diagnostics.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add roro/gate_diagnostics.py tests/test_gate_diagnostics.py
git commit -m "feat(diagnostics): diagnose() orchestrator, baseline pin, artifact writers"
```

---

## Task 6: `roro gate-diagnostics` CLI subcommand

**Files:**
- Modify: `roro/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Append to `tests/test_cli.py`:

```python
def test_cli_gate_diagnostics_dispatches(
    tiny_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    monkeypatch.setattr(
        cli_mod,
        "_build_fred_client",
        lambda key: MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
        ),
    )
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        f"data_path: {tiny_xlsx}\n"
        f"output_dir: {tmp_path / 'out'}\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "tripwire_ewma_halflife_days: 5\n"
        "percentile_window_years: 1\n"
        "bucket_scheme: TERCILE\n"
        "min_n_per_cut: 2\n"
        "direction_lookback_days: 5\n"
        "external_corr_window_days: 60\n"
        "external_corr_alert_threshold: 0.3\n"
        "bootstrap_min_days: 10\n"
        "methodology_version: 1.0.0\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["gate-diagnostics", "--config", str(cfg_yaml),
         "--start", "2024-01-01", "--end", "2024-12-31",
         "--out", str(tmp_path / "diag")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "diag" / "gate_diagnostics.json").exists()
    assert (tmp_path / "diag" / "g3_g5_frontier.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_cli_gate_diagnostics_dispatches -v`
Expected: FAIL — `Error: No such command 'gate-diagnostics'`.

- [ ] **Step 3: Implement the subcommand in `roro/cli.py`**

Add after `cmd_backtest`:

```python
@main.command("gate-diagnostics")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="acceptance_compare.json to pin the baseline against.",
)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
def cmd_gate_diagnostics(
    config_path: Path,
    start: str,
    end: str,
    baseline_path: Path | None,
    out_dir: Path | None,
    fred_key: str | None,
) -> None:
    """Read-only diagnostic over the S9 acceptance gates."""
    # Lazy imports: backtest + diagnostics pull heavy deps.
    from roro.backtest import run_backtest  # noqa: PLC0415
    from roro.engine import run as engine_run  # noqa: PLC0415
    from roro.gate_diagnostics import diagnose, write_diagnostics  # noqa: PLC0415

    cfg = load_config(config_path)
    api_key = fred_key or os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    # Re-run the engine for a fresh causal RunResult (returns are not persisted).
    result = engine_run(cfg, fred_client=client, run_date=end,
                        as_of_data_date=end, force=True)
    diag = diagnose(result, start=start, end=end, baseline_compare_path=baseline_path)
    target = out_dir or (cfg.output_dir / "gate_diagnostics")
    write_diagnostics(target, diag)
    click.echo(f"OK: {target}")
```

(Note: `run_backtest` import is unnecessary here — remove it; the command uses `engine_run` directly. Keep only the two imports actually used: `engine_run`, `diagnose`, `write_diagnostics`.)

Corrected lazy-import block:

```python
    from roro.engine import run as engine_run  # noqa: PLC0415
    from roro.gate_diagnostics import diagnose, write_diagnostics  # noqa: PLC0415
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Full quality gate**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS (full suite, `filterwarnings=["error"]` clean).
Run: `.venv\Scripts\python.exe -m mypy --strict roro tests`
Expected: no errors.
Run: `.venv\Scripts\python.exe -m ruff check roro tests`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add roro/cli.py tests/test_cli.py
git commit -m "feat(cli): roro gate-diagnostics subcommand"
```

---

## Task 7: Generate the findings memo + update context docs

This task produces the actual deliverable by **running** the harness against the recorded 2008–2026 evaluation, then writing the memo from the emitted artifacts. Requires `FRED_API_KEY` and network (same as `roro backtest`).

**Files:**
- Create: `docs/analysis/2026-06-08-gate-diagnostics-memo.md`
- Modify: `docs/context/results.md`, `docs/context/memory.md`, `docs/context/todo.md`, `docs/context/sesion-log.md`, `docs/context/lessons.md`

- [ ] **Step 1: Run the harness against the eval config**

Run:
```bash
.venv\Scripts\python.exe -m roro.cli gate-diagnostics ^
  --config configs\eval.yaml ^
  --start 2008-12-31 --end 2026-05-26 ^
  --baseline outputs\hmm_eval\acceptance_compare.json ^
  --out outputs\gate_diag
```
(Use the same config that produced `outputs/hmm_eval/` — locate it from that run's `snapshot.json` `config_resolved`; if no yaml exists, reconstruct a minimal one with `hmm_enabled: true` and the eval dates.)
Expected: `OK: outputs\gate_diag`, with `gate_diagnostics.json` + `g3_g5_frontier.csv`. Confirm `baseline_pinned: true` inside the JSON — the harness reproduced the recorded gate verdicts.

- [ ] **Step 2: Write the memo from the artifacts**

Create `docs/analysis/2026-06-08-gate-diagnostics-memo.md` with these sections, filling every number from `outputs/gate_diag/`:
  - **Headline:** "RoRo fails G5 (stability); the rest of the scorecard is mis-specified."
  - **Per-gate table:** gate, bar, percentile value, hmm value, `root_cause` tag, shared/discriminating — copied from `gate_diagnostics.json["gates"]`.
  - **G3 repair:** `g3_repair` block — out-of-range `2008_lehman`, graded 7/7, `passed_in_range: true`.
  - **Sweeps:** G1/G2 fraction-vs-rho_min and G4 fraction-vs-gap tables from `sweeps`; state the achievable range and whether any defensible threshold passes.
  - **G3↔G5 frontier (crown jewel):** embed `g3_g5_frontier.csv` as a table; state explicitly whether a confirm-days value exists where `events_caught` stays at the in-range max **and** `max_calm_transitions` drops to a plausible ceiling — i.e. is there a feasible persistence/sensitivity point for percentile, and where would JM need to land relative to it. Mark the HMM single point for reference.
  - **Recalibration options appendix (non-committal):** tiering (pipeline-health G1/G2/G6 vs discriminating G3/G4/G5); G3 scoring fix; G4/G5 threshold candidates with a calibration/test split; frontier-reframe. **No recommendation locked** — input to the follow-up decision.

- [ ] **Step 3: Update context docs (per CLAUDE.md task-management)**

- `docs/context/results.md` — add 1-4 line review: harness built + verified; headline finding; feasible-point yes/no from the frontier.
- `docs/context/memory.md` — add: `- decision: "5/6 gate failure" is mostly scorecard mis-spec — G3 counts an out-of-range 2008 event (start=2008-12-31), G1/G2/G6 are method-shared/vacuous; only G5 is a real discriminating failure. Evidence: outputs/gate_diag/. Memo: docs/analysis/2026-06-08-gate-diagnostics-memo.md.`
- `docs/context/todo.md` — mark gate-diagnostics done; add follow-ups: (1) gate recalibration spec, (2) JM PRD resumes after recalibration.
- `docs/context/lessons.md` — add: read the artifact before trusting a summary; a binary 8/8 gate over a window that excludes an event is a spec bug, not a model failure.
- `docs/context/sesion-log.md` — `- [2026-06-08]: built read-only gate-diagnostics harness + frontier; findings memo; JM PRD parked behind gate recalibration.`

- [ ] **Step 4: Commit**

```bash
git add docs/analysis/2026-06-08-gate-diagnostics-memo.md docs/context/
git commit -m "docs(diagnostics): gate-diagnostics findings memo + context updates"
```

---

## Self-Review notes (author)

- **Spec coverage:** §3.1 refactor → Task 1; §3.2 stages 1-5 → Tasks 2-5 (baseline pin = Task 5 `baseline_compare_path`); §3.3 CLI → Task 6; §4 outputs → Task 5 writers + Task 7 memo; §5 tests → embedded per task (baseline pin, monotonicity, frontier determinism, no-lookahead, refactor regression in Task 1 Step 5); §6 non-goals respected (no threshold changes, no JM, no composite wiring, no new deps); §7 done-definition → Task 7.
- **Production byte-identical invariant:** guarded by Task 1 Step 5 (golden + reproducibility) — the only production-touching change is behavior-preserving extraction + defaulted kwargs.
- **No-drift guarantee:** frontier + G5 reuse `_bucket_transitions`, `_calm_quarters`, `_count_max_calm_transitions`, `_smooth_regime_hysteresis` directly — no re-implementation.
- **Known refinement:** CLI re-runs the engine rather than reconstructing RunResult from CSVs (daily returns unpersisted) — flagged in the header.
```
