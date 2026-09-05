# Regime Attribution (per-asset slope decomposition) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, pure `roro/attribution.py` that decomposes every cut's cross-sectional slope `β̂` **exactly** into per-asset contributions (`β̂ = Σ c_i`, `c_i = h_i·y_i`), a Brinson-style waterfall of `Δβ̂` since the last regime transition, block/quadrant roll-ups, concentration + fragility metrics with alerts, PC1 loadings, and five report figures.

**Architecture:** The WLS slope in `roro/regression.py` is linear in returns, so attribution is an O(n) kernel on the **identical** `DailyPanel` the slope is fitted on (no change to `regression.py`). A pure module computes array-level contributions, wraps them in frozen rows, and an orchestrator builds an `AttributionFrame` (last-date level/delta/rollup/pc1 + full-history concentration). Wiring mirrors the HMM/JM overlays: config fields → engine block → io writers + snapshot block → alerts → report bundle/load/figures/orchestrate. On by default (adds files only); `attribution_enabled=False` keeps every existing artifact byte-identical.

**Tech Stack:** Python 3.12, numpy (<2.1), pandas (<2.3), plotly (<6), Click, pytest + hypothesis. `uv`-managed venv — run everything as `uv run <cmd>` from `C:\Proyectos\RoRo`. Quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest filterwarnings=["error"]`.

**Spec:** `docs/superpowers/specs/2026-09-05-roro-attribution-design.md`. **Branch:** `feat/attribution` (create via `superpowers:using-git-worktrees` at execution time). **Commit style:** conventional (`feat(attrib): …`, `test(attrib): …`, `docs(attrib): …`).

**Spec deviations (deliberate, small):**
1. Anchor date = the day **before** the most recent tercile transition `τ ≤ t` (last day of the previous regime), not "last transition strictly before t". If the transition is today, the waterfall shows what flipped it; if it was 30 days ago, it covers the flip plus the drift since. Still satisfies `anchor < t`.
2. Level rows carry `xbar` and `ybar` (weighted means) so the report can draw the exact WLS line `y = ybar + β̂ (x − xbar)` without refitting.
3. Concentration history is computed from array kernels (no per-row dataclasses) — ~90k panels × O(n) stays under a minute.

---

## File Structure

- **Create** `roro/attribution.py` — pure kernels + orchestrator: `contributions`, `attribute_panel`, `attribute_delta`, `concentration`, `rank_against_prior`, `pc1_loadings`, `find_anchor`, `compute_attribution`. No I/O.
- **Modify** `roro/classify.py` — expose `bucket_label(p, scheme)` (public wrapper over `_bucket`) for the fragility check.
- **Modify** `roro/config.py` — 7 `attribution_*` fields on `EngineConfig`.
- **Modify** `roro/types.py` — `AttributionFrame`, `RunResult.attribution`, `AlertSet.concentration_alerts`.
- **Modify** `roro/engine.py` — attribution block after classification, before alerts.
- **Modify** `roro/io.py` — 5 CSV writers + `snapshot["attribution"]` block; `_write_alerts` gains kind `concentration`.
- **Modify** `roro/alerts.py` — `concentration_alerts` from concentration history × bucket transitions.
- **Create** `roro/report/attribution_figs.py` — 5 pure figure builders (keeps `figures.py` from growing further).
- **Modify** `roro/report/bundle.py`, `load.py`, `orchestrate.py` — optional attribution frames + figure specs.
- **Create** `tests/test_attribution.py`, `tests/report/test_attribution_figs.py`; **extend** `tests/test_config.py`, `tests/test_classify.py`, `tests/test_types.py`, `tests/test_engine.py`, `tests/test_io.py`, `tests/test_alerts.py`, `tests/test_golden.py`, `tests/test_reproducibility.py`, `tests/report/test_bundle.py`, `tests/report/test_e2e.py`; regenerate `tests/golden/2024-Q1/`.
- **Modify** `docs/context/todo.md`, `memory.md`, `results.md`, `sesion-log.md`; **create** `docs/analysis/2026-09-XX-attribution-memo.md` (real-data run).

---

## Task 1: Config fields

**Files:**
- Modify: `roro/config.py:35-49` (after the `jm_*` block, before `fred_api_key`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_attribution_config_defaults_and_roundtrip() -> None:
    cfg = EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o"))
    assert cfg.attribution_enabled is True
    assert cfg.attribution_label_source == "percentile"
    assert cfg.attribution_anchor_lookback_days == 1260
    assert cfg.attribution_fixed_horizon_days == 63
    assert cfg.attribution_top_n == 12
    assert cfg.attribution_top1_alert == 0.5
    assert cfg.attribution_history_global is False
    d = cfg.to_dict()
    for key in (
        "attribution_enabled", "attribution_label_source",
        "attribution_anchor_lookback_days", "attribution_fixed_horizon_days",
        "attribution_top_n", "attribution_top1_alert", "attribution_history_global",
    ):
        assert key in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_attribution_config_defaults_and_roundtrip -v`
Expected: FAIL with `AttributeError: 'EngineConfig' object has no attribute 'attribution_enabled'`

- [ ] **Step 3: Add the fields**

In `roro/config.py`, insert after `jm_random_seed: int = 0` and before `fred_api_key`:

```python
    # Regime attribution (exact per-asset slope decomposition). On by default:
    # adds output files only, never changes existing numeric artifacts.
    attribution_enabled: bool = True
    attribution_label_source: str = "percentile"  # percentile | hmm | jm
    attribution_anchor_lookback_days: int = 1260
    attribution_fixed_horizon_days: int = 63
    attribution_top_n: int = 12
    attribution_top1_alert: float = 0.5
    attribution_history_global: bool = False
```

`load_config`/`to_dict` are generic over `fields()` → nothing else to change.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/config.py tests/test_config.py
git commit -m "feat(attrib): attribution_* config fields on EngineConfig"
```

---

## Task 2: Public `bucket_label` in classify

**Files:**
- Modify: `roro/classify.py:67-73`
- Test: `tests/test_classify.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classify.py`:

```python
from roro.classify import bucket_label
from roro.config import BucketScheme


def test_bucket_label_tercile_and_nan() -> None:
    assert bucket_label(0.1, BucketScheme.TERCILE) == "Risk-off"
    assert bucket_label(0.5, BucketScheme.TERCILE) == "Transitional"
    assert bucket_label(0.9, BucketScheme.TERCILE) == "Risk-on"
    assert bucket_label(float("nan"), BucketScheme.TERCILE) == "Unknown"
    assert bucket_label(0.9, BucketScheme.QUINTILE) == "Q5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classify.py::test_bucket_label_tercile_and_nan -v`
Expected: FAIL with `ImportError: cannot import name 'bucket_label'`

- [ ] **Step 3: Add the public wrapper**

In `roro/classify.py`, directly after `_bucket`:

```python
def bucket_label(p: float, scheme: BucketScheme) -> str:
    """Public bucket label for a percentile under `scheme` (used by attribution fragility)."""
    return _bucket(p, scheme)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_classify.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/classify.py tests/test_classify.py
git commit -m "feat(attrib): expose bucket_label for fragility check"
```

---

## Task 3: Attribution kernel — `contributions` + `attribute_panel`

**Files:**
- Create: `roro/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_attribution.py`:

```python
"""Tests for exact per-asset attribution of the cross-sectional slope."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from roro.attribution import attribute_panel, contributions
from roro.regression import DailyPanel, _wls_slope
from roro.segments import ASSET_EQ, ASSET_FI, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_DATE = pd.Timestamp("2024-01-01")


def _panel(
    vols: FloatArray, rets: FloatArray, weights: FloatArray, *, n_fi: int = 0
) -> DailyPanel:
    n = len(vols)
    series = tuple(
        SeriesId(
            country=f"C{i:02d}",
            segment="DM" if i % 2 == 0 else "EM",
            asset_class=ASSET_FI if i < n_fi else ASSET_EQ,
            mcap=float(weights[i]),
        )
        for i in range(n)
    )
    return DailyPanel(date=_DATE, series=series, returns=rets, vols=vols, weights=weights)


def _finite(n: int, lo: float, hi: float) -> st.SearchStrategy[FloatArray]:
    return hnp.arrays(
        np.float64,
        shape=n,
        elements=st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False),
    )


@given(
    vols=_finite(12, 0.01, 0.9), rets=_finite(12, -0.5, 0.5), w=_finite(12, 0.1, 100.0)
)
@settings(max_examples=50, deadline=None)
def test_contributions_sum_to_wls_slope(vols: FloatArray, rets: FloatArray, w: FloatArray) -> None:
    if np.ptp(vols) < 1e-6:
        return  # degenerate cross-section: no slope to attribute
    p = _panel(vols, rets, w)
    for weighting, wvec in (("cap", w), ("eq", np.ones(12))):
        pc = contributions(p, weighting=weighting, min_n=3)  # type: ignore[arg-type]
        assert pc is not None
        assert abs(pc.c.sum() - _wls_slope(vols, rets, wvec)) < 1e-10
        assert abs(pc.c.sum() - pc.beta) < 1e-12


@given(vols=_finite(12, 0.01, 0.9), w=_finite(12, 0.1, 100.0))
@settings(max_examples=50, deadline=None)
def test_leverage_identities(vols: FloatArray, w: FloatArray) -> None:
    if np.ptp(vols) < 1e-6:
        return
    p = _panel(vols, np.zeros(12), w)
    pc = contributions(p, weighting="cap", min_n=3)
    assert pc is not None
    assert abs(pc.h.sum()) < 1e-10
    assert abs(float(np.sum(pc.h * vols)) - 1.0) < 1e-10


def test_contributions_none_below_min_n_or_degenerate() -> None:
    p = _panel(np.array([0.1, 0.2]), np.array([0.0, 0.1]), np.array([1.0, 1.0]))
    assert contributions(p, weighting="cap", min_n=3) is None
    flat = _panel(np.full(5, 0.2), np.arange(5) / 10.0, np.ones(5))
    assert contributions(flat, weighting="cap", min_n=3) is None


def test_attribute_panel_rows_quadrants_and_order() -> None:
    vols = np.array([0.05, 0.10, 0.30, 0.50])  # xbar (eq) = 0.2375
    rets = np.array([-0.01, 0.02, -0.05, 0.10])
    p = _panel(vols, rets, np.ones(4), n_fi=1)
    rows, beta = attribute_panel(p, weighting="eq", min_n=3)
    assert [r.series for r in rows] == ["C00__FI", "C01__Eq", "C02__Eq", "C03__Eq"]
    assert [r.quadrant for r in rows] == ["LO/-", "LO/+", "HI/-", "HI/+"]
    assert rows[0].block == "DM_FI" and rows[1].block == "EM_Eq"
    assert abs(sum(r.contribution for r in rows) - beta) < 1e-12
    assert abs(sum(r.share for r in rows) - 1.0) < 1e-10
    # sign table: HI/+ and LO/- push beta up; HI/- and LO/+ push it down
    assert rows[3].contribution > 0 and rows[0].contribution > 0
    assert rows[2].contribution < 0 and rows[1].contribution < 0
    assert abs(rows[0].xbar - 0.2375) < 1e-12


def test_attribute_panel_empty_when_degenerate() -> None:
    p = _panel(np.array([0.1, 0.2]), np.array([0.0, 0.1]), np.array([1.0, 1.0]))
    rows, beta = attribute_panel(p, weighting="cap", min_n=3)
    assert rows == [] and np.isnan(beta)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'roro.attribution'`

- [ ] **Step 3: Write the kernel**

Create `roro/attribution.py`:

```python
"""Exact per-asset attribution of the cross-sectional regime slope.

The WLS slope of 3M return on EWMA vol (roro/regression.py) is linear in returns:

    beta = sum_i c_i,   c_i = h_i * y_i,   h_i = w_i (x_i - xbar) / D,
    xbar = sum_i w_i x_i,   D = sum_i w_i (x_i - xbar)^2,   sum_i w_i = 1.

Because sum_i w_i (x_i - xbar) = 0 the intercept drops out, so the decomposition
is exact, additive and residual-free. Everything here consumes the *identical*
DailyPanel the slope is fitted on; regression.py is never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from roro.regression import DailyPanel
from roro.segments import LATAM_COUNTRIES, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Weighting = Literal["cap", "eq"]

WEIGHTINGS: tuple[Weighting, ...] = ("cap", "eq")
_MIN_PANEL: int = 3  # below this a slope + leave-one-out are not meaningful

LEVEL_COLUMNS: tuple[str, ...] = (
    "series", "block", "latam", "vol", "ret3m", "weight", "leverage",
    "contribution", "share", "quadrant", "xbar", "ybar",
)


def series_key(s: SeriesId) -> str:
    """Stable per-asset key: 'Country__Eq' / 'Country__FI' (matches correlation.py suffixes)."""
    return f"{s.country}__{s.asset_class}"


@dataclass(frozen=True)
class PanelContributions:
    """Array-level attribution of one panel under one weighting."""

    w: FloatArray  # normalized weights
    h: FloatArray  # leverage
    c: FloatArray  # contributions, sum == beta
    xbar: float
    ybar: float
    beta: float


@dataclass(frozen=True)
class AttributionRow:
    series: str
    block: str  # DM_Eq | EM_Eq | DM_FI | EM_FI
    latam: bool
    vol: float
    ret3m: float
    weight: float
    leverage: float
    contribution: float
    share: float  # contribution / beta; NaN when beta == 0
    quadrant: str  # HI/+ HI/- LO/+ LO/-
    xbar: float
    ybar: float


def _normalized_weights(panel: DailyPanel, weighting: Weighting) -> FloatArray:
    n = len(panel.returns)
    w = panel.weights if weighting == "cap" else np.ones(n, dtype=np.float64)
    return w / w.sum()


def contributions(
    panel: DailyPanel, *, weighting: Weighting, min_n: int
) -> PanelContributions | None:
    """Exact contributions c_i with sum(c) == WLS slope. None if the panel is degenerate."""
    n = len(panel.returns)
    if n < max(min_n, _MIN_PANEL):
        return None
    w = _normalized_weights(panel, weighting)
    x = panel.vols
    y = panel.returns
    xbar = float(np.sum(w * x))
    d = float(np.sum(w * (x - xbar) ** 2))
    if d <= 0.0:
        return None
    h = w * (x - xbar) / d
    c = h * y
    return PanelContributions(
        w=w, h=h, c=c, xbar=xbar, ybar=float(np.sum(w * y)), beta=float(c.sum())
    )


def _quadrant(vol: float, xbar: float, ret: float) -> str:
    return ("HI" if vol > xbar else "LO") + ("/+" if ret >= 0.0 else "/-")


def attribute_panel(
    panel: DailyPanel, *, weighting: Weighting, min_n: int
) -> tuple[list[AttributionRow], float]:
    """Per-asset rows (sorted by series key) + beta. ([], NaN) when degenerate."""
    pc = contributions(panel, weighting=weighting, min_n=min_n)
    if pc is None:
        return [], float("nan")
    rows = [
        AttributionRow(
            series=series_key(s),
            block=f"{s.segment}_{s.asset_class}",
            latam=s.country in LATAM_COUNTRIES,
            vol=float(panel.vols[i]),
            ret3m=float(panel.returns[i]),
            weight=float(pc.w[i]),
            leverage=float(pc.h[i]),
            contribution=float(pc.c[i]),
            share=float(pc.c[i] / pc.beta) if pc.beta != 0.0 else float("nan"),
            quadrant=_quadrant(float(panel.vols[i]), pc.xbar, float(panel.returns[i])),
            xbar=pc.xbar,
            ybar=pc.ybar,
        )
        for i, s in enumerate(panel.series)
    ]
    rows.sort(key=lambda r: r.series)
    return rows, pc.beta


def rows_to_frame(rows: list[AttributionRow]) -> pd.DataFrame:
    """Rows -> DataFrame with LEVEL_COLUMNS order (empty frame keeps the columns)."""
    return pd.DataFrame([asdict(r) for r in rows], columns=list(LEVEL_COLUMNS))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 5: Quality bars + commit**

Run: `uv run ruff check roro/attribution.py tests/test_attribution.py && uv run mypy roro/attribution.py`
Expected: no errors

```bash
git add roro/attribution.py tests/test_attribution.py
git commit -m "feat(attrib): exact per-asset contributions kernel (sum c_i == WLS slope)"
```

---

## Task 4: `attribute_delta` — Brinson-style waterfall

**Files:**
- Modify: `roro/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attribution.py`:

```python
from roro.attribution import DELTA_COLUMNS, attribute_delta


def test_delta_exact_with_entries_and_exits() -> None:
    vols_a = np.array([0.05, 0.10, 0.30, 0.50, 0.20])
    rets_a = np.array([-0.01, 0.02, -0.05, 0.10, 0.03])
    vols_t = np.array([0.06, 0.12, 0.25, 0.45, 0.33])
    rets_t = np.array([0.00, 0.01, -0.08, 0.15, -0.02])
    w = np.array([5.0, 1.0, 2.0, 3.0, 1.5])
    p_a = _panel(vols_a, rets_a, w)
    p_t = _panel(vols_t, rets_t, w)
    # drop C04 from the anchor panel (entry at t) and C01 from t (exit)
    p_a = DailyPanel(
        date=_DATE, series=p_a.series[:4], returns=rets_a[:4], vols=vols_a[:4], weights=w[:4]
    )
    keep = [0, 2, 3, 4]
    p_t = DailyPanel(
        date=_DATE,
        series=tuple(p_t.series[i] for i in keep),
        returns=rets_t[keep], vols=vols_t[keep], weights=w[keep],
    )
    rows_a, beta_a = attribute_panel(p_a, weighting="cap", min_n=3)
    rows_t, beta_t = attribute_panel(p_t, weighting="cap", min_n=3)
    delta = attribute_delta(rows_t, rows_a, beta_t=beta_t, beta_a=beta_a)
    assert list(delta.columns) == list(DELTA_COLUMNS)
    assert abs(delta["delta_total"].sum() - (beta_t - beta_a)) < 1e-10
    parts = ["effect_return", "effect_position", "effect_interaction", "effect_universe"]
    assert np.allclose(delta[parts].sum(axis=1), delta["delta_total"])
    entry = delta.set_index("series").loc["C04__Eq"]
    exit_ = delta.set_index("series").loc["C01__Eq"]
    assert entry["effect_universe"] != 0.0 and entry["effect_return"] == 0.0
    assert exit_["effect_universe"] != 0.0 and exit_["effect_position"] == 0.0
    both = delta[~delta["series"].isin(["C01__Eq", "C04__Eq"])]
    assert (both["effect_universe"] == 0.0).all()


def test_delta_empty_when_either_beta_nan() -> None:
    p = _panel(np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4), np.ones(4))
    rows, beta = attribute_panel(p, weighting="eq", min_n=3)
    out = attribute_delta(rows, [], beta_t=beta, beta_a=float("nan"))
    assert out.empty and list(out.columns) == list(DELTA_COLUMNS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_attribution.py -k delta -v`
Expected: FAIL with `ImportError: cannot import name 'DELTA_COLUMNS'`

- [ ] **Step 3: Implement**

Append to `roro/attribution.py`:

```python
DELTA_COLUMNS: tuple[str, ...] = (
    "series", "block", "effect_return", "effect_position",
    "effect_interaction", "effect_universe", "delta_total",
)


def attribute_delta(
    rows_t: list[AttributionRow],
    rows_a: list[AttributionRow],
    *,
    beta_t: float,
    beta_a: float,
) -> pd.DataFrame:
    """Brinson-style split of beta_t - beta_a per asset (exact; sums to the delta).

    Common assets: return effect h_a*dy + position effect dh*y_a + interaction dh*dy.
    Entries (only at t) / exits (only at anchor): whole contribution in `effect_universe`.
    Empty frame when either beta is NaN (suppressed panel).
    """
    if not (np.isfinite(beta_t) and np.isfinite(beta_a)):
        return pd.DataFrame(columns=list(DELTA_COLUMNS))
    by_t = {r.series: r for r in rows_t}
    by_a = {r.series: r for r in rows_a}
    out: list[dict[str, object]] = []
    for s in sorted(set(by_t) | set(by_a)):
        rt, ra = by_t.get(s), by_a.get(s)
        eff_ret = eff_pos = eff_int = eff_uni = 0.0
        if rt is not None and ra is not None:
            dh = rt.leverage - ra.leverage
            dy = rt.ret3m - ra.ret3m
            eff_ret = ra.leverage * dy
            eff_pos = dh * ra.ret3m
            eff_int = dh * dy
            block = rt.block
        elif rt is not None:
            eff_uni = rt.contribution
            block = rt.block
        else:
            assert ra is not None
            eff_uni = -ra.contribution
            block = ra.block
        out.append(
            {
                "series": s, "block": block,
                "effect_return": eff_ret, "effect_position": eff_pos,
                "effect_interaction": eff_int, "effect_universe": eff_uni,
                "delta_total": eff_ret + eff_pos + eff_int + eff_uni,
            }
        )
    return pd.DataFrame(out, columns=list(DELTA_COLUMNS))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/attribution.py tests/test_attribution.py
git commit -m "feat(attrib): exact delta-beta waterfall (return/position/interaction/universe)"
```

---

## Task 5: Concentration, leave-one-out beta, fragility rank

**Files:**
- Modify: `roro/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attribution.py`:

```python
from roro.attribution import ConcentrationRow, concentration, rank_against_prior


def test_concentration_bounds_and_top1() -> None:
    vols = np.array([0.05, 0.10, 0.30, 0.90])
    rets = np.array([0.00, 0.01, -0.02, 0.40])
    p = _panel(vols, rets, np.ones(4))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    row = concentration(p, pc, weighting="eq", min_n=3)
    assert isinstance(row, ConcentrationRow)
    assert row.top1_series == "C03__Eq"
    assert 0.25 <= row.hhi <= 1.0
    assert row.top1_share <= row.top5_share <= 1.0 + 1e-12
    # leave-one-out slope drops the dominant asset
    assert np.isfinite(row.beta_ex_top1)
    assert abs(row.beta_ex_top1) < abs(pc.beta)


def test_concentration_beta_ex_top1_nan_when_remaining_below_min_n() -> None:
    p = _panel(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.1, 0.5]), np.ones(3))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    row = concentration(p, pc, weighting="eq", min_n=3)
    assert np.isnan(row.beta_ex_top1)


def test_rank_against_prior_mirrors_classifier() -> None:
    window = np.array([0.1, 0.3, 0.2, np.nan, 0.5])  # last value = today
    # classifier: (count(window <= today) - 1) / (len - 1) = (4 - 1) / 4 = 0.75
    assert abs(rank_against_prior(window, 0.5) - 0.75) < 1e-12
    assert abs(rank_against_prior(window, 0.0) - 0.0) < 1e-12
    assert np.isnan(rank_against_prior(np.array([0.5]), 0.5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_attribution.py -k "concentration or rank" -v`
Expected: FAIL with `ImportError: cannot import name 'ConcentrationRow'`

- [ ] **Step 3: Implement**

Append to `roro/attribution.py`:

```python
_TOP_N: int = 5


@dataclass(frozen=True)
class ConcentrationRow:
    hhi: float
    top1_series: str
    top1_share: float
    top5_share: float
    beta_ex_top1: float


_EMPTY_CONCENTRATION = ConcentrationRow(
    hhi=float("nan"), top1_series="", top1_share=float("nan"),
    top5_share=float("nan"), beta_ex_top1=float("nan"),
)


def _panel_without(panel: DailyPanel, index: int) -> DailyPanel:
    keep = [i for i in range(len(panel.series)) if i != index]
    return DailyPanel(
        date=panel.date,
        series=tuple(panel.series[i] for i in keep),
        returns=panel.returns[keep],
        vols=panel.vols[keep],
        weights=panel.weights[keep],
    )


def concentration(
    panel: DailyPanel, pc: PanelContributions, *, weighting: Weighting, min_n: int
) -> ConcentrationRow:
    """HHI of |c|, top-1/top-5 shares, and the slope re-estimated without the top-1 asset.

    Ties in |c| resolve by panel order (stable argsort) -> deterministic.
    """
    abs_c = np.abs(pc.c)
    total = float(abs_c.sum())
    if total <= 0.0:
        return _EMPTY_CONCENTRATION
    a = abs_c / total
    order = np.argsort(-abs_c, kind="stable")
    top1 = int(order[0])
    ex = contributions(_panel_without(panel, top1), weighting=weighting, min_n=min_n)
    return ConcentrationRow(
        hhi=float(np.sum(a**2)),
        top1_series=series_key(panel.series[top1]),
        top1_share=float(a[top1]),
        top5_share=float(a[order[:_TOP_N]].sum()),
        beta_ex_top1=ex.beta if ex is not None else float("nan"),
    )


def rank_against_prior(window: FloatArray, value: float) -> float:
    """Percentile of `value` against the window's prior values (window[-1] is today).

    Mirrors classify.rolling_percentile: count(prior <= value) / (len(window) - 1),
    NaN comparisons count as False. NaN when there is no prior history.
    """
    prior = window[:-1]
    if prior.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        hits = int(np.sum(prior <= value))
    return float(hits) / float(prior.size)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/attribution.py tests/test_attribution.py
git commit -m "feat(attrib): concentration metrics, leave-one-out slope, fragility rank"
```

---

## Task 6: PC1 loadings + decoupling

**Files:**
- Modify: `roro/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attribution.py`:

```python
from roro.attribution import PC1_COLUMNS, pc1_loadings


def _returns_window(seed: int = 0, n_obs: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_obs)
    data = {
        "A__Eq": common + rng.normal(0, 0.002, n_obs),
        "B__Eq": common + rng.normal(0, 0.002, n_obs),
        "C__FI": -0.2 * common + rng.normal(0, 0.004, n_obs),
        "D__FI": rng.normal(0, 0.006, n_obs),  # independent -> decoupled
    }
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n_obs))


def test_pc1_loadings_identities() -> None:
    out = pc1_loadings(_returns_window())
    assert list(out.columns) == list(PC1_COLUMNS)
    assert abs(out["pc1_load_sq"].sum() - 1.0) < 1e-10
    assert abs(out["var_share"].sum() - 1.0) < 1e-10
    assert abs(out["decoupling"].sum()) < 1e-10
    assert (out["row_mean_corr"].abs() <= 1.0).all()
    assert list(out["series"]) == sorted(out["series"])


def test_pc1_loadings_sign_flip_invariant_and_decoupled_asset() -> None:
    win = _returns_window()
    out = pc1_loadings(win)
    flipped = pc1_loadings(-win)  # eigenvectors may flip sign; squares must not
    assert np.allclose(out["pc1_load_sq"], flipped["pc1_load_sq"])
    by = out.set_index("series")
    assert by.loc["D__FI", "decoupling"] > by.loc["A__Eq", "decoupling"]


def test_pc1_loadings_empty_on_short_or_nan_window() -> None:
    win = _returns_window(n_obs=2)
    assert pc1_loadings(win).empty
    win = _returns_window()
    win.iloc[5, 0] = np.nan
    assert pc1_loadings(win).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_attribution.py -k pc1 -v`
Expected: FAIL with `ImportError: cannot import name 'PC1_COLUMNS'`

- [ ] **Step 3: Implement**

Append to `roro/attribution.py`:

```python
PC1_COLUMNS: tuple[str, ...] = (
    "series", "pc1_load_sq", "var_share", "decoupling", "row_mean_corr",
)
_MIN_OBS_PC1: int = 3
_MIN_COLS_PC1: int = 2


def pc1_loadings(window_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset PC1 loading^2 (sums to 1), variance share (sums to 1), decoupling, row-mean corr.

    Same window/covariance as roro/correlation.py. Columns that are entirely NaN are
    dropped; any remaining NaN makes the covariance undefined -> empty frame.
    """
    empty = pd.DataFrame(columns=list(PC1_COLUMNS))
    arr = window_returns.to_numpy(dtype=np.float64)
    mask = ~np.all(np.isnan(arr), axis=0)
    arr = arr[:, mask]
    cols = [c for c, m in zip(window_returns.columns, mask, strict=True) if m]
    if arr.shape[0] < _MIN_OBS_PC1 or arr.shape[1] < _MIN_COLS_PC1:
        return empty
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.cov(arr, rowvar=False)
        corr = np.corrcoef(arr, rowvar=False)
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(corr))):
        return empty
    trace = float(np.trace(cov))
    if trace <= 0.0:
        return empty
    _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues; last column = PC1
    v1 = eigvecs[:, -1]
    load_sq = v1**2
    var_share = np.diag(cov) / trace
    n = corr.shape[0]
    row_mean = (corr.sum(axis=1) - 1.0) / float(n - 1)
    df = pd.DataFrame(
        {
            "series": [str(c) for c in cols],
            "pc1_load_sq": load_sq,
            "var_share": var_share,
            "decoupling": var_share - load_sq,
            "row_mean_corr": row_mean,
        },
        columns=list(PC1_COLUMNS),
    )
    return df.sort_values("series").reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/attribution.py tests/test_attribution.py
git commit -m "feat(attrib): PC1 loadings, variance share and decoupling score"
```

---

## Task 7: `find_anchor`

**Files:**
- Modify: `roro/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attribution.py`:

```python
from roro.attribution import find_anchor


def _labels() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=12)
    vals = ["Unknown", "Unknown", "Risk-on", "Risk-on", "Risk-on", "Transitional",
            "Transitional", "Risk-off", "Risk-off", "Risk-off", "Risk-off", "Risk-off"]
    return pd.Series(vals, index=idx)


def test_find_anchor_is_day_before_last_transition() -> None:
    lab = _labels()
    t = lab.index[-1]
    a = find_anchor(lab, t, max_lookback_days=1260)
    assert a == lab.index[6]  # transition to Risk-off on idx[7]; anchor = idx[6]
    assert a < t


def test_find_anchor_transition_today_gives_yesterday() -> None:
    lab = _labels()
    t = lab.index[7]
    assert find_anchor(lab, t, max_lookback_days=1260) == lab.index[6]


def test_find_anchor_none_without_transition_or_beyond_lookback() -> None:
    lab = _labels()
    assert find_anchor(lab, lab.index[4], max_lookback_days=1260) is None  # Unknown->Risk-on ignored
    assert find_anchor(lab, lab.index[-1], max_lookback_days=2) is None


def test_find_anchor_ignores_future_rows() -> None:
    lab = _labels()
    t = lab.index[8]
    extended = pd.concat([lab, pd.Series(["Risk-on"], index=[lab.index[-1] + pd.offsets.BDay()])])
    assert find_anchor(lab, t, max_lookback_days=1260) == find_anchor(
        extended, t, max_lookback_days=1260
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_attribution.py -k anchor -v`
Expected: FAIL with `ImportError: cannot import name 'find_anchor'`

- [ ] **Step 3: Implement**

Append to `roro/attribution.py`:

```python
_UNKNOWN_LABEL: str = "Unknown"


def find_anchor(
    labels: pd.Series, t: pd.Timestamp, *, max_lookback_days: int
) -> pd.Timestamp | None:
    """Anchor for the delta waterfall: the day BEFORE the most recent label transition <= t.

    Uses only rows <= t (causal). Transitions from NaN/'Unknown' do not count. The
    transition must lie within the trailing `max_lookback_days` rows. None if no anchor.
    """
    hist = labels.loc[:t].iloc[-(max_lookback_days + 1):]
    known = hist.notna() & (hist != _UNKNOWN_LABEL)
    prev = hist.shift(1)
    prev_known = prev.notna() & (prev != _UNKNOWN_LABEL)
    is_transition = known & prev_known & (hist != prev)
    if not bool(is_transition.any()):
        return None
    pos = int(np.flatnonzero(is_transition.to_numpy())[-1])
    if pos == 0:
        return None
    return pd.Timestamp(hist.index[pos - 1])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/attribution.py tests/test_attribution.py
git commit -m "feat(attrib): causal anchor = day before last regime transition"
```

---

## Task 8: `AttributionFrame` type + `compute_attribution` orchestrator

**Files:**
- Modify: `roro/types.py` (add `AttributionFrame`; extend `RunResult`, `AlertSet`)
- Modify: `roro/attribution.py`
- Test: `tests/test_types.py`, `tests/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_types.py`:

```python
def test_attribution_frame_fields_and_runresult_default() -> None:
    from roro.types import AlertSet, AttributionFrame, RunResult

    names = {f.name for f in fields(AttributionFrame)}
    assert names == {
        "level", "delta", "rollup", "concentration", "pc1", "anchors", "history_global",
    }
    assert RunResult.__dataclass_fields__["attribution"].default is None
    assert "concentration_alerts" in {f.name for f in fields(AlertSet)}
```

(`fields` is already imported in `tests/test_types.py` via `from dataclasses import fields`; if not, add it.)

Append to `tests/test_attribution.py`:

```python
from roro.attribution import compute_attribution
from roro.config import EngineConfig
from roro.io import load_panel, load_prices
from roro.regression import compute_beta_by_segment
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import partition
from roro.classify import classify
from pathlib import Path


def _attribution_inputs(tiny_xlsx: Path, cfg: EngineConfig) -> dict[str, Any]:
    universe = load_panel(tiny_xlsx)
    prices = load_prices(tiny_xlsx)
    eq_ret = total_return_3m(prices.equity_lc, window_days=cfg.return_window_days)
    fi_ret = total_return_3m(prices.fi_lc, window_days=cfg.return_window_days)
    eq_daily = daily_log_returns(prices.equity_lc)
    fi_daily = daily_log_returns(prices.fi_lc)
    eq_vol = ewma_vol(eq_daily, halflife=cfg.ewma_halflife_days)
    fi_vol = ewma_vol(fi_daily, halflife=cfg.ewma_halflife_days)
    cuts = partition(universe)
    dates = pd.DatetimeIndex(prices.equity_lc.index)
    beta = compute_beta_by_segment(
        dates=dates, cuts=cuts, equity_returns_3m=eq_ret, fi_returns_3m=fi_ret,
        equity_vol=eq_vol, fi_vol=fi_vol, min_n=cfg.min_n_per_cut,
    )
    regime = classify(
        beta, bucket_scheme=cfg.bucket_scheme,
        percentile_window_days=cfg.percentile_window_years * 252,
        direction_lookback_days=cfg.direction_lookback_days,
        bootstrap_min_days=cfg.bootstrap_min_days, thin_cuts=frozenset({"LatAm"}),
    )
    return {
        "dates": dates, "cuts": cuts, "equity_returns_3m": eq_ret, "fi_returns_3m": fi_ret,
        "equity_vol": eq_vol, "fi_vol": fi_vol, "daily_log_returns_eq": eq_daily,
        "daily_log_returns_fi": fi_daily, "beta": beta, "regime": regime,
        "anchor_labels": regime.tercile,
    }


def _tiny_cfg(tiny_xlsx: Path, tmp_path: Path) -> EngineConfig:
    return EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10,
    )


def test_compute_attribution_shapes_and_exactness(tiny_xlsx: Path, tmp_path: Path) -> None:
    cfg = _tiny_cfg(tiny_xlsx, tmp_path)
    inputs = _attribution_inputs(tiny_xlsx, cfg)
    af = compute_attribution(cfg=cfg, **inputs)
    # level: last date, all non-degenerate cuts x both weightings
    assert set(af.level["weighting"]) == {"cap", "eq"}
    assert "global" in set(af.level["cut"])
    last = inputs["dates"][-1]
    assert (af.level["date"] == last).all()
    g = af.level[(af.level["cut"] == "global") & (af.level["weighting"] == "cap")]
    beta_cap = inputs["beta"].by_segment["global"].cap_wtd.loc[last, "beta"]
    assert abs(g["contribution"].sum() - beta_cap) < 1e-10
    # concentration: full history, one row per date x cut x weighting that has a slope
    assert {"date", "cut", "weighting", "hhi", "top1_series", "top1_share", "top5_share",
            "beta", "beta_ex_top1", "fragile_flag"} <= set(af.concentration.columns)
    conc_g = af.concentration[(af.concentration["cut"] == "global")
                              & (af.concentration["weighting"] == "cap")]
    assert len(conc_g) == inputs["beta"].by_segment["global"].cap_wtd["beta"].notna().sum()
    assert conc_g["hhi"].between(0.0, 1.0).all()
    # rollups exact within the global regression
    rb = af.rollup[(af.rollup["cut"] == "global") & (af.rollup["weighting"] == "cap")
                   & (af.rollup["group_kind"] == "block")]
    assert abs(rb["contribution_sum"].sum() - beta_cap) < 1e-10
    # delta: fixed horizon exists for global; sums to beta_t - beta_a
    d = af.delta[(af.delta["cut"] == "global") & (af.delta["weighting"] == "cap")
                 & (af.delta["horizon"] == "fixed")]
    assert not d.empty
    assert abs(d["delta_total"].sum() - (d["beta_t"].iloc[0] - d["beta_anchor"].iloc[0])) < 1e-10
    # pc1 for global: loadings sum to 1
    p = af.pc1[af.pc1["cut"] == "global"]
    assert abs(p["pc1_load_sq"].sum() - 1.0) < 1e-10
    assert "global" in af.anchors
    assert af.history_global is None


def test_compute_attribution_history_global_flag(tiny_xlsx: Path, tmp_path: Path) -> None:
    cfg = replace(_tiny_cfg(tiny_xlsx, tmp_path), attribution_history_global=True)
    af = compute_attribution(cfg=cfg, **_attribution_inputs(tiny_xlsx, cfg))
    assert af.history_global is not None
    assert af.history_global.index.name == "date"
    row_sums = af.history_global.sum(axis=1, min_count=1)
    beta = _attribution_inputs(tiny_xlsx, cfg)["beta"].by_segment["global"].cap_wtd["beta"]
    common = row_sums.dropna().index
    assert np.allclose(row_sums.loc[common], beta.loc[common], atol=1e-10)
```

Add `from dataclasses import replace` to the test imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py tests/test_attribution.py -k "attribution_frame or compute_attribution" -v`
Expected: FAIL with `ImportError: cannot import name 'AttributionFrame'`

- [ ] **Step 3: Add the type**

In `roro/types.py`, after `CorrelationFrame`:

```python
@dataclass(frozen=True)
class AttributionFrame:
    """Per-asset attribution of the regime slope (see roro/attribution.py).

    level / delta / rollup / pc1 are LAST-DATE snapshots (long format, all cuts x
    weightings); concentration is FULL HISTORY; history_global is the optional
    date x series matrix of contributions for the global cut (cap-weighted).
    """

    level: pd.DataFrame
    delta: pd.DataFrame
    rollup: pd.DataFrame
    concentration: pd.DataFrame
    pc1: pd.DataFrame
    anchors: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    history_global: pd.DataFrame | None = None
```

Extend `AlertSet` (after `jm_bucket_transitions`):

```python
    concentration_alerts: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "segment", "weighting", "top1_series", "top1_share",
                     "hhi", "fragile_flag", "trigger"]
        )
    )
```

Extend `RunResult` (after `regime_jm`):

```python
    attribution: AttributionFrame | None = None
```

- [ ] **Step 4: Write the orchestrator**

Append to `roro/attribution.py` (add `from functools import partial`, `from roro.classify import bucket_label`, `from roro.config import EngineConfig`, `from roro.regression import daily_panel`, `from roro.segments import ASSET_EQ, ASSET_FI`, `from roro.types import AttributionFrame, BetaBySegment, RegimeFrame` to the imports; `EngineConfig` and `RegimeFrame` are runtime imports here, not TYPE_CHECKING — no cycle: `roro.types` only imports `EngineConfig` under TYPE_CHECKING and `roro.classify` imports `roro.types`, not `roro.attribution`):

```python
CONCENTRATION_COLUMNS: tuple[str, ...] = (
    "date", "cut", "weighting", "n", "beta", "hhi", "top1_series", "top1_share",
    "top5_share", "beta_ex_top1", "pct_today", "pct_ex_top1", "fragile_flag",
)
ROLLUP_COLUMNS: tuple[str, ...] = (
    "date", "cut", "weighting", "group_kind", "group", "contribution_sum", "n",
)
DELTA_META_COLUMNS: tuple[str, ...] = (
    "date", "cut", "weighting", "horizon", "anchor_date", "label_anchor", "label_t",
    "beta_anchor", "beta_t",
)


def _panel_for(
    date: pd.Timestamp,
    series: list[SeriesId],
    *,
    equity_returns_3m: pd.DataFrame,
    fi_returns_3m: pd.DataFrame,
    equity_vol: pd.DataFrame,
    fi_vol: pd.DataFrame,
) -> DailyPanel:
    return daily_panel(
        date=date, series=series, equity_returns=equity_returns_3m,
        fi_returns=fi_returns_3m, equity_vol=equity_vol, fi_vol=fi_vol,
    )


def _rollup(level: pd.DataFrame) -> pd.DataFrame:
    """Block / quadrant / LatAm sums per (date, cut, weighting) from level rows."""
    if level.empty:
        return pd.DataFrame(columns=list(ROLLUP_COLUMNS))
    parts: list[pd.DataFrame] = []
    for kind, col in (("block", "block"), ("quadrant", "quadrant")):
        g = level.groupby(["date", "cut", "weighting", col], sort=True)["contribution"]
        df = g.agg(contribution_sum="sum", n="count").reset_index()
        df = df.rename(columns={col: "group"})
        df.insert(3, "group_kind", kind)
        parts.append(df)
    lat = level[level["latam"]]
    if not lat.empty:
        g = lat.groupby(["date", "cut", "weighting"], sort=True)["contribution"]
        df = g.agg(contribution_sum="sum", n="count").reset_index()
        df.insert(3, "group_kind", "latam")
        df.insert(4, "group", "LatAm")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)[list(ROLLUP_COLUMNS)]
    return out.sort_values(["cut", "weighting", "group_kind", "group"]).reset_index(drop=True)


def _window_returns(
    series: list[SeriesId],
    *,
    daily_log_returns_eq: pd.DataFrame,
    daily_log_returns_fi: pd.DataFrame,
    end: pd.Timestamp,
    window: int,
) -> pd.DataFrame:
    """Same merge as correlation.compute_correlation_panel, sliced to the trailing window."""
    cols_eq = [s.country for s in series if s.asset_class == ASSET_EQ]
    cols_fi = [s.country for s in series if s.asset_class == ASSET_FI]
    eq = daily_log_returns_eq[[c for c in cols_eq if c in daily_log_returns_eq.columns]]
    fi = daily_log_returns_fi[[c for c in cols_fi if c in daily_log_returns_fi.columns]]
    merged = pd.concat([eq.add_suffix("__Eq"), fi.add_suffix("__FI")], axis=1)
    merged = merged.loc[:end]
    return merged.iloc[-window:]


def compute_attribution(
    *,
    cfg: EngineConfig,
    dates: pd.DatetimeIndex,
    cuts: dict[str, list[SeriesId]],
    equity_returns_3m: pd.DataFrame,
    fi_returns_3m: pd.DataFrame,
    equity_vol: pd.DataFrame,
    fi_vol: pd.DataFrame,
    daily_log_returns_eq: pd.DataFrame,
    daily_log_returns_fi: pd.DataFrame,
    beta: BetaBySegment,
    regime: RegimeFrame,
    anchor_labels: pd.DataFrame,
) -> AttributionFrame:
    """Build the AttributionFrame: last-date level/delta/rollup/pc1 + full-history concentration."""
    min_n = cfg.min_n_per_cut
    pct_window = cfg.percentile_window_years * 252
    last = pd.Timestamp(dates[-1])
    level_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    conc_rows: list[dict[str, object]] = []
    pc1_parts: list[pd.DataFrame] = []
    anchors: dict[str, pd.Timestamp | None] = {}
    history_global: pd.DataFrame | None = None

    for cut, series in cuts.items():
        beta_cap = beta.by_segment[cut].cap_wtd["beta"]
        beta_values = beta_cap.reindex(dates).to_numpy(dtype=np.float64)
        pct_today_all = regime.percentile_5y[cut].reindex(dates).to_numpy(dtype=np.float64)

        # functools.partial (not a closure) so ruff B023 does not flag the loop variable.
        panel_at = partial(
            _panel_for, series=series, equity_returns_3m=equity_returns_3m,
            fi_returns_3m=fi_returns_3m, equity_vol=equity_vol, fi_vol=fi_vol,
        )

        # --- full-history concentration (both weightings; fragility on cap only) ---
        hist_c: dict[pd.Timestamp, dict[str, float]] = {}
        for pos, d in enumerate(dates):
            panel = panel_at(pd.Timestamp(d))
            for weighting in WEIGHTINGS:
                pc = contributions(panel, weighting=weighting, min_n=min_n)
                if pc is None:
                    continue
                row = concentration(panel, pc, weighting=weighting, min_n=min_n)
                pct_today = float(pct_today_all[pos])
                pct_ex = float("nan")
                fragile = False
                if weighting == "cap" and np.isfinite(row.beta_ex_top1) and np.isfinite(pct_today):
                    window = beta_values[max(0, pos - pct_window + 1) : pos + 1]
                    pct_ex = rank_against_prior(window, row.beta_ex_top1)
                    fragile = bucket_label(pct_ex, cfg.bucket_scheme) != bucket_label(
                        pct_today, cfg.bucket_scheme
                    )
                conc_rows.append(
                    {
                        "date": pd.Timestamp(d), "cut": cut, "weighting": weighting,
                        "n": len(panel.series), "beta": pc.beta, "hhi": row.hhi,
                        "top1_series": row.top1_series, "top1_share": row.top1_share,
                        "top5_share": row.top5_share, "beta_ex_top1": row.beta_ex_top1,
                        "pct_today": pct_today, "pct_ex_top1": pct_ex, "fragile_flag": fragile,
                    }
                )
                if weighting == "cap" and cut == "global" and cfg.attribution_history_global:
                    hist_c[pd.Timestamp(d)] = {
                        series_key(s): float(pc.c[i]) for i, s in enumerate(panel.series)
                    }
        if cut == "global" and cfg.attribution_history_global:
            history_global = pd.DataFrame.from_dict(hist_c, orient="index").sort_index()
            history_global = history_global.reindex(sorted(history_global.columns), axis=1)
            history_global.index.name = "date"

        # --- last-date level + delta ---
        panel_t = panel_at(last)
        labels = anchor_labels[cut] if cut in anchor_labels.columns else pd.Series(dtype=object)
        anchor = (
            find_anchor(labels, last, max_lookback_days=cfg.attribution_anchor_lookback_days)
            if not labels.empty
            else None
        )
        anchors[cut] = anchor
        fixed_pos = len(dates) - 1 - cfg.attribution_fixed_horizon_days
        fixed_date = pd.Timestamp(dates[fixed_pos]) if fixed_pos >= 0 else None
        for weighting in WEIGHTINGS:
            rows_t, beta_t = attribute_panel(panel_t, weighting=weighting, min_n=min_n)
            if rows_t:
                lv = rows_to_frame(rows_t)
                lv.insert(0, "weighting", weighting)
                lv.insert(0, "cut", cut)
                lv.insert(0, "date", last)
                level_parts.append(lv)
            for horizon, a_date in (("anchor", anchor), ("fixed", fixed_date)):
                if a_date is None:
                    continue
                rows_a, beta_a = attribute_panel(panel_at(a_date), weighting=weighting, min_n=min_n)
                d_df = attribute_delta(rows_t, rows_a, beta_t=beta_t, beta_a=beta_a)
                if d_df.empty:
                    continue
                meta = {
                    "date": last, "cut": cut, "weighting": weighting, "horizon": horizon,
                    "anchor_date": a_date,
                    "label_anchor": labels.get(a_date, _UNKNOWN_LABEL) if not labels.empty else _UNKNOWN_LABEL,
                    "label_t": labels.get(last, _UNKNOWN_LABEL) if not labels.empty else _UNKNOWN_LABEL,
                    "beta_anchor": beta_a, "beta_t": beta_t,
                }
                for i, (k, v) in enumerate(meta.items()):
                    d_df.insert(i, k, v)
                delta_parts.append(d_df)

        # --- PC1 loadings on the trailing return window ---
        win = _window_returns(
            series, daily_log_returns_eq=daily_log_returns_eq,
            daily_log_returns_fi=daily_log_returns_fi, end=last, window=cfg.return_window_days,
        )
        p1 = pc1_loadings(win)
        if not p1.empty:
            p1.insert(0, "cut", cut)
            p1.insert(0, "date", last)
            pc1_parts.append(p1)

    level = (
        pd.concat(level_parts, ignore_index=True)
        if level_parts
        else pd.DataFrame(columns=["date", "cut", "weighting", *LEVEL_COLUMNS])
    )
    delta = (
        pd.concat(delta_parts, ignore_index=True)
        if delta_parts
        else pd.DataFrame(columns=[*DELTA_META_COLUMNS, *DELTA_COLUMNS])
    )
    conc = pd.DataFrame(conc_rows, columns=list(CONCENTRATION_COLUMNS))
    pc1 = (
        pd.concat(pc1_parts, ignore_index=True)
        if pc1_parts
        else pd.DataFrame(columns=["date", "cut", *PC1_COLUMNS])
    )
    return AttributionFrame(
        level=level, delta=delta, rollup=_rollup(level), concentration=conc, pc1=pc1,
        anchors=anchors, history_global=history_global,
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_types.py tests/test_attribution.py -v`
Expected: all PASS

- [ ] **Step 6: Quality bars + commit**

Run: `uv run ruff check roro tests && uv run mypy roro`
Expected: clean. (If ruff flags `PLR0912`/`PLR0915` on `compute_attribution`, split the history loop into `_concentration_history(...)` and the last-date block into `_last_date_snapshot(...)` returning the parts lists — same logic, no behavior change.)

```bash
git add roro/types.py roro/attribution.py tests/test_types.py tests/test_attribution.py
git commit -m "feat(attrib): AttributionFrame + compute_attribution orchestrator"
```

---

## Task 9: Engine wiring

**Files:**
- Modify: `roro/engine.py:96-113` (after the JM block, before correlation) and the `RunResult(...)` call
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
def test_engine_populates_attribution_by_default(tiny_xlsx: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10,
    )
    result = run(cfg, fred_client=_seeded_fred(), run_date="2024-12-31",
                 as_of_data_date="2024-12-31")
    assert result.attribution is not None
    assert "global" in set(result.attribution.level["cut"])
    assert "global" in result.attribution.anchors


def test_engine_attribution_none_when_disabled(tiny_xlsx: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10, attribution_enabled=False,
    )
    result = run(cfg, fred_client=_seeded_fred(), run_date="2024-12-31",
                 as_of_data_date="2024-12-31")
    assert result.attribution is None


def test_engine_attribution_label_source_falls_back_with_warning(
    tiny_xlsx: Path, tmp_path: Path
) -> None:
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10, attribution_label_source="jm",
    )
    result = run(cfg, fred_client=_seeded_fred(), run_date="2024-12-31",
                 as_of_data_date="2024-12-31")
    assert result.attribution is not None
    assert any("attribution_label_source" in w for w in result.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -k attribution -v`
Expected: FAIL with `AttributeError: 'RunResult' object has no attribute 'attribution'` (or `TypeError` on the unknown kwarg)

- [ ] **Step 3: Wire the engine**

In `roro/engine.py` add the import `from roro.attribution import compute_attribution` and, immediately after the `regime_jm = (...)` block (step 5c), insert:

```python
    # 5d) Exact per-asset attribution of the slope (on by default; adds artifacts only).
    attribution = None
    if cfg.attribution_enabled:
        anchor_labels, source_warning = _anchor_labels(
            cfg.attribution_label_source, regime=regime, regime_hmm=regime_hmm, regime_jm=regime_jm
        )
        if source_warning:
            warnings.append(source_warning)
        attribution = compute_attribution(
            cfg=cfg,
            dates=eq_index,
            cuts=cuts,
            equity_returns_3m=eq_ret,
            fi_returns_3m=fi_ret,
            equity_vol=eq_vol,
            fi_vol=fi_vol,
            daily_log_returns_eq=eq_daily,
            daily_log_returns_fi=fi_daily,
            beta=beta,
            regime=regime,
            anchor_labels=anchor_labels,
        )
```

Add at module bottom (after `run`):

```python
def _anchor_labels(
    source: str,
    *,
    regime: RegimeFrame,
    regime_hmm: HmmRegimeFrame | None,
    regime_jm: JmRegimeFrame | None,
) -> tuple[pd.DataFrame, str | None]:
    """Label frame (date x cut) that anchors the attribution waterfall.

    Falls back to the percentile tercile with a warning when the requested overlay
    is not enabled or the source name is unknown.
    """
    if source == "hmm" and regime_hmm is not None:
        return regime_hmm.label, None
    if source == "jm" and regime_jm is not None:
        return regime_jm.label, None
    if source == "percentile":
        return regime.tercile, None
    return (
        regime.tercile,
        f"attribution_label_source='{source}' unavailable; anchors use percentile tercile",
    )
```

Extend the `roro.types` import in `engine.py` with `HmmRegimeFrame, JmRegimeFrame, RegimeFrame`. Pass `attribution=attribution` into `detect_alerts(...)` (Task 10 adds the parameter — until then keep it out) and into `RunResult(...)` after `regime_jm=regime_jm,`:

```python
        attribution=attribution,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all PASS (slow HMM test can be deselected with `-m "not slow"`)

- [ ] **Step 5: Commit**

```bash
git add roro/engine.py tests/test_engine.py
git commit -m "feat(attrib): engine wiring + anchor label source fallback"
```

---

## Task 10: Alerts — concentration

**Files:**
- Modify: `roro/alerts.py`
- Modify: `roro/engine.py` (pass `attribution=attribution` into `detect_alerts`)
- Test: `tests/test_alerts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alerts.py`:

```python
from roro.types import AttributionFrame


def _empty_corr_val() -> tuple[CorrelationFrame, ValidationFrame]:
    return (
        CorrelationFrame(avg_pairwise_3m=pd.DataFrame(), pc1_variance_share=pd.DataFrame()),
        ValidationFrame(rolling_corr_60d=pd.DataFrame(), internal_consistency=pd.DataFrame(),
                        correlation_alerts=pd.DataFrame()),
    )


def test_concentration_alerts_on_transition_day_and_fragile() -> None:
    rf = _regime()  # global flips Transitional->Risk-off on idx[8]; DM_Eq on idx[5]
    idx = rf.tercile.index
    conc = pd.DataFrame(
        [
            # global, transition day, concentrated -> alert (transition_day)
            {"date": idx[8], "cut": "global", "weighting": "cap", "top1_series": "A__Eq",
             "top1_share": 0.7, "hhi": 0.5, "fragile_flag": False},
            # global, calm day, concentrated but not fragile -> no alert
            {"date": idx[3], "cut": "global", "weighting": "cap", "top1_series": "A__Eq",
             "top1_share": 0.7, "hhi": 0.5, "fragile_flag": False},
            # DM_Eq, calm day, fragile -> alert (fragile)
            {"date": idx[2], "cut": "DM_Eq", "weighting": "cap", "top1_series": "B__FI",
             "top1_share": 0.3, "hhi": 0.2, "fragile_flag": True},
            # eq weighting never alerts
            {"date": idx[8], "cut": "global", "weighting": "eq", "top1_series": "A__Eq",
             "top1_share": 0.9, "hhi": 0.8, "fragile_flag": True},
        ]
    )
    af = AttributionFrame(level=pd.DataFrame(), delta=pd.DataFrame(), rollup=pd.DataFrame(),
                          concentration=conc, pc1=pd.DataFrame())
    corr, val = _empty_corr_val()
    out = detect_alerts(regime=rf, correlation=corr, validation=val, attribution=af,
                        top1_alert=0.5)
    ca = out.concentration_alerts
    assert list(ca.columns) == ["date", "segment", "weighting", "top1_series", "top1_share",
                                "hhi", "fragile_flag", "trigger"]
    assert set(zip(ca["segment"], ca["trigger"], strict=True)) == {
        ("global", "transition_day"), ("DM_Eq", "fragile"),
    }


def test_concentration_alerts_empty_without_attribution() -> None:
    corr, val = _empty_corr_val()
    out = detect_alerts(regime=_regime(), correlation=corr, validation=val)
    assert out.concentration_alerts.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_alerts.py -k concentration -v`
Expected: FAIL with `TypeError: detect_alerts() got an unexpected keyword argument 'attribution'`

- [ ] **Step 3: Implement**

In `roro/alerts.py`: import `AttributionFrame` from `roro.types`; add constants and the new function; extend `detect_alerts`:

```python
_CONCENTRATION_COLUMNS: list[str] = [
    "date", "segment", "weighting", "top1_series", "top1_share", "hhi", "fragile_flag", "trigger",
]
_DEFAULT_TOP1_ALERT: float = 0.5


def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
    regime_hmm: HmmRegimeFrame | None = None,
    regime_jm: JmRegimeFrame | None = None,
    attribution: AttributionFrame | None = None,
    top1_alert: float = _DEFAULT_TOP1_ALERT,
) -> AlertSet:
    transitions = _bucket_transitions(regime.tercile)
    return AlertSet(
        bucket_transitions=transitions,
        disagreement_events=_disagreement_events(regime, correlation),
        validation_degradation=_validation_degradation(validation),
        hmm_bucket_transitions=(
            _bucket_transitions(regime_hmm.label)
            if regime_hmm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
        jm_bucket_transitions=(
            _bucket_transitions(regime_jm.label)
            if regime_jm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
        concentration_alerts=(
            _concentration_alerts(attribution.concentration, transitions, top1_alert=top1_alert)
            if attribution is not None
            else pd.DataFrame(columns=_CONCENTRATION_COLUMNS)
        ),
    )


def _concentration_alerts(
    conc: pd.DataFrame, transitions: pd.DataFrame, *, top1_alert: float
) -> pd.DataFrame:
    """Cap-weighted rows where (top1_share > threshold on a bucket-transition day) or fragile."""
    if conc.empty:
        return pd.DataFrame(columns=_CONCENTRATION_COLUMNS)
    cap = conc[conc["weighting"] == "cap"]
    if transitions.empty:
        on_transition = pd.Series(False, index=cap.index)
    else:
        keys = set(zip(transitions["date"], transitions["segment"], strict=True))
        on_transition = pd.Series(
            [(d, s) in keys for d, s in zip(cap["date"], cap["cut"], strict=True)],
            index=cap.index,
        )
    concentrated = cap["top1_share"] > top1_alert
    fragile = cap["fragile_flag"].astype(bool)
    hit = cap[(on_transition & concentrated) | fragile].copy()
    if hit.empty:
        return pd.DataFrame(columns=_CONCENTRATION_COLUMNS)
    trig_transition = (on_transition & concentrated).loc[hit.index]
    hit["trigger"] = ["transition_day" if t else "fragile" for t in trig_transition]
    hit = hit.rename(columns={"cut": "segment"})
    return hit[_CONCENTRATION_COLUMNS].sort_values(["date", "segment"]).reset_index(drop=True)
```

In `roro/engine.py`, extend the `detect_alerts(...)` call:

```python
    alerts = detect_alerts(
        regime=regime,
        correlation=correlation,
        validation=validation,
        regime_hmm=regime_hmm,
        regime_jm=regime_jm,
        attribution=attribution,
        top1_alert=cfg.attribution_top1_alert,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_alerts.py tests/test_engine.py -m "not slow" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/alerts.py roro/engine.py tests/test_alerts.py
git commit -m "feat(attrib): concentration alerts (transition-day + fragile) wired into engine"
```

---

## Task 11: IO — CSV writers, alerts kind, snapshot block

**Files:**
- Modify: `roro/io.py` (`write_run`, `_write_alerts`, `_build_snapshot`, new writers)
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_io.py`:

```python
from roro.types import AttributionFrame


def _result_with_attribution(out_dir: Path) -> RunResult:
    base = _empty_result(out_dir)
    d = pd.Timestamp("2026-05-26")
    level = pd.DataFrame(
        [
            {"date": d, "cut": "global", "weighting": "cap", "series": "A__Eq", "block": "DM_Eq",
             "latam": False, "vol": 0.2, "ret3m": 0.1, "weight": 0.5, "leverage": 1.0,
             "contribution": 0.1, "share": 0.5, "quadrant": "HI/+", "xbar": 0.15, "ybar": 0.05},
            {"date": d, "cut": "global", "weighting": "cap", "series": "B__FI", "block": "DM_FI",
             "latam": False, "vol": 0.1, "ret3m": -0.1, "weight": 0.5, "leverage": -1.0,
             "contribution": 0.1, "share": 0.5, "quadrant": "LO/-", "xbar": 0.15, "ybar": 0.05},
        ]
    )
    delta = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "horizon": "fixed",
          "anchor_date": d - pd.Timedelta(days=90), "label_anchor": "Risk-off",
          "label_t": "Risk-on", "beta_anchor": 0.05, "beta_t": 0.2, "series": "A__Eq",
          "block": "DM_Eq", "effect_return": 0.1, "effect_position": 0.03,
          "effect_interaction": 0.02, "effect_universe": 0.0, "delta_total": 0.15}]
    )
    rollup = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "group_kind": "block",
          "group": "DM_Eq", "contribution_sum": 0.1, "n": 1}]
    )
    conc = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "n": 2, "beta": 0.2, "hhi": 0.5,
          "top1_series": "A__Eq", "top1_share": 0.5, "top5_share": 1.0, "beta_ex_top1": 0.1,
          "pct_today": 0.8, "pct_ex_top1": 0.5, "fragile_flag": True}]
    )
    pc1 = pd.DataFrame(
        [{"date": d, "cut": "global", "series": "A__Eq", "pc1_load_sq": 0.6, "var_share": 0.5,
          "decoupling": -0.1, "row_mean_corr": 0.3}]
    )
    af = AttributionFrame(level=level, delta=delta, rollup=rollup, concentration=conc, pc1=pc1,
                          anchors={"global": d - pd.Timedelta(days=90)})
    alerts = replace(
        base.alerts,
        concentration_alerts=pd.DataFrame(
            [{"date": d, "segment": "global", "weighting": "cap", "top1_series": "A__Eq",
              "top1_share": 0.5, "hhi": 0.5, "fragile_flag": True, "trigger": "fragile"}]
        ),
    )
    return replace(base, attribution=af, alerts=alerts)


def test_write_run_emits_attribution_artifacts(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    path = write_run(_result_with_attribution(out_root), run_date="2026-05-27",
                     out_dir=out_root, as_of_data_date="2026-05-26")
    for name in ("attribution.csv", "attribution_delta.csv", "attribution_rollup.csv",
                 "concentration.csv", "attribution_pc1.csv"):
        assert (path / name).exists(), name
    assert not (path / "attribution_history_global.csv").exists()
    level = pd.read_csv(path / "attribution.csv")
    assert list(level.columns)[:3] == ["date", "cut", "weighting"]
    assert len(level) == 2
    alerts = pd.read_csv(path / "alerts.csv")
    assert "concentration" in set(alerts["kind"])
    snap = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    g = snap["attribution"]["global"]
    assert g["anchor_date"] == "2026-02-25"
    assert g["top1_series"] == "A__Eq" and g["fragile_flag"] is True
    assert [x["series"] for x in g["top3"]] == ["A__Eq", "B__FI"]


def test_write_run_no_attribution_artifacts_when_none(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    path = write_run(_empty_result(out_root), run_date="2026-05-27", out_dir=out_root,
                     as_of_data_date="2026-05-26")
    assert not (path / "attribution.csv").exists()
    snap = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    assert "attribution" not in snap
```

(`replace` is already imported in `tests/test_io.py` via `from dataclasses import replace` — the HMM helper uses it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py -k attribution -v`
Expected: FAIL — `attribution.csv` not written

- [ ] **Step 3: Implement writers**

In `roro/io.py`: import `AttributionFrame` from `roro.types`. In `write_run`, after the JM block:

```python
    if result.attribution is not None:
        _write_attribution(result.attribution, tmp)
```

Add the writers (near the other `_write_*` functions):

```python
_ATTRIBUTION_FILES: tuple[tuple[str, str], ...] = (
    ("level", "attribution.csv"),
    ("delta", "attribution_delta.csv"),
    ("rollup", "attribution_rollup.csv"),
    ("concentration", "concentration.csv"),
    ("pc1", "attribution_pc1.csv"),
)


def _write_attribution(af: AttributionFrame, run_dir: Path) -> None:
    """Five long-format CSVs (+ optional wide global history). Deterministic column order."""
    for attr, name in _ATTRIBUTION_FILES:
        frame: pd.DataFrame = getattr(af, attr)
        frame.to_csv(run_dir / name, index=False)
    if af.history_global is not None:
        af.history_global.to_csv(run_dir / "attribution_history_global.csv", index=True)
```

In `_write_alerts`, after the `validation_degradation` block:

```python
    if not a.concentration_alerts.empty:
        rows.append(a.concentration_alerts.assign(kind="concentration"))
```

In `_build_snapshot`, before `return snapshot`:

```python
    af = result.attribution
    if af is not None and not af.level.empty:
        snapshot["attribution"] = _attribution_snapshot(af)
    return snapshot
```

And the helper:

```python
_SNAPSHOT_TOP: int = 3


def _attribution_snapshot(af: AttributionFrame) -> dict[str, Any]:
    """Per cut (cap-weighted, last date): beta, top-3 contributors, concentration, anchor."""
    out: dict[str, Any] = {}
    level = af.level[af.level["weighting"] == "cap"]
    conc = af.concentration[af.concentration["weighting"] == "cap"]
    for cut in sorted(level["cut"].unique()):
        rows = level[level["cut"] == cut]
        order = rows["contribution"].abs().sort_values(ascending=False, kind="stable").index
        top = rows.reindex(order).head(_SNAPSHOT_TOP)
        c_last = conc[conc["cut"] == cut]
        c_row = c_last.iloc[-1] if not c_last.empty else None
        anchor = af.anchors.get(cut)
        out[cut] = {
            "beta_cap": _safe_float(rows["contribution"].sum()),
            "top3": [
                {"series": str(r["series"]), "contribution": _safe_float(r["contribution"]),
                 "share": _safe_float(r["share"]), "quadrant": str(r["quadrant"])}
                for _, r in top.iterrows()
            ],
            "hhi": _safe_float(c_row["hhi"]) if c_row is not None else None,
            "top1_series": str(c_row["top1_series"]) if c_row is not None else None,
            "top1_share": _safe_float(c_row["top1_share"]) if c_row is not None else None,
            "fragile_flag": bool(c_row["fragile_flag"]) if c_row is not None else None,
            "anchor_date": anchor.strftime("%Y-%m-%d") if anchor is not None else None,
        }
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_io.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/io.py tests/test_io.py
git commit -m "feat(attrib): attribution CSV writers, concentration alert rows, snapshot block"
```

---

## Task 12: Golden + reproducibility + off-switch byte-identity

**Files:**
- Modify: `tests/test_golden.py`, `tests/test_reproducibility.py`
- Regenerate: `tests/golden/2024-Q1/`

- [ ] **Step 1: Prove the off switch keeps old artifacts byte-identical (before regenerating)**

Run: `uv run pytest tests/test_golden.py tests/test_reproducibility.py -v`
Expected: PASS — the existing five CSVs are unchanged by the new code (goldens still match). If this fails, STOP: something changed a pre-existing artifact; fix before continuing.

- [ ] **Step 2: Extend both tests to cover the new files**

In `tests/test_golden.py`, change the `csvs` tuple to:

```python
    csvs = (
        "beta_series.csv",
        "regimes.csv",
        "correlation.csv",
        "external_validation.csv",
        "tripwire.csv",
        "alerts.csv",
        "attribution.csv",
        "attribution_delta.csv",
        "attribution_rollup.csv",
        "concentration.csv",
        "attribution_pc1.csv",
    )
```

In `tests/test_reproducibility.py`, extend the loop tuple with the same six new names (`alerts.csv` + five attribution files) and append this test:

```python
def test_attribution_disabled_keeps_legacy_artifacts_identical(
    tiny_xlsx: Path, tmp_path: Path
) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    client = MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})
    common = dict(
        data_path=tiny_xlsx, ewma_halflife_days=10, return_window_days=21,
        tripwire_window_days=10, percentile_window_years=1, min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    on = EngineConfig(output_dir=tmp_path / "on", **common)
    off = EngineConfig(output_dir=tmp_path / "off", attribution_enabled=False, **common)
    for cfg in (on, off):
        run(cfg, fred_client=client, run_date="2024-12-31", as_of_data_date="2024-12-31",
            force=True)
    for csv in ("beta_series.csv", "regimes.csv", "correlation.csv",
                "external_validation.csv", "tripwire.csv"):
        assert filecmp.cmp(tmp_path / "on" / "2024-12-31" / csv,
                           tmp_path / "off" / "2024-12-31" / csv, shallow=False), csv
    assert not (tmp_path / "off" / "2024-12-31" / "attribution.csv").exists()
    assert (tmp_path / "on" / "2024-12-31" / "attribution.csv").exists()
```

- [ ] **Step 3: Regenerate goldens, then verify**

Run: `uv run pytest tests/test_golden.py --regenerate-goldens -q`
Expected: `1 skipped` ("Regenerated goldens.")

Run: `git status --short tests/golden`
Expected: only **new** files under `tests/golden/2024-Q1/` (`alerts.csv` may already exist and must show no diff; the five pre-existing CSVs must show no diff). If any pre-existing golden shows `M`, STOP and investigate.

Run: `uv run pytest tests/test_golden.py tests/test_reproducibility.py -v`
Expected: all PASS

- [ ] **Step 4: Commit (golden regeneration in its own commit)**

```bash
git add tests/test_golden.py tests/test_reproducibility.py tests/golden/2024-Q1
git commit -m "test(attrib): goldens + reproducibility cover attribution artifacts; off-switch identity"
```

---

## Task 13: Report bundle + load

**Files:**
- Modify: `roro/report/bundle.py`, `roro/report/load.py`
- Test: `tests/report/test_bundle.py`, `tests/report/conftest.py`

- [ ] **Step 1: Add a fixture helper and failing tests**

Append to `tests/report/conftest.py`:

```python
def write_attribution_csvs(run_dir: Path) -> None:
    """Add minimal attribution artifacts (segments match the minimal fixture)."""
    d = pd.Timestamp("2024-12-31")
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    level_rows = []
    delta_rows = []
    pc1_rows = []
    for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
        for weighting in ("cap", "eq"):
            for i, (series, block, quad) in enumerate((
                ("United States__Eq", "DM_Eq", "HI/+"), ("Brazil__Eq", "EM_Eq", "HI/-"),
                ("Germany__FI", "DM_FI", "LO/-"), ("Mexico__FI", "EM_FI", "LO/+"),
            )):
                c = (0.3, -0.1, 0.05, -0.02)[i]
                level_rows.append({
                    "date": d, "cut": seg, "weighting": weighting, "series": series,
                    "block": block, "latam": series.startswith(("Brazil", "Mexico")),
                    "vol": 0.1 * (i + 1), "ret3m": c, "weight": 0.25, "leverage": 1.0,
                    "contribution": c, "share": c / 0.23, "quadrant": quad,
                    "xbar": 0.25, "ybar": 0.05,
                })
                delta_rows.append({
                    "date": d, "cut": seg, "weighting": weighting, "horizon": "fixed",
                    "anchor_date": dates[-64], "label_anchor": "Risk-off", "label_t": "Risk-on",
                    "beta_anchor": 0.1, "beta_t": 0.23, "series": series, "block": block,
                    "effect_return": c / 2, "effect_position": c / 4,
                    "effect_interaction": c / 8, "effect_universe": 0.0,
                    "delta_total": c * 7 / 8,
                })
            if weighting == "cap":
                for series in ("United States__Eq", "Brazil__Eq", "Germany__FI", "Mexico__FI"):
                    pc1_rows.append({
                        "date": d, "cut": seg, "series": series, "pc1_load_sq": 0.25,
                        "var_share": 0.25, "decoupling": 0.0, "row_mean_corr": 0.3,
                    })
    conc_rows = [
        {"date": dt, "cut": seg, "weighting": w, "n": 4, "beta": 0.23, "hhi": 0.4,
         "top1_series": "United States__Eq", "top1_share": 0.6, "top5_share": 1.0,
         "beta_ex_top1": 0.0, "pct_today": 0.8, "pct_ex_top1": 0.4, "fragile_flag": True}
        for dt in dates for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI") for w in ("cap", "eq")
    ]
    rollup_rows = [
        {"date": d, "cut": seg, "weighting": w, "group_kind": "block", "group": blk,
         "contribution_sum": val, "n": 1}
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI") for w in ("cap", "eq")
        for blk, val in (("DM_Eq", 0.3), ("EM_Eq", -0.1), ("DM_FI", 0.05), ("EM_FI", -0.02))
    ]
    pd.DataFrame(level_rows).to_csv(run_dir / "attribution.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(run_dir / "attribution_delta.csv", index=False)
    pd.DataFrame(rollup_rows).to_csv(run_dir / "attribution_rollup.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(run_dir / "concentration.csv", index=False)
    pd.DataFrame(pc1_rows).to_csv(run_dir / "attribution_pc1.csv", index=False)


@pytest.fixture
def attribution_run_dir(minimal_run_dir: Path) -> Path:
    write_attribution_csvs(minimal_run_dir)
    return minimal_run_dir
```

Append to `tests/report/test_bundle.py`:

```python
def test_databundle_attribution_fields_default_none() -> None:
    names = {f.name for f in fields(DataBundle)}
    assert {"attribution_level", "attribution_delta", "attribution_rollup",
            "concentration", "attribution_pc1"} <= names
    for n in ("attribution_level", "attribution_delta", "attribution_rollup",
              "concentration", "attribution_pc1"):
        assert DataBundle.__dataclass_fields__[n].default is None


def test_load_bundle_reads_attribution_when_present(
    attribution_run_dir: Path, tiny_xlsx: Path
) -> None:
    b = load_bundle(attribution_run_dir, tiny_xlsx, window=21)
    assert b.attribution_level is not None and not b.attribution_level.empty
    assert b.concentration is not None
    assert pd.api.types.is_datetime64_any_dtype(b.concentration["date"])
    assert pd.api.types.is_datetime64_any_dtype(b.attribution_delta["anchor_date"])  # type: ignore[index]


def test_load_bundle_attribution_none_when_absent(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    b = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert b.attribution_level is None and b.concentration is None
```

Ensure `tests/report/test_bundle.py` imports `fields` from `dataclasses`, `pd`, `Path`, and `load_bundle` from `roro.report.load`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/report/test_bundle.py -v`
Expected: FAIL — missing bundle fields

- [ ] **Step 3: Implement**

In `roro/report/bundle.py`, append to `DataBundle` after `vol_pct`:

```python
    attribution_level: pd.DataFrame | None = None
    attribution_delta: pd.DataFrame | None = None
    attribution_rollup: pd.DataFrame | None = None
    concentration: pd.DataFrame | None = None
    attribution_pc1: pd.DataFrame | None = None
```

In `roro/report/load.py`, before the `return DataBundle(...)` in `load_bundle`:

```python
    attribution_level = attribution_delta = attribution_rollup = None
    concentration = attribution_pc1 = None
    if (run_dir / "attribution.csv").exists():
        attribution_level = pd.read_csv(run_dir / "attribution.csv", parse_dates=["date"])
        attribution_delta = pd.read_csv(
            run_dir / "attribution_delta.csv", parse_dates=["date", "anchor_date"]
        )
        attribution_rollup = pd.read_csv(run_dir / "attribution_rollup.csv", parse_dates=["date"])
        concentration = pd.read_csv(run_dir / "concentration.csv", parse_dates=["date"])
        attribution_pc1 = pd.read_csv(run_dir / "attribution_pc1.csv", parse_dates=["date"])
```

and pass them into the constructor:

```python
        attribution_level=attribution_level,
        attribution_delta=attribution_delta,
        attribution_rollup=attribution_rollup,
        concentration=concentration,
        attribution_pc1=attribution_pc1,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/report/test_bundle.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/report/bundle.py roro/report/load.py tests/report/conftest.py tests/report/test_bundle.py
git commit -m "feat(attrib): report bundle + load for attribution artifacts"
```

---

## Task 14: Report figures (`roro/report/attribution_figs.py`)

**Files:**
- Create: `roro/report/attribution_figs.py`
- Test: `tests/report/test_attribution_figs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/report/test_attribution_figs.py`:

```python
"""Pure figure-builder tests for the attribution figures."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from roro.report.attribution_figs import (
    QUADRANT_COLORS,
    attribution_bars,
    attribution_scatter,
    attribution_waterfall,
    concentration_timeseries,
    pc1_loadings_bars,
)
from roro.report.bundle import DataBundle
from roro.report.load import load_bundle


@pytest.fixture
def abundle(attribution_run_dir: Path, tiny_xlsx: Path) -> DataBundle:
    return load_bundle(attribution_run_dir, tiny_xlsx, window=21)


def _dropdown_labels(fig: go.Figure) -> list[str]:
    menus = fig.layout.updatemenus
    assert menus, "figure has no dropdown"
    return [b["label"] for b in menus[0]["buttons"]]


def test_bars_default_is_global_cap_and_dropdown_covers_cuts(abundle: DataBundle) -> None:
    fig = attribution_bars(abundle)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text.startswith("Slope attribution — global · cap")
    labels = _dropdown_labels(fig)
    assert "global · cap" in labels and "EM_FI · eq" in labels
    assert set(fig.data[0].marker.color) <= set(QUADRANT_COLORS.values())


def test_waterfall_has_four_effect_traces_and_anchor_in_title(abundle: DataBundle) -> None:
    fig = attribution_waterfall(abundle)
    assert [t.name for t in fig.data] == ["return", "position", "interaction", "universe"]
    assert fig.layout.barmode == "relative"
    assert "anchor" in fig.layout.title.text
    assert "global · cap · fixed" in _dropdown_labels(fig)


def test_scatter_has_wls_line_and_xbar_marker(abundle: DataBundle) -> None:
    fig = attribution_scatter(abundle)
    assert len(fig.data) == 2  # markers + WLS line
    assert fig.data[1].mode == "lines"
    assert any(s["type"] == "line" for s in fig.layout.shapes)
    assert fig.layout.xaxis.title.text == "EWMA vol (annualized)"


def test_concentration_timeseries_two_traces_with_regime_bands(abundle: DataBundle) -> None:
    fig = concentration_timeseries(abundle)
    assert [t.name for t in fig.data] == ["top-1 share", "HHI"]
    assert "global" in _dropdown_labels(fig)
    assert fig.layout.yaxis.range == (0.0, 1.0)


def test_pc1_bars_two_traces_sorted_by_decoupling(abundle: DataBundle) -> None:
    fig = pc1_loadings_bars(abundle)
    assert [t.name for t in fig.data] == ["PC1 loading²", "variance share"]
    assert "global" in _dropdown_labels(fig)


def test_all_figures_height_700_and_simple_white(abundle: DataBundle) -> None:
    for build in (attribution_bars, attribution_waterfall, attribution_scatter,
                  concentration_timeseries, pc1_loadings_bars):
        fig = build(abundle)
        assert fig.layout.height == 700
        # Same convention as test_figures: template presence is the proof (deep lookup is brittle)
        assert fig.layout.template is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/report/test_attribution_figs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'roro.report.attribution_figs'`

- [ ] **Step 3: Implement the five builders**

Create `roro/report/attribution_figs.py`:

```python
"""Pure figure builders for the regime-attribution artifacts (all deterministic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from roro.report.bundle import DataBundle
from roro.report.figures import _band_shapes

QUADRANT_COLORS: dict[str, str] = {
    "HI/+": "#2ca02c",  # classic risk-on
    "HI/-": "#d62728",  # classic risk-off
    "LO/+": "#ff7f0e",  # flight-to-quality
    "LO/-": "#1f77b4",  # havens sold
}
_EFFECTS: tuple[tuple[str, str, str], ...] = (
    ("effect_return", "return", "#1f77b4"),
    ("effect_position", "position", "#ff7f0e"),
    ("effect_interaction", "interaction", "#9467bd"),
    ("effect_universe", "universe", "#7f7f7f"),
)
_BLOCK_ORDER: tuple[str, ...] = ("DM_Eq", "EM_Eq", "DM_FI", "EM_FI")
_HEIGHT: int = 700
_FONT: dict[str, object] = {"family": "system-ui, -apple-system, sans-serif", "size": 13}
_MENU: dict[str, object] = {
    "type": "dropdown", "showactive": True, "x": 1.12, "y": 1.0,
    "xanchor": "left", "yanchor": "top",
}
_TOP_N_DEFAULT: int = 12
_MARKER_MIN: float = 6.0
_MARKER_MAX: float = 40.0


def _layout(title: str, buttons: list[dict[str, object]], **axes: object) -> go.Layout:
    return go.Layout(
        title=title, height=_HEIGHT, template="simple_white", font=_FONT,
        margin={"l": 80, "r": 220, "t": 60, "b": 80},
        updatemenus=[{**_MENU, "buttons": buttons}],
        **axes,
    )


def _combos(df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(set(zip(df["cut"], df["weighting"], strict=True)), key=_combo_key)


def _combo_key(combo: tuple[str, str]) -> tuple[int, int]:
    order = ("global", "DM", "EM", "Equity", "FI", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI", "LatAm")
    cut, weighting = combo
    return (order.index(cut) if cut in order else len(order), 0 if weighting == "cap" else 1)


def attribution_bars(bundle: DataBundle, *, top_n: int = _TOP_N_DEFAULT) -> go.Figure:
    """Horizontal bars of the top-N |c_i| for one (cut, weighting), colored by quadrant."""
    assert bundle.attribution_level is not None
    level = bundle.attribution_level
    per: dict[tuple[str, str], pd.DataFrame] = {}
    for combo in _combos(level):
        sub = level[(level["cut"] == combo[0]) & (level["weighting"] == combo[1])]
        order = sub["contribution"].abs().sort_values(ascending=True, kind="stable").index
        per[combo] = sub.reindex(order).tail(top_n)
    default = ("global", "cap") if ("global", "cap") in per else next(iter(per))

    def _title(combo: tuple[str, str]) -> str:
        full = level[(level["cut"] == combo[0]) & (level["weighting"] == combo[1])]
        beta = float(full["contribution"].sum())
        top1 = float(full["share"].abs().max()) if not full.empty else float("nan")
        return (f"Slope attribution — {combo[0]} · {combo[1]} · β̂ = {beta:.3f} · "
                f"max |share| = {top1:.0%}")

    d0 = per[default]
    trace = go.Bar(
        x=d0["contribution"], y=d0["series"], orientation="h",
        marker={"color": [QUADRANT_COLORS[q] for q in d0["quadrant"]]},
        customdata=np.column_stack([d0["quadrant"], d0["share"], d0["vol"], d0["ret3m"]]),
        hovertemplate=("%{y}<br>c=%{x:.4f}<br>share=%{customdata[1]:.1%}<br>"
                       "quadrant=%{customdata[0]}<br>vol=%{customdata[2]:.2%} "
                       "ret3m=%{customdata[3]:.2%}<extra></extra>"),
    )
    buttons = [
        {
            "method": "update",
            "label": f"{c[0]} · {c[1]}",
            "args": [
                {
                    "x": [per[c]["contribution"]], "y": [per[c]["series"]],
                    "marker": [{"color": [QUADRANT_COLORS[q] for q in per[c]["quadrant"]]}],
                    "customdata": [np.column_stack([per[c]["quadrant"], per[c]["share"],
                                                    per[c]["vol"], per[c]["ret3m"]])],
                },
                {"title": _title(c)},
            ],
        }
        for c in per
    ]
    return go.Figure(
        data=[trace],
        layout=_layout(
            _title(default), buttons,
            xaxis={"title": "contribution to β̂ (share of β̂ can exceed 100%)"},
            yaxis={"title": "asset"},
        ),
    )


def attribution_waterfall(bundle: DataBundle) -> go.Figure:
    """Δβ̂ from anchor to today, stacked into 4 effects per block; dropdown = cut·weighting·horizon."""
    assert bundle.attribution_delta is not None
    delta = bundle.attribution_delta
    keys = sorted(
        set(zip(delta["cut"], delta["weighting"], delta["horizon"], strict=True)),
        key=lambda k: (_combo_key((k[0], k[1])), 0 if k[2] == "anchor" else 1),
    )
    per: dict[tuple[str, str, str], pd.DataFrame] = {}
    meta: dict[tuple[str, str, str], pd.Series] = {}
    for k in keys:
        sub = delta[(delta["cut"] == k[0]) & (delta["weighting"] == k[1]) & (delta["horizon"] == k[2])]
        g = sub.groupby("block")[[e[0] for e in _EFFECTS]].sum()
        per[k] = g.reindex([b for b in _BLOCK_ORDER if b in g.index])
        meta[k] = sub.iloc[0]
    default = ("global", "cap", "anchor") if ("global", "cap", "anchor") in per else keys[0]

    def _title(k: tuple[str, str, str]) -> str:
        m = meta[k]
        a = pd.Timestamp(m["anchor_date"]).strftime("%Y-%m-%d")
        return (f"Δβ̂ waterfall — {k[0]} · {k[1]} · {k[2]} anchor {a} "
                f"({m['label_anchor']} → {m['label_t']}) · "
                f"β̂ {float(m['beta_anchor']):.3f} → {float(m['beta_t']):.3f}")

    g0 = per[default]
    traces = [
        go.Bar(name=label, x=list(g0.index), y=g0[col], marker={"color": color})
        for col, label, color in _EFFECTS
    ]
    buttons = [
        {
            "method": "update",
            "label": f"{k[0]} · {k[1]} · {k[2]}",
            "args": [
                {"x": [list(per[k].index)] * len(_EFFECTS),
                 "y": [per[k][col] for col, _, _ in _EFFECTS]},
                {"title": _title(k)},
            ],
        }
        for k in keys
    ]
    layout = _layout(
        _title(default), buttons,
        xaxis={"title": "block"}, yaxis={"title": "Δβ̂ contribution"},
    )
    layout.barmode = "relative"
    return go.Figure(data=traces, layout=layout)


def attribution_scatter(bundle: DataBundle) -> go.Figure:
    """Vol vs 3M return, marker size ∝ |c_i|, color by quadrant, exact WLS line + x̄ marker."""
    assert bundle.attribution_level is not None
    level = bundle.attribution_level
    per = {
        c: level[(level["cut"] == c[0]) & (level["weighting"] == c[1])].sort_values("series")
        for c in _combos(level)
    }
    default = ("global", "cap") if ("global", "cap") in per else next(iter(per))

    def _size(sub: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        a = sub["contribution"].abs().to_numpy(dtype=float)
        top = float(a.max()) if a.size and a.max() > 0 else 1.0
        return _MARKER_MIN + (_MARKER_MAX - _MARKER_MIN) * a / top

    def _line(sub: pd.DataFrame) -> tuple[list[float], list[float]]:
        beta = float(sub["contribution"].sum())
        xbar, ybar = float(sub["xbar"].iloc[0]), float(sub["ybar"].iloc[0])
        xs = [float(sub["vol"].min()), float(sub["vol"].max())]
        return xs, [ybar + beta * (x - xbar) for x in xs]

    s0 = per[default]
    xs0, ys0 = _line(s0)
    traces = [
        go.Scatter(
            x=s0["vol"], y=s0["ret3m"], mode="markers", text=s0["series"], name="assets",
            marker={"size": _size(s0), "color": [QUADRANT_COLORS[q] for q in s0["quadrant"]],
                    "line": {"width": 0.5, "color": "#333"}},
            hovertemplate="%{text}<br>vol=%{x:.2%} ret3m=%{y:.2%}<extra></extra>",
        ),
        go.Scatter(x=xs0, y=ys0, mode="lines", name="WLS line (regime slope)",
                   line={"color": "#333", "dash": "dash"}),
    ]
    buttons = []
    for c, sub in per.items():
        xs, ys = _line(sub)
        buttons.append({
            "method": "update",
            "label": f"{c[0]} · {c[1]}",
            "args": [
                {"x": [sub["vol"], xs], "y": [sub["ret3m"], ys], "text": [sub["series"], None],
                 "marker": [{"size": _size(sub),
                             "color": [QUADRANT_COLORS[q] for q in sub["quadrant"]],
                             "line": {"width": 0.5, "color": "#333"}}, {}]},
                {"title": f"Vol vs return, sized by |contribution| — {c[0]} · {c[1]}",
                 "shapes": [_xbar_shape(float(sub["xbar"].iloc[0]))]},
            ],
        })
    layout = _layout(
        f"Vol vs return, sized by |contribution| — {default[0]} · {default[1]}", buttons,
        xaxis={"title": "EWMA vol (annualized)"}, yaxis={"title": "3M log return"},
    )
    layout.shapes = [_xbar_shape(float(s0["xbar"].iloc[0]))]
    return go.Figure(data=traces, layout=layout)


def _xbar_shape(xbar: float) -> dict[str, object]:
    return {"type": "line", "x0": xbar, "x1": xbar, "y0": 0, "y1": 1, "xref": "x",
            "yref": "paper", "line": {"color": "#999", "dash": "dot", "width": 1}}


def concentration_timeseries(bundle: DataBundle) -> go.Figure:
    """top-1 share + HHI of |c_i| over time (cap-weighted) with hysteresis-smoothed regime bands."""
    assert bundle.concentration is not None
    conc = bundle.concentration[bundle.concentration["weighting"] == "cap"]
    cuts = [c for c in bundle.seg_tercile.columns if c in set(conc["cut"])]
    per = {c: conc[conc["cut"] == c].sort_values("date") for c in cuts}
    default = "global" if "global" in per else cuts[0]
    s0 = per[default]
    traces = [
        go.Scatter(x=s0["date"], y=s0["top1_share"], mode="lines", name="top-1 share",
                   line={"color": "#d62728"}),
        go.Scatter(x=s0["date"], y=s0["hhi"], mode="lines", name="HHI", line={"color": "#1f77b4"}),
    ]
    buttons = [
        {
            "method": "update",
            "label": c,
            "args": [
                {"x": [per[c]["date"]] * 2, "y": [per[c]["top1_share"], per[c]["hhi"]]},
                {"title": f"Concentration of the slope — {c}",
                 "shapes": _band_shapes(bundle.seg_tercile[c], smooth=True)},
            ],
        }
        for c in cuts
    ]
    layout = _layout(
        f"Concentration of the slope — {default}", buttons,
        xaxis={"title": "Date"}, yaxis={"title": "share of Σ|c|", "range": (0.0, 1.0)},
    )
    layout.shapes = _band_shapes(bundle.seg_tercile[default], smooth=True)
    return go.Figure(data=traces, layout=layout)


def pc1_loadings_bars(bundle: DataBundle) -> go.Figure:
    """Per-asset PC1 loading² vs variance share, sorted by decoupling (havens on the right)."""
    assert bundle.attribution_pc1 is not None
    pc1 = bundle.attribution_pc1
    cuts = sorted(set(pc1["cut"]), key=lambda c: _combo_key((c, "cap")))
    per = {c: pc1[pc1["cut"] == c].sort_values("decoupling") for c in cuts}
    default = "global" if "global" in per else cuts[0]
    s0 = per[default]
    traces = [
        go.Bar(name="PC1 loading²", x=s0["series"], y=s0["pc1_load_sq"],
               marker={"color": "#1f77b4"}),
        go.Bar(name="variance share", x=s0["series"], y=s0["var_share"],
               marker={"color": "#ff7f0e"}),
    ]
    buttons = [
        {
            "method": "update",
            "label": c,
            "args": [
                {"x": [per[c]["series"]] * 2, "y": [per[c]["pc1_load_sq"], per[c]["var_share"]]},
                {"title": f"PC1 loadings vs variance share (sorted by decoupling) — {c}"},
            ],
        }
        for c in cuts
    ]
    layout = _layout(
        f"PC1 loadings vs variance share (sorted by decoupling) — {default}", buttons,
        xaxis={"title": "asset"}, yaxis={"title": "share"},
    )
    layout.barmode = "group"
    return go.Figure(data=traces, layout=layout)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/report/test_attribution_figs.py -v`
Expected: all PASS

- [ ] **Step 5: Quality bars + commit**

Run: `uv run ruff check roro/report/attribution_figs.py tests/report/test_attribution_figs.py && uv run mypy roro/report/attribution_figs.py`
Expected: clean (plotly is `ignore_missing_imports`).

```bash
git add roro/report/attribution_figs.py tests/report/test_attribution_figs.py
git commit -m "feat(attrib): five attribution report figures"
```

---

## Task 15: Orchestrate + e2e

**Files:**
- Modify: `roro/report/orchestrate.py`
- Test: `tests/report/test_e2e.py`, `tests/report/test_cli.py` (no change expected), new test in `tests/report/test_attribution_figs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/report/test_attribution_figs.py`:

```python
from roro.report import build_report


def test_build_report_includes_attribution_sections(
    attribution_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    build_report(attribution_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    for title in ("Slope attribution", "Δβ̂ waterfall", "Vol vs return, sized by |contribution|",
                  "Concentration of the slope", "PC1 loadings vs variance share"):
        assert title in html, title
    assert html.count('class="plotly-graph-div"') == 10


def test_build_report_without_attribution_unchanged(
    minimal_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    build_report(minimal_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    assert "Slope attribution" not in html
    assert html.count('class="plotly-graph-div"') == 5
```

In `tests/report/test_e2e.py`, the regenerated golden dir now carries attribution CSVs, so update:

```python
    # 10 figures rendered (no HMM/JM in this golden run): 3 base + 2 vol heatmaps + 5 attribution
    assert html_text.count('class="plotly-graph-div"') == 10
    assert "Slope attribution" in html_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/report/test_attribution_figs.py tests/report/test_e2e.py -v`
Expected: the two `build_report` tests FAIL (count 5, no attribution sections)

- [ ] **Step 3: Implement**

In `roro/report/orchestrate.py`, add the import:

```python
from roro.report.attribution_figs import (
    attribution_bars,
    attribution_scatter,
    attribution_waterfall,
    concentration_timeseries,
    pc1_loadings_bars,
)
```

and, after the `vol_pct_asset_heatmap` spec is appended and before `html = assemble(...)`:

```python
    if bundle.attribution_level is not None:
        specs.extend([
            FigureSpec(attribution_bars(bundle), "fig_attrib_bars", "Slope attribution"),
            FigureSpec(attribution_waterfall(bundle), "fig_attrib_waterfall", "Δβ̂ waterfall"),
            FigureSpec(attribution_scatter(bundle), "fig_attrib_scatter",
                       "Vol vs return, sized by |contribution|"),
            FigureSpec(concentration_timeseries(bundle), "fig_concentration_ts",
                       "Concentration of the slope"),
            FigureSpec(pc1_loadings_bars(bundle), "fig_pc1_loadings",
                       "PC1 loadings vs variance share"),
        ])
```

- [ ] **Step 4: Run the whole report suite**

Run: `uv run pytest tests/report -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add roro/report/orchestrate.py tests/report/test_attribution_figs.py tests/report/test_e2e.py
git commit -m "feat(attrib): report renders the five attribution figures when artifacts exist"
```

---

## Task 16: Full quality gate

**Files:** none new

- [ ] **Step 1: Run everything**

Run: `uv run pytest -m "not slow" -q`
Expected: all PASS (≈ 224 pre-existing + ~35 new), no warnings escaping (`filterwarnings=error`).

Run: `uv run mypy --strict roro`
Expected: `Success: no issues found`

Run: `uv run ruff check roro tests`
Expected: `All checks passed!`

- [ ] **Step 2: Slow suite once**

Run: `uv run pytest -m slow -q`
Expected: PASS (HMM/JM benches unaffected).

- [ ] **Step 3: Fix anything that surfaced, commit**

```bash
git add -A roro tests
git commit -m "chore(attrib): quality gate green (pytest/mypy/ruff)"
```

(Only if there were fixes; otherwise skip.)

---

## Task 17: Real-data run + memo (AA-6 / AA-7)

**Files:**
- Create: `docs/analysis/2026-09-XX-attribution-memo.md` (use the actual run date)
- Requires: `data.xlsx` present, `FRED_API_KEY` in env (or `--fred-key`)

- [ ] **Step 1: Run the engine with attribution on (default) and global history**

Run (from repo root; adjust the date to today):

```bash
uv run roro run --config configs/default.yaml --date 2026-09-05 --as-of-data-date 2026-05-26 --out outputs/attribution_run --force
```

Expected: `OK: outputs/attribution_run/2026-09-05` and the five attribution CSVs present in that dir.

- [ ] **Step 2: Reproduce the spec §2 snapshot**

Run:

```bash
uv run python -c "import pandas as pd; l=pd.read_csv('outputs/attribution_run/2026-09-05/attribution.csv'); g=l[(l.cut=='global')&(l.weighting=='cap')].sort_values('contribution',key=abs,ascending=False); print(g[['series','quadrant','contribution','share']].head(6).to_string(index=False)); print('beta', round(g.contribution.sum(),4))"
```

Expected (to 4 dp): `beta 0.429`; South Korea__Eq contribution `0.3084` share `0.7187`; Taiwan__Eq `0.0954`; China__Eq `-0.0294`; United States__FI `0.0274`. Any mismatch → STOP, diagnose against the throwaway calculation recorded in the spec.

- [ ] **Step 3: Concentration history statistics**

Run:

```bash
uv run python -c "import pandas as pd; c=pd.read_csv('outputs/attribution_run/2026-09-05/concentration.csv'); g=c[(c.cut=='global')&(c.weighting=='cap')]; print('days', len(g)); print('top1>0.5 share of days', round((g.top1_share>0.5).mean(),3)); print('fragile share', round(g.fragile_flag.mean(),3)); print(g.top1_series.value_counts().head(8).to_string())"
```

Record the three numbers. Decision D3 trigger: if `top1>0.5 share of days` exceeds 0.20, open a separate robust-slope spec (do NOT change the estimator here).

- [ ] **Step 4: Build the report and eyeball the five figures**

Run: `uv run roro report --help` (confirm option names), then:

```bash
uv run roro report outputs/attribution_run/2026-09-05 --out outputs/attribution_report.html
```

Open `outputs/attribution_report.html`; for each of the 7 in-range G3 events (2010 Greek, 2011 Eurozone, 2015 China, 2018 Q4, 2020 COVID, 2022 rate shock, plus the seventh listed in `roro/backtest.py`), note in the memo which block and top-3 assets the concentration/waterfall attribute the transition to. This is the AA-7 PM review.

- [ ] **Step 5: Write the memo**

Create `docs/analysis/2026-09-XX-attribution-memo.md` with sections: Run (command, data date, git sha from `snapshot.json`), Snapshot reproduction (table from Step 2), Concentration statistics (Step 3 numbers + D3 verdict), Event review (Step 4 table: event · date of transition · anchor · driving block · top-3 assets · sensible? yes/no), Open issues.

- [ ] **Step 6: Commit**

```bash
git add docs/analysis/2026-09-XX-attribution-memo.md
git commit -m "docs(attrib): real-data attribution memo (snapshot reproduction, concentration stats, event review)"
```

---

## Task 18: Context docs

**Files:**
- Modify: `docs/context/todo.md`, `docs/context/memory.md`, `docs/context/results.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: todo.md** — add a section:

```markdown
## Regime Attribution (2026-09-05) — DONE (feat/attribution)
Spec: `docs/superpowers/specs/2026-09-05-roro-attribution-design.md` · Plan: `docs/superpowers/plans/2026-09-05-roro-attribution.md`
- [x] A1 kernels: contributions / attribute_panel / attribute_delta / concentration / rank_against_prior / pc1_loadings / find_anchor (property tests: exactness 1e-10, leverage identities)
- [x] A2 AttributionFrame + compute_attribution; engine wiring (on by default); anchor label source fallback
- [x] A3 io: 5 CSVs + snapshot block; alerts kind=concentration; goldens regenerated (new files only)
- [x] A4 report: bundle/load + 5 figures + orchestrate (10 figures when artifacts exist)
- [x] A5 real-data memo: docs/analysis/2026-09-XX-attribution-memo.md
- [ ] Follow-up: D3 robust slope spec only if top-1 > 50% on > 20% of days (see memo)
```

- [ ] **Step 2: memory.md** — append one-liners:

```markdown
- decision: regime attribution = EXACT per-asset decomposition of the WLS slope (c_i = h_i*y_i, h_i = w_i(x_i-xbar)/D, sum == beta to 1e-10) on the identical DailyPanel; regression.py untouched. On by default (adds files only; attribution_enabled=False keeps legacy artifacts byte-identical). Spec: docs/superpowers/specs/2026-09-05-roro-attribution-design.md.
- decision: attribution anchor = day BEFORE the most recent tercile transition <= t (label source percentile by default; hmm/jm when enabled, else fallback + warning); fixed 63d horizon in parallel. Interaction term reported, not folded.
- decision: concentration is REPORTED (HHI, top-1/top-5 share, leave-one-out beta, fragile_flag vs same trailing-5Y thresholds), the estimator is NOT robustified; robust/winsorized slope is a separate decision (D3) gated on the memo statistic.
- constraint: attribution rows/frames are sorted by series key and use stable argsort for ties -> byte-identical CSVs; never iterate dicts/sets into output order.
```

- [ ] **Step 3: results.md** — append (fill the numbers from Task 17):

```markdown
- 2026-09-XX Attribution (feat/attribution, Tasks 1–16): roro/attribution.py + AttributionFrame + engine/io/alerts/report wiring. Exactness + delta + PC1 + anchor property tests green; goldens regenerated (new files only, legacy CSVs unchanged); off-switch byte-identity test green. Full suite green, mypy --strict + ruff clean.
- 2026-09-XX Attribution real-data run: spec §2 snapshot reproduced to 4 dp (Korea Eq 72% of global cap beta on 2026-05-26). Global cap: top-1 > 50% on N% of days, fragile on M% of days. D3 verdict: <open robust-slope spec | not triggered>. Memo: docs/analysis/2026-09-XX-attribution-memo.md.
```

- [ ] **Step 4: sesion-log.md** — append one line:

```markdown
- 2026-09-XX: implemented regime attribution (exact slope decomposition, waterfall, concentration alerts, PC1 loadings, 5 report figures) per 2026-09-05 spec; real-data memo written.
```

- [ ] **Step 5: Commit**

```bash
git add docs/context/todo.md docs/context/memory.md docs/context/results.md docs/context/sesion-log.md
git commit -m "docs(context): record regime attribution implementation + decisions"
```

---

## Self-review against the spec

**Spec coverage**
- §4.1 level attribution → Task 3. §4.2 quadrants → Task 3. §4.3 waterfall (3 effects + universe, anchor + fixed) → Tasks 4, 7, 8. §4.4 roll-ups (block/quadrant/LatAm inside one regression) → Task 8 `_rollup`. §4.5 concentration, leave-one-out, fragility, alert rule → Tasks 5, 8, 10. §4.6 PC1 loadings/decoupling/row-mean corr → Task 6.
- §5.1 module API → Tasks 3–8 (names: `contributions`, `attribute_panel`, `attribute_delta`, `concentration`, `rank_against_prior`, `pc1_loadings`, `find_anchor`, `compute_attribution`). §5.2 types → Task 8. §5.3 config (7 fields) → Task 1. §5.4 engine → Task 9 (+ D4: re-run `daily_panel`). §5.5 outputs (5 CSVs + optional history + snapshot) → Task 11. §5.6 alerts → Task 10. §5.7 five figures → Tasks 14–15.
- §6 tests: exactness / delta exactness / quadrants / leverage identities / concentration bounds / PC1 identities + sign flip / anchor causality / determinism (golden + reproducibility) / off switch → Tasks 3–12. The **jackknife sign check on real data** (§6) is folded into Task 17 Step 3 as a memo statistic (`beta_ex_top1` sign vs `top1` contribution) — add one line to the memo: share of days where `sign(beta - beta_ex_top1) == sign(c_top1)`; compute it from `concentration.csv` (`beta`, `beta_ex_top1`) and `attribution_history_global.csv` if enabled, else from a one-off run with `attribution_history_global: true`.
- §7 AA-1…AA-7 → Tasks 3–4 (AA-1), 12 (AA-2/3), 15 (AA-4), 16 (AA-5), 17 (AA-6/7).
- §10 D1 on-by-default → Task 1; D2 history flag off → Tasks 1, 8; D3 deferred → Task 17 trigger; D4 re-run panels → Task 8; D5 interaction shown → Task 14 (four separate traces).

**Placeholder scan** — Task 17 uses `2026-09-XX` for the memo filename and dates; replace with the actual run date at execution. `uv run roro report --help` is a verification step because the report subcommand's option names are not in this plan's context; the command shown uses the `--out` name visible in `cmd_report`'s message strings.

**Type consistency** — `contributions()` returns `PanelContributions | None` (Task 3) and is consumed as such in Tasks 5 and 8. `concentration(panel, pc, *, weighting, min_n)` signature identical in Tasks 5 and 8. `AttributionFrame` field names (`level, delta, rollup, concentration, pc1, anchors, history_global`) identical in Tasks 8, 10, 11, 13. `detect_alerts(..., attribution=, top1_alert=)` identical in Tasks 10 and 9's engine call (engine edit lives in Task 10). CSV names identical in Tasks 11, 12, 13. Figure builder names identical in Tasks 14 and 15.
