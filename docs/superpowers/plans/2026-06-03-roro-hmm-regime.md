# HMM / Markov-Switching Regime Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 3-state Markov-switching HMM regime classifier on segment β, running in parallel with the existing percentile classifier and compared through the S9 acceptance gates.

**Architecture:** New `roro/regime_hmm.py` owns the whole method. Point-in-time discipline lives inside it (causal-by-construction within one engine run), mirroring how the percentile classifier's trailing rolling window is causal. statsmodels `MarkovRegression` fits params (EM) monthly on an expanding window; the Hamilton filter labels every day. Output is a new `HmmRegimeFrame`, additive and off-by-default (`hmm_enabled=False`).

**Tech Stack:** Python 3.12, statsmodels (new dep), numpy, pandas, pytest + hypothesis. mypy strict, ruff, `filterwarnings=["error"]`.

**Spec:** `docs/superpowers/specs/2026-06-03-roro-hmm-regime-design.md`

---

## Causality note (read before Task 6)

**Filtering is causal; smoothing is not.** In statsmodels, `filtered_marginal_probabilities.iloc[t]` from a filter run over the *whole* passed series equals `P(state_t | β_{0:t})` — it never uses data after `t`. Smoothed probabilities do. We use **only** filtered.

This lets the walk-forward be efficient: within a refit block `[r, r')`, run the filter **once** over `β[:r']` with params estimated at `r` (params depend only on `β[:r]`, and `r ≤ t`, so no leak), then slice rows `[r, r')`. One model construction per refit (~monthly), not per day.

---

## File Structure

- **Create** `roro/regime_hmm.py` — `_FitResult`, `_fit_params`, `_filtered_probs`, `walk_forward`, `classify_hmm`.
- **Create** `tests/test_regime_hmm.py` — core, ordering, causality, cold-start, convergence, synthetic recovery.
- **Create** `tests/test_regime_hmm_cadence.py` — cadence-invariance bench (`@pytest.mark.slow`).
- **Modify** `pyproject.toml` — statsmodels dep + mypy override + `slow` marker.
- **Modify** `roro/types.py` — `HmmRegimeFrame`, `RunResult.regime_hmm`, `AlertSet.hmm_bucket_transitions`.
- **Modify** `roro/config.py` — `hmm_*` fields.
- **Modify** `roro/engine.py` — guarded `classify_hmm` call + thread into `RunResult`.
- **Modify** `roro/alerts.py` — emit `hmm_bucket_transitions`.
- **Modify** `roro/io.py` — `regimes_hmm.csv`, `hmm_refit_log.csv`, snapshot block.
- **Modify** `roro/backtest.py` — parametrize gate scorer over label source; compare report.

---

## Task 1: Add statsmodels dependency + tooling

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the runtime dep**

In `[project].dependencies`, add after the `scipy` line:
```toml
    "statsmodels>=0.14,<0.15",
```

- [ ] **Step 2: Add mypy override**

After the existing scipy override block (`pyproject.toml:66`), add:
```toml
[[tool.mypy.overrides]]
module = "statsmodels.*"
ignore_missing_imports = true
```

- [ ] **Step 3: Register the `slow` marker**

In `[tool.pytest.ini_options]`, add:
```toml
markers = ["slow: long-running benches (deselect with -m 'not slow')"]
```

- [ ] **Step 4: Install and verify import**

Run: `pip install -e ".[dev]"` then `python -c "from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add statsmodels dep for HMM regime classifier"
```

---

## Task 2: Config knobs

**Files:**
- Modify: `roro/config.py:34` (inside `EngineConfig`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:
```python
def test_hmm_defaults_are_off() -> None:
    from roro.config import EngineConfig
    from pathlib import Path

    cfg = EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("out"))
    assert cfg.hmm_enabled is False
    assert cfg.hmm_refit_interval_days == 21
    assert cfg.hmm_min_history_days == 252
    assert cfg.hmm_switching_variance is True
    assert cfg.hmm_window == "expanding"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_hmm_defaults_are_off -v`
Expected: FAIL with `AttributeError: ... has no attribute 'hmm_enabled'`

- [ ] **Step 3: Add the fields**

In `roro/config.py`, inside `EngineConfig` after `methodology_version` (line 34):
```python
    hmm_enabled: bool = False
    hmm_refit_interval_days: int = 21
    hmm_min_history_days: int = 252
    hmm_switching_variance: bool = True
    hmm_window: str = "expanding"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_hmm_defaults_are_off -v`
Expected: PASS

- [ ] **Step 5: Verify YAML round-trip still works**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all). `to_dict`/`load_config` handle the new scalar fields automatically.

- [ ] **Step 6: Commit**

```bash
git add roro/config.py tests/test_config.py
git commit -m "feat(config): add hmm_* knobs (off by default)"
```

---

## Task 3: `HmmRegimeFrame` contract

**Files:**
- Modify: `roro/types.py` (add dataclass; add `RunResult.regime_hmm`)
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_types.py`:
```python
def test_hmm_regime_frame_constructs() -> None:
    import pandas as pd
    from roro.types import HmmRegimeFrame

    idx = pd.bdate_range("2014-01-01", periods=3)
    df = pd.DataFrame({"global": [0, 1, 2]}, index=idx)
    f = HmmRegimeFrame(
        state=df, label=df.astype(str), prob_risk_off=df, prob_transitional=df,
        prob_risk_on=df, confidence=df, n_per_segment=df, thin_cut_flag=df,
        cold_start_flag=df, refit_dates={"global": [idx[0]]},
    )
    assert list(f.label.columns) == ["global"]
    assert f.refit_dates["global"] == [idx[0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py::test_hmm_regime_frame_constructs -v`
Expected: FAIL with `ImportError: cannot import name 'HmmRegimeFrame'`

- [ ] **Step 3: Add the dataclass**

In `roro/types.py`, after `RegimeFrame` (line 67):
```python
@dataclass(frozen=True)
class HmmRegimeFrame:
    state: pd.DataFrame
    label: pd.DataFrame
    prob_risk_off: pd.DataFrame
    prob_transitional: pd.DataFrame
    prob_risk_on: pd.DataFrame
    confidence: pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    cold_start_flag: pd.DataFrame
    refit_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
```

Then add to `RunResult` (after `alerts: AlertSet`, line 100):
```python
    regime_hmm: HmmRegimeFrame | None = None
```
(Place it before the fields that already have defaults — i.e. immediately after `alerts: AlertSet` and before `warnings`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py::test_hmm_regime_frame_constructs -v`
Expected: PASS

- [ ] **Step 5: Verify nothing else broke**

Run: `pytest tests/test_types.py tests/test_engine.py -v`
Expected: PASS (existing `RunResult` constructions omit `regime_hmm`, which defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add roro/types.py tests/test_types.py
git commit -m "feat(types): add HmmRegimeFrame + RunResult.regime_hmm"
```

---

## Task 4: HMM core — fit + state ordering

**Files:**
- Create: `roro/regime_hmm.py`
- Test: `tests/test_regime_hmm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_regime_hmm.py`:
```python
import numpy as np
import pandas as pd

from roro.regime_hmm import _fit_params


def _three_regime_beta(seed: int = 0) -> pd.Series:
    """Concatenate low / mid / high mean blocks with small noise."""
    rng = np.random.default_rng(seed)
    blocks = [
        rng.normal(-0.8, 0.05, 400),
        rng.normal(0.0, 0.05, 400),
        rng.normal(0.8, 0.05, 400),
    ]
    vals = np.concatenate(blocks)
    idx = pd.bdate_range("2010-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, name="beta")


def test_fit_orders_states_by_mean_ascending() -> None:
    beta = _three_regime_beta()
    fit = _fit_params(beta, switching_variance=True)
    assert fit.converged
    # perm maps ordered-rank -> raw regime index; means under perm must ascend.
    ordered_means = fit.means[fit.perm]
    assert ordered_means[0] < ordered_means[1] < ordered_means[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_hmm.py::test_fit_orders_states_by_mean_ascending -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'roro.regime_hmm'`

- [ ] **Step 3: Write the module skeleton + fit**

Create `roro/regime_hmm.py`:
```python
"""3-state Markov-switching HMM regime classifier on segment beta.

Parallel to roro.classify. Causal-by-construction: params re-estimated on an
expanding window (monthly), states inferred by the Hamilton filter daily.
Filtered probabilities only — never smoothed (smoothing peeks at the future).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from roro.config import EngineConfig
from roro.types import BetaBySegment, HmmRegimeFrame

_K_REGIMES = 3
_ORDERED_LABELS = ("Risk-off", "Transitional", "Risk-on")
_UNKNOWN = "Unknown"


@dataclass(frozen=True)
class _FitResult:
    params: np.ndarray        # statsmodels param vector (raw regime order)
    perm: np.ndarray          # ordered-rank -> raw regime index (sort by mean asc)
    means: np.ndarray         # per-raw-regime fitted mean (const)
    converged: bool


def _build_model(beta: pd.Series, *, switching_variance: bool) -> MarkovRegression:
    return MarkovRegression(
        endog=beta.to_numpy(dtype=float),
        k_regimes=_K_REGIMES,
        trend="c",
        switching_variance=switching_variance,
    )


def _fit_params(beta: pd.Series, *, switching_variance: bool) -> _FitResult:
    """Fit one HMM; return params, mean-sorted permutation, convergence flag.

    Degenerate/non-converged fits return converged=False with NaN params so the
    caller can fall back to the previous good params.
    """
    model = _build_model(beta, switching_variance=switching_variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = model.fit(maxiter=200, disp=False)
        except Exception:  # noqa: BLE001 - degenerate EM -> caller falls back
            nan = np.full(_K_REGIMES, np.nan)
            return _FitResult(params=nan, perm=np.arange(_K_REGIMES), means=nan, converged=False)

    means = np.array([float(res.params[f"const[{i}]"]) for i in range(_K_REGIMES)])
    converged = bool(np.all(np.isfinite(res.params.to_numpy()))) and not np.any(np.isnan(means))
    perm = np.argsort(means)  # ascending; perm[0] = lowest-mean (Risk-off)
    return _FitResult(
        params=res.params.to_numpy(dtype=float),
        perm=perm,
        means=means,
        converged=converged,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_hmm.py::test_fit_orders_states_by_mean_ascending -v`
Expected: PASS

- [ ] **Step 5: Add a determinism test**

Add to `tests/test_regime_hmm.py`:
```python
def test_fit_is_deterministic() -> None:
    beta = _three_regime_beta()
    a = _fit_params(beta, switching_variance=True)
    b = _fit_params(beta, switching_variance=True)
    np.testing.assert_allclose(a.params, b.params)
    np.testing.assert_array_equal(a.perm, b.perm)
```

Run: `pytest tests/test_regime_hmm.py -v`
Expected: PASS (statsmodels `MarkovRegression.fit` default `search_reps=0` → deterministic start).

- [ ] **Step 6: Commit**

```bash
git add roro/regime_hmm.py tests/test_regime_hmm.py
git commit -m "feat(regime_hmm): HMM core fit with mean-sorted state ordering"
```

---

## Task 5: HMM core — filtered probabilities (frozen params)

**Files:**
- Modify: `roro/regime_hmm.py`
- Test: `tests/test_regime_hmm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime_hmm.py`:
```python
from roro.regime_hmm import _filtered_probs


def test_filtered_probs_shape_and_simplex() -> None:
    beta = _three_regime_beta()
    fit = _fit_params(beta, switching_variance=True)
    probs = _filtered_probs(beta, fit, switching_variance=True)
    assert probs.shape == (len(beta), 3)
    # rows sum to ~1 (probability simplex)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # last block should be dominated by the high-mean (Risk-on, column 2) state
    assert probs[-1, 2] > probs[-1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_hmm.py::test_filtered_probs_shape_and_simplex -v`
Expected: FAIL with `ImportError: cannot import name '_filtered_probs'`

- [ ] **Step 3: Implement filtered-probs with ordering applied**

Add to `roro/regime_hmm.py`:
```python
def _filtered_probs(
    beta: pd.Series, fit: _FitResult, *, switching_variance: bool
) -> np.ndarray:
    """Causal filtered P(state_t | beta_{0:t}) as a (T, 3) array in ORDERED columns.

    Columns are [Risk-off, Transitional, Risk-on] via fit.perm. Uses .filter()
    with frozen params (no re-estimation). Filtering is causal: row t uses only
    beta[:t].
    """
    model = _build_model(beta, switching_variance=switching_variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.filter(fit.params)
    raw = np.asarray(res.filtered_marginal_probabilities)
    # Normalize to (T, k) regardless of statsmodels minor-version orientation.
    if raw.shape[0] == _K_REGIMES and raw.shape[1] != _K_REGIMES:
        raw = raw.T
    return raw[:, fit.perm]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_hmm.py::test_filtered_probs_shape_and_simplex -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roro/regime_hmm.py tests/test_regime_hmm.py
git commit -m "feat(regime_hmm): causal filtered probabilities in ordered columns"
```

---

## Task 6: Walk-forward engine (per-segment causal series)

**Files:**
- Modify: `roro/regime_hmm.py`
- Test: `tests/test_regime_hmm.py`

- [ ] **Step 1: Write the failing causality test (the critical one)**

Add to `tests/test_regime_hmm.py`:
```python
from roro.regime_hmm import walk_forward


def test_walk_forward_is_causal_no_lookahead() -> None:
    """Label at date t must not change when future data is appended."""
    beta = _three_regime_beta()
    cut = 900  # inside the series, past min_history
    full = walk_forward(
        beta, refit_interval_days=21, min_history_days=252, switching_variance=True
    )
    truncated = walk_forward(
        beta.iloc[:cut], refit_interval_days=21, min_history_days=252, switching_variance=True
    )
    # Labels on the overlapping, non-cold-start region must be identical.
    common = truncated["label"].index[truncated["label"] != _UNKNOWN]
    common = common[common < beta.index[cut - 1]]  # exclude the last refit-boundary day
    pd.testing.assert_series_equal(
        full["label"].loc[common], truncated["label"].loc[common], check_names=False
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_hmm.py::test_walk_forward_is_causal_no_lookahead -v`
Expected: FAIL with `ImportError: cannot import name 'walk_forward'`

- [ ] **Step 3: Implement walk-forward**

Add to `roro/regime_hmm.py`:
```python
def walk_forward(
    beta: pd.Series,
    *,
    refit_interval_days: int,
    min_history_days: int,
    switching_variance: bool,
) -> dict[str, object]:
    """Causal per-segment HMM labels over the full beta index.

    Returns a dict with keys: state, label, prob_risk_off, prob_transitional,
    prob_risk_on, confidence, cold_start, refit_dates. Each value (except
    refit_dates: list[Timestamp]) is a pandas object aligned to beta.index.

    Refit clock: every refit_interval_days (trading days) starting at
    min_history_days, params are re-estimated on beta[:t]. Filter clock: daily.
    Within a refit block [r, r'), filtered probs come from one .filter() over
    beta[:r'] with params(beta[:r]) (causal — see module Causality note).
    NaN beta rows are dropped before fitting and emitted as Unknown.
    """
    full_index = beta.index
    clean = beta.dropna()
    n = len(clean)

    probs = np.full((n, _K_REGIMES), np.nan)
    cold = np.ones(n, dtype=bool)
    refit_dates: list[pd.Timestamp] = []

    last_good: _FitResult | None = None
    r = min_history_days
    while r < n:
        block_end = min(r + refit_interval_days, n)
        fit = _fit_params(clean.iloc[:r], switching_variance=switching_variance)
        if fit.converged:
            last_good = fit
            refit_dates.append(pd.Timestamp(clean.index[r]))
        used = fit if fit.converged else last_good
        if used is not None:
            block_probs = _filtered_probs(
                clean.iloc[:block_end], used, switching_variance=switching_variance
            )
            probs[r:block_end] = block_probs[r:block_end]
            cold[r:block_end] = False
        r = block_end

    state_idx = np.where(np.isnan(probs).any(axis=1), -1, probs.argmax(axis=1))
    labels = np.array([_ORDERED_LABELS[i] if i >= 0 else _UNKNOWN for i in state_idx])
    confidence = np.where(np.isnan(probs).any(axis=1), np.nan, probs.max(axis=1))

    def _series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=clean.index).reindex(full_index)

    return {
        "state": _series(np.where(state_idx < 0, np.nan, state_idx)),
        "label": _series(labels).fillna(_UNKNOWN),
        "prob_risk_off": _series(probs[:, 0]),
        "prob_transitional": _series(probs[:, 1]),
        "prob_risk_on": _series(probs[:, 2]),
        "confidence": _series(confidence),
        "cold_start": _series(cold).fillna(True).astype(bool),
        "refit_dates": refit_dates,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_hmm.py::test_walk_forward_is_causal_no_lookahead -v`
Expected: PASS

- [ ] **Step 5: Add cold-start + recovery tests**

Add to `tests/test_regime_hmm.py`:
```python
def test_walk_forward_cold_start_is_unknown() -> None:
    beta = _three_regime_beta()
    out = walk_forward(
        beta, refit_interval_days=21, min_history_days=252, switching_variance=True
    )
    assert (out["label"].iloc[:252] == _UNKNOWN).all()
    assert out["cold_start"].iloc[:252].all()


def test_walk_forward_recovers_known_regimes() -> None:
    beta = _three_regime_beta()
    out = walk_forward(
        beta, refit_interval_days=21, min_history_days=252, switching_variance=True
    )
    # Final high-mean block should classify Risk-on on most non-cold days.
    tail = out["label"].iloc[-200:]
    assert (tail == "Risk-on").mean() > 0.8
```

Run: `pytest tests/test_regime_hmm.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add roro/regime_hmm.py tests/test_regime_hmm.py
git commit -m "feat(regime_hmm): causal walk-forward (monthly refit, daily filter)"
```

---

## Task 7: `classify_hmm` per-segment orchestration

**Files:**
- Modify: `roro/regime_hmm.py`
- Test: `tests/test_regime_hmm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime_hmm.py`:
```python
from roro.config import EngineConfig
from roro.regime_hmm import classify_hmm
from roro.types import BetaBySegment, BetaFrame
from pathlib import Path


def _bbs_from_beta(beta: pd.Series) -> BetaBySegment:
    cap = pd.DataFrame({"beta": beta, "r2": 0.5, "n": 20}, index=beta.index)
    bf = BetaFrame(cap_wtd=cap, eq_wtd=cap, slope_spread=pd.Series(0.0, index=beta.index))
    return BetaBySegment(by_segment={"global": bf, "LatAm": bf})


def test_classify_hmm_returns_frame_with_segments() -> None:
    beta = _three_regime_beta()
    cfg = EngineConfig(
        data_path=Path("d.xlsx"), output_dir=Path("out"),
        hmm_enabled=True, hmm_min_history_days=252, hmm_refit_interval_days=42,
    )
    frame = classify_hmm(_bbs_from_beta(beta), cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    assert set(frame.label.columns) == {"global", "LatAm"}
    assert frame.thin_cut_flag["LatAm"].iloc[-1]
    assert not frame.thin_cut_flag["global"].iloc[-1]
    assert "global" in frame.refit_dates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_hmm.py::test_classify_hmm_returns_frame_with_segments -v`
Expected: FAIL with `ImportError: cannot import name 'classify_hmm'`

- [ ] **Step 3: Implement classify_hmm**

Add to `roro/regime_hmm.py`:
```python
def classify_hmm(
    bbs: BetaBySegment,
    *,
    cfg: EngineConfig,
    thin_cuts: frozenset[str],
) -> HmmRegimeFrame:
    """Run the walk-forward HMM per segment; assemble an HmmRegimeFrame."""
    state, label, p_off, p_tr, p_on, conf, cold, nseg, thin = (
        {} for _ in range(9)
    )
    refit_dates: dict[str, list[pd.Timestamp]] = {}

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        out = walk_forward(
            beta,
            refit_interval_days=cfg.hmm_refit_interval_days,
            min_history_days=cfg.hmm_min_history_days,
            switching_variance=cfg.hmm_switching_variance,
        )
        state[cut] = out["state"]
        label[cut] = out["label"]
        p_off[cut] = out["prob_risk_off"]
        p_tr[cut] = out["prob_transitional"]
        p_on[cut] = out["prob_risk_on"]
        conf[cut] = out["confidence"]
        cold[cut] = out["cold_start"]
        nseg[cut] = bf.cap_wtd["n"]
        thin[cut] = pd.Series(cut in thin_cuts, index=beta.index)
        refit_dates[cut] = out["refit_dates"]

    return HmmRegimeFrame(
        state=pd.DataFrame(state),
        label=pd.DataFrame(label),
        prob_risk_off=pd.DataFrame(p_off),
        prob_transitional=pd.DataFrame(p_tr),
        prob_risk_on=pd.DataFrame(p_on),
        confidence=pd.DataFrame(conf),
        n_per_segment=pd.DataFrame(nseg),
        thin_cut_flag=pd.DataFrame(thin),
        cold_start_flag=pd.DataFrame(cold),
        refit_dates=refit_dates,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_hmm.py::test_classify_hmm_returns_frame_with_segments -v`
Expected: PASS

- [ ] **Step 5: Run full module test + type check**

Run: `pytest tests/test_regime_hmm.py -v && mypy roro/regime_hmm.py`
Expected: PASS, no mypy errors.

- [ ] **Step 6: Commit**

```bash
git add roro/regime_hmm.py tests/test_regime_hmm.py
git commit -m "feat(regime_hmm): classify_hmm per-segment orchestration"
```

---

## Task 8: Convergence-failure handling test

**Files:**
- Test: `tests/test_regime_hmm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime_hmm.py`:
```python
def test_degenerate_beta_does_not_crash_or_warn() -> None:
    # Near-constant beta: 3-state fit is ill-posed; must degrade gracefully.
    idx = pd.bdate_range("2010-01-01", periods=800)
    beta = pd.Series(np.full(800, 0.5) + 1e-9, index=idx, name="beta")
    out = walk_forward(
        beta, refit_interval_days=42, min_history_days=252, switching_variance=True
    )
    # No exception, output spans the full index, no warning escaped to error.
    assert len(out["label"]) == len(beta)
```

- [ ] **Step 2: Run test to verify it passes (behavior already implemented)**

Run: `pytest tests/test_regime_hmm.py::test_degenerate_beta_does_not_crash_or_warn -v`
Expected: PASS — `_fit_params` catches exceptions → `converged=False`; `walk_forward` falls back to `last_good` (or stays cold if none). `filterwarnings=error` does not trip because warnings are suppressed inside `_fit_params`/`_filtered_probs`.

If it FAILS because some warning still escapes, widen the `warnings.simplefilter("ignore")` scope to wrap the whole loop body in `walk_forward`, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_regime_hmm.py
git commit -m "test(regime_hmm): degenerate beta degrades without crash/warning"
```

---

## Task 9: Engine wiring

**Files:**
- Modify: `roro/engine.py:84-92` (after the percentile `classify` block)
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_engine.py` (uses the existing `tiny_xlsx` fixture + the project's fake FRED client pattern — copy the client construction from the existing engine test in this file):
```python
def test_engine_populates_regime_hmm_when_enabled(tiny_xlsx, monkeypatch) -> None:
    # Reuse this file's existing fake FRED client + run helper; only the
    # hmm_enabled flag and the assertion are new.
    from roro.config import EngineConfig
    cfg = _engine_cfg(tiny_xlsx)  # existing helper in this test module
    cfg = EngineConfig(**{**cfg.__dict__, "hmm_enabled": True,
                          "hmm_min_history_days": 120, "hmm_refit_interval_days": 60})
    result = _run_engine(cfg)     # existing helper in this test module
    assert result.regime_hmm is not None
    assert "global" in result.regime_hmm.label.columns


def test_engine_regime_hmm_none_when_disabled(tiny_xlsx) -> None:
    cfg = _engine_cfg(tiny_xlsx)
    result = _run_engine(cfg)
    assert result.regime_hmm is None
```

> If `_engine_cfg`/`_run_engine` helpers don't exist in `tests/test_engine.py`, inline the same construction the existing engine test uses (fake FRED client + `engine.run(...)`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_engine_populates_regime_hmm_when_enabled -v`
Expected: FAIL (`result.regime_hmm is None`).

- [ ] **Step 3: Wire it into the engine**

In `roro/engine.py`, add the import near the other classify import (line 8):
```python
from roro.regime_hmm import classify_hmm
```
Then after the percentile `classify(...)` block (ends line 92), add:
```python
    # 5b) Optional HMM regime classifier (parallel method, off by default).
    regime_hmm = (
        classify_hmm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
        if cfg.hmm_enabled
        else None
    )
```
Then in the `RunResult(...)` construction (line 141), add the field:
```python
        regime_hmm=regime_hmm,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (both new tests + existing).

- [ ] **Step 5: Commit**

```bash
git add roro/engine.py tests/test_engine.py
git commit -m "feat(engine): wire classify_hmm behind hmm_enabled flag"
```

---

## Task 10: Alerts — HMM bucket transitions

**Files:**
- Modify: `roro/types.py` (`AlertSet`), `roro/alerts.py`
- Test: `tests/test_alerts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_alerts.py`:
```python
def test_detect_alerts_emits_hmm_transitions() -> None:
    import pandas as pd
    from roro.alerts import detect_alerts
    from roro.types import (CorrelationFrame, HmmRegimeFrame, RegimeFrame,
                            ValidationFrame)

    idx = pd.bdate_range("2014-01-01", periods=3)
    empty = pd.DataFrame(index=idx)
    rf = RegimeFrame(percentile_5y=empty, tercile=empty, quintile=empty,
                     direction=empty, n_per_segment=empty, thin_cut_flag=empty,
                     bootstrap_flag=empty)
    labels = pd.DataFrame({"global": ["Risk-off", "Risk-off", "Risk-on"]}, index=idx)
    hmm = HmmRegimeFrame(state=labels, label=labels, prob_risk_off=empty,
                         prob_transitional=empty, prob_risk_on=empty, confidence=empty,
                         n_per_segment=empty, thin_cut_flag=empty, cold_start_flag=empty)
    cf = CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty)
    vf = ValidationFrame(rolling_corr_60d=empty, internal_consistency=empty,
                         correlation_alerts=empty)
    out = detect_alerts(regime=rf, correlation=cf, validation=vf, regime_hmm=hmm)
    assert (out.hmm_bucket_transitions["to_bucket"] == "Risk-on").any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_alerts.py::test_detect_alerts_emits_hmm_transitions -v`
Expected: FAIL (`detect_alerts` has no `regime_hmm` param / `AlertSet` has no `hmm_bucket_transitions`).

- [ ] **Step 3: Extend AlertSet + detect_alerts**

In `roro/types.py`, add to `AlertSet` (after `validation_degradation`, line 86):
```python
    hmm_bucket_transitions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "segment", "from_bucket", "to_bucket"]
        )
    )
```

In `roro/alerts.py`, change the signature and body of `detect_alerts` (line 12):
```python
def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
    regime_hmm: HmmRegimeFrame | None = None,
) -> AlertSet:
    return AlertSet(
        bucket_transitions=_bucket_transitions(regime.tercile),
        disagreement_events=_disagreement_events(regime, correlation),
        validation_degradation=_validation_degradation(validation),
        hmm_bucket_transitions=(
            _bucket_transitions(regime_hmm.label)
            if regime_hmm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
    )
```
Add `HmmRegimeFrame` to the import on line 7:
```python
from roro.types import AlertSet, CorrelationFrame, HmmRegimeFrame, RegimeFrame, ValidationFrame
```

> `_bucket_transitions` ignores `Unknown`→label flips only if `prev.notna()`; since labels are strings, an `Unknown`→`Risk-on` flip WILL register. That is acceptable (cold-start exit is a real first signal), but note it for the compare report.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_alerts.py -v`
Expected: PASS (new + existing; existing `detect_alerts` calls omit `regime_hmm`, defaulting to empty transitions).

- [ ] **Step 5: Wire the engine call**

In `roro/engine.py`, update the `detect_alerts(...)` call (line 133):
```python
    alerts = detect_alerts(
        regime=regime, correlation=correlation, validation=validation, regime_hmm=regime_hmm
    )
```
Run: `pytest tests/test_engine.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add roro/types.py roro/alerts.py roro/engine.py tests/test_alerts.py
git commit -m "feat(alerts): emit hmm_bucket_transitions from HMM labels"
```

---

## Task 11: Output artifacts (CSV + snapshot)

**Files:**
- Modify: `roro/io.py` (`write_run`, new writers, `_build_snapshot`)
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_io.py`:
```python
def test_write_run_emits_hmm_artifacts(tmp_path) -> None:
    import pandas as pd
    from roro.io import write_run
    # Build a minimal RunResult with regime_hmm populated. Reuse this module's
    # existing _minimal_run_result() helper if present; otherwise construct one
    # the same way the existing write_run test does, then attach regime_hmm.
    result = _minimal_run_result_with_hmm(tmp_path)  # see helper note below
    out = write_run(result, run_date="2026-06-03", out_dir=tmp_path / "runs",
                    as_of_data_date="2026-06-03", force=True)
    assert (out / "regimes_hmm.csv").exists()
    assert (out / "hmm_refit_log.csv").exists()
    df = pd.read_csv(out / "regimes_hmm.csv")
    assert {"date", "segment", "label", "p_risk_off"}.issubset(df.columns)
```

> Helper `_minimal_run_result_with_hmm`: copy the existing minimal-RunResult builder in `tests/test_io.py`, then set `regime_hmm=HmmRegimeFrame(...)` with one segment ("global") and a 3-row index. If no builder exists, construct `RunResult` directly mirroring `tests/test_engine.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io.py::test_write_run_emits_hmm_artifacts -v`
Expected: FAIL (files not written).

- [ ] **Step 3: Add writers + wire into write_run**

In `roro/io.py`, add `HmmRegimeFrame` to the types import (line 17 block). After `_write_tripwire` (line 251), add:
```python
def _write_regime_hmm(hf: "HmmRegimeFrame", path: Path) -> None:
    cols = "date,segment,state,label,p_risk_off,p_transitional,p_risk_on,confidence,cold_start,thin_cut\n"
    if hf.label.empty:
        path.write_text(cols, encoding="utf-8")
        return
    merged = _melt_with_date(hf.label, "label")
    for name, frame in (
        ("state", hf.state),
        ("p_risk_off", hf.prob_risk_off),
        ("p_transitional", hf.prob_transitional),
        ("p_risk_on", hf.prob_risk_on),
        ("confidence", hf.confidence),
        ("cold_start", hf.cold_start_flag),
        ("thin_cut", hf.thin_cut_flag),
    ):
        merged = merged.merge(_melt_with_date(frame, name), on=["date", "segment"], how="left")
    merged.to_csv(path, index=False)


def _write_hmm_refit_log(hf: "HmmRegimeFrame", path: Path) -> None:
    rows = [
        {"segment": seg, "refit_date": d}
        for seg, dates in hf.refit_dates.items()
        for d in dates
    ]
    df = pd.DataFrame(rows, columns=["segment", "refit_date"])
    df.to_csv(path, index=False)
```
Then inside `write_run`, after the `_write_tripwire(...)` line (line 128), add:
```python
    if result.regime_hmm is not None:
        _write_regime_hmm(result.regime_hmm, tmp / "regimes_hmm.csv")
        _write_hmm_refit_log(result.regime_hmm, tmp / "hmm_refit_log.csv")
```

- [ ] **Step 4: Add snapshot block**

In `_build_snapshot` (line 254), before the `return`, build an optional block and include it:
```python
    hmm_block: dict[str, Any] = {}
    if result.regime_hmm is not None and not result.regime_hmm.label.empty:
        hf = result.regime_hmm
        last = hf.label.index[-1]
        for seg in hf.label.columns:
            hmm_block[seg] = {
                "label": hf.label.loc[last, seg],
                "p_risk_off": _safe_float(hf.prob_risk_off.loc[last, seg]),
                "p_transitional": _safe_float(hf.prob_transitional.loc[last, seg]),
                "p_risk_on": _safe_float(hf.prob_risk_on.loc[last, seg]),
                "confidence": _safe_float(hf.confidence.loc[last, seg]),
            }
```
Add `"regime_hmm": hmm_block` to the returned dict. Add this helper above `_build_snapshot`:
```python
def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`
Expected: PASS. Existing runs (regime_hmm None) write no HMM files and an empty `regime_hmm` snapshot block.

- [ ] **Step 6: Commit**

```bash
git add roro/io.py tests/test_io.py
git commit -m "feat(io): write regimes_hmm.csv, hmm_refit_log.csv, snapshot block"
```

---

## Task 12: Backtest — parametrized scorer + compare report

**Files:**
- Modify: `roro/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest.py`:
```python
def test_evaluate_gates_accepts_explicit_label_source() -> None:
    import pandas as pd
    from roro.backtest import _evaluate_gates
    # Reuse this module's existing backtest RunResult builder.
    result = _backtest_result()  # existing helper
    gates = _evaluate_gates(
        result,
        labels=result.regime.tercile,
        transitions=result.alerts.bucket_transitions,
    )
    assert set(gates) == {"G1_vix", "G2_bbb", "G3_events", "G4_segmentation_lift",
                          "G5_stability", "G6_internal"}
```

> If `_backtest_result` doesn't exist, build a `RunResult` the way the existing backtest test does.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py::test_evaluate_gates_accepts_explicit_label_source -v`
Expected: FAIL (`_evaluate_gates` takes only `result`).

- [ ] **Step 3: Parametrize the scorer over a label source**

In `roro/backtest.py`, change `_evaluate_gates` (line 106) and the gate functions that read `result.regime.tercile` / `result.alerts.bucket_transitions` to take explicit `labels`/`transitions`:
```python
def _evaluate_gates(
    result: RunResult,
    *,
    labels: pd.DataFrame,
    transitions: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    gates["G1_vix"] = _gate_external_corr(result, series_id="VIXCLS",
                                          rho_min=_G1_VIX_RHO_MIN, fraction_min=_G1_FRACTION_MIN)
    gates["G2_bbb"] = _gate_external_corr(result, series_id="BAMLC0A4CBBB",
                                          rho_min=_G2_BBB_RHO_MIN, fraction_min=_G2_FRACTION_MIN)
    gates["G3_events"] = _gate_events(labels)
    gates["G4_segmentation_lift"] = _gate_segmentation_lift(labels)
    gates["G5_stability"] = _gate_stability(result, transitions)
    gates["G6_internal"] = _gate_internal_consistency(result)
    return gates
```
Update `_gate_events(result)` → `_gate_events(terc: pd.DataFrame)` (use `terc` instead of `result.regime.tercile` at line 146). Update `_gate_segmentation_lift(result)` → `_gate_segmentation_lift(terc: pd.DataFrame)` (line 170). Update `_gate_stability(result)` → `_gate_stability(result, transitions: pd.DataFrame)` and use the passed `transitions` instead of `result.alerts.bucket_transitions` (line 183-184). Leave G1/G2/G6 reading from `result` (external/internal validation are method-shared in v1.1).

- [ ] **Step 4: Update the single caller + add the HMM pass**

In `run_backtest` (line 89), replace the single `_evaluate_gates(result)` call:
```python
    gates = _evaluate_gates(
        result,
        labels=result.regime.tercile,
        transitions=result.alerts.bucket_transitions,
    )
    report: dict[str, Any] = {
        "start": start, "end": end,
        "methodology_version": cfg.methodology_version,
        "gates": gates,
        "all_passed": all(bool(v.get("passed", False)) for v in gates.values()),
    }
    (cfg.output_dir / "acceptance_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    if result.regime_hmm is not None:
        hmm_gates = _evaluate_gates(
            result,
            labels=result.regime_hmm.label,
            transitions=result.alerts.hmm_bucket_transitions,
        )
        hmm_report = {**report, "method": "hmm", "gates": hmm_gates,
                      "all_passed": all(bool(v.get("passed", False)) for v in hmm_gates.values())}
        (cfg.output_dir / "acceptance_report_hmm.json").write_text(
            json.dumps(hmm_report, indent=2, default=str), encoding="utf-8")
        compare = {
            g: {"percentile": gates[g], "hmm": hmm_gates[g]} for g in gates
        }
        (cfg.output_dir / "acceptance_compare.json").write_text(
            json.dumps(compare, indent=2, default=str), encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v && mypy roro/backtest.py`
Expected: PASS, no mypy errors.

- [ ] **Step 6: Commit**

```bash
git add roro/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): score both methods through shared gates + compare report"
```

---

## Task 13: Full suite + golden/reproducibility check

**Files:** none (verification task)

- [ ] **Step 1: Run the whole suite (fast subset)**

Run: `pytest -m "not slow" -q`
Expected: PASS. If `tests/test_golden.py` or `tests/test_reproducibility.py` fail because a snapshot now contains an (empty) `regime_hmm` block, regenerate goldens: `pytest tests/test_golden.py --regenerate-goldens` then re-run and inspect the diff is only the additive `regime_hmm: {}` key.

- [ ] **Step 2: Type + lint gate**

Run: `mypy roro && ruff check roro tests`
Expected: clean.

- [ ] **Step 3: Commit any golden updates**

```bash
git add tests/
git commit -m "test: regenerate goldens for additive regime_hmm snapshot key"
```

---

## Task 14: Cadence-invariance bench

**Files:**
- Create: `tests/test_regime_hmm_cadence.py`

- [ ] **Step 1: Write the bench (slow)**

Create `tests/test_regime_hmm_cadence.py`:
```python
import numpy as np
import pandas as pd
import pytest

from roro.regime_hmm import walk_forward


def _long_three_regime_beta(seed: int = 7, n_per: int = 1200) -> pd.Series:
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(m, 0.06, n_per) for m in (-0.7, 0.0, 0.7)]
    vals = np.concatenate(blocks)
    idx = pd.bdate_range("2008-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, name="beta")


@pytest.mark.slow
def test_monthly_matches_daily_label_agreement() -> None:
    beta = _long_three_regime_beta()
    daily = walk_forward(beta, refit_interval_days=1, min_history_days=252,
                         switching_variance=True)["label"]
    monthly = walk_forward(beta, refit_interval_days=21, min_history_days=252,
                           switching_variance=True)["label"]
    mask = (daily != "Unknown") & (monthly != "Unknown")
    agreement = float((daily[mask] == monthly[mask]).mean())
    assert agreement >= 0.98, f"cadence label agreement {agreement:.3f} < 0.98"
```

- [ ] **Step 2: Run the bench**

Run: `pytest tests/test_regime_hmm_cadence.py -m slow -v`
Expected: PASS (agreement ≥ 0.98). If it fails, the bench has surfaced a real cadence sensitivity — record the achieved agreement, raise `hmm_refit_interval_days` default in `config.py`, and note it in the results log (next step).

- [ ] **Step 3: Log the result**

Append one line to `docs/context/results.md` recording the achieved label agreement and the chosen production cadence.

- [ ] **Step 4: Commit**

```bash
git add tests/test_regime_hmm_cadence.py docs/context/results.md
git commit -m "test(regime_hmm): cadence-invariance bench (monthly vs daily)"
```

---

## Task 15: Real-data backtest comparison + decision

**Files:** none (analysis task; updates docs)

- [ ] **Step 1: Run the backtest with HMM enabled on real data**

Set `hmm_enabled: true` in the backtest config (or pass as an override) and run the project's backtest entry point over the full 2008–2024 window. Confirm `acceptance_report_hmm.json` and `acceptance_compare.json` are written.

- [ ] **Step 2: Apply the decision rule (spec §8)**

HMM becomes the production default ONLY if it passes all 6 gates AND improves G5 (stability) without regressing G3 (8/8 events). Otherwise percentile stays default; HMM ships as overlay.

- [ ] **Step 3: Record the decision**

Append the gate-by-gate comparison and the chosen default to `docs/context/results.md`, add a one-line `decision:` to `docs/context/memory.md`, and check off H1–H7 in `docs/context/todo.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/context/
git commit -m "docs: HMM vs percentile backtest comparison + production-default decision"
```

---

## Self-Review

**Spec coverage:** §3 architecture → Tasks 4–7; §4 config → Task 2; §5 HMM core (model, ordering, argmax, convergence, determinism) → Tasks 4, 5, 8; §6 walk-forward + cadence → Tasks 6, 14; §7 integration (engine, alerts, io) → Tasks 9, 10, 11; §8 backtest comparison + decision rule → Tasks 12, 15; §9 testing → embedded per task + Tasks 8, 14; §10 risks (determinism, warnings, label-switching, thin cuts) → Tasks 1, 4, 5, 8; §12 dep → Task 1. All covered.

**Placeholder scan:** No TBD/TODO. Test-helper reuse notes (`_engine_cfg`, `_backtest_result`, `_minimal_run_result_with_hmm`) point at existing per-file builders with an explicit fallback ("construct as the existing test does") — not deferred work.

**Type consistency:** `_FitResult(params, perm, means, converged)` used identically in Tasks 4–6. `walk_forward(...)` return-dict keys (`state`, `label`, `prob_risk_off`, `prob_transitional`, `prob_risk_on`, `confidence`, `cold_start`, `refit_dates`) consumed consistently in Task 7. `HmmRegimeFrame` field names match across types.py (Task 3), classify_hmm (Task 7), alerts (Task 10), io (Task 11). `_ORDERED_LABELS` vocabulary matches the percentile `{Risk-off, Transitional, Risk-on}` so gates/alerts are method-agnostic.
