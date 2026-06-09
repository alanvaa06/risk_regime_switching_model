# Statistical Jump Model Regime Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, causal 3-state **statistical jump model** regime classifier (`roro/regime_jm.py`) mirroring the HMM contract, off by default, with a 3-way report band toggle (Percentile↔HMM↔JM) + a JM state-probability figure.

**Architecture:** A vendored numpy-only JM core (`roro/jump_model.py`: seeded k-means++ + O(T·K²) Viterbi DP + coordinate descent) is wrapped by a causal `walk_forward` (expanding/rolling refit + per-block frozen-scaler forward-DP online inference) and a per-segment `classify_jm`, then wired through engine/io/alerts/backtest/report exactly parallel to the HMM. Determinism (RoRo's #1 invariant) comes from `SeedSequence(seed).spawn` + PCG64, strict tie-breaks, and per-refit mean-sort canonicalization.

**Tech Stack:** Python 3.12, numpy (pinned <2.1), pandas (<2.3), Click, pytest. `uv`-managed `.venv` — use `.venv\Scripts\python.exe`. Quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest filterwarnings=["error"]`.

**Spec:** `docs/superpowers/specs/2026-06-09-jm-regime-design.md`. **Integration contract:** `docs/superpowers/jm-integration-contract.md`. **HMM template to mirror:** `roro/regime_hmm.py`.

---

## File Structure

- **Create** `roro/jump_model.py` — vendored numpy-only JM core (pure; no pandas/sklearn). Discrete + (J3) continuous.
- **Create** `roro/regime_jm.py` — `walk_forward` + `classify_jm` (pandas; mirrors `regime_hmm.py`).
- **Modify** `roro/config.py` — `jm_*` fields on `EngineConfig`.
- **Modify** `roro/types.py` — `JmRegimeFrame`, `RunResult.regime_jm`, `AlertSet.jm_bucket_transitions`.
- **Modify** `roro/engine.py`, `roro/alerts.py`, `roro/io.py`, `roro/backtest.py` — JM wiring (each a minimal HMM parallel).
- **Modify** `roro/report/bundle.py`, `load.py`, `figures.py`, `html.py`, `orchestrate.py` — 3-way band toggle + JM probability figure.
- **Create** `tests/test_jump_model.py`, `tests/test_regime_jm.py`, `tests/test_regime_jm_cadence.py`; extend `tests/test_config.py`, `tests/test_types.py`, `tests/test_io.py`, `tests/test_backtest.py`, `tests/report/conftest.py`, `tests/report/test_orchestrate.py`.

---

## Task 1: JM core — DP primitives (`roro/jump_model.py`)

**Files:**
- Create: `roro/jump_model.py`
- Test: `tests/test_jump_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_jump_model.py`:

```python
"""Tests for the vendored numpy-only statistical jump model core."""

from __future__ import annotations

import numpy as np

from roro.jump_model import (
    _forward_values,
    _loss_matrix,
    _viterbi_path,
    online_states,
)


def test_loss_matrix_shape_and_values() -> None:
    y = np.array([0.0, 1.0, 2.0])
    c = np.array([0.0, 2.0])
    lm = _loss_matrix(y, c)
    assert lm.shape == (3, 2)
    assert np.isclose(lm[1, 0], 0.5 * 1.0)  # 0.5*(1-0)^2
    assert np.isclose(lm[2, 1], 0.0)        # 0.5*(2-2)^2


def test_viterbi_lambda_zero_is_nearest_centroid() -> None:
    y = np.array([0.0, 0.1, 5.0, 5.1, 0.0])
    c = np.array([0.0, 5.0])
    path = _viterbi_path(_loss_matrix(y, c), 0.0)
    assert path.tolist() == [0, 0, 1, 1, 0]  # no jump cost -> pure assignment


def test_viterbi_huge_lambda_collapses_to_single_state() -> None:
    y = np.array([0.0, 5.0, 0.0, 5.0])
    c = np.array([0.0, 5.0])
    path = _viterbi_path(_loss_matrix(y, c), 1e6)
    assert len(set(path.tolist())) == 1  # one transition too costly -> single state


def test_forward_values_prefix_stable_no_lookahead() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    c = np.array([-1.0, 0.0, 1.0])
    lm = _loss_matrix(y, c)
    full = _forward_values(lm, 5.0)
    prefix = _forward_values(lm[:120], 5.0)
    assert np.allclose(full[:120], prefix)  # values[e] depends only on rows <= e


def test_online_states_equals_forward_argmin() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    c = np.array([-1.0, 0.0, 1.0])
    st = online_states(y, c, 2.0)
    vals = _forward_values(_loss_matrix(y, c), 2.0)
    assert st.tolist() == vals.argmin(axis=1).tolist()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roro.jump_model'`.

- [ ] **Step 3: Implement the DP primitives**

Create `roro/jump_model.py`:

```python
"""Vendored deterministic statistical jump model (discrete; continuous in J3).

Pure numpy -- no pandas, no scikit-learn. The "jump" is a regime-transition
penalty in an unsupervised clustering objective (Bemporad et al. 2018, Automatica
96:11-21; Nystrup et al. 2020, ESWA 150; Shu-Mulvey 2024, J. Asset Management /
arXiv:2402.05272). It is UNRELATED to jump-diffusion option pricing -- same word,
different machinery. Algorithm adapted (Apache-2.0) from the `jumpmodels` package.

Determinism is the contract: every result is a pure function of
(y, seed, jump_penalty, k, n_init, max_iter, tol). Seeding uses numpy
SeedSequence/PCG64 only; there is no global RNG, shuffle, or unseeded draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NDArrayF = np.ndarray  # float64 (T,) or (T,k); annotated loosely for mypy --strict


def _loss_matrix(y: _NDArrayF, centroids: _NDArrayF) -> _NDArrayF:
    """(T,) y, (k,) centroids -> (T,k) of 0.5*(y_t - theta_k)^2 (scaled L2 loss)."""
    diff = y[:, None] - centroids[None, :]
    return 0.5 * diff * diff


def _forward_values(loss_mx: _NDArrayF, jump_penalty: float) -> _NDArrayF:
    """Forward DP value matrix (T,k). values[t] depends only on loss_mx[:t+1].

    values[t,s] = loss_mx[t,s] + min_j (values[t-1,j] + lambda*1{j!=s}).
    The per-index argmin of this matrix is the causal online filter (no lookahead).
    """
    t_len, k = loss_mx.shape
    penalty = jump_penalty * (1.0 - np.eye(k))
    values = np.empty((t_len, k), dtype=float)
    values[0] = loss_mx[0]
    for t in range(1, t_len):
        values[t] = loss_mx[t] + (values[t - 1][:, None] + penalty).min(axis=0)
    return values


def _viterbi_path(loss_mx: _NDArrayF, jump_penalty: float) -> _NDArrayF:
    """Offline (two-sided) optimal state path via backward reconstruction.

    Uses future rows during backtracking -> FIT-ONLY (on a closed historical
    window). NEVER use for the live signal; online inference uses _forward_values
    argmin instead. numpy argmin returns the first minimal index -> deterministic.
    """
    t_len, k = loss_mx.shape
    penalty = jump_penalty * (1.0 - np.eye(k))
    values = _forward_values(loss_mx, jump_penalty)
    assign = np.empty(t_len, dtype=np.intp)
    assign[t_len - 1] = int(values[t_len - 1].argmin())
    for t in range(t_len - 1, 0, -1):
        assign[t - 1] = int((values[t - 1] + penalty[:, assign[t]]).argmin())
    return assign


def online_states(y: _NDArrayF, centroids: _NDArrayF, jump_penalty: float) -> _NDArrayF:
    """Causal per-index state = argmin of the forward DP value (no backward pass)."""
    values = _forward_values(_loss_matrix(y, centroids), jump_penalty)
    return values.argmin(axis=1).astype(np.intp)
```

- [ ] **Step 4: Run to verify pass + quality**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m mypy --strict roro/jump_model.py` and `.venv\Scripts\python.exe -m ruff check roro/jump_model.py tests/test_jump_model.py`
Expected: clean. (If mypy flags the loose `np.ndarray` annotations, add `# type: ignore[type-arg]` on the `_NDArrayF` alias line exactly as `roro/regime_hmm.py:28` does.)

- [ ] **Step 5: Commit**

```bash
git add roro/jump_model.py tests/test_jump_model.py
git commit -m "feat(jm): vendored JM core DP primitives (loss/forward-values/viterbi/online)"
```

---

## Task 2: JM core — k-means++ init, coordinate descent, fit

**Files:**
- Modify: `roro/jump_model.py`
- Test: `tests/test_jump_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_jump_model.py`:

```python
from roro.jump_model import JumpFit, fit_jump_model


def test_lambda_zero_matches_kmeans_assignment() -> None:
    # Two well-separated clusters; lambda=0 -> labels = nearest centroid (k-means).
    rng = np.random.default_rng(7)
    y = np.concatenate([rng.normal(-5, 0.1, 50), rng.normal(5, 0.1, 50)])
    fit = fit_jump_model(y, k=2, jump_penalty=0.0, seed=0)
    # canonical: centroids ascending -> state 0 is the -5 cluster
    assert fit.centroids[0] < fit.centroids[1]
    assert (fit.labels[:50] == 0).all()
    assert (fit.labels[50:] == 1).all()


def test_fit_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    a = fit_jump_model(y, k=3, jump_penalty=10.0, seed=0)
    b = fit_jump_model(y, k=3, jump_penalty=10.0, seed=0)
    assert np.array_equal(a.labels, b.labels)
    assert np.allclose(a.centroids, b.centroids)
    assert a.objective == b.objective


def test_fit_recovers_two_regimes_with_persistence() -> None:
    # Persistent regime structure: 100 low then 100 high; jump penalty keeps it clean.
    y = np.concatenate([np.full(100, -3.0), np.full(100, 3.0)]) + \
        np.random.default_rng(5).normal(0, 0.2, 200)
    fit = fit_jump_model(y, k=2, jump_penalty=20.0, seed=0)
    assert fit.converged
    # exactly one transition in the recovered path
    assert int((fit.labels[1:] != fit.labels[:-1]).sum()) == 1


def test_fit_canonical_state_ordering_by_mean() -> None:
    rng = np.random.default_rng(9)
    y = np.concatenate([rng.normal(2, 0.1, 60), rng.normal(-2, 0.1, 60)])
    fit = fit_jump_model(y, k=2, jump_penalty=5.0, seed=0)
    assert fit.centroids[0] < fit.centroids[1]  # ascending -> Risk-off lowest


def test_fit_handles_degenerate_single_value_window() -> None:
    # All-identical input -> empty-cluster reseed must not NaN or crash.
    y = np.full(40, 1.5)
    fit = fit_jump_model(y, k=3, jump_penalty=5.0, seed=0)
    assert np.all(np.isfinite(fit.centroids))
    assert fit.labels.shape == (40,)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py -k "lambda_zero_matches or deterministic or recovers or canonical or degenerate" -v`
Expected: FAIL — `ImportError: cannot import name 'fit_jump_model'`.

- [ ] **Step 3: Implement init + descent + fit**

Add to `roro/jump_model.py` (the `JumpFit` dataclass near the top after imports, the helpers before `online_states`):

```python
@dataclass(frozen=True)
class JumpFit:
    centroids: _NDArrayF   # (k,) sorted ascending (canonical: state 0 = lowest mean)
    labels: _NDArrayF      # (T,) intp in [0,k)
    objective: float
    converged: bool


def _kmeanspp_init(y: _NDArrayF, k: int, rng: np.random.Generator) -> _NDArrayF:
    """Seeded k-means++ on a 1-D series. Pure function of (y, k, rng state)."""
    t_len = y.shape[0]
    centers = np.empty(k, dtype=float)
    centers[0] = y[int(rng.integers(t_len))]
    d2 = (y - centers[0]) ** 2
    for j in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:  # all points coincide with chosen centers
            centers[j] = y[int(rng.integers(t_len))]
        else:
            cumulative = np.cumsum(d2 / total)
            idx = int(np.searchsorted(cumulative, rng.random(), side="right"))
            centers[j] = y[min(idx, t_len - 1)]
        d2 = np.minimum(d2, (y - centers[j]) ** 2)
    return centers


def _update_centroids(
    y: _NDArrayF, labels: _NDArrayF, k: int, prev: _NDArrayF
) -> _NDArrayF:
    """M-step: occupied states -> cluster mean; empty states -> sequential
    farthest-point reseed (recomputed after each, so two empties never collide)."""
    t_len = y.shape[0]
    centroids = prev.astype(float).copy()
    is_set = np.zeros(k, dtype=bool)
    for c in range(k):
        mask = labels == c
        if bool(mask.any()):
            centroids[c] = float(y[mask].mean())
            is_set[c] = True
    if bool(is_set.all()):
        return centroids
    claimed = np.zeros(t_len, dtype=bool)
    for c in range(k):
        if is_set[c]:
            continue
        occupied = centroids[is_set]
        d2 = ((y[:, None] - occupied[None, :]) ** 2).min(axis=1)
        d2 = np.where(claimed, -np.inf, d2)
        idx = int(d2.argmax())  # farthest unclaimed point, first-index tie-break
        centroids[c] = y[idx]
        claimed[idx] = True
        is_set[c] = True
    return centroids


def _objective(loss_mx: _NDArrayF, labels: _NDArrayF, jump_penalty: float) -> float:
    fit_loss = float(loss_mx[np.arange(labels.shape[0]), labels].sum())
    jumps = int((labels[1:] != labels[:-1]).sum())
    return fit_loss + jump_penalty * jumps


def _fit_once(
    y: _NDArrayF, k: int, jump_penalty: float, max_iter: int, tol: float,
    rng: np.random.Generator,
) -> tuple[_NDArrayF, _NDArrayF, float]:
    """One restart of coordinate descent. Returns (centroids, labels, objective)."""
    centroids = _kmeanspp_init(y, k, rng)
    loss_mx = _loss_matrix(y, centroids)
    labels = _viterbi_path(loss_mx, jump_penalty)  # fit on a CLOSED window -> two-sided OK
    obj = _objective(loss_mx, labels, jump_penalty)
    for _ in range(max_iter):
        centroids = _update_centroids(y, labels, k, centroids)
        loss_mx = _loss_matrix(y, centroids)
        new_labels = _viterbi_path(loss_mx, jump_penalty)
        new_obj = _objective(loss_mx, new_labels, jump_penalty)
        stop = bool(np.array_equal(new_labels, labels)) or (obj - new_obj) < tol
        labels, obj = new_labels, new_obj
        if stop:
            break
    return centroids, labels, obj


def fit_jump_model(
    y: _NDArrayF, *, k: int = 3, jump_penalty: float = 50.0, n_init: int = 10,
    max_iter: int = 30, tol: float = 1e-8, seed: int = 0,
) -> JumpFit:
    """Fit a discrete K-state jump model with n_init seeded restarts.

    Deterministic: pure function of (y, seed, jump_penalty, k, n_init, max_iter, tol).
    Restarts are seeded by SeedSequence(seed).spawn(n_init); the strict-lowest-objective
    restart wins (lowest-index on ties). States are canonicalized ascending by centroid.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    children = np.random.SeedSequence(seed).spawn(n_init)
    best: tuple[_NDArrayF, _NDArrayF, float] | None = None
    for child in children:
        rng = np.random.Generator(np.random.PCG64(child))
        centroids, labels, obj = _fit_once(y, k, jump_penalty, max_iter, tol, rng)
        if best is None or obj < best[2]:  # strict '<' -> first/lowest-index wins ties
            best = (centroids, labels, obj)
    assert best is not None
    centroids, labels, obj = best
    converged = bool(np.all(np.isfinite(centroids)) and np.isfinite(obj))
    perm = np.argsort(centroids, kind="stable")
    inv = np.argsort(perm, kind="stable")
    return JumpFit(
        centroids=centroids[perm],
        labels=inv[labels].astype(np.intp),
        objective=float(obj),
        converged=converged,
    )
```

- [ ] **Step 4: Run to verify pass + quality**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py -v`
Expected: PASS (all core tests).
Run: `.venv\Scripts\python.exe -m mypy --strict roro/jump_model.py` and `.venv\Scripts\python.exe -m ruff check roro/jump_model.py tests/test_jump_model.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add roro/jump_model.py tests/test_jump_model.py
git commit -m "feat(jm): k-means++ init, coordinate descent, deterministic fit"
```

---

## Task 3: Config fields + types (`config.py`, `types.py`)

**Files:**
- Modify: `roro/config.py`, `roro/types.py`
- Test: `tests/test_config.py`, `tests/test_types.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_jm_config_defaults_and_roundtrip() -> None:
    from roro.config import EngineConfig

    cfg = EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o"))
    assert cfg.jm_enabled is False
    assert cfg.jm_jump_penalty == 50.0
    assert cfg.jm_n_states == 3
    assert cfg.jm_window == "expanding"
    d = cfg.to_dict()
    for key in ("jm_enabled", "jm_jump_penalty", "jm_refit_interval_days",
                "jm_min_history_days", "jm_window", "jm_rolling_window_days",
                "jm_continuous", "jm_n_init", "jm_max_iter", "jm_tol", "jm_random_seed"):
        assert key in d
```

(`Path` and `EngineConfig` import conventions already exist in `tests/test_config.py`.) Append to `tests/test_types.py`:

```python
def test_jm_regime_frame_fields() -> None:
    import pandas as pd

    from roro.types import JmRegimeFrame

    empty = pd.DataFrame()
    f = JmRegimeFrame(
        state=empty, label=empty, prob_risk_off=empty, prob_transitional=empty,
        prob_risk_on=empty, confidence=empty, n_per_segment=empty,
        thin_cut_flag=empty, cold_start_flag=empty,
    )
    assert f.refit_dates == {}
    assert f.jump_penalty_used == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_jm_config_defaults_and_roundtrip tests/test_types.py::test_jm_regime_frame_fields -v`
Expected: FAIL — `AttributeError`/`ImportError` (`jm_enabled` / `JmRegimeFrame` missing).

- [ ] **Step 3: Implement config + type**

In `roro/config.py`, add these fields to `EngineConfig` immediately after the `hmm_*` block (the five `hmm_` fields). Keep them plain frozen-dataclass fields — `to_dict()` and `load_config` are generic over `fields()`, so no other change is needed:

```python
    # Statistical Jump Model overlay (parallel method, off by default).
    jm_enabled: bool = False
    jm_jump_penalty: float = 50.0
    jm_n_states: int = 3
    jm_refit_interval_days: int = 21
    jm_min_history_days: int = 252
    jm_window: str = "expanding"
    jm_rolling_window_days: int = 2000
    jm_continuous: bool = False
    jm_n_init: int = 10
    jm_max_iter: int = 30
    jm_tol: float = 1e-8
    jm_random_seed: int = 0
```

In `roro/types.py`, add the `JmRegimeFrame` dataclass immediately after `HmmRegimeFrame` (around line 81). It is structurally identical to `HmmRegimeFrame` plus `jump_penalty_used`:

```python
@dataclass(frozen=True)
class JmRegimeFrame:
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
    jump_penalty_used: dict[str, float] = field(default_factory=dict)
```

In `roro/types.py`, add `regime_jm` to `RunResult` right after `regime_hmm` (line 120):

```python
    regime_jm: "JmRegimeFrame | None" = None
```

And add `jm_bucket_transitions` to `AlertSet` (after `hmm_bucket_transitions`, line 105):

```python
    jm_bucket_transitions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "segment", "from_bucket", "to_bucket"]
        )
    )
```

- [ ] **Step 4: Run to verify pass + full config/types suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/config.py roro/types.py tests/test_config.py tests/test_types.py
git commit -m "feat(jm): EngineConfig jm_* fields + JmRegimeFrame/RunResult/AlertSet contracts"
```

---

## Task 4: Causal `walk_forward` (`roro/regime_jm.py`)

**Files:**
- Create: `roro/regime_jm.py`
- Test: `tests/test_regime_jm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_regime_jm.py`:

```python
"""Tests for the causal JM walk-forward + per-segment classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.regime_jm import walk_forward


def _two_regime_series(n: int = 600) -> pd.Series:
    rng = np.random.default_rng(11)
    lo = rng.normal(-2.0, 0.3, n // 2)
    hi = rng.normal(2.0, 0.3, n - n // 2)
    vals = np.concatenate([lo, hi])
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(vals, index=idx)


def test_walk_forward_returns_hmm_parity_keys() -> None:
    out = walk_forward(_two_regime_series(), jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=252, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert set(out) == {"state", "label", "prob_risk_off", "prob_transitional",
                        "prob_risk_on", "confidence", "cold_start", "refit_dates"}
    assert out["label"].iloc[:252].eq("Unknown").all()         # warmup
    assert bool(out["cold_start"].iloc[:252].all())
    assert out["label"].isin({"Risk-off", "Transitional", "Risk-on", "Unknown"}).all()


def test_walk_forward_deterministic() -> None:
    s = _two_regime_series()
    kw = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["label"], b["label"])
    pd.testing.assert_series_equal(a["state"], b["state"])


def test_walk_forward_no_lookahead() -> None:
    # Label at date t is invariant to appending future rows (causal online inference).
    s = _two_regime_series(600)
    kw = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    full = walk_forward(s, **kw)["label"]
    prefix = walk_forward(s.iloc[:500], **kw)["label"]
    pd.testing.assert_series_equal(full.iloc[:500], prefix)


def test_walk_forward_drops_nan_emits_unknown() -> None:
    s = _two_regime_series(400)
    s.iloc[300] = np.nan
    out = walk_forward(s, jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=120, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert out["label"].iloc[300] == "Unknown"
    assert bool(out["cold_start"].iloc[300])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_regime_jm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roro.regime_jm'`.

- [ ] **Step 3: Implement `walk_forward`**

Create `roro/regime_jm.py`:

```python
"""3-state statistical jump model regime classifier on segment beta.

Parallel to roro.regime_hmm. Causal-by-construction: centroids re-estimated on a
point-in-time window (expanding monthly, or rolling), states inferred by the
forward-DP online filter daily. Backward (two-sided) reconstruction is used only
to FIT on a closed historical window -- never for the live signal.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from roro.jump_model import fit_jump_model, online_states

_ORDERED_LABELS = ("Risk-off", "Transitional", "Risk-on")
_UNKNOWN = "Unknown"


def _causal_scaler(window: np.ndarray) -> tuple[float, float]:
    """Mean/std (ddof=0) of the in-window data; std==0 -> 1.0 (avoid div-by-zero)."""
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    return mean, (std if std > 0.0 else 1.0)


def walk_forward(
    beta: pd.Series,
    *,
    jump_penalty: float,
    refit_interval_days: int,
    min_history_days: int,
    n_states: int,
    window: str,
    rolling_window_days: int,
    continuous: bool,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> dict[str, Any]:
    """Expanding/rolling refit + per-block frozen-scaler forward-DP online inference."""
    full_index = beta.index
    clean = beta.dropna()
    n = len(clean)
    cvals = clean.to_numpy(dtype=float)

    probs = np.full((n, n_states), np.nan)
    cold = np.ones(n, dtype=bool)
    refit_dates: list[pd.Timestamp] = []
    last_good: tuple[np.ndarray, float, float] | None = None  # (centroids, mean, std)

    r = min_history_days
    while r < n:
        block_end = min(r + refit_interval_days, n)
        fit_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
        fit_win = cvals[fit_lo:r]
        mean, std = _causal_scaler(fit_win)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_jump_model(
                (fit_win - mean) / std, k=n_states, jump_penalty=jump_penalty,
                n_init=n_init, max_iter=max_iter, tol=tol, seed=seed,
            )
        if fit.converged:
            last_good = (fit.centroids, mean, std)
            refit_dates.append(pd.Timestamp(clean.index[r]))
        used = (fit.centroids, mean, std) if fit.converged else last_good
        if used is not None:
            centroids, u_mean, u_std = used
            inf_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
            inf_win = (cvals[inf_lo:block_end] - u_mean) / u_std
            states = online_states(inf_win, centroids, jump_penalty)
            # local index of global position e is (e - inf_lo)
            for e in range(r, block_end):
                s = int(states[e - inf_lo])
                probs[e, :] = 0.0
                probs[e, s] = 1.0
                cold[e] = False
        r = block_end

    state_idx = np.where(np.isnan(probs).any(axis=1), -1, probs.argmax(axis=1))
    labels = [_ORDERED_LABELS[i] if i >= 0 else _UNKNOWN for i in state_idx]
    confidence = np.where(np.isnan(probs).any(axis=1), np.nan, probs.max(axis=1))

    def _series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=clean.index).reindex(full_index)

    return {
        "state": _series(np.where(state_idx < 0, np.nan, state_idx.astype(float))),
        "label": _series(pd.Series(labels, index=clean.index)).fillna(_UNKNOWN),
        "prob_risk_off": _series(probs[:, 0]),
        "prob_transitional": _series(probs[:, 1]),
        "prob_risk_on": _series(probs[:, 2]),
        "confidence": _series(confidence),
        "cold_start": pd.Series(cold, index=clean.index).reindex(
            full_index, fill_value=True
        ).astype(bool),
        "refit_dates": refit_dates,
    }
```

> Note for the implementer: this is the **discrete** path (one-hot probs, `confidence=1.0`). The continuous path is added in Task 9 (J3) behind `continuous=True`; for now `continuous` is accepted but only the discrete branch is exercised. `n_states` is parameterized but the label vocabulary assumes 3 — keep `n_states=3` (config default); a non-3 value is out of scope.

- [ ] **Step 4: Run to verify pass + quality**

Run: `.venv\Scripts\python.exe -m pytest tests/test_regime_jm.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m mypy --strict roro/regime_jm.py` and `.venv\Scripts\python.exe -m ruff check roro/regime_jm.py tests/test_regime_jm.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add roro/regime_jm.py tests/test_regime_jm.py
git commit -m "feat(jm): causal walk_forward (frozen-scaler forward-DP online inference)"
```

---

## Task 5: `classify_jm` per-segment orchestration

**Files:**
- Modify: `roro/regime_jm.py`
- Test: `tests/test_regime_jm.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_regime_jm.py`:

```python
from roro.config import EngineConfig
from roro.regime_jm import classify_jm
from roro.types import BetaBySegment, BetaFrame, JmRegimeFrame


def _bbs(n: int = 400) -> BetaBySegment:
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(2)
    beta = np.concatenate([rng.normal(-2, 0.3, n // 2), rng.normal(2, 0.3, n - n // 2)])
    cap = pd.DataFrame({"beta": beta, "n": np.full(n, 12)}, index=idx)
    bf = BetaFrame(cap_wtd=cap, eq_wtd=pd.DataFrame(), slope_spread=pd.Series(dtype=float))
    return BetaBySegment(by_segment={"global": bf, "LatAm": bf})


def test_classify_jm_returns_frame() -> None:
    cfg = EngineConfig(data_path=__import__("pathlib").Path("d.xlsx"),
                       output_dir=__import__("pathlib").Path("o"),
                       jm_min_history_days=120, jm_refit_interval_days=40, jm_n_init=4)
    frame = classify_jm(_bbs(), cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    assert isinstance(frame, JmRegimeFrame)
    assert "global" in frame.label.columns and "LatAm" in frame.label.columns
    assert bool(frame.thin_cut_flag["LatAm"].all())
    assert not bool(frame.thin_cut_flag["global"].any())
    assert frame.label["global"].isin({"Risk-off", "Transitional", "Risk-on", "Unknown"}).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_regime_jm.py::test_classify_jm_returns_frame -v`
Expected: FAIL — `ImportError: cannot import name 'classify_jm'`.

- [ ] **Step 3: Implement `classify_jm`** (mirrors `classify_hmm`, regime_hmm.py:179-227)

First add the imports `classify_jm` needs to the top of `roro/regime_jm.py` (these were intentionally omitted in Task 4 because `walk_forward` doesn't use them — adding them now keeps `ruff` F401 clean at every step):

```python
from roro.config import EngineConfig
from roro.types import BetaBySegment, JmRegimeFrame
```

Then add the function to `roro/regime_jm.py`:

```python
def classify_jm(
    bbs: BetaBySegment, *, cfg: EngineConfig, thin_cuts: frozenset[str]
) -> JmRegimeFrame:
    """Per-segment JM classification mirroring classify_hmm's loop + frame shape."""
    state: dict[str, pd.Series] = {}
    label: dict[str, pd.Series] = {}
    p_off: dict[str, pd.Series] = {}
    p_tr: dict[str, pd.Series] = {}
    p_on: dict[str, pd.Series] = {}
    conf: dict[str, pd.Series] = {}
    nseg: dict[str, pd.Series] = {}
    thin: dict[str, pd.Series] = {}
    cold: dict[str, pd.Series] = {}
    refit_dates: dict[str, list[pd.Timestamp]] = {}
    penalty_used: dict[str, float] = {}

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        out = walk_forward(
            beta, jump_penalty=cfg.jm_jump_penalty,
            refit_interval_days=cfg.jm_refit_interval_days,
            min_history_days=cfg.jm_min_history_days, n_states=cfg.jm_n_states,
            window=cfg.jm_window, rolling_window_days=cfg.jm_rolling_window_days,
            continuous=cfg.jm_continuous, n_init=cfg.jm_n_init,
            max_iter=cfg.jm_max_iter, tol=cfg.jm_tol, seed=cfg.jm_random_seed,
        )
        state[cut] = out["state"]
        label[cut] = out["label"]
        p_off[cut] = out["prob_risk_off"]
        p_tr[cut] = out["prob_transitional"]
        p_on[cut] = out["prob_risk_on"]
        conf[cut] = out["confidence"]
        nseg[cut] = bf.cap_wtd["n"].reindex(beta.index)
        thin[cut] = pd.Series(cut in thin_cuts, index=beta.index)
        cold[cut] = out["cold_start"]
        refit_dates[cut] = out["refit_dates"]
        penalty_used[cut] = cfg.jm_jump_penalty

    return JmRegimeFrame(
        state=pd.DataFrame(state), label=pd.DataFrame(label),
        prob_risk_off=pd.DataFrame(p_off), prob_transitional=pd.DataFrame(p_tr),
        prob_risk_on=pd.DataFrame(p_on), confidence=pd.DataFrame(conf),
        n_per_segment=pd.DataFrame(nseg), thin_cut_flag=pd.DataFrame(thin),
        cold_start_flag=pd.DataFrame(cold), refit_dates=refit_dates,
        jump_penalty_used=penalty_used,
    )
```

- [ ] **Step 4: Run to verify pass + quality**

Run: `.venv\Scripts\python.exe -m pytest tests/test_regime_jm.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m mypy --strict roro/regime_jm.py` and `.venv\Scripts\python.exe -m ruff check roro/regime_jm.py tests/test_regime_jm.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add roro/regime_jm.py tests/test_regime_jm.py
git commit -m "feat(jm): classify_jm per-segment orchestration"
```

---

## Task 6: Engine + alerts + io wiring

**Files:**
- Modify: `roro/engine.py`, `roro/alerts.py`, `roro/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_io.py` (mirror the existing HMM io test; build a minimal `RunResult` with a `regime_jm`). Use the existing test helpers/fixtures in that file for constructing frames; the assertion is that `regimes_jm.csv` + `jm_refit_log.csv` are written and carry the expected columns:

```python
def test_write_run_emits_jm_files(tmp_path: Path) -> None:
    import pandas as pd

    from roro.io import _write_regime_jm
    idx = pd.bdate_range("2020-01-01", periods=3)
    wide = lambda v: pd.DataFrame({"global": v}, index=idx)  # noqa: E731
    from roro.types import JmRegimeFrame
    jf = JmRegimeFrame(
        state=wide([0.0, 1.0, 2.0]), label=wide(["Risk-off", "Transitional", "Risk-on"]),
        prob_risk_off=wide([1.0, 0.0, 0.0]), prob_transitional=wide([0.0, 1.0, 0.0]),
        prob_risk_on=wide([0.0, 0.0, 1.0]), confidence=wide([1.0, 1.0, 1.0]),
        n_per_segment=wide([12, 12, 12]), thin_cut_flag=wide([False, False, False]),
        cold_start_flag=wide([False, False, False]),
        refit_dates={"global": [idx[1]]},
    )
    out = tmp_path / "regimes_jm.csv"
    _write_regime_jm(jf, out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "segment", "state", "label", "p_risk_off",
                                "p_transitional", "p_risk_on", "confidence",
                                "cold_start", "thin_cut"]
    assert len(df) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_io.py::test_write_run_emits_jm_files -v`
Expected: FAIL — `ImportError: cannot import name '_write_regime_jm'`.

- [ ] **Step 3: Implement the wiring**

**`roro/engine.py`** — add the import and the block after the HMM block (engine.py:100), then pass `regime_jm` to alerts and `RunResult`:

```python
from roro.regime_jm import classify_jm  # near the regime_hmm import (line 20)
```
```python
    # 5c) Optional JM regime classifier (parallel method, off by default).
    regime_jm = (
        classify_jm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
        if cfg.jm_enabled
        else None
    )
```
In the `detect_alerts(...)` call add `regime_jm=regime_jm`; in the `RunResult(...)` constructor add `regime_jm=regime_jm`.

**`roro/alerts.py`** — add the `regime_jm` param + `jm_bucket_transitions` (reuse `_bucket_transitions`):

```python
def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
    regime_hmm: HmmRegimeFrame | None = None,
    regime_jm: "JmRegimeFrame | None" = None,
) -> AlertSet:
```
Inside the `AlertSet(...)` constructor add:
```python
        jm_bucket_transitions=(
            _bucket_transitions(regime_jm.label)
            if regime_jm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
```
(Add `JmRegimeFrame` to the `from roro.types import ...` line in alerts.py.)

**`roro/io.py`** — in `write_run`, after the HMM guard (io.py:131-133) add:
```python
    if result.regime_jm is not None:
        _write_regime_jm(result.regime_jm, tmp / "regimes_jm.csv")
        _write_jm_refit_log(result.regime_jm, tmp / "jm_refit_log.csv")
```
Add the two writers (clone `_write_regime_hmm` io.py:259 and `_write_hmm_refit_log` io.py:285, swapping the frame; column header identical):
```python
def _write_regime_jm(jf: "JmRegimeFrame", path: Path) -> None:
    cols = (
        "date,segment,state,label,p_risk_off,p_transitional,"
        "p_risk_on,confidence,cold_start,thin_cut\n"
    )
    frames = {
        "label": jf.label, "state": jf.state, "p_risk_off": jf.prob_risk_off,
        "p_transitional": jf.prob_transitional, "p_risk_on": jf.prob_risk_on,
        "confidence": jf.confidence, "cold_start": jf.cold_start_flag,
        "thin_cut": jf.thin_cut_flag,
    }
    merged = _merge_regime_long(frames)  # reuse the same melt+merge helper the HMM writer uses
    path.write_text(cols, encoding="utf-8")
    merged[["date", "segment", "state", "label", "p_risk_off", "p_transitional",
            "p_risk_on", "confidence", "cold_start", "thin_cut"]].to_csv(
        path, mode="a", header=False, index=False
    )


def _write_jm_refit_log(jf: "JmRegimeFrame", path: Path) -> None:
    rows = [{"segment": seg, "refit_date": d}
            for seg, dates in jf.refit_dates.items() for d in dates]
    pd.DataFrame(rows, columns=["segment", "refit_date"]).to_csv(path, index=False)
```
> Implementer: open `roro/io.py` and read `_write_regime_hmm` (line 259) first — reproduce its EXACT melt/merge mechanism (whatever helper or inline `_melt_with_date` it uses) so `regimes_jm.csv` is byte-shaped identically to `regimes_hmm.csv`. The pseudo-helper `_merge_regime_long` above stands for that existing mechanism; do not invent a new one. Round `p_*`/`confidence` to 10 decimals before writing (spec §3.7/§8). Also add the `snapshot["regime_jm"]` last-row block in `_build_snapshot` (io.py:315), cloning the `regime_hmm` block with `result.regime_jm`.

(Add `JmRegimeFrame` to io.py's `from roro.types import ...`.)

- [ ] **Step 4: Run to verify pass + byte-identical guard**

Run: `.venv\Scripts\python.exe -m pytest tests/test_io.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m pytest tests/test_golden.py tests/test_reproducibility.py tests/test_engine.py -q`
Expected: PASS — with `jm_enabled=False` (default), engine/io output is byte-identical (AJ-1).

- [ ] **Step 5: Commit**

```bash
git add roro/engine.py roro/alerts.py roro/io.py tests/test_io.py
git commit -m "feat(jm): engine + alerts + io wiring (regimes_jm.csv, jm_refit_log.csv, snapshot)"
```

---

## Task 7: Golden fixture + determinism regression (J2)

**Files:**
- Modify: `tests/test_regime_jm.py` (or a new `tests/test_regime_jm_golden.py`)
- Test: as above

- [ ] **Step 1: Write the determinism + golden test**

Append to `tests/test_regime_jm.py`. The determinism test runs the full engine twice with `jm_enabled=True` over the tiny fixture and asserts byte-identical `regimes_jm.csv`. Reuse the `tiny_xlsx` fixture + `MockFredClient` pattern from `tests/test_backtest.py`:

```python
def test_engine_jm_csv_is_byte_identical(tiny_xlsx, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from roro.config import EngineConfig
    from roro.engine import run as engine_run
    from roro.fred_client import FRED_SERIES_IDS, MockFredClient

    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}

    def _run(out: str) -> bytes:
        cfg = EngineConfig(data_path=tiny_xlsx, output_dir=tmp_path / out,
                           ewma_halflife_days=10, return_window_days=21,
                           tripwire_window_days=10, percentile_window_years=1,
                           min_n_per_cut=2, bootstrap_min_days=10, jm_enabled=True,
                           jm_min_history_days=120, jm_refit_interval_days=60, jm_n_init=4)
        engine_run(cfg, fred_client=MockFredClient(seeded=seeded),
                   run_date="2024-12-31", as_of_data_date="2024-12-31", force=True)
        return (tmp_path / out / "2024-12-31" / "regimes_jm.csv").read_bytes()

    assert _run("a") == _run("b")
```

- [ ] **Step 2: Run to verify it passes** (the code already exists from Tasks 4-6)

Run: `.venv\Scripts\python.exe -m pytest tests/test_regime_jm.py::test_engine_jm_csv_is_byte_identical -v`
Expected: PASS. If it FAILS on byte-identity, the determinism contract is broken — STOP and diagnose the seeding/tie-break (do not weaken the assertion).

- [ ] **Step 3: Commit**

```bash
git add tests/test_regime_jm.py
git commit -m "test(jm): engine-level byte-identical regimes_jm.csv determinism gate (AJ-2)"
```

---

## Task 8: Backtest 3-way scoring (J4)

**Files:**
- Modify: `roro/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing test** (mirror `test_run_backtest_writes_hmm_compare_reports`)

Append to `tests/test_backtest.py`:

```python
@pytest.mark.slow
def test_run_backtest_writes_jm_compare(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "bt", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10, jm_enabled=True,
        jm_min_history_days=120, jm_refit_interval_days=60, jm_n_init=4,
    )
    run_backtest(cfg, fred_client=MockFredClient(seeded=seeded),
                 start="2024-01-01", end="2024-12-31")
    assert (tmp_path / "bt" / "acceptance_report_jm.json").exists()
    compare = json.loads((tmp_path / "bt" / "acceptance_compare.json").read_text())
    assert "jm" in compare["G3_events"]
```

(`json` is already imported in test_backtest? if not, add `import json`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py::test_run_backtest_writes_jm_compare -v`
Expected: FAIL — `acceptance_report_jm.json` missing / `"jm"` not in compare.

- [ ] **Step 3: Implement** in `roro/backtest.py` `run_backtest`, after the HMM block (backtest.py:107-125):

```python
    if result.regime_jm is not None:
        jm_gates = _evaluate_gates(
            result, labels=result.regime_jm.label,
            transitions=result.alerts.jm_bucket_transitions,
        )
        jm_report = {
            **report, "method": "jm", "gates": jm_gates,
            "all_passed": all(bool(v.get("passed", False)) for v in jm_gates.values()),
        }
        (cfg.output_dir / "acceptance_report_jm.json").write_text(
            json.dumps(jm_report, indent=2, default=str), encoding="utf-8"
        )
```
Then extend the `acceptance_compare.json` writer so each gate dict gains a `"jm"` key when JM is present. Replace the existing compare-construction (inside the `if result.regime_hmm is not None:` block) so it conditionally includes jm:
```python
        compare = {
            g: {
                "percentile": gates[g], "hmm": hmm_gates[g],
                **({"jm": jm_gates[g]} if result.regime_jm is not None else {}),
            }
            for g in gates
        }
```
> Implementer: `jm_gates` must be in scope where `compare` is built. If the HMM block and JM block are separate, compute `jm_gates` before the compare dict, or move the compare construction after both. Keep the percentile-only and hmm-only paths working (compare is only written when HMM present today — preserve that; add jm as an extra key).

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v -m "not slow"` then `... -m slow -k jm_compare`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/backtest.py tests/test_backtest.py
git commit -m "feat(jm): backtest scores JM through G1-G6 + 3-way acceptance_compare"
```

---

## Task 9: Continuous JM (CJM) soft probabilities (J3)

**Files:**
- Modify: `roro/jump_model.py`, `roro/regime_jm.py`
- Test: `tests/test_jump_model.py`, `tests/test_regime_jm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_jump_model.py`:

```python
from roro.jump_model import discretize_simplex, online_soft_states


def test_discretize_simplex_is_deterministic_and_valid() -> None:
    g1 = discretize_simplex(3, 0.05)
    g2 = discretize_simplex(3, 0.05)
    assert np.array_equal(g1, g2)            # fixed enumeration order
    assert g1.shape[1] == 3
    assert np.allclose(g1.sum(axis=1), 1.0)  # all on the simplex
    assert (g1 >= -1e-12).all()


def test_online_soft_states_sum_to_one_and_causal() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=120)
    c = np.array([-1.0, 0.0, 1.0])
    grid = discretize_simplex(3, 0.1)
    soft_full = online_soft_states(y, c, 5.0, grid)
    soft_prefix = online_soft_states(y[:80], c, 5.0, grid)
    assert np.allclose(soft_full.sum(axis=1), 1.0)
    assert np.allclose(soft_full[:80], soft_prefix)  # no lookahead
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py -k "simplex or soft" -v`
Expected: FAIL — `ImportError: cannot import name 'discretize_simplex'`.

- [ ] **Step 3: Implement the CJM grid + soft online inference** in `roro/jump_model.py`:

```python
def discretize_simplex(k: int, grid_size: float) -> _NDArrayF:
    """Fixed lexicographic grid of prob vectors with components in multiples of
    grid_size summing to 1. Deterministic enumeration order (grid index -> vector)."""
    steps = int(round(1.0 / grid_size))
    rows: list[list[float]] = []

    def _recurse(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append([(p / steps) for p in (*prefix, remaining)])
            return
        for take in range(remaining + 1):
            _recurse([*prefix, take], remaining - take, slots - 1)

    _recurse([], steps, k)
    return np.asarray(rows, dtype=float)


def _cjm_loss_rows(loss_mx: _NDArrayF, grid: _NDArrayF) -> _NDArrayF:
    """Per-day loss of each grid vector: sum_k grid[n,k]*loss_mx[t,k].

    Broadcast-and-sum, NOT a BLAS matmul, so D=1 stays bit-stable across BLAS
    thread counts (spec section 8)."""
    return (loss_mx[:, None, :] * grid[None, :, :]).sum(axis=2)


def online_soft_states(
    y: _NDArrayF, centroids: _NDArrayF, jump_penalty: float, grid: _NDArrayF
) -> _NDArrayF:
    """Causal soft state vectors (T,k): forward DP over the fixed simplex grid."""
    loss_mx = _loss_matrix(y, centroids)
    cjm_loss = _cjm_loss_rows(loss_mx, grid)                 # (T, N_grid)
    l1 = np.abs(grid[:, None, :] - grid[None, :, :]).sum(axis=2)  # (N,N)
    penalty = jump_penalty * (l1 * l1)                       # L1-squared
    t_len, n_grid = cjm_loss.shape
    values = np.empty((t_len, n_grid), dtype=float)
    values[0] = cjm_loss[0]
    for t in range(1, t_len):
        values[t] = cjm_loss[t] + (values[t - 1][:, None] + penalty).min(axis=0)
    chosen = values.argmin(axis=1)
    return grid[chosen]
```

Then wire the CJM branch into `walk_forward` (`roro/regime_jm.py`). Replace the discrete-only emission inside the block loop with a branch on `continuous`:

```python
        if used is not None:
            centroids, u_mean, u_std = used
            inf_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
            inf_win = (cvals[inf_lo:block_end] - u_mean) / u_std
            if continuous:
                grid = discretize_simplex(n_states, 0.05)
                soft = online_soft_states(inf_win, centroids, jump_penalty, grid)
                for e in range(r, block_end):
                    probs[e, :] = np.round(soft[e - inf_lo], 10)
                    cold[e] = False
            else:
                states = online_states(inf_win, centroids, jump_penalty)
                for e in range(r, block_end):
                    probs[e, :] = 0.0
                    probs[e, int(states[e - inf_lo])] = 1.0
                    cold[e] = False
```
Add `discretize_simplex, online_soft_states` to the `from roro.jump_model import ...` line. `confidence`/`state`/`label` derivation (argmax of probs) already handles both paths.

Append a CJM determinism + no-lookahead test to `tests/test_regime_jm.py`:

```python
def test_walk_forward_cjm_deterministic_and_causal() -> None:
    s = _two_regime_series(500)
    kw = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=200,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=True, n_init=4, max_iter=30, tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["prob_risk_off"], b["prob_risk_off"])
    full = walk_forward(s, **kw)["prob_risk_on"]
    prefix = walk_forward(s.iloc[:420], **kw)["prob_risk_on"]
    pd.testing.assert_series_equal(full.iloc[:420], prefix)
```

- [ ] **Step 4: Run to verify pass + quality**

Run: `.venv\Scripts\python.exe -m pytest tests/test_jump_model.py tests/test_regime_jm.py -v`
Expected: PASS.
Run: `.venv\Scripts\python.exe -m mypy --strict roro/jump_model.py roro/regime_jm.py` and `ruff check` on both + their tests.
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add roro/jump_model.py roro/regime_jm.py tests/test_jump_model.py tests/test_regime_jm.py
git commit -m "feat(jm): continuous JM soft probabilities (BLAS-free simplex DP)"
```

---

## Task 10: Report — bundle + load (J5a)

**Files:**
- Modify: `roro/report/bundle.py`, `roro/report/load.py`
- Test: `tests/report/test_load.py`

- [ ] **Step 1: Write failing test**

Append to `tests/report/test_load.py` (mirror the HMM load test; write a `regimes_jm.csv` into a temp run dir and assert the bundle picks up `seg_jm_label` etc.). Reuse the existing run-dir fixture pattern in that test file:

```python
def test_load_picks_up_jm_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pandas as pd
    from roro.report.load import load_bundle  # use whatever the module's loader is named

    # ... build a minimal run dir with regimes.csv + beta_series.csv (required) ...
    # then write regimes_jm.csv with the HMM-identical schema:
    pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=2).repeat(1),
        "segment": ["global", "global"], "state": [0.0, 2.0],
        "label": ["Risk-off", "Risk-on"], "p_risk_off": [1.0, 0.0],
        "p_transitional": [0.0, 0.0], "p_risk_on": [0.0, 1.0],
        "confidence": [1.0, 1.0], "cold_start": [False, False],
        "thin_cut": [False, False],
    }).to_csv(run_dir / "regimes_jm.csv", index=False)
    bundle = load_bundle(run_dir)  # match the real signature
    assert bundle.seg_jm_label is not None
    assert "global" in bundle.seg_jm_label.columns
```

> Implementer: open `tests/report/test_load.py` + `roro/report/load.py` first to match the exact loader name/signature and the minimal required-CSV setup; this test mirrors the existing HMM-load test one-for-one.

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py::test_load_picks_up_jm_csv -v`
Expected: FAIL — `AttributeError: ... seg_jm_label`.

- [ ] **Step 3: Implement**

**`roro/report/bundle.py`** — add four optional fields after the `seg_hmm_*` block (bundle.py:33):
```python
    seg_jm_label: pd.DataFrame | None = None
    seg_jm_p_off: pd.DataFrame | None = None
    seg_jm_p_tr: pd.DataFrame | None = None
    seg_jm_p_on: pd.DataFrame | None = None
```
**`roro/report/load.py`** — after the HMM pivot block (load.py:150-159) add the parallel JM block and pass the four to the `DataBundle(...)` constructor:
```python
    seg_jm_label = seg_jm_p_off = seg_jm_p_tr = seg_jm_p_on = None
    jm_path = run_dir / "regimes_jm.csv"
    if jm_path.exists():
        jm = pd.read_csv(jm_path, parse_dates=["date"])
        seg_jm_label = jm.pivot(index="date", columns="segment", values="label").sort_index()
        seg_jm_p_off = jm.pivot(index="date", columns="segment", values="p_risk_off").sort_index()
        seg_jm_p_tr = jm.pivot(index="date", columns="segment", values="p_transitional").sort_index()
        seg_jm_p_on = jm.pivot(index="date", columns="segment", values="p_risk_on").sort_index()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/report/bundle.py roro/report/load.py tests/report/test_load.py
git commit -m "feat(jm): report bundle + load for regimes_jm.csv"
```

---

## Task 11: Report — 3-way band toggle + JM probability figure (J5b)

**Files:**
- Modify: `roro/report/figures.py`, `roro/report/html.py`, `roro/report/orchestrate.py`
- Test: `tests/report/conftest.py`, `tests/report/test_orchestrate.py`

- [ ] **Step 1: Write failing tests**

In `tests/report/conftest.py`, add a `write_jm_csv(run_dir)` helper cloning `write_hmm_csv` (conftest.py:66-83), filename `regimes_jm.csv`, identical columns. In `tests/report/test_orchestrate.py` add:

```python
def test_build_report_includes_jm_when_present(minimal_run_dir, tiny_xlsx, tmp_path, write_jm_csv) -> None:  # type: ignore[no-untyped-def]
    write_jm_csv(minimal_run_dir)
    out = tmp_path / "r.html"
    build_report(minimal_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    assert "JM regime probabilities" in html
    assert 'value="jm"' in html            # third band-toggle option
    assert "fig_jm_probs" in html


def test_build_report_byte_identical_without_jm(minimal_run_dir, tiny_xlsx, tmp_path) -> None:  # type: ignore[no-untyped-def]
    a, b = tmp_path / "a.html", tmp_path / "b.html"
    build_report(minimal_run_dir, tiny_xlsx, a, window=21)
    build_report(minimal_run_dir, tiny_xlsx, b, window=21)
    assert a.read_bytes() == b.read_bytes()  # determinism preserved (no JM present)
```

> Implementer: match the exact fixture names (`minimal_run_dir`, `write_jm_csv`, `tiny_xlsx`) used by the existing HMM report tests in this directory; register `write_jm_csv` as a fixture in conftest.py the same way `write_hmm_csv` is.

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_orchestrate.py -k jm -v`
Expected: FAIL — "JM regime probabilities" not found.

- [ ] **Step 3: Implement**

**`roro/report/figures.py`**:
- Clone `regime_probability_area` (figures.py:862-948) → `regime_probability_area_jm`: swap the four `seg_hmm_*` accessors for `seg_jm_*`, the assert to `bundle.seg_jm_label is not None`, the layout/button title prefix to `"JM regime probabilities"`. Everything else (the `_PROB_TRACE_ORDER`, `stackgroup="p"`, `REGIME_COLORS`, segment dropdown, `yaxis [0,1]`) identical. (Add a `_PROB_TRACE_ORDER_JM` analog mapping to the `seg_jm_*` attr names, or parameterize the existing one — keep it simple: a small dedicated tuple.)
- Extend `beta_band_lookup` (figures.py:684-703): for each segment, add `"jm"` when `bundle.seg_jm_label is not None and seg in bundle.seg_jm_label.columns`, value `_band_shapes(bundle.seg_jm_label[seg], smooth=False)`. Guard each method independently so HMM-only or JM-only runs both work.

**`roro/report/html.py`** — in `_band_toggle_markup` (html.py:38-88) add the third option to the `<select>`:
```html
        <option value="jm">Jump Model</option>
```
(No JS change — `apply()` already keys `byMethod[method]`.)

**`roro/report/orchestrate.py`** — after the HMM gate (orchestrate.py:48-52) add:
```python
    if bundle.seg_jm_label is not None:
        specs.append(FigureSpec(regime_probability_area_jm(bundle), "fig_jm_probs",
                                "JM regime probabilities"))
        lookup = beta_band_lookup(bundle)  # now includes jm; safe if also called for HMM
```
Import `regime_probability_area_jm`. Ensure `lookup` is computed when EITHER hmm or jm is present (the existing HMM branch already computes it; if only JM is present, compute it in the JM branch). `beta_band_lookup` must tolerate `seg_hmm_label is None` (HMM-absent) — make its hmm shapes conditional too.

- [ ] **Step 4: Run to verify pass + byte-identical HTML**

Run: `.venv\Scripts\python.exe -m pytest tests/report/ -v`
Expected: PASS (JM figure present when CSV exists; byte-identical when absent).

- [ ] **Step 5: Commit**

```bash
git add roro/report/figures.py roro/report/html.py roro/report/orchestrate.py tests/report/
git commit -m "feat(jm): report 3-way band toggle (Percentile/HMM/JM) + JM probability figure"
```

---

## Task 12: End-to-end run + verification (J6)

This produces the deliverable: an HTML report showing JM regime bands (in the 3-way toggle) + the JM probability figure on real data. Requires `FRED_API_KEY` + network (same as `roro backtest`). Run via the `roro` console script (NOT `python -m roro.cli`, which has no `__main__` dispatch).

- [ ] **Step 1: Full quality gate**

Run: `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
Expected: all pass.
Run: `.venv\Scripts\python.exe -m mypy --strict roro` and `.venv\Scripts\python.exe -m ruff check roro tests`
Expected: clean (pre-existing mypy errors in `tests/test_regime_hmm.py` + `tests/report/test_figures.py` are out of scope — do not fix here).

- [ ] **Step 2: Create the JM eval config**

Create `configs/jm-eval.yaml` (copy `configs/eval.yaml`, set `jm_enabled: true`, `jm_continuous: true` for soft bands, keep `hmm_enabled: true` so the report shows all three methods):

```yaml
# (all fields from configs/eval.yaml) plus:
jm_enabled: true
jm_continuous: true
jm_jump_penalty: 50.0
```

- [ ] **Step 3: Run the engine + build the report (BLAS pinned for determinism)**

PowerShell:
```powershell
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
.venv\Scripts\roro.exe run --config configs\jm-eval.yaml --date 2026-05-26 --as-of-data-date 2026-05-26 --force
.venv\Scripts\roro.exe report --run-dir outputs\<run_date_dir> --out outputs\jm_report.html
```
Expected: `regimes_jm.csv` + `jm_refit_log.csv` present in the run dir; `outputs\jm_report.html` built.

- [ ] **Step 4: Verify the visualization**

Open `outputs\jm_report.html` (or grep it) and confirm: (a) the segment-β chart's band-source `<select>` has three options (Percentile / HMM / Jump Model) and switching to "Jump Model" repaints the regime bands; (b) a "JM regime probabilities" stacked-area figure (`fig_jm_probs`) renders Risk-off/Transitional/Risk-on summing to 1. Capture the two facts in `docs/context/results.md`.

- [ ] **Step 5: Determinism spot-check + commit config**

Run the engine twice (BLAS pinned) and diff `regimes_jm.csv`:
```powershell
.venv\Scripts\roro.exe run --config configs\jm-eval.yaml --date 2026-05-26 --as-of-data-date 2026-05-26 --out outputs\jm_a --force
.venv\Scripts\roro.exe run --config configs\jm-eval.yaml --date 2026-05-26 --as-of-data-date 2026-05-26 --out outputs\jm_b --force
# assert the two regimes_jm.csv are byte-identical
```
```bash
git add configs/jm-eval.yaml docs/context/results.md
git commit -m "chore(jm): jm-eval config + end-to-end report verification (JM bands + prob figure)"
```

---

## Self-Review notes (author)

- **Spec coverage:** §3 algorithm → Tasks 1-2 (+ CJM Task 9); §3.6/§3.8 walk_forward → Task 4; §4.2 type + §5 config → Task 3; §4.3 classify_jm → Task 5; §6 engine/io/alerts → Task 6; backtest 3-way → Task 8; §7 report viz → Tasks 10-11; §8 determinism → seeded fit (T2) + byte-identical gates (T6,T7) + CJM BLAS-free (T9) + BLAS pin (T12); §9 acceptance AJ-1 (T6 byte-identical), AJ-2 (T7), AJ-3 (no-lookahead T4/T9), AJ-6 (T8); cadence AJ-4 deferred to a `@pytest.mark.slow` bench (add in T12 follow-up or a J6 stretch — noted).
- **Determinism is the throughline:** every fit goes through SeedSequence/PCG64 (T2); byte-identical CSV asserted at engine level (T7) for discrete and (T9) for CJM; CJM avoids BLAS by broadcast-sum; T12 pins BLAS threads.
- **Mechanical-mirror tasks (6,8,10,11)** point the implementer at the exact HMM template line to clone and show the JM-specific code; the implementer MUST read the cited HMM function first to reproduce its exact serialization (esp. io.py melt mechanism) — flagged inline.
- **Known follow-ups (not blocking the deliverable):** λ re-calibration on standardized features (S-JM3) and the cadence-invariance bench (AJ-4) are diagnostic refinements; the deliverable (JM regimes in HTML) lands at Task 12.
```
