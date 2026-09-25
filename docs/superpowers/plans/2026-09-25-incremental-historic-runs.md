# Incremental Historic Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run lands in `outputs/historic/<config>/results_<last-data-date>/`. A new-data run copies closed HMM/JM refit blocks from the newest checkpoint and recomputes only the open block, and it produces CSVs byte-identical to a full rerun. The feature ships as `roro update` plus a one-click `run_roro.py`/`run_roro.bat`.

**Architecture:** New `roro/resume.py` (pure resume primitives + history check). `walk_forward` in `regime_hmm.py`/`regime_jm.py` gains an optional `prior=` that seeds closed blocks. `engine.run` gains `data_until=` and `resume=`. New `roro/historic.py` holds checkpoint discovery, the pure `plan_update()` decision, and the `run_update()` orchestrator. The CLI and the runner script both call `run_update()`.

**Tech Stack:** Python 3.12, pandas 2.2, numpy, statsmodels (HMM), vendored JM, click, pytest + hypothesis, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md`

**Conventions (read before starting):**
- Python: always `.venv/Scripts/python.exe` (uv-managed 3.12), never the global Python.
- Console output ASCII-only (`->`, `[ok]`, `[x]`). No unicode arrows in `print`/`echo`.
- Determinism house rule: sorted keys, stable sorts, never iterate a dict directly into output order.
- `pytest` runs with `filterwarnings = error`. Any warning fails a test.
- Commit after every task. Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch: `feat/incremental-historic` (already created; the spec is committed there).

## File map

| File | Status | Responsibility |
|---|---|---|
| `roro/types.py` | modify | + `SegmentPrior`, `ResumeState` |
| `roro/resume.py` | create | `resume_block_start`, `seed_prior_rows`, `prior_refits_before`, `verify_beta_history`, `HistoryRevisedError` |
| `roro/regime_hmm.py` | modify | `walk_forward(prior=)`, `classify_hmm(prior=)` |
| `roro/regime_jm.py` | modify | extract `_fit_at`; `walk_forward(prior=)`, `classify_jm(prior=)` |
| `roro/io.py` | modify | `beta_long`, `cut_prices`, `read_resume_state` |
| `roro/engine.py` | modify | `run(data_until=, resume=)` |
| `roro/historic.py` | create | `TypeRun`, `UpdateMode`, `Checkpoint`, `UpdatePlan`, `UpdateOutcome`, `parse_user_date`, `available_configs`, `find_checkpoint`, `config_changes`, `plan_update`, `friendly_error`, `run_update` |
| `roro/cli.py` | modify | `roro update` subcommand |
| `run_roro.py`, `run_roro.bat` | create | colleague runner |
| `.gitattributes` | create | `*.bat` CRLF |
| `tests/conftest.py` | modify | `build_xlsx`, `random_walk_prices`, `rw_xlsx` fixture |
| `tests/test_resume.py`, `tests/test_regime_hmm_resume.py`, `tests/test_regime_jm_resume.py`, `tests/test_io_resume.py`, `tests/test_engine_resume.py`, `tests/test_historic.py`, `tests/test_historic_update.py`, `tests/test_run_roro.py` | create | tests |
| `tests/test_cli.py` | modify | `roro update` smoke |
| `README.md`, `docs/context/*.md` | modify | docs |

---

### Task 1: Resume primitives + `SegmentPrior`

**Files:**
- Modify: `roro/types.py` (add after `JmRegimeFrame`)
- Create: `roro/resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_resume.py`:

```python
"""Resume primitives: block alignment, checkpoint row seeding, refit-date filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roro.resume import (
    HistoryRevisedError,
    prior_refits_before,
    resume_block_start,
    seed_prior_rows,
)
from roro.types import SegmentPrior


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.mark.parametrize(
    ("n_old", "expected"),
    [
        (100, None),  # checkpoint still in warmup -> nothing to reuse
        (252, None),  # first new row is the very first block start -> nothing to reuse
        (253, 252),  # one day into block 0
        (314, 252),  # last day of block 0 already processed, first new row still in block 0
        (315, 315),  # first new row opens block 1 exactly on its boundary
        (316, 315),
        (400, 378),
        (500, None),  # no new rows at all
    ],
)
def test_resume_block_start(n_old: int, expected: int | None) -> None:
    idx = _idx(500)
    got = resume_block_start(
        idx, idx[n_old - 1], min_history_days=252, refit_interval_days=63
    )
    assert got == expected


def _prior(n: int = 10) -> SegmentPrior:
    idx = _idx(n)
    probs = pd.DataFrame(
        {
            "p_risk_off": np.linspace(0.1, 0.2, n),
            "p_transitional": np.linspace(0.3, 0.4, n),
            "p_risk_on": np.linspace(0.6, 0.4, n),
        },
        index=idx,
    )
    cold = pd.Series([True] * 3 + [False] * (n - 3), index=idx)
    refits = (idx[3], idx[6])
    return SegmentPrior(probs=probs, cold_start=cold, refit_dates=refits, last_date=idx[-1])


def test_seed_prior_rows_returns_ordered_columns() -> None:
    prior = _prior()
    probs, cold = seed_prior_rows(prior, prior.probs.index[:5])
    np.testing.assert_array_equal(
        probs, prior.probs[["p_risk_off", "p_transitional", "p_risk_on"]].to_numpy()[:5]
    )
    np.testing.assert_array_equal(cold, [True, True, True, False, False])


def test_seed_prior_rows_missing_date_raises() -> None:
    prior = _prior()
    wanted = pd.DatetimeIndex([prior.probs.index[0], pd.Timestamp("2030-01-01")])
    with pytest.raises(HistoryRevisedError, match="2030-01-01"):
        seed_prior_rows(prior, wanted)


def test_prior_refits_before_is_strict() -> None:
    prior = _prior()
    idx = prior.probs.index
    assert prior_refits_before(prior, idx[6]) == [idx[3]]
    assert prior_refits_before(prior, idx[7]) == [idx[3], idx[6]]
    assert prior_refits_before(prior, idx[3]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_resume.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'roro.resume'`

- [ ] **Step 3: Add `SegmentPrior` to `roro/types.py`** (insert directly after the `JmRegimeFrame` class)

```python
@dataclass(frozen=True)
class SegmentPrior:
    """One segment's HMM or JM output read back from a checkpoint run folder.

    ``probs`` columns are ordered (p_risk_off, p_transitional, p_risk_on) and indexed
    by date; ``refit_dates`` are the converged refit dates from the refit log.
    """

    probs: pd.DataFrame
    cold_start: pd.Series
    refit_dates: tuple[pd.Timestamp, ...]
    last_date: pd.Timestamp
```

- [ ] **Step 4: Create `roro/resume.py`**

```python
"""Resume primitives for incremental runs: block alignment, checkpoint seeding.

The HMM/JM walk-forwards fit at the start of each refit block on data strictly
before it, then infer the block causally. Rows in closed blocks are therefore a
pure function of the data up to the block's end: they can be copied from a
checkpoint and the loop restarted at the open block, reproducing a full rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from roro.types import SegmentPrior

PROB_COLUMNS: tuple[str, str, str] = ("p_risk_off", "p_transitional", "p_risk_on")


class HistoryRevisedError(RuntimeError):
    """Recomputed history differs from the checkpoint (source data was revised)."""


def resume_block_start(
    index: pd.Index,
    last_date: pd.Timestamp,
    *,
    min_history_days: int,
    refit_interval_days: int,
) -> int | None:
    """Position of the refit block containing the first row after ``last_date``.

    Returns None when nothing can be reused: the checkpoint ends inside the warmup
    (or exactly at the first block start), or there are no new rows.
    """
    n_old = int(index.searchsorted(last_date, side="right"))
    if n_old <= min_history_days or n_old >= len(index):
        return None
    k = (n_old - min_history_days) // refit_interval_days
    return min_history_days + k * refit_interval_days


def seed_prior_rows(
    prior: SegmentPrior, dates: pd.Index
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Checkpoint probabilities (ordered columns) and cold-start flags for ``dates``."""
    missing = dates.difference(prior.probs.index)
    if len(missing) > 0:
        raise HistoryRevisedError(
            f"checkpoint has no overlay row for {pd.Timestamp(missing[0]).date()}"
        )
    probs = prior.probs.loc[dates, list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    cold = prior.cold_start.loc[dates].to_numpy(dtype=bool)
    return probs, cold


def prior_refits_before(prior: SegmentPrior, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    """Converged refit dates strictly before ``cutoff`` (the open block's first date)."""
    return [d for d in prior.refit_dates if d < cutoff]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_resume.py -v`
Expected: 11 passed

- [ ] **Step 6: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/resume.py roro/types.py tests/test_resume.py && .venv/Scripts/python.exe -m mypy roro/resume.py roro/types.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add roro/resume.py roro/types.py tests/test_resume.py
git commit -m "feat(resume): block alignment + checkpoint seeding primitives

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: HMM walk-forward resume

**Files:**
- Modify: `roro/regime_hmm.py` (`walk_forward`, `classify_hmm`, imports)
- Test: `tests/test_regime_hmm_resume.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_regime_hmm_resume.py`:

```python
"""HMM walk-forward resume: output must equal a run without a checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import roro.regime_hmm as hmm_mod
from roro.config import EngineConfig
from roro.regime_hmm import classify_hmm, walk_forward
from roro.types import BetaBySegment, BetaFrame, SegmentPrior

_KW: dict[str, Any] = dict(
    refit_interval_days=60, min_history_days=150, switching_variance=True
)
_SERIES_KEYS = (
    "state", "label", "prob_risk_off", "prob_transitional", "prob_risk_on",
    "confidence", "cold_start",
)


def _beta(n: int = 420, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    third = n // 3
    vals = np.concatenate(
        [
            rng.normal(-0.8, 0.05, third),
            rng.normal(0.0, 0.05, third),
            rng.normal(0.8, 0.05, n - 2 * third),
        ]
    )
    return pd.Series(vals, index=pd.bdate_range("2012-01-02", periods=n), name="beta")


def _prior_from(out: dict[str, Any], last_date: pd.Timestamp) -> SegmentPrior:
    """What a checkpoint folder holds for this segment (rows up to last_date)."""
    probs = pd.DataFrame(
        {
            "p_risk_off": out["prob_risk_off"],
            "p_transitional": out["prob_transitional"],
            "p_risk_on": out["prob_risk_on"],
        }
    )
    return SegmentPrior(
        probs=probs.loc[:last_date],
        cold_start=out["cold_start"].loc[:last_date],
        refit_dates=tuple(out["refit_dates"]),
        last_date=last_date,
    )


def _assert_same(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in _SERIES_KEYS:
        pd.testing.assert_series_equal(a[key], b[key], check_exact=True, obj=key)
    assert a["refit_dates"] == b["refit_dates"]


@pytest.mark.parametrize("split", [100, 200, 210, 330, 419])
def test_resume_equals_full(split: int) -> None:
    beta = _beta()
    last = beta.index[split - 1]
    checkpoint = walk_forward(beta.loc[:last], **_KW)
    resumed = walk_forward(beta, prior=_prior_from(checkpoint, last), **_KW)
    _assert_same(resumed, walk_forward(beta, **_KW))


def test_resume_rebuilds_last_good_when_open_block_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = hmm_mod._fit_params

    def flaky(beta: pd.Series, *, switching_variance: bool) -> Any:
        fit = real(beta, switching_variance=switching_variance)
        if len(beta) == 330:  # the fit that opens the resumed block
            return hmm_mod._FitResult(
                params=fit.params, perm=fit.perm, means=fit.means, converged=False
            )
        return fit

    monkeypatch.setattr(hmm_mod, "_fit_params", flaky)
    beta = _beta()
    last = beta.index[349]  # n_old=350 -> open block starts at 330
    checkpoint = walk_forward(beta.loc[:last], **_KW)
    resumed = walk_forward(beta, prior=_prior_from(checkpoint, last), **_KW)
    full = walk_forward(beta, **_KW)
    _assert_same(resumed, full)
    assert not np.isnan(resumed["prob_risk_on"].iloc[330])  # fallback params were used


def test_classify_hmm_prior_is_per_segment() -> None:
    beta = _beta()
    cap = pd.DataFrame({"beta": beta, "r2": 0.5, "n": 20}, index=beta.index)
    bf = BetaFrame(cap_wtd=cap, eq_wtd=cap, slope_spread=pd.Series(0.0, index=beta.index))
    bbs = BetaBySegment(by_segment={"global": bf, "LatAm": bf})
    cfg = EngineConfig(
        data_path=Path("d.xlsx"),
        output_dir=Path("o"),
        hmm_refit_interval_days=60,
        hmm_min_history_days=150,
    )
    last = beta.index[299]
    checkpoint = walk_forward(beta.loc[:last], **_KW)
    # Only "global" has a checkpoint; "LatAm" must fall back to a full walk-forward.
    frame = classify_hmm(
        bbs, cfg=cfg, thin_cuts=frozenset({"LatAm"}),
        prior={"global": _prior_from(checkpoint, last)},
    )
    full = classify_hmm(bbs, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    pd.testing.assert_frame_equal(frame.prob_risk_on, full.prob_risk_on, check_exact=True)
    assert frame.refit_dates == full.refit_dates
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_hmm_resume.py -v`
Expected: FAIL, `TypeError: walk_forward() got an unexpected keyword argument 'prior'`

- [ ] **Step 3: Implement.** In `roro/regime_hmm.py`:

Imports: add `from collections.abc import Mapping` and

```python
from roro.resume import prior_refits_before, resume_block_start, seed_prior_rows
from roro.types import BetaBySegment, HmmRegimeFrame, SegmentPrior
```

(replace the existing `from roro.types import BetaBySegment, HmmRegimeFrame`).

Replace `walk_forward`'s signature, docstring tail and loop head with:

```python
def walk_forward(
    beta: pd.Series,
    *,
    refit_interval_days: int,
    min_history_days: int,
    switching_variance: bool,
    prior: SegmentPrior | None = None,
) -> dict[str, object]:
    """Causal per-segment HMM labels over the full beta index.

    Returns a dict with keys: state, label, prob_risk_off, prob_transitional,
    prob_risk_on, confidence, cold_start, refit_dates. Each value (except
    refit_dates: list[Timestamp]) is a pandas object aligned to beta.index.

    Refit clock: every refit_interval_days (trading days) starting at
    min_history_days, params are re-estimated on beta[:t]. Filter clock: daily.
    Within a refit block [r, r'), filtered probs come from one .filter() over
    beta[:r'] with params(beta[:r]) (causal). NaN beta rows are dropped before
    fitting and emitted as Unknown.

    prior: a checkpoint of this segment. Rows of closed refit blocks are copied
    and the loop restarts at the open block, so the output equals a run without
    prior whenever beta up to prior.last_date is unchanged.
    """
    full_index = beta.index
    clean = beta.dropna()
    n = len(clean)

    probs = np.full((n, _K_REGIMES), np.nan)
    cold = np.ones(n, dtype=bool)
    refit_dates: list[pd.Timestamp] = []

    last_good: _FitResult | None = None
    r = min_history_days
    resume_at = (
        resume_block_start(
            clean.index,
            prior.last_date,
            min_history_days=min_history_days,
            refit_interval_days=refit_interval_days,
        )
        if prior is not None
        else None
    )
    if prior is not None and resume_at is not None:
        probs[:resume_at], cold[:resume_at] = seed_prior_rows(prior, clean.index[:resume_at])
        refit_dates = prior_refits_before(prior, pd.Timestamp(clean.index[resume_at]))
        r = resume_at
    while r < n:
        block_end = min(r + refit_interval_days, n)
        fit = _fit_params(clean.iloc[:r], switching_variance=switching_variance)
        if fit.converged:
            last_good = fit
            refit_dates.append(pd.Timestamp(clean.index[r]))
        elif last_good is None and refit_dates:
            # Resumed run only (a fresh run appends refit_dates together with
            # last_good): rebuild the fallback a full run carried into this block.
            at = int(clean.index.searchsorted(refit_dates[-1]))
            fallback = _fit_params(clean.iloc[:at], switching_variance=switching_variance)
            if fallback.converged:
                last_good = fallback
        used = fit if fit.converged else last_good
```

The rest of the loop body (`if used is not None: ...`, `r = block_end`) and everything after it stays unchanged.

In `classify_hmm`, add the parameter and pass it through:

```python
def classify_hmm(
    bbs: BetaBySegment,
    *,
    cfg: EngineConfig,
    thin_cuts: frozenset[str],
    prior: Mapping[str, SegmentPrior] | None = None,
) -> HmmRegimeFrame:
    """Run the walk-forward HMM per segment; assemble an HmmRegimeFrame.

    prior: per-segment checkpoint rows (see walk_forward); a missing segment runs
    in full.
    """
```

and in the loop:

```python
        out = walk_forward(
            beta,
            refit_interval_days=cfg.hmm_refit_interval_days,
            min_history_days=cfg.hmm_min_history_days,
            switching_variance=cfg.hmm_switching_variance,
            prior=prior.get(cut) if prior is not None else None,
        )
```

- [ ] **Step 4: Run new + existing HMM tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_hmm_resume.py tests/test_regime_hmm.py -v`
Expected: all pass (existing slow HMM tests included; they prove `prior=None` is unchanged)

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/regime_hmm.py tests/test_regime_hmm_resume.py && .venv/Scripts/python.exe -m mypy roro/regime_hmm.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/regime_hmm.py tests/test_regime_hmm_resume.py
git commit -m "feat(hmm): resume walk-forward from checkpoint rows (exact vs full)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: JM walk-forward resume

**Files:**
- Modify: `roro/regime_jm.py`
- Test: `tests/test_regime_jm_resume.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_regime_jm_resume.py`:

```python
"""JM walk-forward resume (discrete, continuous, rolling): equals a run without prior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import roro.regime_jm as jm_mod
from roro.config import EngineConfig
from roro.jump_model import JumpFit
from roro.regime_jm import classify_jm, walk_forward
from roro.types import BetaBySegment, BetaFrame, SegmentPrior

_SERIES_KEYS = (
    "state", "label", "prob_risk_off", "prob_transitional", "prob_risk_on",
    "confidence", "cold_start",
)


def _kw(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        jump_penalty=20.0, refit_interval_days=40, min_history_days=120, n_states=3,
        window="expanding", rolling_window_days=2000, continuous=False, n_init=4,
        max_iter=30, tol=1e-8, seed=0,
    )
    base.update(over)
    return base


def _series(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(11)
    vals = np.concatenate([rng.normal(-2.0, 0.3, n // 2), rng.normal(2.0, 0.3, n - n // 2)])
    return pd.Series(vals, index=pd.bdate_range("2015-01-01", periods=n))


def _prior_from(out: dict[str, Any], last_date: pd.Timestamp) -> SegmentPrior:
    probs = pd.DataFrame(
        {
            "p_risk_off": out["prob_risk_off"],
            "p_transitional": out["prob_transitional"],
            "p_risk_on": out["prob_risk_on"],
        }
    )
    return SegmentPrior(
        probs=probs.loc[:last_date],
        cold_start=out["cold_start"].loc[:last_date],
        refit_dates=tuple(out["refit_dates"]),
        last_date=last_date,
    )


def _assert_same(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in _SERIES_KEYS:
        pd.testing.assert_series_equal(a[key], b[key], check_exact=True, obj=key)
    assert a["refit_dates"] == b["refit_dates"]


@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"continuous": True},
        {"window": "rolling", "rolling_window_days": 150},
    ],
    ids=["discrete", "continuous", "rolling"],
)
@pytest.mark.parametrize("split", [60, 150, 160, 285, 399])
def test_resume_equals_full(variant: dict[str, Any], split: int) -> None:
    s = _series()
    kw = _kw(**variant)
    last = s.index[split - 1]
    checkpoint = walk_forward(s.loc[:last], **kw)
    resumed = walk_forward(s, prior=_prior_from(checkpoint, last), **kw)
    _assert_same(resumed, walk_forward(s, **kw))


def test_resume_rebuilds_last_good_when_open_block_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = jm_mod.fit_jump_model

    def flaky(x: Any, **kwargs: Any) -> JumpFit:
        fit = real(x, **kwargs)
        if len(x) == 280:  # expanding window at r=280 -> the resumed open block
            return JumpFit(centroids=fit.centroids, labels=fit.labels,
                           objective=fit.objective, converged=False)
        return fit

    monkeypatch.setattr(jm_mod, "fit_jump_model", flaky)
    s = _series()
    kw = _kw()
    last = s.index[289]  # n_old=290 -> open block starts at 280
    checkpoint = walk_forward(s.loc[:last], **kw)
    resumed = walk_forward(s, prior=_prior_from(checkpoint, last), **kw)
    _assert_same(resumed, walk_forward(s, **kw))
    assert resumed["label"].iloc[280] != "Unknown"  # fallback centroids were used


def test_resume_does_less_work(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = jm_mod.fit_jump_model

    def counting(x: Any, **kwargs: Any) -> JumpFit:
        calls["n"] += 1
        return real(x, **kwargs)

    monkeypatch.setattr(jm_mod, "fit_jump_model", counting)
    s = _series()
    last = s.index[389]
    checkpoint = walk_forward(s.loc[:last], **_kw())
    calls["n"] = 0
    walk_forward(s, prior=_prior_from(checkpoint, last), **_kw())
    assert calls["n"] == 1  # only the open block (start 360) is re-fit


def test_classify_jm_prior_is_per_segment() -> None:
    s = _series()
    cap = pd.DataFrame({"beta": s, "n": np.full(len(s), 12)}, index=s.index)
    bf = BetaFrame(cap_wtd=cap, eq_wtd=pd.DataFrame(), slope_spread=pd.Series(dtype=float))
    bbs = BetaBySegment(by_segment={"global": bf, "LatAm": bf})
    cfg = EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o"),
                       jm_min_history_days=120, jm_refit_interval_days=40, jm_n_init=4,
                       jm_jump_penalty=20.0)
    last = s.index[299]
    checkpoint = walk_forward(s.loc[:last], **_kw())
    frame = classify_jm(bbs, cfg=cfg, thin_cuts=frozenset({"LatAm"}),
                        prior={"global": _prior_from(checkpoint, last)})
    full = classify_jm(bbs, cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    pd.testing.assert_frame_equal(frame.prob_risk_on, full.prob_risk_on, check_exact=True)
    assert frame.refit_dates == full.refit_dates
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_jm_resume.py -v`
Expected: FAIL, `TypeError: walk_forward() got an unexpected keyword argument 'prior'`

- [ ] **Step 3: Implement.** In `roro/regime_jm.py`:

Imports: add `from collections.abc import Mapping` and `from functools import partial`. Change the jump_model import to include `JumpFit`:

```python
from roro.jump_model import (
    JumpFit,
    discretize_simplex,
    fit_jump_model,
    online_soft_states,
    online_states,
)
from roro.resume import prior_refits_before, resume_block_start, seed_prior_rows
from roro.types import BetaBySegment, JmRegimeFrame, SegmentPrior
```

Add after `_causal_scaler`:

```python
def _fit_at(
    cvals: NDArray[np.float64],
    r: int,
    *,
    window: str,
    rolling_window_days: int,
    n_states: int,
    jump_penalty: float,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[JumpFit, float, float]:
    """Fit on the point-in-time window ending before position r -> (fit, mean, std)."""
    fit_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
    fit_win = cvals[fit_lo:r]
    mean, std = _causal_scaler(fit_win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_jump_model(
            (fit_win - mean) / std,
            k=n_states,
            jump_penalty=jump_penalty,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
    return fit, mean, std
```

Replace `walk_forward` from its signature through the line `used = (fit.centroids, mean, std) if fit.converged else last_good` with:

```python
def walk_forward(
    beta: pd.Series[Any],
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
    prior: SegmentPrior | None = None,
) -> dict[str, Any]:
    """Expanding/rolling refit + per-block frozen-scaler forward-DP online inference.

    prior: a checkpoint of this segment. Rows of closed refit blocks are copied
    and the loop restarts at the open block, so the output equals a run without
    prior whenever beta up to prior.last_date is unchanged.
    """
    full_index = beta.index
    clean = beta.dropna()
    n = len(clean)
    cvals = clean.to_numpy(dtype=np.float64)

    probs = np.full((n, n_states), np.nan)
    cold = np.ones(n, dtype=bool)
    refit_dates: list[pd.Timestamp] = []
    last_good: tuple[NDArray[np.float64], float, float] | None = None

    grid = discretize_simplex(n_states, 0.05) if continuous else None
    fit_at = partial(
        _fit_at,
        cvals,
        window=window,
        rolling_window_days=rolling_window_days,
        n_states=n_states,
        jump_penalty=jump_penalty,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )

    r = min_history_days
    resume_at = (
        resume_block_start(
            clean.index,
            prior.last_date,
            min_history_days=min_history_days,
            refit_interval_days=refit_interval_days,
        )
        if prior is not None
        else None
    )
    if prior is not None and resume_at is not None:
        probs[:resume_at], cold[:resume_at] = seed_prior_rows(prior, clean.index[:resume_at])
        refit_dates = prior_refits_before(prior, pd.Timestamp(clean.index[resume_at]))
        r = resume_at
    while r < n:
        block_end = min(r + refit_interval_days, n)
        fit, mean, std = fit_at(r)
        if fit.converged:
            last_good = (fit.centroids, mean, std)
            refit_dates.append(pd.Timestamp(clean.index[r]))
        elif last_good is None and refit_dates:
            # Resumed run only (a fresh run appends refit_dates together with
            # last_good): rebuild the fallback a full run carried into this block.
            fb, fb_mean, fb_std = fit_at(int(clean.index.searchsorted(refit_dates[-1])))
            if fb.converged:
                last_good = (fb.centroids, fb_mean, fb_std)
        used = (fit.centroids, mean, std) if fit.converged else last_good
```

In the rest of the loop body, delete the now-redundant line `cvals = clean.to_numpy(dtype=np.float64)` (it moved above the loop). Everything else stays the same.

In `classify_jm`, add `prior: Mapping[str, SegmentPrior] | None = None` after `thin_cuts`, then pass `prior=prior.get(cut) if prior is not None else None` to `walk_forward`. Extend its docstring with: "prior: per-segment checkpoint rows (see walk_forward); a missing segment runs in full."

- [ ] **Step 4: Run new + existing JM tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_jm_resume.py tests/test_regime_jm.py tests/test_jump_model.py -v`
Expected: all pass. The existing `test_walk_forward_last_good_fallback` must still see the same call count: `prior=None` never enters the new `elif`.

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/regime_jm.py tests/test_regime_jm_resume.py && .venv/Scripts/python.exe -m mypy roro/regime_jm.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/regime_jm.py tests/test_regime_jm_resume.py
git commit -m "feat(jm): resume walk-forward from checkpoint rows (exact vs full)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Random-walk test fixture

`tiny_xlsx` has linear-ramp prices (smooth betas) and is pinned by goldens. HMM/JM resume at engine level needs realistic, noisy data. Refactor the xlsx builder so `tiny_xlsx` output stays byte-identical, and add `rw_xlsx`.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Refactor `tests/conftest.py`**

Replace the body of `tiny_xlsx` and add the helpers. Final module (keep `pytest_addoption` and `_write_two_header_sheet` exactly as they are):

```python
"""Shared pytest fixtures: tiny synthetic dataset that mirrors data.xlsx structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COUNTRIES: list[str] = ["United States", "Brazil", "Germany", "Mexico"]
_EQ_TICKERS: list[str] = ["SPX Index", "MXBR Index", "MXDE Index", "MXMX Index"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--regenerate-goldens", action="store_true", default=False)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": ["United States", "Brazil", "Germany", "Mexico", "DM", "LatAm"],
            "Segment": ["DM", "EM", "DM", "EM", "DM", "EM"],
            "Equity Index": ["SPX", "MXBR", "MXDE", "MXMX", "MXWO", "MXLA"],
            "Equity Index Curreny": ["USD"] * 6,
            "Bond Index": ["LBUSTRUU", "I00", "I05", "I05M", "I35", "H04"],
            "Bond Index Curreny": ["USD"] * 6,
            "Local Curreny": ["USD", "BRL", "EUR", "MXN", "USD", "USD"],
            "Curr": [1.0, 5.0, 1.16, 20.0, 1.0, 1.0],
            "Pair": ["USD", "USDBRL", "USDEUR", "USDMXN", "USD", "USD"],
            "Equity_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Equity_Mkt_Cap": [100, 10, 20, 5, 130, 15],
            "FI_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Fixed_Income_Mkt_Cap": [50, 5, 10, 3, 65, 8],
            "Equity_Mkt_Cap_Val": [100, 10, 20, 5, 130, 15],
            "Fixed_Income_Mkt_Cap_Val": [50, 5, 10, 3, 65, 8],
        }
    )


def build_xlsx(path: Path, eq_data: pd.DataFrame, fi_data: pd.DataFrame) -> Path:
    """Write a data.xlsx-shaped workbook (Panel + Equity_LC + Fixed_Income_LC)."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        _panel().to_excel(w, sheet_name="Panel", index=False)
        _write_two_header_sheet(w, "Equity_LC", _EQ_TICKERS, COUNTRIES, eq_data)
        _write_two_header_sheet(w, "Fixed_Income_LC", _EQ_TICKERS, COUNTRIES, fi_data)
    return path


def random_walk_prices(
    dates: pd.DatetimeIndex, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded geometric random walks with distinct per-country vols (noisy betas)."""
    rng = np.random.default_rng(seed)
    n, k = len(dates), len(COUNTRIES)
    eq_vol = np.array([0.008, 0.018, 0.011, 0.015])
    fi_vol = np.array([0.002, 0.005, 0.003, 0.004])
    eq_ret = rng.normal(0.0003, 1.0, (n, k)) * eq_vol
    fi_ret = rng.normal(0.0001, 1.0, (n, k)) * fi_vol
    eq = pd.DataFrame(100.0 * np.exp(np.cumsum(eq_ret, axis=0)), index=dates, columns=COUNTRIES)
    fi = pd.DataFrame(200.0 * np.exp(np.cumsum(fi_ret, axis=0)), index=dates, columns=COUNTRIES)
    return eq, fi


RW_DATES: pd.DatetimeIndex = pd.bdate_range("2020-01-01", "2024-12-31")
RW_SEED: int = 7


@pytest.fixture
def tiny_xlsx(tmp_path: Path) -> Path:
    """Build a 4-country + 2-composite tiny xlsx that mirrors data.xlsx layout."""
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    eq_data = pd.DataFrame({c: range(100, 100 + len(dates)) for c in COUNTRIES}, index=dates)
    fi_data = pd.DataFrame({c: range(200, 200 + len(dates)) for c in COUNTRIES}, index=dates)
    return build_xlsx(tmp_path / "tiny.xlsx", eq_data, fi_data)


@pytest.fixture
def rw_xlsx(tmp_path: Path) -> Path:
    """Same layout as tiny_xlsx, seeded random-walk prices (noisy, realistic betas)."""
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    return build_xlsx(tmp_path / "rw.xlsx", eq, fi)
```

- [ ] **Step 2: Prove `tiny_xlsx` is unchanged**

Run: `.venv/Scripts/python.exe -m pytest tests/test_golden.py tests/test_reproducibility.py -v`
Expected: all pass (goldens unchanged, no `--regenerate-goldens`)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test(fixtures): build_xlsx helper + seeded random-walk rw_xlsx fixture

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: io — `beta_long`, `cut_prices`, `read_resume_state`, `verify_beta_history`

**Files:**
- Modify: `roro/types.py` (add `ResumeState` after `SegmentPrior`)
- Modify: `roro/io.py`
- Modify: `roro/resume.py` (add `verify_beta_history`)
- Test: `tests/test_io_resume.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_io_resume.py`:

```python
"""Checkpoint read-back (exact float round-trip) and the beta history check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.io import (
    _write_hmm_refit_log,
    _write_regime_hmm,
    beta_long,
    cut_prices,
    load_prices,
    read_resume_state,
)
from roro.resume import HistoryRevisedError, verify_beta_history
from roro.types import HmmRegimeFrame


def _hmm_frame() -> HmmRegimeFrame:
    idx = pd.bdate_range("2021-01-01", periods=30)
    rng = np.random.default_rng(5)
    raw = rng.random((30, 3))
    raw /= raw.sum(axis=1, keepdims=True)  # full-precision floats, not round numbers
    raw[:4] = np.nan

    def frame(col: int) -> pd.DataFrame:
        return pd.DataFrame({"global": raw[:, col], "EM": raw[::-1, col]}, index=idx)

    labels = pd.DataFrame("Risk-on", index=idx, columns=["global", "EM"])
    cold = pd.DataFrame(False, index=idx, columns=["global", "EM"])
    cold.iloc[:4] = True
    return HmmRegimeFrame(
        state=pd.DataFrame(2.0, index=idx, columns=["global", "EM"]),
        label=labels,
        prob_risk_off=frame(0),
        prob_transitional=frame(1),
        prob_risk_on=frame(2),
        confidence=frame(2),
        n_per_segment=pd.DataFrame(10, index=idx, columns=["global", "EM"]),
        thin_cut_flag=pd.DataFrame(False, index=idx, columns=["global", "EM"]),
        cold_start_flag=cold,
        refit_dates={"global": [idx[4], idx[20]], "EM": [idx[4]]},
    )


def test_overlay_prior_round_trips_exactly(tmp_path: Path) -> None:
    hf = _hmm_frame()
    _write_regime_hmm(hf, tmp_path / "regimes_hmm.csv")
    _write_hmm_refit_log(hf, tmp_path / "hmm_refit_log.csv")
    beta = pd.DataFrame(
        {"date": hf.label.index, "segment": "global", "scheme": "cap_wtd", "beta": 0.1}
    )
    beta.to_csv(tmp_path / "beta_series.csv", index=False)
    last = hf.label.index[-1]
    state = read_resume_state(tmp_path, last)
    assert state.jm is None
    assert state.hmm is not None
    assert sorted(state.hmm) == ["EM", "global"]
    prior = state.hmm["global"]
    np.testing.assert_array_equal(
        prior.probs["p_risk_on"].to_numpy(), hf.prob_risk_on["global"].to_numpy()
    )
    np.testing.assert_array_equal(
        prior.cold_start.to_numpy(), hf.cold_start_flag["global"].to_numpy()
    )
    assert prior.refit_dates == tuple(hf.refit_dates["global"])
    assert prior.last_date == last


def test_beta_long_round_trips_through_checkpoint(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10,
    )
    result = engine_run(
        cfg, fred_client=MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}),
        run_date="r", as_of_data_date="2024-12-31", force=True,
    )
    state = read_resume_state(tmp_path / "out" / "r", pd.Timestamp("2024-12-31"))
    # Exact equality with the in-memory betas proves the CSV round trip is lossless.
    verify_beta_history(beta_long(result.beta), state.beta_series,
                        through=pd.Timestamp("2024-12-31"))


def _long(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.DataFrame(
        {"date": dates, "segment": "global", "scheme": "cap_wtd", "beta": values}
    )


def test_verify_beta_history_accepts_equal_and_nan() -> None:
    old = _long([0.1, float("nan"), 0.3])
    new = _long([0.1, float("nan"), 0.3, 0.4])  # an extra (new) date is ignored
    verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_verify_beta_history_flags_changed_value() -> None:
    old = _long([0.1, 0.2, 0.3])
    new = _long([0.1, 0.2 + 1e-9, 0.3])
    with pytest.raises(HistoryRevisedError, match="2021-01-04 global cap_wtd"):
        verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_verify_beta_history_flags_missing_date() -> None:
    old = _long([0.1, 0.2, 0.3])
    new = old.drop(index=1)
    with pytest.raises(HistoryRevisedError, match="rows differ"):
        verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_cut_prices(tiny_xlsx: Path) -> None:
    cut = cut_prices(load_prices(tiny_xlsx), pd.Timestamp("2023-06-30"))
    assert cut.equity_lc.index.max() == pd.Timestamp("2023-06-30")
    assert cut.fi_lc.index.max() == pd.Timestamp("2023-06-30")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_io_resume.py -v`
Expected: FAIL, `ImportError: cannot import name 'beta_long' from 'roro.io'`

- [ ] **Step 3: Add `ResumeState` to `roro/types.py`** (after `SegmentPrior`)

```python
@dataclass(frozen=True)
class ResumeState:
    """Checkpoint content a RESUME run needs (see roro/historic.py)."""

    checkpoint_date: pd.Timestamp
    beta_series: pd.DataFrame
    hmm: dict[str, SegmentPrior] | None = None
    jm: dict[str, SegmentPrior] | None = None
```

- [ ] **Step 4: Implement in `roro/io.py`**

Add `ResumeState` and `SegmentPrior` to the `from roro.types import (...)` block, then add:

```python
_BETA_COLUMNS: tuple[str, ...] = (
    "date", "segment", "scheme", "beta", "r2", "n", "suppressed", "singular",
)
_PROB_COLUMNS: list[str] = ["p_risk_off", "p_transitional", "p_risk_on"]


def cut_prices(prices: PriceFrame, until: pd.Timestamp) -> PriceFrame:
    """Prices on or before ``until`` (the data_until cut of an update run)."""
    return PriceFrame(equity_lc=prices.equity_lc.loc[:until], fi_lc=prices.fi_lc.loc[:until])


def beta_long(bbs: BetaBySegment) -> pd.DataFrame:
    """Long (date, segment, scheme, beta, r2, n, suppressed, singular) frame of beta_series.csv."""
    cap = _stack_segment_frame({k: v.cap_wtd for k, v in bbs.by_segment.items()}, "cap_wtd")
    eq = _stack_segment_frame({k: v.eq_wtd for k, v in bbs.by_segment.items()}, "eq_wtd")
    if cap.empty and eq.empty:
        return pd.DataFrame(columns=list(_BETA_COLUMNS))
    return pd.concat([cap, eq], ignore_index=True)


def read_resume_state(run_dir: Path, last_date: pd.Timestamp) -> ResumeState:
    """Everything a RESUME run needs from a checkpoint folder (exact float round trip)."""
    beta = pd.read_csv(
        run_dir / "beta_series.csv", parse_dates=["date"], float_precision="round_trip"
    )
    return ResumeState(
        checkpoint_date=last_date,
        beta_series=beta,
        hmm=_read_overlay_prior(
            run_dir / "regimes_hmm.csv", run_dir / "hmm_refit_log.csv", last_date
        ),
        jm=_read_overlay_prior(
            run_dir / "regimes_jm.csv", run_dir / "jm_refit_log.csv", last_date
        ),
    )


def _read_overlay_prior(
    rows_path: Path, log_path: Path, last_date: pd.Timestamp
) -> dict[str, SegmentPrior] | None:
    if not rows_path.exists():
        return None
    rows = pd.read_csv(rows_path, parse_dates=["date"], float_precision="round_trip")
    log = (
        pd.read_csv(log_path, parse_dates=["refit_date"])
        if log_path.exists()
        else pd.DataFrame(columns=["segment", "refit_date"])
    )
    out: dict[str, SegmentPrior] = {}
    for seg in sorted(rows["segment"].astype(str).unique()):
        g = rows.loc[rows["segment"] == seg].set_index("date").sort_index()
        refits = log.loc[log["segment"] == seg, "refit_date"]
        out[seg] = SegmentPrior(
            probs=g[_PROB_COLUMNS],
            cold_start=g["cold_start"].astype(bool),
            refit_dates=tuple(pd.Timestamp(d) for d in refits),
            last_date=last_date,
        )
    return out
```

Replace `_write_beta` with (output bytes unchanged):

```python
def _write_beta(bbs: BetaBySegment, path: Path) -> None:
    frame = beta_long(bbs)
    if frame.empty:
        path.write_text(",".join(_BETA_COLUMNS) + "\n", encoding="utf-8")
        return
    frame.to_csv(path, index=False)
```

- [ ] **Step 5: Add `verify_beta_history` to `roro/resume.py`**

```python
BETA_TOLERANCE: float = 1e-12
_BETA_KEYS: list[str] = ["date", "segment", "scheme"]


def verify_beta_history(
    current: pd.DataFrame, checkpoint: pd.DataFrame, *, through: pd.Timestamp
) -> None:
    """Raise HistoryRevisedError unless current betas up to ``through`` equal the checkpoint's.

    Both frames are long beta_series layout. Same (date, segment, scheme) rows,
    |diff| <= BETA_TOLERANCE, NaN == NaN.
    """
    cur = (
        current.loc[current["date"] <= through]
        .set_index(_BETA_KEYS)["beta"]
        .astype(float)
        .sort_index()
    )
    old = checkpoint.set_index(_BETA_KEYS)["beta"].astype(float).sort_index()
    if not cur.index.equals(old.index):
        first = cur.index.symmetric_difference(old.index)[0]
        raise HistoryRevisedError(
            f"date/segment rows differ from checkpoint (first: "
            f"{pd.Timestamp(first[0]).date()} {first[1]} {first[2]})"
        )
    same = ((cur - old).abs() <= BETA_TOLERANCE) | (cur.isna() & old.isna())
    if not bool(same.all()):
        d, seg, scheme = same.index[~same.to_numpy()][0]
        raise HistoryRevisedError(f"beta changed on {pd.Timestamp(d).date()} {seg} {scheme}")
```

- [ ] **Step 6: Run tests (new + goldens: `_write_beta` must stay byte-identical)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_io_resume.py tests/test_io.py tests/test_golden.py -v`
Expected: all pass

- [ ] **Step 7: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro tests/test_io_resume.py && .venv/Scripts/python.exe -m mypy roro/io.py roro/resume.py roro/types.py`
Expected: no errors

- [ ] **Step 8: Commit**

```bash
git add roro/io.py roro/resume.py roro/types.py tests/test_io_resume.py
git commit -m "feat(io): read checkpoint resume state + beta history check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Engine — `data_until` + `resume`

**Files:**
- Modify: `roro/engine.py`
- Test: `tests/test_engine_resume.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_engine_resume.py`:

```python
"""engine.run data_until cut + resume plumbing + revised-history detection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.io import read_resume_state
from roro.resume import HistoryRevisedError


def _fred() -> MockFredClient:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _cfg(xlsx: Path, out: Path) -> EngineConfig:
    return EngineConfig(
        data_path=xlsx, output_dir=out, ewma_halflife_days=10, return_window_days=21,
        tripwire_window_days=10, percentile_window_years=1, min_n_per_cut=2,
        bootstrap_min_days=10, jm_enabled=True, jm_min_history_days=250,
        jm_refit_interval_days=120, jm_n_init=4,
    )


def test_data_until_cuts_every_artifact(rw_xlsx: Path, tmp_path: Path) -> None:
    engine_run(_cfg(rw_xlsx, tmp_path), fred_client=_fred(), run_date="r",
               as_of_data_date="2023-06-15", force=True, data_until="2023-06-15")
    for name in ("beta_series.csv", "regimes.csv", "regimes_jm.csv", "tripwire.csv"):
        dates = pd.read_csv(tmp_path / "r" / name, parse_dates=["date"])["date"]
        assert dates.max() == pd.Timestamp("2023-06-15"), name


def test_resume_with_revised_beta_raises(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _cfg(rw_xlsx, tmp_path)
    engine_run(cfg, fred_client=_fred(), run_date="cp", as_of_data_date="2023-06-15",
               force=True, data_until="2023-06-15")
    state = read_resume_state(tmp_path / "cp", pd.Timestamp("2023-06-15"))
    tampered = state.beta_series.copy()
    row = tampered.index[tampered["beta"].notna()][100]
    tampered.loc[row, "beta"] += 1e-6
    with pytest.raises(HistoryRevisedError, match="beta changed"):
        engine_run(cfg, fred_client=_fred(), run_date="new", as_of_data_date="2024-12-31",
                   force=True, resume=replace(state, beta_series=tampered))
    assert not (tmp_path / "new").exists()  # nothing written on a revised history
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_engine_resume.py -v`
Expected: FAIL, `TypeError: run() got an unexpected keyword argument 'data_until'`

- [ ] **Step 3: Implement in `roro/engine.py`**

Imports: add `beta_long` and `cut_prices` to the `from roro.io import (...)` block. Add `from roro.resume import verify_beta_history`. Add `ResumeState` to the `from roro.types import (...)` block.

Signature and docstring:

```python
def run(
    cfg: EngineConfig,
    *,
    fred_client: FredClient,
    run_date: str,
    as_of_data_date: str,
    force: bool = False,
    data_until: str | None = None,
    resume: ResumeState | None = None,
) -> RunResult:
    """Execute the full RoRo pipeline and persist outputs under ``cfg.output_dir``.

    Steps: load + validate inputs -> FRED pull -> returns/vol kernels ->
    segment partition -> cross-sectional regression -> classification ->
    correlation panel -> external + internal validation -> tripwire ->
    alerts -> write run.

    data_until: drop prices after this date (YYYY-MM-DD) before any computation.
    resume: checkpoint state. Betas up to its date must equal the checkpoint's
    (else HistoryRevisedError, nothing written); HMM/JM then resume from it.
    """
```

Step 1 of the body becomes:

```python
    universe = load_panel(cfg.data_path)
    warnings.extend(validate_universe(universe))
    prices = load_prices(cfg.data_path)
    if data_until is not None:
        prices = cut_prices(prices, pd.Timestamp(data_until))
    warnings.extend(validate_prices(prices))
```

Directly after the `beta = compute_beta_by_segment(...)` call, add:

```python
    # 4b) Resume guard: the checkpoint is only valid if history is unchanged.
    if resume is not None:
        verify_beta_history(beta_long(beta), resume.beta_series, through=resume.checkpoint_date)
```

Pass the priors in steps 5b/5c:

```python
    regime_hmm = (
        classify_hmm(
            beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}),
            prior=resume.hmm if resume is not None else None,
        )
        if cfg.hmm_enabled
        else None
    )

    regime_jm = (
        classify_jm(
            beta, cfg=cfg, thin_cuts=frozenset({"LatAm"}),
            prior=resume.jm if resume is not None else None,
        )
        if cfg.jm_enabled
        else None
    )
```

- [ ] **Step 4: Run tests (new + goldens + engine suite)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_engine_resume.py tests/test_engine.py tests/test_golden.py tests/test_reproducibility.py -v`
Expected: all pass

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/engine.py tests/test_engine_resume.py && .venv/Scripts/python.exe -m mypy roro/engine.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/engine.py tests/test_engine_resume.py
git commit -m "feat(engine): data_until cut + resume from checkpoint with history guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `roro/historic.py` — pure pieces (dates, configs, checkpoints, planning, errors)

**Files:**
- Create: `roro/historic.py`
- Test: `tests/test_historic.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_historic.py`:

```python
"""Pure historic-run helpers: date parsing, checkpoint discovery, run planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from roro.historic import (
    Checkpoint,
    TypeRun,
    UpdateMode,
    UpdatePlan,
    available_configs,
    config_changes,
    find_checkpoint,
    friendly_error,
    parse_user_date,
    plan_update,
)


@pytest.mark.parametrize("text", ["2026-09-25", "25-09-2026", " 25-09-2026 "])
def test_parse_user_date_accepts_both_formats(text: str) -> None:
    assert parse_user_date(text) == pd.Timestamp("2026-09-25")


@pytest.mark.parametrize("text", ["09/25/2026", "2026-13-01", "31-02-2026", "25-9-2026", ""])
def test_parse_user_date_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="DD-MM-YYYY"):
        parse_user_date(text)


def test_available_configs_sorted_yaml_stems(tmp_path: Path) -> None:
    for name in ("b.yaml", "a.yaml", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert list(available_configs(tmp_path)) == ["a", "b"]


def _mk(root: Path, name: str, *, snapshot: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if snapshot:
        (d / "snapshot.json").write_text(json.dumps({"config_resolved": {}}), encoding="utf-8")
    return d


def test_find_checkpoint_newest_by_name_date(tmp_path: Path) -> None:
    older = _mk(tmp_path, "results_2026-09-18")
    newest = _mk(tmp_path, "results_2026-09-25")
    _mk(tmp_path, "results_2026-10-01.tmp")               # interrupted write
    _mk(tmp_path, "results_2026-10-02", snapshot=False)   # incomplete folder
    _mk(tmp_path, "scratch")
    (older / "touched.txt").write_text("x", encoding="utf-8")  # newer mtime must not matter
    cp = find_checkpoint(tmp_path)
    assert cp is not None
    assert cp.run_dir == newest
    assert cp.last_date == pd.Timestamp("2026-09-25")
    assert cp.snapshot == {"config_resolved": {}}


def test_find_checkpoint_missing_or_empty(tmp_path: Path) -> None:
    assert find_checkpoint(tmp_path / "nope") is None
    assert find_checkpoint(tmp_path) is None


def test_config_changes_ignores_paths_and_key() -> None:
    old = {"a": 1, "output_dir": "x", "data_path": "d1", "fred_api_key": None, "b": 2}
    new = {"a": 1, "output_dir": "y", "data_path": "d2", "fred_api_key": "k", "b": 3, "c": 0}
    assert config_changes(old, new) == ["b", "c"]


_CFG: dict[str, Any] = {"jm_enabled": True, "methodology_version": "1.0.0"}


def _cp(last: str = "2026-09-18", cfg: dict[str, Any] | None = None) -> Checkpoint:
    return Checkpoint(
        run_dir=Path(f"results_{last}"),
        last_date=pd.Timestamp(last),
        snapshot={"config_resolved": _CFG if cfg is None else cfg},
    )


def _plan(**over: Any) -> UpdatePlan:
    kw: dict[str, Any] = dict(
        type_run=TypeRun.NEW_DATA, checkpoint=_cp(), config_now=_CFG,
        data_last=pd.Timestamp("2026-09-25"), data_until=None,
    )
    kw.update(over)
    return plan_update(**kw)


def test_plan_all_forces_full() -> None:
    plan = _plan(type_run=TypeRun.ALL)
    assert plan.mode is UpdateMode.FULL
    assert "requested" in plan.reason


def test_plan_without_checkpoint_is_full() -> None:
    plan = _plan(checkpoint=None)
    assert plan.mode is UpdateMode.FULL
    assert "no checkpoint" in plan.reason


def test_plan_config_change_is_full_and_names_keys() -> None:
    plan = _plan(checkpoint=_cp(cfg={**_CFG, "methodology_version": "0.9.0"}))
    assert plan.mode is UpdateMode.FULL
    assert "methodology_version" in plan.reason


def test_plan_up_to_date() -> None:
    plan = _plan(checkpoint=_cp("2026-09-25"))
    assert plan.mode is UpdateMode.UP_TO_DATE
    assert "type_run='all'" not in plan.reason


def test_plan_up_to_date_hints_when_data_until_is_earlier() -> None:
    plan = _plan(data_last=pd.Timestamp("2026-09-10"), data_until=pd.Timestamp("2026-09-10"))
    assert plan.mode is UpdateMode.UP_TO_DATE
    assert "type_run='all'" in plan.reason


def test_plan_resume() -> None:
    plan = _plan()
    assert plan.mode is UpdateMode.RESUME
    assert "results_2026-09-18" in plan.reason


def test_friendly_error_locked_file() -> None:
    msg = friendly_error(PermissionError(13, "Permission denied", "data.xlsx"))
    assert msg is not None
    assert "data.xlsx" in msg and "Excel" in msg


def test_friendly_error_unknown_is_none() -> None:
    assert friendly_error(ValueError("boom")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_historic.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'roro.historic'`

- [ ] **Step 3: Create `roro/historic.py`** (pure part; Task 8 appends `run_update`)

```python
"""Incremental historic runs: checkpoint discovery, run planning, orchestration.

Layout: <historic_root>/<config-stem>/results_<YYYY-MM-DD>/, where the date is the
last data date processed. A RESUME run copies closed HMM/JM refit blocks from the
newest checkpoint; everything else recomputes on the full data. Spec:
docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_HISTORIC_ROOT: Path = Path("outputs") / "historic"

_RESULTS_DIR = re.compile(r"^results_(\d{4}-\d{2}-\d{2})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DMY_DATE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
#: Config keys that do not change results (paths, secrets): never force a FULL run.
_CONFIG_KEYS_IGNORED: frozenset[str] = frozenset({"output_dir", "data_path", "fred_api_key"})


class TypeRun(StrEnum):
    NEW_DATA = "new_data"
    ALL = "all"


class UpdateMode(StrEnum):
    FULL = "full"
    RESUME = "resume"
    UP_TO_DATE = "up_to_date"


@dataclass(frozen=True)
class Checkpoint:
    run_dir: Path
    last_date: pd.Timestamp
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class UpdatePlan:
    mode: UpdateMode
    reason: str


@dataclass(frozen=True)
class UpdateOutcome:
    mode: UpdateMode
    reason: str
    run_dir: Path | None
    dates_added: int
    elapsed_seconds: float


def parse_user_date(text: str) -> pd.Timestamp:
    """Parse DD-MM-YYYY or YYYY-MM-DD (the 4-digit year's position disambiguates)."""
    s = text.strip()
    message = f"Invalid date {text!r}: use DD-MM-YYYY (25-09-2026) or YYYY-MM-DD (2026-09-25)"
    if _ISO_DATE.match(s):
        fmt = "%Y-%m-%d"
    elif _DMY_DATE.match(s):
        fmt = "%d-%m-%Y"
    else:
        raise ValueError(message)
    try:
        return pd.Timestamp(datetime.strptime(s, fmt))
    except ValueError:
        raise ValueError(message) from None


def available_configs(configs_dir: Path) -> dict[str, Path]:
    """Config name (file stem) -> path, sorted by name."""
    return {p.stem: p for p in sorted(configs_dir.glob("*.yaml"))}


def find_checkpoint(historic_dir: Path) -> Checkpoint | None:
    """Newest complete ``results_YYYY-MM-DD`` folder, by the date in its name.

    Skips ``.tmp`` folders (interrupted writes) and folders without snapshot.json.
    """
    if not historic_dir.is_dir():
        return None
    found: list[tuple[pd.Timestamp, Path]] = []
    for p in historic_dir.iterdir():
        m = _RESULTS_DIR.match(p.name)
        if m and p.is_dir() and (p / "snapshot.json").is_file():
            found.append((pd.Timestamp(m.group(1)), p))
    if not found:
        return None
    last_date, run_dir = max(found)
    snapshot = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    return Checkpoint(run_dir=run_dir, last_date=last_date, snapshot=snapshot)


def config_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Sorted config keys whose value differs (paths and secrets ignored)."""
    keys = (set(old) | set(new)) - _CONFIG_KEYS_IGNORED
    return sorted(k for k in keys if old.get(k) != new.get(k))


def plan_update(
    *,
    type_run: TypeRun,
    checkpoint: Checkpoint | None,
    config_now: Mapping[str, Any],
    data_last: pd.Timestamp,
    data_until: pd.Timestamp | None,
) -> UpdatePlan:
    """Decide FULL / RESUME / UP_TO_DATE. First matching rule wins (spec section 6)."""
    if type_run is TypeRun.ALL:
        return UpdatePlan(UpdateMode.FULL, "full history requested")
    if checkpoint is None:
        return UpdatePlan(UpdateMode.FULL, "no checkpoint yet")
    changed = config_changes(checkpoint.snapshot.get("config_resolved", {}), config_now)
    if changed:
        return UpdatePlan(
            UpdateMode.FULL,
            f"config changed since {checkpoint.run_dir.name}: {', '.join(changed)}",
        )
    if checkpoint.last_date >= data_last:
        reason = f"already processed through {checkpoint.last_date.date()}"
        if data_until is not None and data_until < checkpoint.last_date:
            reason += "; use type_run='all' to rebuild as of an earlier date"
        return UpdatePlan(UpdateMode.UP_TO_DATE, reason)
    return UpdatePlan(UpdateMode.RESUME, f"resuming from {checkpoint.run_dir.name}")


def friendly_error(exc: BaseException) -> str | None:
    """Plain-English message for errors a non-developer can fix; None otherwise."""
    if isinstance(exc, PermissionError):
        return f"Cannot open {exc.filename}: close it in Excel (or any other program) and retry"
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {exc.filename}"
    return None
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_historic.py -v`
Expected: all pass

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/historic.py tests/test_historic.py && .venv/Scripts/python.exe -m mypy roro/historic.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/historic.py tests/test_historic.py
git commit -m "feat(historic): checkpoint discovery, run planning, user date parsing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `run_update` orchestrator + engine-level equality

**Files:**
- Modify: `roro/historic.py` (append orchestrator)
- Test: `tests/test_historic_update.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_historic_update.py`:

```python
"""run_update end to end: RESUME output is byte-identical to a FULL rerun (AI-1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import roro.regime_jm as jm_mod
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.historic import TypeRun, UpdateMode, UpdateOutcome, run_update
from tests.conftest import RW_DATES, RW_SEED, build_xlsx, random_walk_prices

T0 = pd.Timestamp("2023-06-15")  # checkpoint date, inside a JM refit block


def _fred() -> MockFredClient:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _config(tmp_path: Path, xlsx: Path, *, hmm: bool = False, jm: bool = True,
            penalty: float = 50.0) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"data_path: {xlsx.as_posix()}\n"
        "output_dir: ignored\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "percentile_window_years: 1\n"
        "min_n_per_cut: 2\n"
        "bootstrap_min_days: 10\n"
        f"hmm_enabled: {str(hmm).lower()}\n"
        "hmm_min_history_days: 250\n"
        "hmm_refit_interval_days: 250\n"
        f"jm_enabled: {str(jm).lower()}\n"
        "jm_continuous: true\n"
        f"jm_jump_penalty: {penalty}\n"
        "jm_min_history_days: 250\n"
        "jm_refit_interval_days: 120\n"
        "jm_n_init: 4\n",
        encoding="utf-8",
    )
    return path


def _update(cfg: Path, root: Path, type_run: TypeRun,
            until: pd.Timestamp | None = None) -> UpdateOutcome:
    return run_update(cfg, type_run=type_run, data_until=until, build_report=False,
                      fred_client=_fred(), historic_root=root, echo=lambda _m: None)


def _assert_same_csvs(a: Path, b: Path) -> None:
    files_a = sorted(p.name for p in a.glob("*.csv"))
    files_b = sorted(p.name for p in b.glob("*.csv"))
    assert files_a == files_b
    for name in files_a:
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} differs"


def test_resume_equals_full_rerun(rw_xlsx: Path, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    first = _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    assert first.mode is UpdateMode.FULL
    assert first.run_dir is not None and first.run_dir.name == "results_2023-06-15"

    calls = {"n": 0}
    real = jm_mod.fit_jump_model

    def counting(x: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(x, **kwargs)

    monkeypatch.setattr(jm_mod, "fit_jump_model", counting)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    resume_fits = calls["n"]
    calls["n"] = 0
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    full_fits = calls["n"]

    assert resumed.mode is UpdateMode.RESUME
    assert resumed.run_dir is not None and full.run_dir is not None
    assert resumed.run_dir.name == full.run_dir.name == "results_2024-12-31"
    _assert_same_csvs(resumed.run_dir, full.run_dir)
    assert resume_fits < full_fits  # the resume really skipped closed blocks
    assert resumed.dates_added == int((RW_DATES > T0).sum())

    update = json.loads((resumed.run_dir / "snapshot.json").read_text(encoding="utf-8"))["update"]
    assert update["mode"] == "resume"
    assert update["checkpoint"] == "results_2023-06-15"
    assert update["dates_added"] == resumed.dates_added


def test_up_to_date_writes_nothing(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    _update(cfg, tmp_path / "a", TypeRun.ALL)
    before = sorted(p.name for p in (tmp_path / "a" / "cfg").iterdir())
    again = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    assert again.mode is UpdateMode.UP_TO_DATE
    assert again.run_dir is None
    assert sorted(p.name for p in (tmp_path / "a" / "cfg").iterdir()) == before


def test_config_change_forces_full(rw_xlsx: Path, tmp_path: Path) -> None:
    _update(_config(tmp_path, rw_xlsx), tmp_path / "a", TypeRun.ALL, until=T0)
    out = _update(_config(tmp_path, rw_xlsx, penalty=30.0), tmp_path / "a", TypeRun.NEW_DATA)
    assert out.mode is UpdateMode.FULL
    assert "jm_jump_penalty" in out.reason


def test_revised_history_falls_back_to_full(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    eq.iloc[300, 1] *= 1.05  # vendor revises one old Brazil equity print
    build_xlsx(rw_xlsx, eq, fi)
    out = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    assert out.mode is UpdateMode.FULL
    assert "history revised" in out.reason
    assert out.run_dir is not None
    update = json.loads((out.run_dir / "snapshot.json").read_text(encoding="utf-8"))["update"]
    assert update["mode"] == "full"


@pytest.mark.slow
def test_hmm_resume_equals_full_rerun(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx, hmm=True, jm=False)
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    assert resumed.mode is UpdateMode.RESUME
    assert resumed.run_dir is not None and full.run_dir is not None
    _assert_same_csvs(resumed.run_dir, full.run_dir)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_historic_update.py -v -m "not slow"`
Expected: FAIL, `ImportError: cannot import name 'run_update' from 'roro.historic'`

- [ ] **Step 3: Append the orchestrator to `roro/historic.py`**

Add these imports to the top of the module (merge with the existing ones):

```python
import time
from collections.abc import Callable

from roro.config import EngineConfig, load_config
from roro.config import to_dict as config_to_dict
from roro.engine import run as engine_run
from roro.fred_client import FredClient
from roro.io import cut_prices, load_prices, read_resume_state
from roro.resume import HistoryRevisedError
from roro.types import ResumeState
```

and append:

```python
def run_update(
    config_path: Path,
    *,
    type_run: TypeRun,
    data_until: pd.Timestamp | None,
    build_report: bool,
    fred_client: FredClient,
    historic_root: Path = DEFAULT_HISTORIC_ROOT,
    echo: Callable[[str], None] = print,
) -> UpdateOutcome:
    """Run one config into <historic_root>/<config-stem>/results_<last-data-date>/.

    RESUME reuses the newest checkpoint; a revised history falls back to FULL.
    UP_TO_DATE writes nothing. All console output is ASCII.
    """
    started = time.perf_counter()
    name = config_path.stem
    historic_dir = historic_root / name
    cfg = load_config(config_path, overrides={"output_dir": historic_dir})
    dates = _data_dates(cfg.data_path, data_until)
    data_last = pd.Timestamp(dates.max())
    checkpoint = find_checkpoint(historic_dir)
    plan = plan_update(
        type_run=type_run,
        checkpoint=checkpoint,
        config_now=_json_config(cfg),
        data_last=data_last,
        data_until=data_until,
    )
    echo(f"[{plan.mode.value.upper()}] {name}: {plan.reason}")
    if plan.mode is UpdateMode.UP_TO_DATE:
        return UpdateOutcome(plan.mode, plan.reason, None, 0, time.perf_counter() - started)

    mode, reason = plan.mode, plan.reason
    resume = (
        read_resume_state(checkpoint.run_dir, checkpoint.last_date)
        if plan.mode is UpdateMode.RESUME and checkpoint is not None
        else None
    )
    try:
        run_dir = _run_engine(cfg, fred_client, data_last, resume)
    except HistoryRevisedError as exc:
        mode = UpdateMode.FULL
        reason = f"history revised in {cfg.data_path.name} ({exc}) -> full rerun"
        resume = None
        echo(f"[{mode.value.upper()}] {name}: {reason}")
        run_dir = _run_engine(cfg, fred_client, data_last, None)

    since = checkpoint.last_date if resume is not None and checkpoint is not None else None
    new_dates = dates[dates > since] if since is not None else dates
    _stamp_update(
        run_dir,
        {
            "mode": mode.value,
            "reason": reason,
            "checkpoint": checkpoint.run_dir.name if since is not None and checkpoint else None,
            "dates_added": len(new_dates),
            "first_new_date": f"{new_dates.min():%Y-%m-%d}" if len(new_dates) else None,
        },
    )
    if build_report:
        # Lazy import: the report stack pulls plotly and is only needed here.
        from roro.report import build_report as build_report_html  # noqa: PLC0415

        echo("building report.html ...")
        build_report_html(run_dir, cfg.data_path, run_dir / "report.html")
    elapsed = time.perf_counter() - started
    echo(f"[ok] {run_dir} ({len(new_dates)} new dates, {elapsed:.0f}s)")
    return UpdateOutcome(mode, reason, run_dir, len(new_dates), elapsed)


def _data_dates(data_path: Path, data_until: pd.Timestamp | None) -> pd.DatetimeIndex:
    prices = load_prices(data_path)
    if data_until is not None:
        prices = cut_prices(prices, data_until)
    dates = pd.DatetimeIndex(prices.equity_lc.index)
    if dates.empty:
        raise ValueError(f"no data in {data_path} on or before {data_until}")
    return dates


def _json_config(cfg: EngineConfig) -> dict[str, Any]:
    """Config as snapshot.json stores it (JSON round trip), for like-for-like comparison."""
    loaded: dict[str, Any] = json.loads(json.dumps(config_to_dict(cfg), default=str))
    return loaded


def _run_engine(
    cfg: EngineConfig,
    fred_client: FredClient,
    data_last: pd.Timestamp,
    resume: ResumeState | None,
) -> Path:
    """One engine run into cfg.output_dir/results_<data_last>/ (atomic tmp + rename)."""
    stamp = f"{data_last:%Y-%m-%d}"
    engine_run(
        cfg,
        fred_client=fred_client,
        run_date=f"results_{stamp}",
        as_of_data_date=stamp,
        force=True,  # a same-named folder without snapshot.json is not a valid checkpoint
        data_until=stamp,
        resume=resume,
    )
    return cfg.output_dir / f"results_{stamp}"


def _stamp_update(run_dir: Path, info: dict[str, Any]) -> None:
    """Record how this folder was produced in snapshot.json["update"]."""
    path = run_dir / "snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["update"] = info
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
```

- [ ] **Step 4: Run tests (including the slow HMM test)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_historic_update.py -v`
Expected: 5 passed. If `test_resume_equals_full_rerun` fails with a byte diff, **STOP and debug** (superpowers:systematic-debugging). Do not loosen the comparison: exact equality is the spec's acceptance AI-1.

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/historic.py tests/test_historic_update.py && .venv/Scripts/python.exe -m mypy roro/historic.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/historic.py tests/test_historic_update.py
git commit -m "feat(historic): run_update orchestrator; RESUME == FULL byte-identical

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: CLI `roro update`

**Files:**
- Modify: `roro/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_cli.py`)

```python
def test_cli_update_full_then_up_to_date(
    rw_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    monkeypatch.setattr(
        cli_mod,
        "_build_fred_client",
        lambda key: MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
        ),
    )
    cfg_yaml = tmp_path / "smoke.yaml"
    cfg_yaml.write_text(
        f"data_path: {rw_xlsx.as_posix()}\n"
        "output_dir: ignored\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "percentile_window_years: 1\n"
        "min_n_per_cut: 2\n"
        "bootstrap_min_days: 10\n",
        encoding="utf-8",
    )
    root = tmp_path / "hist"
    base = ["update", "--config", str(cfg_yaml), "--no-report", "--historic-root", str(root)]
    runner = CliRunner()

    first = runner.invoke(main, [*base, "--data-until", "15-06-2023"])
    assert first.exit_code == 0, first.output
    assert "[FULL]" in first.output
    assert (root / "smoke" / "results_2023-06-15" / "snapshot.json").exists()

    second = runner.invoke(main, base)
    assert second.exit_code == 0, second.output
    assert "[RESUME]" in second.output

    third = runner.invoke(main, base)
    assert third.exit_code == 0, third.output
    assert "[UP_TO_DATE]" in third.output

    bad = runner.invoke(main, [*base, "--data-until", "2023/06/15"])
    assert bad.exit_code != 0
    assert "DD-MM-YYYY" in bad.output
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::test_cli_update_full_then_up_to_date -v`
Expected: FAIL, `No such command 'update'`

- [ ] **Step 3: Implement.** Append to `roro/cli.py` (and update the module docstring to mention `roro update`):

```python
@main.command("update")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--full", is_flag=True, help="Reprocess the full history (ignore checkpoints).")
@click.option(
    "--data-until",
    default=None,
    help="Only use data up to this date (DD-MM-YYYY or YYYY-MM-DD). Default: latest.",
)
@click.option("--no-report", is_flag=True, help="Skip building report.html.")
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
@click.option(
    "--historic-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs") / "historic",
    show_default=True,
)
def cmd_update(
    config_path: Path,
    full: bool,
    data_until: str | None,
    no_report: bool,
    fred_key: str | None,
    historic_root: Path,
) -> None:
    """Process only dates not yet in outputs/historic/<config>/ (or everything with --full)."""
    # Lazy import: historic pulls the engine; keeps `roro --help` fast.
    from roro.historic import (  # noqa: PLC0415
        TypeRun,
        friendly_error,
        parse_user_date,
        run_update,
    )

    try:
        until = parse_user_date(data_until) if data_until else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--data-until") from None
    client = _build_fred_client(fred_key or os.environ.get("FRED_API_KEY", ""))
    try:
        run_update(
            config_path,
            type_run=TypeRun.ALL if full else TypeRun.NEW_DATA,
            data_until=until,
            build_report=not no_report,
            fred_client=client,
            historic_root=historic_root,
            echo=click.echo,
        )
    except OSError as exc:
        message = friendly_error(exc)
        if message is None:
            raise
        raise click.ClickException(message) from None
```

- [ ] **Step 4: Run CLI tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check roro/cli.py tests/test_cli.py && .venv/Scripts/python.exe -m mypy roro/cli.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add roro/cli.py tests/test_cli.py
git commit -m "feat(cli): roro update (incremental historic runs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: One-click runner (`run_roro.py` + `run_roro.bat`)

**Files:**
- Create: `run_roro.py`, `run_roro.bat`, `.gitattributes`
- Test: `tests/test_run_roro.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_run_roro.py`:

```python
"""run_roro.py parameter validation and dispatch (run_update is stubbed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from roro.historic import TypeRun, UpdateMode, UpdateOutcome

REPO = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_roro", REPO / "run_roro.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.chdir(tmp_path)  # main() chdirs to the repo; monkeypatch restores cwd
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    return _load()


def test_bad_config_lists_valid(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "CONFIG", "nope")
    assert runner.main() == 2
    out = capsys.readouterr().out
    assert "not found" in out and "default" in out


def test_bad_type_run(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                      capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "TYPE_RUN", "everything")
    assert runner.main() == 2
    assert "TYPE_RUN" in capsys.readouterr().out


def test_bad_date(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "DATA_UNTIL", "2026/09/25")
    assert runner.main() == 2
    assert "DD-MM-YYYY" in capsys.readouterr().out


def test_dispatches_to_run_update(runner: ModuleType,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run_update(config_path: Path, **kwargs: Any) -> UpdateOutcome:
        seen["config_path"] = config_path
        seen.update(kwargs)
        return UpdateOutcome(UpdateMode.UP_TO_DATE, "stub", None, 0, 0.0)

    monkeypatch.setattr(runner, "run_update", fake_run_update)
    monkeypatch.setattr(runner, "FredApiClient", lambda api_key: object())
    monkeypatch.setattr(runner, "CONFIG", "default")
    monkeypatch.setattr(runner, "TYPE_RUN", "all")
    monkeypatch.setattr(runner, "DATA_UNTIL", "25-09-2026")
    assert runner.main() == 0
    assert seen["config_path"] == REPO / "configs" / "default.yaml"
    assert seen["type_run"] is TypeRun.ALL
    assert str(seen["data_until"].date()) == "2026-09-25"


def test_known_error_prints_friendly_message(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def locked(config_path: Path, **kwargs: Any) -> UpdateOutcome:
        raise PermissionError(13, "Permission denied", "data.xlsx")

    monkeypatch.setattr(runner, "run_update", locked)
    monkeypatch.setattr(runner, "FredApiClient", lambda api_key: object())
    assert runner.main() == 1
    assert "close it in Excel" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_run_roro.py -v`
Expected: FAIL, `FileNotFoundError` for `run_roro.py`

- [ ] **Step 3: Create `run_roro.py`** (repo root)

```python
"""RoRo one-click update.

How to use:
  1. Edit the PARAMETERS block below (Notepad is fine).
  2. Double-click run_roro.bat (runs this file with the project's .venv).

Results land in outputs/historic/<CONFIG>/results_<last data date>/
(CSV files + report.html). With TYPE_RUN = "new_data" only dates not processed
yet are computed; the folder still holds the complete history.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from roro.fred_client import FredApiClient
from roro.historic import (
    TypeRun,
    available_configs,
    friendly_error,
    parse_user_date,
    run_update,
)

# ============================ PARAMETERS ============================
CONFIG = "jm-eval"  # default | eval | jm-only | jm-eval | attribution-history
TYPE_RUN = "new_data"  # "new_data" = only unprocessed dates | "all" = full history
DATA_UNTIL: str | None = None  # None = latest in data.xlsx, or "25-09-2026" / "2026-09-25"
BUILD_REPORT = True  # also write report.html into the results folder
# ====================================================================

REPO = Path(__file__).resolve().parent


def main() -> int:
    """Validate the parameters, then run. Exit code 0 ok, 1 run failed, 2 bad parameters."""
    os.chdir(REPO)  # configs use repo-relative paths (data.xlsx, outputs/)
    load_dotenv(REPO / ".env")
    configs = available_configs(REPO / "configs")
    if CONFIG not in configs:
        print(f"[x] CONFIG={CONFIG!r} not found. Valid: {', '.join(configs)}")
        return 2
    try:
        type_run = TypeRun(TYPE_RUN)
    except ValueError:
        print("[x] TYPE_RUN must be 'new_data' or 'all'")
        return 2
    try:
        until = parse_user_date(DATA_UNTIL) if DATA_UNTIL else None
    except ValueError as exc:
        print(f"[x] {exc}")
        return 2
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("[x] Add FRED_API_KEY to .env (see .env.example)")
        return 2
    try:
        run_update(
            configs[CONFIG],
            type_run=type_run,
            data_until=until,
            build_report=BUILD_REPORT,
            fred_client=FredApiClient(api_key=api_key),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort handler of a double-click runner
        message = friendly_error(exc)
        if message is None:
            traceback.print_exc()
        else:
            print(f"[x] {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create `run_roro.bat`** (repo root)

```bat
@echo off
rem Double-click to run RoRo with the parameters set in run_roro.py.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [x] .venv not found. Open a terminal in this folder and run: uv sync
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_roro.py
echo.
pause
```

- [ ] **Step 5: Create `.gitattributes`** (cmd.exe needs CRLF in .bat files)

```
*.bat text eol=crlf
```

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_run_roro.py -v`
Expected: 5 passed

- [ ] **Step 7: Lint + types**

Run: `.venv/Scripts/python.exe -m ruff check run_roro.py tests/test_run_roro.py && .venv/Scripts/python.exe -m mypy run_roro.py`
Expected: no errors

- [ ] **Step 8: Manual smoke of the .bat parameter path** (no real run: point it at a bad config to exercise the launcher and the pause)

Temporarily set `CONFIG = "nope"` in `run_roro.py`, then run `cmd //c run_roro.bat < NUL` from Git Bash. Expected output contains `[x] CONFIG='nope' not found. Valid: attribution-history, default, eval, jm-eval, jm-only`. Revert `CONFIG` to `"jm-eval"` afterwards (`git diff run_roro.py` must be empty before committing).

- [ ] **Step 9: Commit**

```bash
git add run_roro.py run_roro.bat .gitattributes tests/test_run_roro.py
git commit -m "feat(runner): one-click run_roro.py + run_roro.bat for colleagues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Docs

**Files:**
- Modify: `README.md` (insert a new subsection after "### Daily run", before "### Regime attribution")
- Modify: `docs/context/todo.md`, `docs/context/results.md`, `docs/context/memory.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: README subsection** (insert verbatim)

````markdown
### Incremental historic runs (recommended for refreshes)

`roro update` keeps one folder per config under `outputs/historic/` and, when `data.xlsx` gains new dates, computes only what is new:

```bash
# Process only dates not yet in outputs/historic/jm-eval/
roro update --config configs/jm-eval.yaml

# Rebuild the full history (e.g. after changing a parameter)
roro update --config configs/jm-eval.yaml --full

# Process data only up to a date; skip the HTML report
roro update --config configs/jm-eval.yaml --data-until 25-09-2026 --no-report
```

Layout: `outputs/historic/<config-name>/results_<last-data-date>/` holds the **complete** history (all CSVs + `snapshot.json` + `report.html`). Older folders are kept; delete or archive them by hand.

How it decides: `--full` → full run; no previous folder → full run; config changed since the last folder → full run; no new dates → nothing written; otherwise **resume**. Resume recomputes the fast stages on the full data (about 1 minute) and re-fits only the last open HMM/JM refit block instead of the whole 2008→today walk-forward. The result is byte-identical to a full rerun. If old prices in `data.xlsx` were revised, the run detects it and falls back to a full run automatically. `snapshot.json["update"]` records which path ran and why.

**One click (for colleagues):** edit the four parameters at the top of `run_roro.py` (`CONFIG`, `TYPE_RUN = "new_data" | "all"`, `DATA_UNTIL`, `BUILD_REPORT`), then double-click `run_roro.bat`. It needs the project `.venv` (`uv sync`) and `FRED_API_KEY` in `.env`.
````

- [ ] **Step 2: Context files** (one line each, list format)

`docs/context/memory.md` append:
```
- decision: incremental runs = checkpoint resume (spec docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md). Only HMM/JM walk-forwards resume (closed refit blocks copied from the newest outputs/historic/<config>/results_<date>/, open block re-fit); every other stage recomputes on full data. RESUME output must be byte-identical to FULL; revised beta history (|diff|>1e-12) forces FULL.
- decision: historic layout outputs/historic/<config-stem>/results_<last-data-date>/; CSV is the checkpoint (read back with float_precision="round_trip"), no Excel output; folders never auto-deleted; config diff (ignoring output_dir/data_path/fred_api_key) forces FULL.
```

`docs/context/todo.md`: tick the I1-I12 boxes added at plan time (see below) as tasks complete.

`docs/context/results.md` and `docs/context/sesion-log.md`: add the outcome lines in Task 12 once numbers exist.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/context/memory.md docs/context/todo.md
git commit -m "docs: incremental historic runs usage + decisions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Full verification + real-data bench

- [ ] **Step 1: Full non-slow suite + static checks**

Run: `.venv/Scripts/python.exe -m pytest -m "not slow" -q`
Expected: all pass (previous count 278 + the new tests)

Run: `.venv/Scripts/python.exe -m pytest -m slow tests/test_historic_update.py tests/test_regime_hmm.py -q`
Expected: all pass

Run: `.venv/Scripts/python.exe -m ruff check roro tests run_roro.py && .venv/Scripts/python.exe -m mypy roro run_roro.py`
Expected: no errors

- [ ] **Step 2: Real-data bench (long: roughly 2 full runs + 1 resume. Run in the background, needs `FRED_API_KEY` in `.env`)**

Find the last date in `data.xlsx` and the business day 20 trading days earlier:

```bash
.venv/Scripts/python.exe -c "from roro.io import load_prices; ix = load_prices('data.xlsx').equity_lc.index; print(ix[-1].date(), ix[-21].date())"
```

With `LAST` and `T0` from that output:

```bash
.venv/Scripts/roro.exe update --config configs/jm-eval.yaml --full --data-until <T0> --no-report
.venv/Scripts/roro.exe update --config configs/jm-eval.yaml --no-report
.venv/Scripts/roro.exe update --config configs/jm-eval.yaml --full --no-report --historic-root outputs/historic_check
```

Each command's last line prints `[ok] <dir> (<n> new dates, <seconds>s)`. Record all three timings.

Byte-compare RESUME vs FULL on real data:

```bash
.venv/Scripts/python.exe -c "import pathlib,sys; a=pathlib.Path('outputs/historic/jm-eval/results_<LAST>'); b=pathlib.Path('outputs/historic_check/jm-eval/results_<LAST>'); bad=[p.name for p in sorted(a.glob('*.csv')) if p.read_bytes()!=(b/p.name).read_bytes()]; print('[ok] identical' if not bad else f'[x] differ: {bad}'); sys.exit(1 if bad else 0)"
```

Expected: `[ok] identical`. On `[x]`, STOP and debug before claiming done.

- [ ] **Step 3: Record results**

`docs/context/results.md` append (fill the measured numbers):
```
- 2026-09-25 Incremental historic runs (feat/incremental-historic): roro/resume.py + walk_forward(prior=) HMM/JM + engine(data_until, resume) + roro/historic.py (plan/run_update) + `roro update` + run_roro.py/.bat. RESUME == FULL byte-identical (synthetic JM/CJM/rolling/HMM + real jm-eval). Real data jm-eval: FULL to T0 <s1>s, RESUME (20 new days) <s2>s, FULL to LAST <s3>s -> <s3/s2>x speedup. Non-slow suite <N> green, mypy+ruff clean.
```

`docs/context/sesion-log.md` append:
```
- 2026-09-25: incremental historic runs built (spec -> plan -> TDD): checkpoint resume of HMM/JM, outputs/historic/<config>/results_<date>/, roro update, one-click run_roro.bat; real-data RESUME byte-identical to FULL, <speedup>x faster.
```

Tick I12 in `docs/context/todo.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/context/results.md docs/context/sesion-log.md docs/context/todo.md
git commit -m "docs(context): incremental historic runs verified on real data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Hand off** via superpowers:finishing-a-development-branch (merge/PR decision is the user's).
