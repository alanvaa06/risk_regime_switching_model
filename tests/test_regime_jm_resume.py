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


def test_resume_when_restated_block_start_turns_nan() -> None:
    """Checkpoint's last row opened a refit block; it is now NaN (restated).

    A full rerun never refits on that date, so its checkpoint refit date must not
    be copied.
    """
    s = _series()
    kw = _kw()
    last = s.index[200]  # block start (120 + 2 * 40)
    checkpoint = walk_forward(s.loc[:last], **kw)
    assert last in checkpoint["refit_dates"]
    s.iloc[200] = np.nan
    resumed = walk_forward(s, prior=_prior_from(checkpoint, last), **kw)
    full = walk_forward(s, **kw)
    _assert_same(resumed, full)
    assert last not in resumed["refit_dates"]


@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"continuous": True},
        {"window": "rolling", "rolling_window_days": 150},
    ],
    ids=["discrete", "continuous", "rolling"],
)
def test_resume_equals_full_with_nan_betas(variant: dict[str, Any]) -> None:
    """NaN in warmup, a copied block, and the recomputed part."""
    s = _series()
    s.iloc[[50, 170, 300, 395]] = np.nan
    kw = _kw(**variant)
    last = s.index[389]
    checkpoint = walk_forward(s.loc[:last], **kw)
    resumed = walk_forward(s, prior=_prior_from(checkpoint, last), **kw)
    _assert_same(resumed, walk_forward(s, **kw))


def test_resume_rebuilds_last_good_when_open_block_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = jm_mod.fit_jump_model  # type: ignore[attr-defined]

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
    real = jm_mod.fit_jump_model  # type: ignore[attr-defined]

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
