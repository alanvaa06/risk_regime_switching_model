"""HMM walk-forward resume: output must equal a run without a checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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


@pytest.mark.parametrize(
    "split",
    [
        pytest.param(100, marks=pytest.mark.slow),
        pytest.param(200, marks=pytest.mark.slow),
        pytest.param(210, marks=pytest.mark.slow),
        pytest.param(330, marks=pytest.mark.slow),
    ],
)
def test_resume_equals_full(split: int) -> None:
    beta = _beta()
    last = beta.index[split - 1]
    checkpoint = walk_forward(beta.loc[:last], **_KW)
    resumed = walk_forward(beta, prior=_prior_from(checkpoint, last), **_KW)
    _assert_same(resumed, walk_forward(beta, **_KW))


def test_resume_equals_full_with_nan_betas() -> None:
    """NaN rows in the warmup, a copied block, and the recomputed part."""
    beta = _beta()
    beta.iloc[[60, 180, 305, 400]] = np.nan
    last = beta.index[418]  # split=419
    checkpoint = walk_forward(beta.loc[:last], **_KW)
    resumed = walk_forward(beta, prior=_prior_from(checkpoint, last), **_KW)
    _assert_same(resumed, walk_forward(beta, **_KW))


def test_resume_when_restated_block_start_turns_nan() -> None:
    """Checkpoint's last row opened a refit block; it is now NaN (restated).

    A full rerun never refits on that date, so its checkpoint refit date must not
    be copied.
    """
    rng = np.random.default_rng(0)
    vals = np.concatenate(
        [rng.normal(-1, 0.3, 22), rng.normal(1, 0.3, 22), rng.normal(0, 0.3, 22)]
    )
    beta = pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=66), name="beta")
    kw: dict[str, Any] = dict(
        refit_interval_days=10, min_history_days=40, switching_variance=False
    )
    last = beta.index[60]  # block start (40 + 2 * 10)
    checkpoint = walk_forward(beta.loc[:last], **kw)
    assert last in cast(list[pd.Timestamp], checkpoint["refit_dates"])
    beta.iloc[60] = np.nan
    resumed = walk_forward(beta, prior=_prior_from(checkpoint, last), **kw)
    full = walk_forward(beta, **kw)
    _assert_same(resumed, full)
    assert last not in cast(list[pd.Timestamp], resumed["refit_dates"])


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
    resumed_p_on = cast(pd.Series, resumed["prob_risk_on"])
    assert not np.isnan(resumed_p_on.iloc[330])  # fallback params were used


@pytest.mark.slow
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
