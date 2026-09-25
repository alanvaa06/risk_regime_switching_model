"""3-state statistical jump model regime classifier on segment beta.

Parallel to roro.regime_hmm. Causal-by-construction: centroids re-estimated on a
point-in-time window (expanding monthly, or rolling), states inferred by the
forward-DP online filter daily. Backward (two-sided) reconstruction is used only
to FIT on a closed historical window -- never for the live signal.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from roro.config import EngineConfig
from roro.jump_model import (
    JumpFit,
    discretize_simplex,
    fit_jump_model,
    online_soft_states,
    online_states,
)
from roro.resume import (
    HistoryRevisedError,
    prior_refits_before,
    resume_block_start,
    seed_prior_rows,
)
from roro.types import BetaBySegment, JmRegimeFrame, SegmentPrior

_ORDERED_LABELS = ("Risk-off", "Transitional", "Risk-on")
_UNKNOWN = "Unknown"


def _causal_scaler(window: NDArray[np.float64]) -> tuple[float, float]:
    """Mean/std (ddof=0) of the in-window data; std==0 -> 1.0 (avoid div-by-zero)."""
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    return mean, (std if std > 0.0 else 1.0)


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
            if not fb.converged:
                raise HistoryRevisedError(
                    f"refit at {refit_dates[-1].date()} no longer converges"
                )
            last_good = (fb.centroids, fb_mean, fb_std)
        used = (fit.centroids, mean, std) if fit.converged else last_good
        if used is not None:
            centroids, u_mean, u_std = used
            inf_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
            inf_win = (cvals[inf_lo:block_end] - u_mean) / u_std
            if continuous:
                assert grid is not None
                soft = online_soft_states(inf_win, centroids, jump_penalty, grid)
                for e in range(r, block_end):
                    probs[e, :] = np.round(soft[e - inf_lo], 10)
                    cold[e] = False
            else:
                states = online_states(inf_win, centroids, jump_penalty)
                for e in range(r, block_end):
                    s = int(states[e - inf_lo])
                    probs[e, :] = 0.0
                    probs[e, s] = 1.0
                    cold[e] = False
        r = block_end

    state_idx = np.where(np.isnan(probs).any(axis=1), -1, probs.argmax(axis=1))
    labels = [_ORDERED_LABELS[i] if i >= 0 else _UNKNOWN for i in state_idx]
    confidence = np.where(np.isnan(probs).any(axis=1), np.nan, probs.max(axis=1))

    def _series(values: NDArray[Any]) -> pd.Series[Any]:
        return pd.Series(values, index=clean.index).reindex(full_index)

    return {
        "state": _series(
            np.where(state_idx < 0, np.nan, state_idx.astype(np.float64))
        ),
        "label": pd.Series(labels, index=clean.index)
        .reindex(full_index)
        .fillna(_UNKNOWN),
        "prob_risk_off": _series(probs[:, 0]),
        "prob_transitional": _series(probs[:, 1]),
        "prob_risk_on": _series(probs[:, 2]),
        "confidence": _series(confidence),
        "cold_start": pd.Series(cold, index=clean.index)
        .reindex(full_index, fill_value=True)
        .astype(bool),
        "refit_dates": refit_dates,
    }


def classify_jm(
    bbs: BetaBySegment,
    *,
    cfg: EngineConfig,
    thin_cuts: frozenset[str],
    prior: Mapping[str, SegmentPrior] | None = None,
) -> JmRegimeFrame:
    """Per-segment JM classification mirroring classify_hmm's loop + frame shape.

    prior: per-segment checkpoint rows (see walk_forward); a missing segment runs
    in full.
    """
    state: dict[str, object] = {}
    label: dict[str, object] = {}
    p_off: dict[str, object] = {}
    p_tr: dict[str, object] = {}
    p_on: dict[str, object] = {}
    conf: dict[str, object] = {}
    nseg: dict[str, object] = {}
    thin: dict[str, object] = {}
    cold: dict[str, object] = {}
    refit_dates: dict[str, list[pd.Timestamp]] = {}
    penalty_used: dict[str, float] = {}

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        out = walk_forward(
            beta,
            jump_penalty=cfg.jm_jump_penalty,
            refit_interval_days=cfg.jm_refit_interval_days,
            min_history_days=cfg.jm_min_history_days,
            n_states=cfg.jm_n_states,
            window=cfg.jm_window,
            rolling_window_days=cfg.jm_rolling_window_days,
            continuous=cfg.jm_continuous,
            n_init=cfg.jm_n_init,
            max_iter=cfg.jm_max_iter,
            tol=cfg.jm_tol,
            seed=cfg.jm_random_seed,
            prior=prior.get(cut) if prior is not None else None,
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
        refit_dates[cut] = cast(list[pd.Timestamp], out["refit_dates"])
        penalty_used[cut] = cfg.jm_jump_penalty

    return JmRegimeFrame(
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
        jump_penalty_used=penalty_used,
    )
