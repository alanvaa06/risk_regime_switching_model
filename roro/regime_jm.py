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
from numpy.typing import NDArray

from roro.jump_model import fit_jump_model, online_states

_ORDERED_LABELS = ("Risk-off", "Transitional", "Risk-on")
_UNKNOWN = "Unknown"


def _causal_scaler(window: NDArray[np.float64]) -> tuple[float, float]:
    """Mean/std (ddof=0) of the in-window data; std==0 -> 1.0 (avoid div-by-zero)."""
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    return mean, (std if std > 0.0 else 1.0)


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
) -> dict[str, Any]:
    """Expanding/rolling refit + per-block frozen-scaler forward-DP online inference."""
    full_index = beta.index
    clean = beta.dropna()
    n = len(clean)

    probs = np.full((n, n_states), np.nan)
    cold = np.ones(n, dtype=bool)
    refit_dates: list[pd.Timestamp] = []
    last_good: tuple[NDArray[np.float64], float, float] | None = None

    r = min_history_days
    while r < n:
        block_end = min(r + refit_interval_days, n)
        fit_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
        fit_win = clean.to_numpy(dtype=np.float64)[fit_lo:r]
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
        if fit.converged:
            last_good = (fit.centroids, mean, std)
            refit_dates.append(pd.Timestamp(clean.index[r]))
        used = (fit.centroids, mean, std) if fit.converged else last_good
        if used is not None:
            centroids, u_mean, u_std = used
            inf_lo = 0 if window == "expanding" else max(0, r - rolling_window_days)
            cvals = clean.to_numpy(dtype=np.float64)
            inf_win = (cvals[inf_lo:block_end] - u_mean) / u_std
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
