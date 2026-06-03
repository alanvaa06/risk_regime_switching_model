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

_K_REGIMES = 3
_ORDERED_LABELS = ("Risk-off", "Transitional", "Risk-on")
_UNKNOWN = "Unknown"


@dataclass(frozen=True)
class _FitResult:
    params: np.ndarray  # type: ignore[type-arg]
    perm: np.ndarray  # type: ignore[type-arg]
    means: np.ndarray  # type: ignore[type-arg]
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

    NOTE: In statsmodels 0.14.x MarkovRegression, res.params is a plain
    np.ndarray.  The regime-mean positions are read via model.param_names so
    that the extraction is robust to any ordering variation.
    """
    # Build model BEFORE the try so the exception path can size the sentinel.
    model = _build_model(beta, switching_variance=switching_variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = model.fit(maxiter=200, disp=False)
        except Exception:  # noqa: BLE001 - degenerate EM -> caller falls back
            # FIX 2: size sentinel by the model's actual param vector length so
            # that _filtered_probs can call model.filter(fit.params) without a
            # shape mismatch.
            nan_params = np.full(len(model.param_names), np.nan)
            return _FitResult(
                params=nan_params,
                perm=np.arange(_K_REGIMES),
                means=np.full(_K_REGIMES, np.nan),
                converged=False,
            )

    raw_params: np.ndarray = np.asarray(res.params, dtype=float)  # type: ignore[type-arg]
    param_names: list[str] = list(model.param_names)

    # Extract per-regime constant (mean) using the model's own param name list
    means = np.array(
        [float(raw_params[param_names.index(f"const[{i}]")]) for i in range(_K_REGIMES)]
    )

    # FIX 1: AND in statsmodels' own convergence verdict.  A fit that exhausts
    # maxiter returns finite but meaningless params; mle_retvals['converged']
    # catches that case.  The isnan(means) term is redundant once raw_params are
    # all finite, so it is dropped.
    sm_converged = bool(res.mle_retvals.get("converged", True))
    converged = sm_converged and bool(np.all(np.isfinite(raw_params)))
    perm = np.argsort(means)  # ascending; perm[0] = lowest-mean (Risk-off)
    return _FitResult(
        params=raw_params,
        perm=perm,
        means=means,
        converged=converged,
    )


def _filtered_probs(
    beta: pd.Series, fit: _FitResult, *, switching_variance: bool
) -> np.ndarray:  # type: ignore[type-arg]
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
