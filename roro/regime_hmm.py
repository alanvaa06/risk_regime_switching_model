"""3-state Markov-switching HMM regime classifier on segment beta.

Parallel to roro.classify. Causal-by-construction: params re-estimated on an
expanding window (monthly), states inferred by the Hamilton filter daily.
Filtered probabilities only — never smoothed (smoothing peeks at the future).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from roro.config import EngineConfig
from roro.resume import (
    HistoryRevisedError,
    prior_refits_before,
    resume_block_start,
    seed_prior_rows,
)
from roro.types import BetaBySegment, HmmRegimeFrame, SegmentPrior

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
            if not fallback.converged:
                raise HistoryRevisedError(
                    f"refit at {refit_dates[-1].date()} no longer converges"
                )
            last_good = fallback
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

    def _series(values: np.ndarray) -> pd.Series:  # type: ignore[type-arg]
        return pd.Series(values, index=clean.index).reindex(full_index)

    return {
        "state": _series(np.where(state_idx < 0, np.nan, state_idx)),
        "label": _series(labels).fillna(_UNKNOWN),
        "prob_risk_off": _series(probs[:, 0]),
        "prob_transitional": _series(probs[:, 1]),
        "prob_risk_on": _series(probs[:, 2]),
        "confidence": _series(confidence),
        "cold_start": pd.Series(cold, index=clean.index)
        .reindex(full_index, fill_value=True)
        .astype(bool),
        "refit_dates": refit_dates,
    }


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
    state: dict[str, object] = {}
    label: dict[str, object] = {}
    p_off: dict[str, object] = {}
    p_tr: dict[str, object] = {}
    p_on: dict[str, object] = {}
    conf: dict[str, object] = {}
    cold: dict[str, object] = {}
    nseg: dict[str, object] = {}
    thin: dict[str, object] = {}
    refit_dates: dict[str, list[pd.Timestamp]] = {}

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        out = walk_forward(
            beta,
            refit_interval_days=cfg.hmm_refit_interval_days,
            min_history_days=cfg.hmm_min_history_days,
            switching_variance=cfg.hmm_switching_variance,
            prior=prior.get(cut) if prior is not None else None,
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
        refit_dates[cut] = cast(list[pd.Timestamp], out["refit_dates"])

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
