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
from numpy.typing import NDArray


@dataclass(frozen=True)
class JumpFit:
    centroids: NDArray[np.float64]   # (k,) sorted ascending (state 0 = lowest mean)
    labels: NDArray[np.intp]         # (T,) in [0,k)
    objective: float
    converged: bool


def _loss_matrix(
    y: NDArray[np.float64], centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(T,) y, (k,) centroids -> (T,k) of 0.5*(y_t - theta_k)^2 (scaled L2 loss)."""
    diff: NDArray[np.float64] = y[:, None] - centroids[None, :]
    result: NDArray[np.float64] = 0.5 * diff * diff
    return result


def _forward_values(
    loss_mx: NDArray[np.float64], jump_penalty: float
) -> NDArray[np.float64]:
    """Forward DP value matrix (T,k). values[t] depends only on loss_mx[:t+1].

    values[t,s] = loss_mx[t,s] + min_j (values[t-1,j] + lambda*1{j!=s}).
    The per-index argmin of this matrix is the causal online filter (no lookahead).
    """
    t_len, k = loss_mx.shape
    penalty: NDArray[np.float64] = jump_penalty * (1.0 - np.eye(k))
    values: NDArray[np.float64] = np.empty((t_len, k), dtype=np.float64)
    values[0] = loss_mx[0]
    for t in range(1, t_len):
        values[t] = loss_mx[t] + (values[t - 1][:, None] + penalty).min(axis=0)
    return values


def _viterbi_path(
    loss_mx: NDArray[np.float64], jump_penalty: float
) -> NDArray[np.intp]:
    """Offline (two-sided) optimal state path via backward reconstruction.

    Uses future rows during backtracking -> FIT-ONLY (on a closed historical
    window). NEVER use for the live signal; online inference uses _forward_values
    argmin instead. numpy argmin returns the first minimal index -> deterministic.
    """
    t_len, k = loss_mx.shape
    penalty: NDArray[np.float64] = jump_penalty * (1.0 - np.eye(k))
    values = _forward_values(loss_mx, jump_penalty)
    assign: NDArray[np.intp] = np.empty(t_len, dtype=np.intp)
    assign[t_len - 1] = int(values[t_len - 1].argmin())
    for t in range(t_len - 1, 0, -1):
        assign[t - 1] = int((values[t - 1] + penalty[:, assign[t]]).argmin())
    return assign


def _kmeanspp_init(
    y: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Seeded k-means++ on a 1-D series. Pure function of (y, k, rng state)."""
    t_len = y.shape[0]
    centers = np.empty(k, dtype=np.float64)
    centers[0] = y[int(rng.integers(t_len))]
    d2: NDArray[np.float64] = (y - centers[0]) ** 2
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
    y: NDArray[np.float64],
    labels: NDArray[np.intp],
    k: int,
    prev: NDArray[np.float64],
) -> NDArray[np.float64]:
    """M-step: occupied states -> cluster mean; empty states -> sequential
    farthest-point reseed (recomputed after each, so two empties never collide)."""
    t_len = y.shape[0]
    centroids: NDArray[np.float64] = prev.astype(np.float64).copy()
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
        d2_reseed: NDArray[np.float64] = (
            ((y[:, None] - occupied[None, :]) ** 2).min(axis=1)
        )
        d2_reseed = np.where(claimed, -np.inf, d2_reseed)
        idx = int(d2_reseed.argmax())  # farthest unclaimed point, first-index tie-break
        centroids[c] = y[idx]
        claimed[idx] = True
        is_set[c] = True
    return centroids


def _objective(
    loss_mx: NDArray[np.float64], labels: NDArray[np.intp], jump_penalty: float
) -> float:
    fit_loss = float(loss_mx[np.arange(labels.shape[0]), labels].sum())
    jumps = int((labels[1:] != labels[:-1]).sum())
    return fit_loss + jump_penalty * jumps


def _fit_once(
    y: NDArray[np.float64],
    k: int,
    jump_penalty: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.intp], float]:
    """One restart of coordinate descent. Returns (centroids, labels, objective)."""
    centroids = _kmeanspp_init(y, k, rng)
    loss_mx = _loss_matrix(y, centroids)
    labels = _viterbi_path(loss_mx, jump_penalty)
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
    y: NDArray[np.float64],
    *,
    k: int = 3,
    jump_penalty: float = 50.0,
    n_init: int = 10,
    max_iter: int = 30,
    tol: float = 1e-8,
    seed: int = 0,
) -> JumpFit:
    """Fit a discrete K-state jump model with n_init seeded restarts.

    Deterministic: pure function of (y, seed, jump_penalty, k, n_init, max_iter, tol).
    Restarts seeded by SeedSequence(seed).spawn(n_init); strict-lowest-objective
    restart wins (lowest-index on ties). States canonicalized ascending by centroid.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    children = np.random.SeedSequence(seed).spawn(n_init)
    best: tuple[NDArray[np.float64], NDArray[np.intp], float] | None = None
    for child in children:
        rng = np.random.Generator(np.random.PCG64(child))
        centroids, labels, obj = _fit_once(y, k, jump_penalty, max_iter, tol, rng)
        if best is None or obj < best[2]:  # strict '<' -> first/lowest-index wins ties
            best = (centroids, labels, obj)
    assert best is not None
    centroids, labels, obj = best
    converged = bool(np.all(np.isfinite(centroids)) and np.isfinite(obj))
    perm: NDArray[np.intp] = np.argsort(centroids, kind="stable").astype(np.intp)
    inv: NDArray[np.intp] = np.argsort(perm, kind="stable").astype(np.intp)
    return JumpFit(
        centroids=centroids[perm],
        labels=inv[labels].astype(np.intp),
        objective=float(obj),
        converged=converged,
    )


def online_states(
    y: NDArray[np.float64], centroids: NDArray[np.float64], jump_penalty: float
) -> NDArray[np.intp]:
    """Causal per-index state = argmin of the forward DP value (no backward pass)."""
    values = _forward_values(_loss_matrix(y, centroids), jump_penalty)
    result: NDArray[np.intp] = values.argmin(axis=1).astype(np.intp)
    return result
