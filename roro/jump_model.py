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

import numpy as np
from numpy.typing import NDArray


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
    values: NDArray[np.float64] = np.empty((t_len, k), dtype=float)
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


def online_states(
    y: NDArray[np.float64], centroids: NDArray[np.float64], jump_penalty: float
) -> NDArray[np.intp]:
    """Causal per-index state = argmin of the forward DP value (no backward pass)."""
    values = _forward_values(_loss_matrix(y, centroids), jump_penalty)
    result: NDArray[np.intp] = values.argmin(axis=1).astype(np.intp)
    return result
