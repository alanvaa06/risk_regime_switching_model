"""Tests for the vendored numpy-only statistical jump model core."""

from __future__ import annotations

import numpy as np

from roro.jump_model import (
    _forward_values,
    _kmeanspp_init,
    _loss_matrix,
    _objective,
    _update_centroids,
    _viterbi_path,
    fit_jump_model,
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


def test_online_states_returns_intp() -> None:
    # Folded-in review nit: guard the online_states dtype contract.
    y = np.array([0.0, 1.0, 2.0])
    c = np.array([0.0, 2.0])
    assert online_states(y, c, 1.0).dtype == np.intp


def test_update_centroids_sequential_reseed_two_empty() -> None:
    # k=3 but labels only use state 1 -> states 0 and 2 are empty and must reseed
    # to DISTINCT farthest points (not collide on the same point).
    y = np.array([0.0, 0.1, 0.2, 5.0, 9.0])
    labels = np.array([1, 1, 1, 1, 1], dtype=np.intp)
    prev = np.array([0.1, 0.1, 0.1])
    out = _update_centroids(y, labels, 3, prev)
    assert np.all(np.isfinite(out))
    # the two reseeded empties must be the two farthest-from-occupied DISTINCT points
    assert len(set(np.round(out, 6))) >= 2  # not all identical -> no collision


def test_kmeanspp_init_all_coincident_points() -> None:
    y = np.full(20, 3.0)
    rng = np.random.default_rng(0)
    centers = _kmeanspp_init(y, 3, rng)
    assert centers.shape == (3,)
    assert np.all(np.isfinite(centers))
    assert np.allclose(centers, 3.0)  # all coincide -> all centers == the only value


def test_kmeanspp_init_is_seed_reproducible() -> None:
    y = np.random.default_rng(1).normal(size=100)
    a = _kmeanspp_init(y, 3, np.random.Generator(np.random.PCG64(np.random.SeedSequence(0))))
    b = _kmeanspp_init(y, 3, np.random.Generator(np.random.PCG64(np.random.SeedSequence(0))))
    assert np.array_equal(a, b)


def test_objective_fit_loss_plus_jump_penalty() -> None:
    loss_mx = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0], dtype=np.intp)  # picks 0.0, 0.0, 0.0 -> fit_loss 0
    # transitions: 0->1 (jump), 1->0 (jump) = 2 jumps
    assert _objective(loss_mx, labels, 5.0) == 0.0 + 5.0 * 2
