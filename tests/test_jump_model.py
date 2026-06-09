"""Tests for the vendored numpy-only statistical jump model core."""

from __future__ import annotations

import numpy as np

from roro.jump_model import (
    _forward_values,
    _loss_matrix,
    _viterbi_path,
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
