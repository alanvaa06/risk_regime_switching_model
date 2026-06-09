"""Tests for the causal JM walk-forward + per-segment classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.regime_jm import walk_forward


def _two_regime_series(n: int = 600) -> pd.Series:
    rng = np.random.default_rng(11)
    lo = rng.normal(-2.0, 0.3, n // 2)
    hi = rng.normal(2.0, 0.3, n - n // 2)
    vals = np.concatenate([lo, hi])
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(vals, index=idx)


def test_walk_forward_returns_hmm_parity_keys() -> None:
    out = walk_forward(_two_regime_series(), jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=252, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert set(out) == {"state", "label", "prob_risk_off", "prob_transitional",
                        "prob_risk_on", "confidence", "cold_start", "refit_dates"}
    assert out["label"].iloc[:252].eq("Unknown").all()         # warmup
    assert bool(out["cold_start"].iloc[:252].all())
    assert out["label"].isin({"Risk-off", "Transitional", "Risk-on", "Unknown"}).all()


def test_walk_forward_deterministic() -> None:
    s = _two_regime_series()
    kw = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["label"], b["label"])
    pd.testing.assert_series_equal(a["state"], b["state"])


def test_walk_forward_no_lookahead() -> None:
    # Label at date t is invariant to appending future rows (causal online inference).
    s = _two_regime_series(600)
    kw = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    full = walk_forward(s, **kw)["label"]
    prefix = walk_forward(s.iloc[:500], **kw)["label"]
    pd.testing.assert_series_equal(full.iloc[:500], prefix)


def test_walk_forward_drops_nan_emits_unknown() -> None:
    s = _two_regime_series(400)
    s.iloc[300] = np.nan
    out = walk_forward(s, jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=120, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert out["label"].iloc[300] == "Unknown"
    assert bool(out["cold_start"].iloc[300])
