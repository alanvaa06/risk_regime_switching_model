"""Tests for the causal JM walk-forward + per-segment classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import roro.regime_jm as _jm_mod
from roro.jump_model import JumpFit
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
    kw: dict[str, Any] = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["label"], b["label"])
    pd.testing.assert_series_equal(a["state"], b["state"])


def test_walk_forward_no_lookahead() -> None:
    # Label at date t is invariant to appending future rows (causal online inference).
    s = _two_regime_series(600)
    kw: dict[str, Any] = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
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


def test_walk_forward_rolling_no_lookahead_and_deterministic() -> None:
    # Rolling window: causal + deterministic, mirroring the expanding no-lookahead test.
    s = _two_regime_series(500)
    kw: dict[str, Any] = dict(
        jump_penalty=20.0, refit_interval_days=21, min_history_days=200,
        n_states=3, window="rolling", rolling_window_days=120,
        continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0,
    )
    full = walk_forward(s, **kw)["label"]
    prefix = walk_forward(s.iloc[:400], **kw)["label"]
    pd.testing.assert_series_equal(full.iloc[:400], prefix)  # no lookahead under rolling
    a = walk_forward(s, **kw)
    b = walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["state"], b["state"])    # deterministic


def test_walk_forward_last_good_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # When later refits do not converge, the block must reuse the last converged
    # (centroids, scaler) -- emitting real labels, not NaN/Unknown, and recording
    # only the converged refit date.
    real_fit = _jm_mod.fit_jump_model  # type: ignore[attr-defined]
    calls = {"n": 0}

    def fake_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        f = real_fit(*args, **kwargs)
        if calls["n"] == 1:
            return f  # first refit converges
        return JumpFit(centroids=f.centroids, labels=f.labels,
                       objective=f.objective, converged=False)  # later refits "fail"

    monkeypatch.setattr(_jm_mod, "fit_jump_model", fake_fit)
    out = walk_forward(_two_regime_series(400), jump_penalty=20.0,
                       refit_interval_days=40, min_history_days=120, n_states=3,
                       window="expanding", rolling_window_days=2000, continuous=False,
                       n_init=4, max_iter=30, tol=1e-8, seed=0)
    assert calls["n"] >= 2                                   # multiple refits occurred
    assert len(out["refit_dates"]) == 1                     # only the converged refit recorded
    active = out["label"].iloc[120:]
    assert (active != "Unknown").any()                      # last_good kept producing labels
    assert not active.isna().any()
