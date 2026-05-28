"""beta_vs_global kernel unit + property tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from roro.report.beta_vs_global import compute_beta_vs_global

RNG_SEED = 12345


def _make_returns(n_days: int, n_series: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"s{i}" for i in range(n_series)]
    data = rng.normal(loc=0.0, scale=0.01, size=(n_days, n_series))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_beta_of_constant_series_is_zero() -> None:
    """Constant returns (zero variance) yield 0 covariance, β=0 after window fills."""
    returns = _make_returns(200, 3)
    returns["constant"] = 0.0
    weights = pd.Series({c: 1.0 for c in returns.columns})

    beta = compute_beta_vs_global(returns, weights, window=63)

    # After window fills, constant column should be 0
    assert beta["constant"].iloc[-1] == 0.0


def test_beta_of_proxy_is_one() -> None:
    """A series equal to the global proxy has β = 1."""
    returns = _make_returns(200, 4)
    weights = pd.Series({c: 1.0 for c in returns.columns})
    proxy = returns.mean(axis=1)
    returns["proxy_clone"] = proxy
    weights["proxy_clone"] = 0.0  # exclude from proxy build so it stays identical

    beta = compute_beta_vs_global(returns, weights, window=63)

    assert np.isclose(beta["proxy_clone"].iloc[-1], 1.0, atol=1e-9)


def test_window_fill_at_correct_index() -> None:
    """First non-NaN β is at the window-th row, not earlier."""
    returns = _make_returns(80, 3)
    weights = pd.Series({c: 1.0 for c in returns.columns})

    beta = compute_beta_vs_global(returns, weights, window=63)

    # Rows 0..61 NaN; row 62 (the 63rd) first valid
    assert beta.iloc[62].notna().all()
    assert beta.iloc[61].isna().all()


def test_beta_invariant_to_proxy_scale() -> None:
    """Scaling returns multiplicatively does not change β (cov/var ratio)."""
    returns = _make_returns(200, 4)
    weights = pd.Series({c: 1.0 for c in returns.columns})

    beta_base = compute_beta_vs_global(returns, weights, window=63)
    beta_scaled = compute_beta_vs_global(returns * 5.0, weights, window=63)

    pd.testing.assert_frame_equal(beta_base, beta_scaled, atol=1e-9)


@settings(max_examples=20, deadline=None)
@given(scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False))
def test_property_scale_invariance(scale: float) -> None:
    returns = _make_returns(150, 3)
    weights = pd.Series({c: 1.0 for c in returns.columns})

    base = compute_beta_vs_global(returns, weights, window=63)
    scaled = compute_beta_vs_global(returns * scale, weights, window=63)

    pd.testing.assert_frame_equal(base, scaled, atol=1e-9)
