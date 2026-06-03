import numpy as np
import pandas as pd

from roro.regime_hmm import _fit_params


def _three_regime_beta(seed: int = 0) -> pd.Series:
    """Concatenate low / mid / high mean blocks with small noise."""
    rng = np.random.default_rng(seed)
    blocks = [
        rng.normal(-0.8, 0.05, 400),
        rng.normal(0.0, 0.05, 400),
        rng.normal(0.8, 0.05, 400),
    ]
    vals = np.concatenate(blocks)
    idx = pd.bdate_range("2010-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, name="beta")


def test_fit_orders_states_by_mean_ascending() -> None:
    beta = _three_regime_beta()
    fit = _fit_params(beta, switching_variance=True)
    assert fit.converged
    ordered_means = fit.means[fit.perm]
    assert ordered_means[0] < ordered_means[1] < ordered_means[2]


def test_fit_is_deterministic() -> None:
    beta = _three_regime_beta()
    a = _fit_params(beta, switching_variance=True)
    b = _fit_params(beta, switching_variance=True)
    np.testing.assert_allclose(a.params, b.params)
    np.testing.assert_array_equal(a.perm, b.perm)
