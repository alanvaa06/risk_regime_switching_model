"""Cadence-invariance bench: monthly refit should match daily refit labels closely.

Proves that refit_interval_days=21 (monthly) produces nearly identical regime
labels as refit_interval_days=1 (daily) on well-separated 3-regime data, so
monthly is a safe, cheap default for production.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from roro.regime_hmm import walk_forward


def _three_regime_beta(seed: int = 7, n_per: int = 167) -> pd.Series:
    """Well-separated 3-regime beta (~500 days) so daily refit stays tractable."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(m, 0.06, n_per) for m in (-0.7, 0.0, 0.7)]
    vals = np.concatenate(blocks)
    idx = pd.bdate_range("2014-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, name="beta")


@pytest.mark.slow
def test_monthly_matches_daily_label_agreement() -> None:
    beta = _three_regime_beta()
    daily = cast(
        pd.Series,
        walk_forward(beta, refit_interval_days=1, min_history_days=120, switching_variance=True)[
            "label"
        ],
    )
    monthly = cast(
        pd.Series,
        walk_forward(beta, refit_interval_days=21, min_history_days=120, switching_variance=True)[
            "label"
        ],
    )
    mask = (daily != "Unknown") & (monthly != "Unknown")
    agreement = float((daily[mask] == monthly[mask]).mean())
    print(f"agreement={agreement:.4f}")
    assert agreement >= 0.98, f"cadence label agreement {agreement:.3f} < 0.98"
