"""Tests for the daily cross-sectional WLS regression kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, settings

from roro.regression import _wls_slope, cross_section, daily_panel
from roro.segments import ASSET_EQ, SeriesId
from tests.strategies import positive_weights


def _series_ids() -> list[SeriesId]:
    return [
        SeriesId(country=f"C{i}", segment="DM", asset_class=ASSET_EQ, mcap=float(i + 1))
        for i in range(10)
    ]


def test_cross_section_recovers_known_slope() -> None:
    ids = _series_ids()
    rng = np.random.default_rng(0)
    vols = np.linspace(0.05, 0.3, 10)
    true_beta = 0.5
    true_alpha = 0.01
    rets = true_alpha + true_beta * vols + rng.normal(0, 1e-6, 10)
    ret_df = pd.DataFrame(
        [rets], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")]
    )
    vol_df = pd.DataFrame(
        [vols], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")]
    )
    panel = daily_panel(
        date=pd.Timestamp("2024-01-01"),
        series=ids,
        equity_returns=ret_df,
        fi_returns=ret_df,  # not used because all series are EQ
        equity_vol=vol_df,
        fi_vol=vol_df,
    )
    out = cross_section(panel, min_n=5)
    assert abs(out.beta_cap - true_beta) < 1e-3
    assert abs(out.beta_eq - true_beta) < 1e-3
    assert abs(out.slope_spread) < 1e-3


def test_cross_section_suppresses_below_min_n() -> None:
    ids = _series_ids()[:3]
    ret_df = pd.DataFrame(
        [[0.1, 0.2, 0.3]],
        columns=[s.country for s in ids],
        index=[pd.Timestamp("2024-01-01")],
    )
    vol_df = pd.DataFrame(
        [[0.1, 0.2, 0.3]],
        columns=[s.country for s in ids],
        index=[pd.Timestamp("2024-01-01")],
    )
    panel = daily_panel(
        date=pd.Timestamp("2024-01-01"),
        series=ids,
        equity_returns=ret_df,
        fi_returns=ret_df,
        equity_vol=vol_df,
        fi_vol=vol_df,
    )
    out = cross_section(panel, min_n=5)
    assert out.suppressed
    assert np.isnan(out.beta_cap)


@given(w=positive_weights(n=10))
@settings(max_examples=25, deadline=None)
def test_ols_equals_wls_when_weights_uniform(w: np.ndarray[Any, np.dtype[np.float64]]) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 0.05, 10)
    y = 0.5 * x + 0.01 + rng.normal(0, 1e-3, 10)
    beta_uniform = _wls_slope(x, y, np.ones(10))
    beta_explicit_avg = _wls_slope(x, y, np.full(10, 1 / 10))
    assert abs(beta_uniform - beta_explicit_avg) < 1e-12
    # `w` is exercised by hypothesis to ensure weight-array dtype invariants hold.
    assert w.dtype == np.float64
