"""Tests for exact per-asset attribution of the cross-sectional slope."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from roro.attribution import attribute_panel, contributions
from roro.regression import DailyPanel, _wls_slope
from roro.segments import ASSET_EQ, ASSET_FI, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_DATE = pd.Timestamp("2024-01-01")


def _panel(
    vols: FloatArray, rets: FloatArray, weights: FloatArray, *, n_fi: int = 0
) -> DailyPanel:
    n = len(vols)
    series = tuple(
        SeriesId(
            country=f"C{i:02d}",
            segment="DM" if i % 2 == 0 else "EM",
            asset_class=ASSET_FI if i < n_fi else ASSET_EQ,
            mcap=float(weights[i]),
        )
        for i in range(n)
    )
    return DailyPanel(date=_DATE, series=series, returns=rets, vols=vols, weights=weights)


def _finite(n: int, lo: float, hi: float) -> st.SearchStrategy[FloatArray]:
    return hnp.arrays(
        np.float64,
        shape=n,
        elements=st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False),
    )


@given(
    vols=_finite(12, 0.01, 0.9), rets=_finite(12, -0.5, 0.5), w=_finite(12, 0.1, 100.0)
)
@settings(max_examples=50, deadline=None)
def test_contributions_sum_to_wls_slope(vols: FloatArray, rets: FloatArray, w: FloatArray) -> None:
    if np.ptp(vols) < 1e-6:
        return  # degenerate cross-section: no slope to attribute
    p = _panel(vols, rets, w)
    for weighting, wvec in (("cap", w), ("eq", np.ones(12))):
        pc = contributions(p, weighting=weighting, min_n=3)  # type: ignore[arg-type]
        assert pc is not None
        assert abs(pc.c.sum() - _wls_slope(vols, rets, wvec)) < 1e-10
        assert abs(pc.c.sum() - pc.beta) < 1e-12


@given(vols=_finite(12, 0.01, 0.9), w=_finite(12, 0.1, 100.0))
@settings(max_examples=50, deadline=None)
def test_leverage_identities(vols: FloatArray, w: FloatArray) -> None:
    if np.ptp(vols) < 1e-6:
        return
    p = _panel(vols, np.zeros(12), w)
    pc = contributions(p, weighting="cap", min_n=3)
    assert pc is not None
    assert abs(pc.h.sum()) < 1e-10
    assert abs(float(np.sum(pc.h * vols)) - 1.0) < 1e-10


def test_contributions_none_below_min_n_or_degenerate() -> None:
    p = _panel(np.array([0.1, 0.2]), np.array([0.0, 0.1]), np.array([1.0, 1.0]))
    assert contributions(p, weighting="cap", min_n=3) is None
    flat = _panel(np.full(5, 0.2), np.arange(5) / 10.0, np.ones(5))
    assert contributions(flat, weighting="cap", min_n=3) is None


def test_attribute_panel_rows_quadrants_and_order() -> None:
    vols = np.array([0.05, 0.10, 0.30, 0.50])  # xbar (eq) = 0.2375
    rets = np.array([-0.01, 0.02, -0.05, 0.10])
    p = _panel(vols, rets, np.ones(4), n_fi=1)
    rows, beta = attribute_panel(p, weighting="eq", min_n=3)
    assert [r.series for r in rows] == ["C00__FI", "C01__Eq", "C02__Eq", "C03__Eq"]
    assert [r.quadrant for r in rows] == ["LO/-", "LO/+", "HI/-", "HI/+"]
    assert rows[0].block == "DM_FI" and rows[1].block == "EM_Eq"
    assert abs(sum(r.contribution for r in rows) - beta) < 1e-12
    assert abs(sum(r.share for r in rows) - 1.0) < 1e-10
    # sign table: HI/+ and LO/- push beta up; HI/- and LO/+ push it down
    assert rows[3].contribution > 0 and rows[0].contribution > 0
    assert rows[2].contribution < 0 and rows[1].contribution < 0
    assert abs(rows[0].xbar - 0.2375) < 1e-12


def test_attribute_panel_empty_when_degenerate() -> None:
    p = _panel(np.array([0.1, 0.2]), np.array([0.0, 0.1]), np.array([1.0, 1.0]))
    rows, beta = attribute_panel(p, weighting="cap", min_n=3)
    assert rows == [] and np.isnan(beta)
