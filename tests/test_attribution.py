"""Tests for exact per-asset attribution of the cross-sectional slope."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from roro.attribution import DELTA_COLUMNS, attribute_delta, attribute_panel, contributions
from roro.regression import DailyPanel, _wls_slope
from roro.segments import ASSET_EQ, ASSET_FI, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_DATE = pd.Timestamp("2024-01-01")
# Comfortably above the production degeneracy floor so the exactness properties below
# cover well-conditioned panels only; the near-degenerate boundary has its own explicit test.
_MIN_PTP: float = 1e-4


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
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
def test_contributions_sum_to_wls_slope(vols: FloatArray, rets: FloatArray, w: FloatArray) -> None:
    assume(float(np.ptp(vols)) >= _MIN_PTP)
    p = _panel(vols, rets, w)
    for weighting, wvec in (("cap", w), ("eq", np.ones(12))):
        pc = contributions(p, weighting=weighting, min_n=3)  # type: ignore[arg-type]
        assert pc is not None
        assert abs(pc.c.sum() - _wls_slope(vols, rets, wvec)) < 1e-10
        assert abs(pc.c.sum() - pc.beta) < 1e-12


@given(vols=_finite(12, 0.01, 0.9), w=_finite(12, 0.1, 100.0))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
def test_leverage_identities(vols: FloatArray, w: FloatArray) -> None:
    assume(float(np.ptp(vols)) >= _MIN_PTP)
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


def test_contributions_none_for_near_duplicate_vols() -> None:
    rng = np.random.default_rng(7)
    vols = 0.2 + rng.normal(0.0, 1e-10, 8)
    rets = rng.normal(0.0, 0.05, 8)
    p = _panel(vols, rets, np.ones(8))
    assert contributions(p, weighting="cap", min_n=3) is None
    assert contributions(p, weighting="eq", min_n=3) is None
    # a genuinely dispersed panel of the same size is fine
    ok = _panel(np.linspace(0.05, 0.5, 8), rets, np.ones(8))
    assert contributions(ok, weighting="cap", min_n=3) is not None


def test_delta_exact_with_entries_and_exits() -> None:
    vols_a = np.array([0.05, 0.10, 0.30, 0.50, 0.20])
    rets_a = np.array([-0.01, 0.02, -0.05, 0.10, 0.03])
    vols_t = np.array([0.06, 0.12, 0.25, 0.45, 0.33])
    rets_t = np.array([0.00, 0.01, -0.08, 0.15, -0.02])
    w = np.array([5.0, 1.0, 2.0, 3.0, 1.5])
    p_a = _panel(vols_a, rets_a, w)
    p_t = _panel(vols_t, rets_t, w)
    # drop C04 from the anchor panel (entry at t) and C01 from t (exit)
    p_a = DailyPanel(
        date=_DATE, series=p_a.series[:4], returns=rets_a[:4], vols=vols_a[:4], weights=w[:4]
    )
    keep = [0, 2, 3, 4]
    p_t = DailyPanel(
        date=_DATE,
        series=tuple(p_t.series[i] for i in keep),
        returns=rets_t[keep], vols=vols_t[keep], weights=w[keep],
    )
    rows_a, beta_a = attribute_panel(p_a, weighting="cap", min_n=3)
    rows_t, beta_t = attribute_panel(p_t, weighting="cap", min_n=3)
    delta = attribute_delta(rows_t, rows_a, beta_t=beta_t, beta_a=beta_a)
    assert list(delta.columns) == list(DELTA_COLUMNS)
    assert abs(delta["delta_total"].sum() - (beta_t - beta_a)) < 1e-10
    parts = ["effect_return", "effect_position", "effect_interaction", "effect_universe"]
    assert np.allclose(delta[parts].sum(axis=1), delta["delta_total"])
    entry = delta.set_index("series").loc["C04__Eq"]
    exit_ = delta.set_index("series").loc["C01__Eq"]
    assert entry["effect_universe"] != 0.0 and entry["effect_return"] == 0.0
    assert exit_["effect_universe"] != 0.0 and exit_["effect_position"] == 0.0
    both = delta[~delta["series"].isin(["C01__Eq", "C04__Eq"])]
    assert (both["effect_universe"] == 0.0).all()


def test_delta_empty_when_either_beta_nan() -> None:
    p = _panel(np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4), np.ones(4))
    rows, beta = attribute_panel(p, weighting="eq", min_n=3)
    out = attribute_delta(rows, [], beta_t=beta, beta_a=float("nan"))
    assert out.empty and list(out.columns) == list(DELTA_COLUMNS)
