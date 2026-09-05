"""Tests for exact per-asset attribution of the cross-sectional slope."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from roro.attribution import (
    _EMPTY_CONCENTRATION,
    DELTA_COLUMNS,
    PC1_COLUMNS,
    ConcentrationRow,
    attribute_delta,
    attribute_panel,
    concentration,
    contributions,
    find_anchor,
    pc1_loadings,
    rank_against_prior,
)
from roro.regression import DailyPanel, _wls_slope
from roro.segments import ASSET_EQ, ASSET_FI, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_DATE = pd.Timestamp("2024-01-01")
# Comfortably above the production degeneracy floor so the exactness properties below
# cover well-conditioned panels only; the degenerate floor is covered by the explicit
# near-duplicate test.
_MIN_PTP: float = 1e-2


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
        slope = _wls_slope(vols, rets, wvec)
        assert abs(pc.c.sum() - slope) <= 1e-10 * max(1.0, abs(slope))
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


def test_concentration_bounds_and_top1() -> None:
    vols = np.array([0.05, 0.10, 0.30, 0.90])
    rets = np.array([0.00, 0.01, -0.02, 0.40])
    p = _panel(vols, rets, np.ones(4))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    row = concentration(p, pc, weighting="eq", min_n=3)
    assert isinstance(row, ConcentrationRow)
    assert row.top1_series == "C03__Eq"
    assert 0.25 <= row.hhi <= 1.0
    assert row.top1_share <= row.top5_share <= 1.0 + 1e-12
    # leave-one-out slope drops the dominant asset
    assert np.isfinite(row.beta_ex_top1)
    assert abs(row.beta_ex_top1) < abs(pc.beta)


def test_concentration_beta_ex_top1_nan_when_remaining_below_min_n() -> None:
    p = _panel(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.1, 0.5]), np.ones(3))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    row = concentration(p, pc, weighting="eq", min_n=3)
    assert np.isnan(row.beta_ex_top1)


def test_rank_against_prior_mirrors_classifier() -> None:
    window = np.array([0.1, 0.3, 0.2, np.nan, 0.5])  # last value = today
    # classifier: (count(window <= today) - 1) / (len - 1) = (4 - 1) / 4 = 0.75
    assert abs(rank_against_prior(window, 0.5) - 0.75) < 1e-12
    assert abs(rank_against_prior(window, 0.0) - 0.0) < 1e-12
    assert np.isnan(rank_against_prior(np.array([0.5]), 0.5))


def test_concentration_zero_total_returns_sentinel() -> None:
    p = _panel(np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4), np.ones(4))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    assert concentration(p, pc, weighting="eq", min_n=3) == _EMPTY_CONCENTRATION


def test_concentration_tie_break_is_lowest_index() -> None:
    # symmetric vols around xbar with equal |ret| -> |c| ties between C00 and C03.
    # Uses integer vols (exactly representable in binary) so the tie is bit-exact:
    # with 0.1/0.2/0.3/0.4 the leverage h0/h3 are not exact negatives of each other
    # (decimal fractions aren't exact in binary), so abs(c[0]) and abs(c[3]) differ
    # by ~5e-17 and the "tie" silently resolves by magnitude instead of index.
    vols = np.array([1.0, 2.0, 3.0, 4.0])
    rets = np.array([-0.1, 0.0, 0.0, 0.1])
    p = _panel(vols, rets, np.ones(4))
    pc = contributions(p, weighting="eq", min_n=3)
    assert pc is not None
    assert abs(abs(pc.c[0]) - abs(pc.c[3])) < 1e-15
    assert concentration(p, pc, weighting="eq", min_n=3).top1_series == "C00__Eq"


def _returns_window(seed: int = 0, n_obs: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_obs)
    data = {
        "A__Eq": common + rng.normal(0, 0.002, n_obs),
        "B__Eq": common + rng.normal(0, 0.002, n_obs),
        "C__FI": -0.2 * common + rng.normal(0, 0.004, n_obs),
        "D__FI": rng.normal(0, 0.006, n_obs),  # independent -> decoupled
    }
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n_obs))


def test_pc1_loadings_identities() -> None:
    out = pc1_loadings(_returns_window())
    assert list(out.columns) == list(PC1_COLUMNS)
    assert abs(out["pc1_load_sq"].sum() - 1.0) < 1e-10
    assert abs(out["var_share"].sum() - 1.0) < 1e-10
    assert abs(out["decoupling"].sum()) < 1e-10
    assert (out["row_mean_corr"].abs() <= 1.0).all()
    assert list(out["series"]) == sorted(out["series"])


def test_pc1_loadings_sign_flip_invariant_and_decoupled_asset() -> None:
    win = _returns_window()
    out = pc1_loadings(win)
    flipped = pc1_loadings(-win)  # eigenvectors may flip sign; squares must not
    assert np.allclose(out["pc1_load_sq"], flipped["pc1_load_sq"])
    by = out.set_index("series")["decoupling"]
    assert float(by.loc["D__FI"]) > float(by.loc["A__Eq"])


def test_pc1_loadings_empty_on_short_or_nan_window() -> None:
    win = _returns_window(n_obs=2)
    assert pc1_loadings(win).empty
    win = _returns_window()
    win.iloc[5, 0] = np.nan
    assert pc1_loadings(win).empty


def test_pc1_loadings_zero_variance_asset_keeps_loadings() -> None:
    win = _returns_window()
    win["E__FI"] = 0.0  # flat price: zero variance, corr undefined for this row
    out = pc1_loadings(win).set_index("series")
    assert len(out) == 5
    assert abs(out["pc1_load_sq"].sum() - 1.0) < 1e-10
    assert abs(out["var_share"].sum() - 1.0) < 1e-10
    assert float(cast(Any, out.at["E__FI", "pc1_load_sq"])) < 1e-12
    assert float(cast(Any, out.at["E__FI", "var_share"])) == 0.0
    assert np.isnan(float(cast(Any, out.at["E__FI", "row_mean_corr"])))
    assert np.isfinite(float(cast(Any, out.at["A__Eq", "row_mean_corr"])))


def _labels() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=12)
    vals = ["Unknown", "Unknown", "Risk-on", "Risk-on", "Risk-on", "Transitional",
            "Transitional", "Risk-off", "Risk-off", "Risk-off", "Risk-off", "Risk-off"]
    return pd.Series(vals, index=idx)


def test_find_anchor_is_day_before_last_transition() -> None:
    lab = _labels()
    t = lab.index[-1]
    a = find_anchor(lab, t, max_lookback_days=1260)
    assert a == lab.index[6]  # transition to Risk-off on idx[7]; anchor = idx[6]
    assert a < t


def test_find_anchor_transition_today_gives_yesterday() -> None:
    lab = _labels()
    t = lab.index[7]
    assert find_anchor(lab, t, max_lookback_days=1260) == lab.index[6]


def test_find_anchor_none_without_transition_or_beyond_lookback() -> None:
    lab = _labels()
    # Unknown->Risk-on ignored
    assert find_anchor(lab, lab.index[4], max_lookback_days=1260) is None
    assert find_anchor(lab, lab.index[-1], max_lookback_days=2) is None


def test_find_anchor_ignores_future_rows() -> None:
    lab = _labels()
    t = lab.index[8]
    extended = pd.concat([lab, pd.Series(["Risk-on"], index=[lab.index[-1] + pd.offsets.BDay()])])
    assert find_anchor(lab, t, max_lookback_days=1260) == find_anchor(
        extended, t, max_lookback_days=1260
    )
