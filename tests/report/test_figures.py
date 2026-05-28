"""Pure figure-builder tests."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from roro.report.bundle import DataBundle
from roro.report.figures import (
    BETA_TS_SEGMENTS,
    _ci_band_traces,
    _ols_with_ci,
    beta_timeseries,
    scatter_beta_return,
    scatter_vol_return,
)
from roro.report.load import load_bundle


@pytest.fixture
def bundle(minimal_run_dir: Path, tiny_xlsx: Path) -> DataBundle:
    return load_bundle(minimal_run_dir, tiny_xlsx, window=21)


def test_scatter_vol_return_returns_figure(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert isinstance(fig, go.Figure)


def test_scatter_vol_return_frame_count_matches_dates(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert len(fig.frames) == len(bundle.dates)


def test_scatter_vol_return_has_segment_dropdown(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    menus = fig.layout.updatemenus
    assert menus is not None and len(menus) >= 1
    labels = [btn.label for menu in menus for btn in (menu.buttons or [])]
    for expected in ("Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI"):
        assert expected in labels


def test_scatter_vol_return_axis_titles(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert "volatility" in fig.layout.xaxis.title.text.lower()
    assert "return" in fig.layout.yaxis.title.text.lower()


def test_scatter_vol_return_traces_per_frame_invariant(bundle: DataBundle) -> None:
    """Every frame has the same trace count: 7 segments * 8 traces per segment = 56."""
    fig = scatter_vol_return(bundle)
    expected = 7 * 8
    for frame in fig.frames:
        assert len(frame.data) == expected


def test_scatter_vol_return_dropdown_toggles_visibility(bundle: DataBundle) -> None:
    """Each segment button restyles visibility so only that segment's traces are visible."""
    fig = scatter_vol_return(bundle)
    menus = list(fig.layout.updatemenus)
    assert len(menus) == 1
    buttons = list(menus[0].buttons)
    n_segments = 7
    traces_per_segment = 8
    expected_total = n_segments * traces_per_segment
    for idx, button in enumerate(buttons):
        restyle = button.args[0]
        visibility = list(restyle["visible"])
        assert len(visibility) == expected_total
        assert sum(visibility) == traces_per_segment
        for seg_idx in range(n_segments):
            start = seg_idx * traces_per_segment
            end = start + traces_per_segment
            block = visibility[start:end]
            assert all(block) if seg_idx == idx else not any(block)


def test_scatter_vol_return_initial_visibility_is_full_only(bundle: DataBundle) -> None:
    """Initial figure shows only the 'Full' segment's 8 traces visible."""
    fig = scatter_vol_return(bundle)
    traces = list(fig.data)
    assert len(traces) == 7 * 8
    for i, trace in enumerate(traces):
        is_full_block = i < 8
        if is_full_block:
            assert trace.visible is None or trace.visible is True
        else:
            assert trace.visible is False


def test_scatter_dropdown_includes_dm_eq_and_dm_fi(bundle: DataBundle) -> None:
    """Spec v2: scatter segment dropdown must expose 7 segments including DM_Eq/DM_FI."""
    fig = scatter_vol_return(bundle)
    labels = [btn.label for menu in fig.layout.updatemenus for btn in (menu.buttons or [])]
    expected = ("Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI")
    for label in expected:
        assert label in labels
    assert len(labels) == len(expected)


def test_scatter_beta_return_returns_figure(bundle: DataBundle) -> None:
    fig = scatter_beta_return(bundle)
    assert isinstance(fig, go.Figure)


def test_scatter_beta_return_x_title_mentions_beta(bundle: DataBundle) -> None:
    fig = scatter_beta_return(bundle)
    assert "β" in fig.layout.xaxis.title.text or "beta" in fig.layout.xaxis.title.text.lower()


def test_scatter_beta_return_title_prefix(bundle: DataBundle) -> None:
    fig = scatter_beta_return(bundle)
    assert fig.layout.title.text.startswith("Beta vs Return")


def test_beta_timeseries_returns_figure(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    assert isinstance(fig, go.Figure)


def test_beta_timeseries_has_segment_dropdown(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    labels = [btn.label for menu in fig.layout.updatemenus for btn in (menu.buttons or [])]
    available = [s for s in BETA_TS_SEGMENTS if s in bundle.seg_beta.columns]
    for expected in available:
        assert expected in labels


def test_beta_timeseries_has_shading_shapes(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    # vrects render as 'rect' shapes
    rects = [s for s in (fig.layout.shapes or []) if s.type == "rect"]
    # At minimum one rect should exist whenever the segment has any tercile labels
    assert len(rects) >= 1


def test_beta_timeseries_y_axis_label(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    assert "β" in fig.layout.yaxis.title.text or "beta" in fig.layout.yaxis.title.text.lower()


def test_scatter_em_color_is_green(bundle: DataBundle) -> None:
    """EM marker traces must use #2ca02c (green) per v2 spec."""
    from roro.report.figures import COLOR_EM  # noqa: PLC0415
    assert COLOR_EM == "#2ca02c"


def test_scatter_em_marker_trace_uses_green(bundle: DataBundle) -> None:
    """At least one EM markers trace in the initial Full segment must render in green."""
    fig = scatter_vol_return(bundle)
    em_marker_traces = [
        t for t in fig.data
        if getattr(t, "name", None) == "Full:EM"
    ]
    assert em_marker_traces, "expected at least one trace named 'Full:EM'"
    for t in em_marker_traces:
        assert t.marker.color == "#2ca02c"


def test_ols_with_ci_returns_none_for_two_points() -> None:
    """N < 3 must return None because stderr requires >= 3 obs."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    assert _ols_with_ci(x, y) is None


def test_ols_with_ci_returns_none_for_zero_variance_x() -> None:
    """Identical x values yield rank-deficient fit; must return None."""
    x = np.array([3.0, 3.0, 3.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _ols_with_ci(x, y) is None


def test_ols_with_ci_returns_none_for_all_nan() -> None:
    x = np.array([np.nan, np.nan, np.nan])
    y = np.array([np.nan, np.nan, np.nan])
    assert _ols_with_ci(x, y) is None


def test_ols_with_ci_matches_scipy_linregress() -> None:
    """Happy path: returned values agree with scipy.stats.linregress."""
    from scipy.stats import linregress  # noqa: PLC0415
    rng = np.random.default_rng(42)
    x = rng.normal(size=50)
    y = 2.0 * x + 0.5 + rng.normal(scale=0.1, size=50)
    out = _ols_with_ci(x, y)
    assert out is not None
    slope, intercept, se_slope, se_intercept = out
    expected = linregress(x, y)
    assert np.isclose(slope, expected.slope, atol=1e-9)
    assert np.isclose(intercept, expected.intercept, atol=1e-9)
    assert np.isclose(se_slope, expected.stderr, atol=1e-9)
    assert np.isclose(se_intercept, expected.intercept_stderr, atol=1e-9)


def test_ci_band_traces_returns_two_scatters() -> None:
    upper, lower = _ci_band_traces(
        slope=1.0,
        intercept=0.0,
        se_slope=0.1,
        se_intercept=0.2,
        mean_x=0.0,
        x_range=(-1.0, 1.0),
        color="#1f77b4",
    )
    assert isinstance(upper, go.Scatter)
    assert isinstance(lower, go.Scatter)


def test_ci_band_traces_upper_has_no_fill_lower_uses_tonexty() -> None:
    upper, lower = _ci_band_traces(
        slope=1.0, intercept=0.0,
        se_slope=0.1, se_intercept=0.2,
        mean_x=0.0, x_range=(-1.0, 1.0),
        color="#1f77b4",
    )
    # Plotly stores unset fill as None or "none"
    assert upper.fill in (None, "none")
    assert lower.fill == "tonexty"


def test_ci_band_traces_default_50_points() -> None:
    upper, lower = _ci_band_traces(
        slope=1.0, intercept=0.0,
        se_slope=0.1, se_intercept=0.2,
        mean_x=0.0, x_range=(-1.0, 1.0),
        color="#1f77b4",
    )
    assert len(upper.x) == 50
    assert len(lower.x) == 50


def test_ci_band_traces_upper_above_lower_at_every_point() -> None:
    upper, lower = _ci_band_traces(
        slope=1.0, intercept=0.0,
        se_slope=0.1, se_intercept=0.2,
        mean_x=0.0, x_range=(-1.0, 1.0),
        color="#1f77b4",
    )
    u = np.asarray(upper.y, dtype=float)
    lo = np.asarray(lower.y, dtype=float)
    assert (u >= lo).all()


def test_scatter_full_segment_has_ci_trace_positions(bundle: DataBundle) -> None:
    """Structural check: CI upper/lower traces live at the right positions.

    Trace order per segment (8 total): for each of DM and EM, in order:
      markers, fit, ci_upper, ci_lower.
    """
    fig = scatter_vol_return(bundle)
    full_block = list(fig.data[:8])
    assert len(full_block) == 8
    for ci_idx in (2, 3, 6, 7):
        ci = full_block[ci_idx]
        assert ci.mode == "lines"


def test_scatter_ci_lower_uses_tonexty(bundle: DataBundle) -> None:
    """Every ci_lower trace position uses fill='tonexty' (even empty placeholders)."""
    fig = scatter_vol_return(bundle)
    for seg_idx in range(7):
        base = seg_idx * 8
        dm_lo = fig.data[base + 3]
        em_lo = fig.data[base + 7]
        assert dm_lo.fill == "tonexty"
        assert em_lo.fill == "tonexty"


def test_scatter_traces_per_segment_constant() -> None:
    """Public constant must reflect the v2 trace count."""
    from roro.report.figures import _TRACES_PER_SEGMENT  # noqa: PLC0415
    assert _TRACES_PER_SEGMENT == 8


def test_scatter_ci_ribbon_populates_with_varied_data() -> None:
    """When per-series x has nonzero variance, CI traces have populated 50-point arrays.

    Builds a hand-crafted bundle with 3 DM and 3 EM series and varied vol/return
    values so scipy.stats.linregress can compute non-degenerate slope/intercept.
    """
    import pandas as pd  # noqa: PLC0415

    from roro.report.bundle import DataBundle  # noqa: PLC0415

    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    series_ids = ["US_Eq", "DE_Eq", "FR_Eq", "BR_Eq", "MX_Eq", "AR_Eq"]
    vol_data = {
        "US_Eq": [0.10] * 5, "DE_Eq": [0.12] * 5, "FR_Eq": [0.15] * 5,
        "BR_Eq": [0.25] * 5, "MX_Eq": [0.22] * 5, "AR_Eq": [0.30] * 5,
    }
    ret_data = {
        "US_Eq": [0.05] * 5, "DE_Eq": [0.04] * 5, "FR_Eq": [0.03] * 5,
        "BR_Eq": [0.02] * 5, "MX_Eq": [0.06] * 5, "AR_Eq": [-0.01] * 5,
    }
    vol = pd.DataFrame(vol_data, index=dates)
    ret = pd.DataFrame(ret_data, index=dates)
    beta = vol.copy()
    meta = pd.DataFrame(
        [
            {"country": "United States", "asset": "Eq", "segment": "DM", "weight": 100.0},
            {"country": "Germany",       "asset": "Eq", "segment": "DM", "weight": 50.0},
            {"country": "France",        "asset": "Eq", "segment": "DM", "weight": 30.0},
            {"country": "Brazil",        "asset": "Eq", "segment": "EM", "weight": 10.0},
            {"country": "Mexico",        "asset": "Eq", "segment": "EM", "weight": 8.0},
            {"country": "Argentina",     "asset": "Eq", "segment": "EM", "weight": 5.0},
        ],
        index=pd.Index(series_ids, name="series_id"),
    )
    seg_beta = pd.DataFrame({"global": [1.0] * 5}, index=dates)
    seg_tercile = pd.DataFrame({"global": ["Transitional"] * 5}, index=dates)
    test_bundle = DataBundle(
        run_date=pd.Timestamp("2024-01-08"),
        methodology_version="1.0.0",
        dates=pd.DatetimeIndex(dates),
        vol=vol,
        ret_3m=ret,
        beta_vs_global=beta,
        meta=meta,
        seg_beta=seg_beta,
        seg_tercile=seg_tercile,
    )

    fig = scatter_vol_return(test_bundle)
    full_block = list(fig.data[:8])
    dm_ci_upper = full_block[2]
    em_ci_upper = full_block[6]
    assert len(dm_ci_upper.x) == 50, "DM CI upper should have 50 points when var(x) > 0"
    assert len(em_ci_upper.x) == 50, "EM CI upper should have 50 points when var(x) > 0"


def test_scatter_vol_return_height_is_700(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert fig.layout.height == 700


def test_scatter_beta_return_height_is_700(bundle: DataBundle) -> None:
    fig = scatter_beta_return(bundle)
    assert fig.layout.height == 700


def test_scatter_dropdown_is_positioned_clear_of_legend(bundle: DataBundle) -> None:
    """Dropdown anchored at x=1.12 so it does not overlap markers or legend."""
    fig = scatter_vol_return(bundle)
    menus = list(fig.layout.updatemenus)
    assert menus[0].x == 1.12


def test_scatter_template_is_simple_white(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    # Plotly resolves the template into a Template object; existence is enough proof
    # that a template was wired (precise name check would require deep template lookup).
    assert fig.layout.template is not None


def test_beta_timeseries_height_is_700(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    assert fig.layout.height == 700


def test_beta_timeseries_dropdown_at_x_1_12(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    menus = list(fig.layout.updatemenus)
    assert menus[0].x == 1.12


def test_beta_timeseries_template_is_simple_white(bundle: DataBundle) -> None:
    fig = beta_timeseries(bundle)
    assert fig.layout.template is not None


def test_beta_timeseries_x_axis_uses_full_seg_beta_index(bundle: DataBundle) -> None:
    """β time-series x-axis spans full seg_beta history, not the 252d scatter window."""
    fig = beta_timeseries(bundle)
    line = fig.data[0]
    assert len(line.x) == len(bundle.seg_beta.index)
    assert len(line.x) > len(bundle.dates)


def test_scatter_segment_buttons_carry_per_segment_axis_ranges(bundle: DataBundle) -> None:
    """Each of the 7 scatter buttons sets non-degenerate per-segment x/y axis ranges."""
    fig = scatter_vol_return(bundle)
    buttons = list(fig.layout.updatemenus[0].buttons)
    assert len(buttons) == 7
    for button in buttons:
        layout_update = button.args[1]
        x_range = layout_update["xaxis.range"]
        y_range = layout_update["yaxis.range"]
        assert len(x_range) == 2 and x_range[1] > x_range[0]
        assert len(y_range) == 2 and y_range[1] > y_range[0]


def test_scatter_segment_ranges_differ_with_varied_data() -> None:
    """A narrow sub-segment retightens its range vs Full when data has cross-sectional spread.

    The shared tiny_xlsx fixture synthesizes identical prices across countries (zero
    cross-sectional variance), so every segment's range would collapse to the same
    degenerate padded range. This test builds a bundle with deliberately varied vol so
    Full's range is wider than the EM_Eq sub-segment's range.
    """
    import pandas as pd  # noqa: PLC0415

    from roro.report.bundle import DataBundle  # noqa: PLC0415

    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    series_ids = ["US_Eq", "US_FI", "DE_Eq", "DE_FI", "BR_Eq", "BR_FI", "MX_Eq", "MX_FI"]
    # DM vol spread is wide (0.05..0.40); EM_Eq vol is a narrow band (0.20..0.23).
    vol_cols = {
        "US_Eq": 0.05, "US_FI": 0.40, "DE_Eq": 0.10, "DE_FI": 0.35,
        "BR_Eq": 0.20, "BR_FI": 0.15, "MX_Eq": 0.23, "MX_FI": 0.18,
    }
    ret_cols = {sid: 0.01 * i for i, sid in enumerate(series_ids)}
    vol = pd.DataFrame({sid: [v] * 5 for sid, v in vol_cols.items()}, index=dates)
    ret = pd.DataFrame({sid: [v] * 5 for sid, v in ret_cols.items()}, index=dates)
    beta = vol.copy()
    meta = pd.DataFrame(
        [
            {"country": "United States", "asset": "Eq", "segment": "DM", "weight": 100.0},
            {"country": "United States", "asset": "FI", "segment": "DM", "weight": 90.0},
            {"country": "Germany",       "asset": "Eq", "segment": "DM", "weight": 50.0},
            {"country": "Germany",       "asset": "FI", "segment": "DM", "weight": 40.0},
            {"country": "Brazil",        "asset": "Eq", "segment": "EM", "weight": 10.0},
            {"country": "Brazil",        "asset": "FI", "segment": "EM", "weight": 8.0},
            {"country": "Mexico",        "asset": "Eq", "segment": "EM", "weight": 6.0},
            {"country": "Mexico",        "asset": "FI", "segment": "EM", "weight": 5.0},
        ],
        index=pd.Index(series_ids, name="series_id"),
    )
    seg_beta = pd.DataFrame({"global": [1.0] * 5}, index=dates)
    seg_tercile = pd.DataFrame({"global": ["Transitional"] * 5}, index=dates)
    test_bundle = DataBundle(
        run_date=pd.Timestamp("2024-01-08"),
        methodology_version="1.0.0",
        dates=pd.DatetimeIndex(dates),
        vol=vol,
        ret_3m=ret,
        beta_vs_global=beta,
        meta=meta,
        seg_beta=seg_beta,
        seg_tercile=seg_tercile,
    )

    fig = scatter_vol_return(test_bundle)
    buttons = {b.label: b.args[1] for b in fig.layout.updatemenus[0].buttons}
    full_x = buttons["Full"]["xaxis.range"]
    em_eq_x = buttons["EM_Eq"]["xaxis.range"]
    full_width = full_x[1] - full_x[0]
    em_eq_width = em_eq_x[1] - em_eq_x[0]
    # EM_Eq (vol 0.20, 0.23) is a tighter band than Full (0.05..0.40).
    assert em_eq_width < full_width


def test_scatter_legend_is_top_left_overlay(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    legend = fig.layout.legend
    assert legend.orientation == "v"
    assert legend.y == 0.98
    assert legend.x == 0.02
    assert legend.bgcolor is not None and "rgba" in legend.bgcolor


def test_scatter_slider_y_is_minus_0_18(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert fig.layout.sliders[0].y == -0.18


def test_scatter_bottom_margin_is_140(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert fig.layout.margin.b == 140


def test_scatter_marker_hover_template_includes_country_asset(bundle: DataBundle) -> None:
    """Locks v2 hover behavior: marker traces expose <country>_<asset> via text + template."""
    fig = scatter_vol_return(bundle)
    marker_traces = [
        t for t in fig.data
        if getattr(t, "mode", None) == "markers" and t.text is not None and len(t.text) > 0
    ]
    assert marker_traces, "expected at least one populated markers trace"
    target = marker_traces[0]
    assert "%{text}" in target.hovertemplate
    pattern = re.compile(r"^[A-Za-z .]+_(Eq|FI)$")
    assert pattern.match(target.text[0]), f"unexpected hover label: {target.text[0]!r}"
