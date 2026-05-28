"""Pure figure-builder tests."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from roro.report.bundle import DataBundle
from roro.report.figures import (
    BETA_TS_SEGMENTS,
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
    """Every frame has the same trace count: len(SCATTER_SEGMENTS) * _TRACES_PER_SEGMENT."""
    fig = scatter_vol_return(bundle)
    # v2: 7 segments * 4 traces per segment (CI ribbon arrives in Task 5)
    expected = 7 * 4
    for frame in fig.frames:
        assert len(frame.data) == expected


def test_scatter_vol_return_dropdown_toggles_visibility(bundle: DataBundle) -> None:
    """Each segment button restyles visibility so only that segment's traces are visible."""
    fig = scatter_vol_return(bundle)
    menus = list(fig.layout.updatemenus)
    assert len(menus) == 1
    buttons = list(menus[0].buttons)
    n_segments = 7  # v2
    expected_total = n_segments * 4  # CI traces arrive in Task 5
    for idx, button in enumerate(buttons):
        restyle = button.args[0]
        visibility = list(restyle["visible"])
        assert len(visibility) == expected_total
        assert sum(visibility) == 4
        for seg_idx in range(n_segments):
            start = seg_idx * 4
            end = start + 4
            block = visibility[start:end]
            assert all(block) if seg_idx == idx else not any(block)


def test_scatter_vol_return_initial_visibility_is_full_only(bundle: DataBundle) -> None:
    """Initial figure shows only the 'Full' segment's traces visible."""
    fig = scatter_vol_return(bundle)
    traces = list(fig.data)
    # v2: 7 segments * 4 traces = 28 total
    assert len(traces) == 7 * 4
    for i, trace in enumerate(traces):
        is_full_block = i < 4
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
