"""Pure figure-builder tests."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from roro.report.bundle import DataBundle
from roro.report.figures import _TRACES_PER_SEGMENT, SCATTER_SEGMENTS, scatter_vol_return
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
    for expected in ("Full", "DM", "EM", "EM_Eq", "EM_FI"):
        assert expected in labels


def test_scatter_vol_return_axis_titles(bundle: DataBundle) -> None:
    fig = scatter_vol_return(bundle)
    assert "volatility" in fig.layout.xaxis.title.text.lower()
    assert "return" in fig.layout.yaxis.title.text.lower()


def test_scatter_vol_return_traces_per_frame_invariant(bundle: DataBundle) -> None:
    """Every frame has the same trace count: len(SCATTER_SEGMENTS) * _TRACES_PER_SEGMENT."""
    fig = scatter_vol_return(bundle)
    expected = len(SCATTER_SEGMENTS) * _TRACES_PER_SEGMENT
    for frame in fig.frames:
        assert len(frame.data) == expected


def test_scatter_vol_return_dropdown_toggles_visibility(bundle: DataBundle) -> None:
    """Each segment button restyles visibility so only that segment's traces are visible."""
    fig = scatter_vol_return(bundle)
    menus = list(fig.layout.updatemenus)
    assert len(menus) == 1
    buttons = list(menus[0].buttons)
    n_segments = len(SCATTER_SEGMENTS)
    expected_total = n_segments * _TRACES_PER_SEGMENT
    for idx, button in enumerate(buttons):
        # button.args is a tuple/list: (restyle_dict, layout_dict)
        restyle = button.args[0]
        visibility = list(restyle["visible"])
        assert len(visibility) == expected_total
        assert sum(visibility) == _TRACES_PER_SEGMENT
        # Active segment's slice is True, others False
        for seg_idx in range(n_segments):
            start = seg_idx * _TRACES_PER_SEGMENT
            end = start + _TRACES_PER_SEGMENT
            block = visibility[start:end]
            assert all(block) if seg_idx == idx else not any(block)


def test_scatter_vol_return_initial_visibility_is_full_only(bundle: DataBundle) -> None:
    """Initial figure shows only the 'Full' segment's traces visible."""
    fig = scatter_vol_return(bundle)
    traces = list(fig.data)
    # Full is the first segment in SCATTER_SEGMENTS, so traces[0:_TRACES_PER_SEGMENT] are visible
    for i, trace in enumerate(traces):
        is_full_block = i < _TRACES_PER_SEGMENT
        # `visible` may be True/False or unset (None) — None counts as visible by default
        if is_full_block:
            assert trace.visible is None or trace.visible is True
        else:
            assert trace.visible is False
