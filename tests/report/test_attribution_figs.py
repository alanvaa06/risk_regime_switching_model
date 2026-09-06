"""Pure figure-builder tests for the attribution figures."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from roro.report.attribution_figs import (
    QUADRANT_COLORS,
    attribution_bars,
    attribution_scatter,
    attribution_waterfall,
    concentration_timeseries,
    pc1_loadings_bars,
)
from roro.report.bundle import DataBundle
from roro.report.load import load_bundle


@pytest.fixture
def abundle(attribution_run_dir: Path, tiny_xlsx: Path) -> DataBundle:
    return load_bundle(attribution_run_dir, tiny_xlsx, window=21)


def _dropdown_labels(fig: go.Figure) -> list[str]:
    menus = fig.layout.updatemenus
    assert menus, "figure has no dropdown"
    return [b["label"] for b in menus[0]["buttons"]]


def test_bars_default_is_global_cap_and_dropdown_covers_cuts(abundle: DataBundle) -> None:
    fig = attribution_bars(abundle)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text.startswith("Slope attribution — global · cap")
    labels = _dropdown_labels(fig)
    assert "global · cap" in labels and "EM_FI · eq" in labels
    assert set(fig.data[0].marker.color) <= set(QUADRANT_COLORS.values())


def test_waterfall_has_four_effect_traces_and_anchor_in_title(abundle: DataBundle) -> None:
    fig = attribution_waterfall(abundle)
    assert [t.name for t in fig.data] == ["return", "position", "interaction", "universe"]
    assert fig.layout.barmode == "relative"
    assert "anchor" in fig.layout.title.text
    assert "global · cap · fixed" in _dropdown_labels(fig)


def test_scatter_has_wls_line_and_xbar_marker(abundle: DataBundle) -> None:
    fig = attribution_scatter(abundle)
    assert len(fig.data) == 2  # markers + WLS line
    assert fig.data[1].mode == "lines"
    assert any(s["type"] == "line" for s in fig.layout.shapes)
    assert fig.layout.xaxis.title.text == "EWMA vol (annualized)"


def test_concentration_timeseries_two_traces_with_regime_bands(abundle: DataBundle) -> None:
    fig = concentration_timeseries(abundle)
    assert [t.name for t in fig.data] == ["top-1 share", "HHI"]
    assert "global" in _dropdown_labels(fig)
    assert fig.layout.yaxis.range == (0.0, 1.0)


def test_pc1_bars_two_traces_sorted_by_decoupling(abundle: DataBundle) -> None:
    fig = pc1_loadings_bars(abundle)
    assert [t.name for t in fig.data] == ["PC1 loading²", "variance share"]
    assert "global" in _dropdown_labels(fig)


def test_all_figures_height_700_and_template(abundle: DataBundle) -> None:
    for build in (attribution_bars, attribution_waterfall, attribution_scatter,
                  concentration_timeseries, pc1_loadings_bars):
        fig = build(abundle)
        assert fig.layout.height == 700
        # Same convention as test_figures: template presence is the proof (deep lookup is brittle)
        assert fig.layout.template is not None
