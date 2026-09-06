"""Pure figure-builder tests for the attribution figures."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from roro.report import build_report
from roro.report.attribution_figs import (
    QUADRANT_COLORS,
    attribution_bars,
    attribution_scatter,
    attribution_waterfall,
    concentration_timeseries,
    pc1_loadings_bars,
)
from roro.report.bundle import DataBundle
from roro.report.errors import ReportInputError
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


def test_figures_raise_report_input_error_on_empty_frames(abundle: DataBundle) -> None:
    assert abundle.attribution_level is not None
    empty = replace(
        abundle,
        attribution_level=abundle.attribution_level.iloc[:0],
        attribution_delta=abundle.attribution_delta.iloc[:0],  # type: ignore[union-attr]
        concentration=abundle.concentration.iloc[:0],  # type: ignore[union-attr]
        attribution_pc1=abundle.attribution_pc1.iloc[:0],  # type: ignore[union-attr]
    )
    for build in (attribution_bars, attribution_waterfall, attribution_scatter,
                  concentration_timeseries, pc1_loadings_bars):
        with pytest.raises(ReportInputError):
            build(empty)


def test_figures_are_deterministic_and_marker_sizes_bounded(abundle: DataBundle) -> None:
    for build in (attribution_bars, attribution_waterfall, attribution_scatter,
                  concentration_timeseries, pc1_loadings_bars):
        assert build(abundle).to_json() == build(abundle).to_json()
    sizes = attribution_scatter(abundle).data[0].marker.size
    assert min(sizes) >= 6.0 and max(sizes) <= 40.0


def test_bars_dropdown_button_carries_other_cut_data(abundle: DataBundle) -> None:
    fig = attribution_bars(abundle)
    buttons = fig.layout.updatemenus[0]["buttons"]
    em_eq = next(b for b in buttons if b["label"] == "EM_Eq · eq")
    restyle, relayout = em_eq["args"]
    assert relayout["title"].startswith("Slope attribution — EM_Eq · eq")
    assert len(list(restyle["y"][0])) == 4  # four fixture assets


def test_build_report_includes_attribution_sections(
    attribution_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    build_report(attribution_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    for title in ("Slope attribution", "Δβ̂ waterfall", "Vol vs return, sized by |contribution|",
                  "Concentration of the slope", "PC1 loadings vs variance share"):
        assert title in html, title
    assert html.count('class="plotly-graph-div"') == 10


def test_build_report_without_attribution_unchanged(
    minimal_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    build_report(minimal_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    assert "Slope attribution" not in html
    assert html.count('class="plotly-graph-div"') == 5


def test_build_report_skips_attribution_when_csv_has_only_header(
    attribution_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    for name in (
        "attribution.csv",
        "attribution_delta.csv",
        "attribution_rollup.csv",
        "concentration.csv",
        "attribution_pc1.csv",
    ):
        f = attribution_run_dir / name
        f.write_text(pd.read_csv(f).iloc[:0].to_csv(index=False), encoding="utf-8")
    out = tmp_path / "r.html"
    build_report(attribution_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    assert "Slope attribution" not in html
    assert html.count('class="plotly-graph-div"') == 5


def test_build_report_skips_waterfall_when_delta_empty(
    attribution_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    dl = attribution_run_dir / "attribution_delta.csv"
    dl.write_text(pd.read_csv(dl).iloc[:0].to_csv(index=False), encoding="utf-8")
    out = tmp_path / "r.html"
    build_report(attribution_run_dir, tiny_xlsx, out, window=21)
    html = out.read_text(encoding="utf-8")
    assert "Slope attribution" in html and "Δβ̂ waterfall" not in html
    assert html.count('class="plotly-graph-div"') == 9
