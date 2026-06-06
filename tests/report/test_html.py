"""HTML assembly tests."""
from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd
import plotly.graph_objects as go

from roro.report.html import FigureSpec, assemble


def _dummy_fig() -> go.Figure:
    return go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])


def _spec(fig: go.Figure, i: int = 0) -> FigureSpec:
    return FigureSpec(figure=fig, div_id=f"fig_{i}", title=f"Section {i}")


def _specs(n: int) -> list[FigureSpec]:
    return [_spec(_dummy_fig(), i) for i in range(n)]


def test_assemble_returns_string() -> None:
    out = assemble(
        _specs(3),
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert isinstance(out, str)


def test_assemble_contains_three_plotly_divs() -> None:
    out = assemble(
        _specs(3),
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert out.count('class="plotly-graph-div"') == 3


def test_assemble_contains_run_date_and_version() -> None:
    out = assemble(
        _specs(3),
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert "2024-03-29" in out
    assert "1.0.0" in out


def test_assemble_is_valid_html5() -> None:
    out = assemble(
        _specs(3),
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )

    class _P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.errors: list[str] = []

        def error(self, message: str) -> None:
            self.errors.append(message)

    parser = _P()
    parser.feed(out)
    assert parser.errors == []
    assert out.lstrip().startswith("<!DOCTYPE html>")


def test_assemble_rejects_empty() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="at least one figure"):
        assemble([], run_date=pd.Timestamp("2024-03-29"), methodology_version="1.0.0")


def test_assemble_container_max_width_is_1280() -> None:
    out = assemble(
        _specs(3),
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert "max-width: 1280px" in out


def test_assemble_accepts_four_figures_with_hmm_toggle() -> None:
    specs = [
        _spec(_dummy_fig(), 0), _spec(_dummy_fig(), 1),
        FigureSpec(_dummy_fig(), "fig_beta_ts", "Segment β with regime bands"),
        FigureSpec(_dummy_fig(), "fig_hmm_probs", "HMM regime probabilities"),
    ]
    lookup = {"global": {"percentile": [], "hmm": [{"type": "rect", "x0": "2020-01-02",
              "x1": "2020-02-01", "y0": 0, "y1": 1, "fillcolor": "rgba(1,1,1,0.5)",
              "line": {"width": 0}, "layer": "below"}]}}
    html = assemble(specs, run_date=pd.Timestamp("2024-12-31"), methodology_version="1.0.0",
                    beta_div_id="fig_beta_ts", beta_band_lookup=lookup)
    assert "HMM regime probabilities" in html
    assert 'id="hmm-band-method"' in html
    assert "plotly_relayout" in html
    assert "fig_beta_ts" in html
    assert html.index('id="hmm-band-method"') < html.index("Segment β with regime bands")
    assert html.index("Segment β with regime bands") < html.index("HMM regime probabilities")


def test_assemble_three_figures_unchanged_without_lookup() -> None:
    html = assemble(_specs(3), run_date=pd.Timestamp("2024-12-31"), methodology_version="1.0.0")
    assert "hmm-band-method" not in html
    assert "HMM regime probabilities" not in html
