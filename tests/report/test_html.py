"""HTML assembly tests."""
from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd
import plotly.graph_objects as go

from roro.report.html import assemble


def _dummy_fig() -> go.Figure:
    return go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])


def test_assemble_returns_string() -> None:
    out = assemble(
        [_dummy_fig(), _dummy_fig(), _dummy_fig()],
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert isinstance(out, str)


def test_assemble_contains_three_plotly_divs() -> None:
    out = assemble(
        [_dummy_fig(), _dummy_fig(), _dummy_fig()],
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert out.count('class="plotly-graph-div"') == 3


def test_assemble_contains_run_date_and_version() -> None:
    out = assemble(
        [_dummy_fig(), _dummy_fig(), _dummy_fig()],
        run_date=pd.Timestamp("2024-03-29"),
        methodology_version="1.0.0",
    )
    assert "2024-03-29" in out
    assert "1.0.0" in out


def test_assemble_is_valid_html5() -> None:
    out = assemble(
        [_dummy_fig(), _dummy_fig(), _dummy_fig()],
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


def test_assemble_requires_exactly_three_figures() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="exactly 3"):
        assemble(
            [_dummy_fig()],
            run_date=pd.Timestamp("2024-03-29"),
            methodology_version="1.0.0",
        )
