"""Verify roro.report package is importable."""
from __future__ import annotations


def test_report_package_importable() -> None:
    import roro.report  # noqa: F401, PLC0415


def test_plotly_available() -> None:
    import plotly  # noqa: F401, PLC0415
