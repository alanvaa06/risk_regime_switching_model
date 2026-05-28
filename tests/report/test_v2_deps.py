"""Verify scipy is importable for the CI-ribbon kernel in viz v2."""
from __future__ import annotations


def test_scipy_available() -> None:
    import scipy  # noqa: F401, PLC0415


def test_scipy_linregress_importable() -> None:
    from scipy.stats import linregress  # noqa: F401, PLC0415
