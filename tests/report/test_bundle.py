"""DataBundle contract tests."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from roro.report.bundle import DataBundle


def _make_bundle() -> DataBundle:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    empty = pd.DataFrame(index=idx)
    meta = pd.DataFrame(
        {"country": [], "asset": [], "segment": [], "weight": []},
    )
    return DataBundle(
        run_date=pd.Timestamp("2024-01-04"),
        methodology_version="1.0.0",
        dates=idx,
        vol=empty,
        ret_3m=empty,
        beta_vs_global=empty,
        meta=meta,
        seg_beta=empty,
        seg_tercile=empty,
    )


def test_databundle_is_frozen() -> None:
    bundle = _make_bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.run_date = pd.Timestamp("2025-01-01")  # type: ignore[misc]


def test_databundle_fields_present() -> None:
    bundle = _make_bundle()
    assert bundle.methodology_version == "1.0.0"
    assert bundle.run_date == pd.Timestamp("2024-01-04")
    assert len(bundle.dates) == 3
