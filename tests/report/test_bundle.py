"""DataBundle contract tests."""
from __future__ import annotations

import dataclasses
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from roro.report.bundle import DataBundle
from roro.report.load import load_bundle


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


def test_databundle_attribution_fields_default_none() -> None:
    names = {f.name for f in fields(DataBundle)}
    assert {"attribution_level", "attribution_delta", "attribution_rollup",
            "concentration", "attribution_pc1"} <= names
    for n in ("attribution_level", "attribution_delta", "attribution_rollup",
              "concentration", "attribution_pc1"):
        assert DataBundle.__dataclass_fields__[n].default is None


def test_load_bundle_reads_attribution_when_present(
    attribution_run_dir: Path, tiny_xlsx: Path
) -> None:
    b = load_bundle(attribution_run_dir, tiny_xlsx, window=21)
    assert b.attribution_level is not None and not b.attribution_level.empty
    assert b.concentration is not None and b.attribution_delta is not None
    assert pd.api.types.is_datetime64_any_dtype(b.concentration["date"])
    assert pd.api.types.is_datetime64_any_dtype(b.attribution_delta["anchor_date"])


def test_load_bundle_attribution_none_when_absent(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    b = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert b.attribution_level is None and b.concentration is None
