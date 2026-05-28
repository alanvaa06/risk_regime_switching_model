"""load_bundle integration tests using existing tiny_xlsx fixture + an in-test run dir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from roro.report.errors import ReportInputError
from roro.report.load import load_bundle


def test_load_bundle_happy_path(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)

    assert bundle.methodology_version == "1.0.0"
    assert bundle.run_date == pd.Timestamp("2024-03-29")
    assert len(bundle.dates) > 0
    assert "global" in bundle.seg_beta.columns
    assert "global" in bundle.seg_tercile.columns
    # series_id format: "<country>_<asset>"
    assert any("_Eq" in c for c in bundle.vol.columns)
    assert any("_FI" in c for c in bundle.vol.columns)
    # meta has required columns
    assert {"country", "asset", "segment", "weight"}.issubset(bundle.meta.columns)


def test_load_bundle_missing_regimes_csv(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    (minimal_run_dir / "regimes.csv").unlink()

    with pytest.raises(ReportInputError, match="regimes.csv"):
        load_bundle(minimal_run_dir, tiny_xlsx, window=21)


def test_load_bundle_missing_snapshot(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    (minimal_run_dir / "snapshot.json").unlink()

    with pytest.raises(ReportInputError, match="snapshot.json"):
        load_bundle(minimal_run_dir, tiny_xlsx, window=21)
