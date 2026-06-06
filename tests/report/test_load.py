"""load_bundle integration tests using existing tiny_xlsx fixture + an in-test run dir."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roro.report.errors import ReportInputError
from roro.report.load import load_bundle
from tests.report.conftest import write_hmm_csv


def test_load_bundle_happy_path(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)

    assert bundle.methodology_version == "1.0.0"
    assert bundle.run_date == pd.Timestamp("2024-12-31")
    assert len(bundle.dates) > 0
    assert "global" in bundle.seg_beta.columns
    assert "global" in bundle.seg_tercile.columns
    # series_id format: "<country>_<asset>"
    assert any("_Eq" in c for c in bundle.vol.columns)
    assert any("_FI" in c for c in bundle.vol.columns)
    # meta has required columns
    assert {"country", "asset", "segment", "weight"}.issubset(bundle.meta.columns)
    # Segment frames must overlap bundle.dates — silent NaN reindex would hide this
    assert bundle.seg_beta["global"].notna().any()
    assert bundle.seg_tercile["global"].notna().any()


def test_load_bundle_missing_regimes_csv(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    (minimal_run_dir / "regimes.csv").unlink()

    with pytest.raises(ReportInputError, match="regimes.csv"):
        load_bundle(minimal_run_dir, tiny_xlsx, window=21)


def test_load_bundle_missing_snapshot(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    (minimal_run_dir / "snapshot.json").unlink()

    with pytest.raises(ReportInputError, match="snapshot.json"):
        load_bundle(minimal_run_dir, tiny_xlsx, window=21)


def test_load_bundle_missing_beta_series_csv(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    (minimal_run_dir / "beta_series.csv").unlink()

    with pytest.raises(ReportInputError, match="beta_series.csv"):
        load_bundle(minimal_run_dir, tiny_xlsx, window=21)


def test_load_bundle_seg_beta_carries_full_history(
    minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    """seg_beta / seg_tercile must span the full run-dir CSV history, not the 252d window."""
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    # The conftest fixture writes beta_series.csv over 2020-01-02..2024-12-31 (~1300 bdays),
    # while the per-series window is 21 days. Full history must exceed the scatter window.
    assert len(bundle.seg_beta.index) > len(bundle.dates)
    assert len(bundle.seg_tercile.index) > len(bundle.dates)
    assert bundle.seg_beta["global"].notna().any()
    assert bundle.seg_tercile["global"].notna().any()


def test_load_bundle_reads_hmm_when_present(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    write_hmm_csv(minimal_run_dir)
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.seg_hmm_label is not None
    assert bundle.seg_hmm_p_off is not None
    assert "global" in bundle.seg_hmm_label.columns
    assert bundle.seg_hmm_p_off.shape == bundle.seg_hmm_label.shape
    assert bundle.seg_hmm_p_tr is not None and bundle.seg_hmm_p_on is not None


def test_load_bundle_hmm_none_when_absent(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.seg_hmm_label is None
    assert bundle.seg_hmm_p_off is None
    assert bundle.seg_hmm_p_tr is None
    assert bundle.seg_hmm_p_on is None


def test_load_bundle_vol_pct_full_history_with_warmup(
    minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.vol_pct is not None
    assert len(bundle.vol_pct.index) > len(bundle.dates)
    vals = bundle.vol_pct.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert (finite >= 0).all() and (finite <= 1).all()
