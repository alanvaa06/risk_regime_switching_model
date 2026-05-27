"""Integration regression: engine output must match committed golden CSVs.

Regenerate with: pytest --regenerate-goldens
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient

GOLDEN_DIR = Path(__file__).parent / "golden" / "2024-Q1"


def _seeded_fred() -> MockFredClient:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _run_against(tmp_path: Path, tiny_xlsx: Path) -> Path:
    cfg = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "out",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    run(
        cfg,
        fred_client=_seeded_fred(),
        run_date="2024-03-31",
        as_of_data_date="2024-03-31",
        force=True,
    )
    return cfg.output_dir / "2024-03-31"


def test_golden_2024_q1(tiny_xlsx: Path, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    actual = _run_against(tmp_path, tiny_xlsx)
    if request.config.getoption("--regenerate-goldens"):
        if GOLDEN_DIR.exists():
            shutil.rmtree(GOLDEN_DIR)
        shutil.copytree(actual, GOLDEN_DIR, ignore=shutil.ignore_patterns("snapshot.json"))
        pytest.skip("Regenerated goldens.")

    if not GOLDEN_DIR.exists():
        pytest.skip("Goldens not yet generated; run with --regenerate-goldens.")

    csvs = (
        "beta_series.csv",
        "regimes.csv",
        "correlation.csv",
        "external_validation.csv",
        "tripwire.csv",
    )
    for csv in csvs:
        assert filecmp.cmp(actual / csv, GOLDEN_DIR / csv, shallow=False), (
            f"{csv} drifted from golden"
        )
