"""Reproducibility test: two runs with identical inputs produce byte-identical CSVs."""

from __future__ import annotations

import filecmp
from pathlib import Path

import pandas as pd

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def test_two_runs_identical_csvs(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
    client = MockFredClient(seeded=seeded)
    cfg_a = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "a",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    cfg_b = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "b",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    run(
        cfg_a,
        fred_client=client,
        run_date="2024-12-31",
        as_of_data_date="2024-12-31",
        force=True,
    )
    run(
        cfg_b,
        fred_client=client,
        run_date="2024-12-31",
        as_of_data_date="2024-12-31",
        force=True,
    )

    for csv in (
        "beta_series.csv",
        "regimes.csv",
        "correlation.csv",
        "external_validation.csv",
        "tripwire.csv",
    ):
        a = tmp_path / "a" / "2024-12-31" / csv
        b = tmp_path / "b" / "2024-12-31" / csv
        assert filecmp.cmp(a, b, shallow=False), f"{csv} differs between runs"
