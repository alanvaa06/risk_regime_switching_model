"""Engine orchestrator integration test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def _seeded_fred() -> MockFredClient:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def test_engine_run_produces_outputs(tiny_xlsx: Path, tmp_path: Path) -> None:
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
    result = run(
        cfg,
        fred_client=_seeded_fred(),
        run_date="2024-12-31",
        as_of_data_date="2024-12-31",
    )
    assert (cfg.output_dir / "2024-12-31").exists()
    assert "global" in result.beta.by_segment
    assert result.config.methodology_version == "1.0.0"
