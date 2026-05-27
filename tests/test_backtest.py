"""Tests for the backtest harness and PRD §10 acceptance gates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.backtest import EVENTS, run_backtest
from roro.config import EngineConfig
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def test_events_documented_match_prd() -> None:
    names = {e.name for e in EVENTS}
    assert "2020_COVID" in names
    assert "2022_rate_shock_jan" in names or "2022_rate_shock" in names


def test_run_backtest_writes_acceptance_report(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
    client = MockFredClient(seeded=seeded)
    cfg = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "bt",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    report = run_backtest(cfg, fred_client=client, start="2024-01-01", end="2024-12-31")
    assert "gates" in report
    assert {
        "G1_vix",
        "G2_bbb",
        "G3_events",
        "G4_segmentation_lift",
        "G5_stability",
        "G6_internal",
    } <= set(report["gates"])
    assert (tmp_path / "bt" / "acceptance_report.json").exists()
