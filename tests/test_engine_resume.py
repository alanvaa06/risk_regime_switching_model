"""engine.run data_until cut + resume plumbing + revised-history detection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.io import read_resume_state
from roro.resume import HistoryRevisedError


def _fred() -> MockFredClient:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _cfg(xlsx: Path, out: Path) -> EngineConfig:
    return EngineConfig(
        data_path=xlsx, output_dir=out, ewma_halflife_days=10, return_window_days=21,
        tripwire_window_days=10, percentile_window_years=1, min_n_per_cut=2,
        bootstrap_min_days=10, jm_enabled=True, jm_min_history_days=250,
        jm_refit_interval_days=120, jm_n_init=4,
    )


def test_data_until_cuts_every_artifact(rw_xlsx: Path, tmp_path: Path) -> None:
    engine_run(_cfg(rw_xlsx, tmp_path), fred_client=_fred(), run_date="r",
               as_of_data_date="2023-06-15", force=True, data_until="2023-06-15")
    for name in ("beta_series.csv", "regimes.csv", "regimes_jm.csv", "tripwire.csv"):
        dates = pd.read_csv(tmp_path / "r" / name, parse_dates=["date"])["date"]
        assert dates.max() == pd.Timestamp("2023-06-15"), name


def test_resume_with_revised_beta_raises(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _cfg(rw_xlsx, tmp_path)
    engine_run(cfg, fred_client=_fred(), run_date="cp", as_of_data_date="2023-06-15",
               force=True, data_until="2023-06-15")
    state = read_resume_state(tmp_path / "cp", pd.Timestamp("2023-06-15"))
    tampered = state.beta_series.copy()
    row = tampered.index[tampered["beta"].notna()][100]
    tampered.loc[row, "beta"] += 1e-6
    with pytest.raises(HistoryRevisedError, match="beta changed"):
        engine_run(cfg, fred_client=_fred(), run_date="new", as_of_data_date="2024-12-31",
                   force=True, resume=replace(state, beta_series=tampered))
    assert not (tmp_path / "new").exists()  # nothing written on a revised history


def test_resume_matches_full_in_memory(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _cfg(rw_xlsx, tmp_path)
    engine_run(cfg, fred_client=_fred(), run_date="cp", as_of_data_date="2023-06-15",
               force=True, data_until="2023-06-15")
    state = read_resume_state(tmp_path / "cp", pd.Timestamp("2023-06-15"))
    engine_run(cfg, fred_client=_fred(), run_date="res", as_of_data_date="2024-12-31",
               force=True, resume=state)
    engine_run(cfg, fred_client=_fred(), run_date="full", as_of_data_date="2024-12-31",
               force=True)
    assert (
        (tmp_path / "res" / "regimes_jm.csv").read_bytes()
        == (tmp_path / "full" / "regimes_jm.csv").read_bytes()
    )
    assert (
        (tmp_path / "res" / "jm_refit_log.csv").read_bytes()
        == (tmp_path / "full" / "jm_refit_log.csv").read_bytes()
    )
