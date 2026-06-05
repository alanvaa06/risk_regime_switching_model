"""Tests for the backtest harness and PRD §10 acceptance gates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from roro.backtest import EVENTS, _evaluate_gates, run_backtest
from roro.config import EngineConfig
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.types import (
    AlertSet,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    RegimeFrame,
    ReturnsFrame,
    RunResult,
    Universe,
    ValidationFrame,
    VolFrame,
)


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


def test_evaluate_gates_accepts_explicit_label_source() -> None:
    empty = pd.DataFrame()
    # tercile / label frames need a DatetimeIndex so _gate_events can compare
    # index values against pd.Timestamp objects without TypeError.
    empty_dt = pd.DataFrame(index=pd.DatetimeIndex([]))
    empty_s = pd.Series(dtype=float)
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=empty_s)
    bbs = BetaBySegment(by_segment={"global": bf})
    result = RunResult(
        config=EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o")),
        universe=Universe(countries=empty, composites=empty),
        returns=ReturnsFrame(log_returns_3m=empty, daily_log_returns=empty),
        vol=VolFrame(ewma_sigma_annualized=empty),
        beta=bbs,
        regime=RegimeFrame(
            percentile_5y=empty,
            tercile=empty_dt,
            quintile=empty,
            direction=empty,
            n_per_segment=empty,
            thin_cut_flag=empty,
            bootstrap_flag=empty,
        ),
        correlation=CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty),
        validation=ValidationFrame(
            rolling_corr_60d=empty,
            internal_consistency=empty,
            correlation_alerts=empty,
        ),
        tripwire=bbs,
        alerts=AlertSet(
            bucket_transitions=empty,
            disagreement_events=empty,
            validation_degradation=empty,
        ),
    )
    gates = _evaluate_gates(
        result,
        labels=result.regime.tercile,
        transitions=result.alerts.bucket_transitions,
    )
    assert set(gates) == {
        "G1_vix",
        "G2_bbb",
        "G3_events",
        "G4_segmentation_lift",
        "G5_stability",
        "G6_internal",
    }


@pytest.mark.slow
def test_run_backtest_writes_hmm_compare_reports(tiny_xlsx: Path, tmp_path: Path) -> None:
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
        hmm_enabled=True,
        hmm_min_history_days=120,
        hmm_refit_interval_days=60,
    )
    run_backtest(cfg, fred_client=client, start="2024-01-01", end="2024-12-31")
    assert (tmp_path / "bt" / "acceptance_report.json").exists()
    assert (tmp_path / "bt" / "acceptance_report_hmm.json").exists()
    assert (tmp_path / "bt" / "acceptance_compare.json").exists()
