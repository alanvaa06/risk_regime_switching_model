import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.io import load_panel, load_prices, write_run
from roro.types import (
    AlertSet,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    HmmRegimeFrame,
    RegimeFrame,
    ReturnsFrame,
    RunResult,
    Universe,
    ValidationFrame,
    VolFrame,
)


def test_panel_splits_countries_and_composites(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert set(u.countries["Country"]) == {"United States", "Brazil", "Germany", "Mexico"}
    assert set(u.composites["Country"]) == {"DM", "LatAm"}


def test_panel_carries_mcap_val_columns(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert "Equity_Mkt_Cap_Val" in u.countries.columns
    assert "Fixed_Income_Mkt_Cap_Val" in u.countries.columns
    assert (u.countries["Equity_Mkt_Cap_Val"] > 0).all()


def test_load_panel_missing_column_raises_sorted(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xlsx"
    # Missing Equity_Mkt_Cap_Val and Fixed_Income_Mkt_Cap_Val
    df = pd.DataFrame({"Country": ["US"], "Segment": ["DM"]})
    with pd.ExcelWriter(bad, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Panel", index=False)

    with pytest.raises(ValueError, match=r"Equity_Mkt_Cap_Val.*Fixed_Income_Mkt_Cap_Val"):
        load_panel(bad)


def test_prices_uses_country_row_as_header(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert "United States" in pf.equity_lc.columns
    assert "Brazil" in pf.equity_lc.columns
    # Header row must be country names, not tickers
    assert "SPX Index" not in pf.equity_lc.columns
    assert isinstance(pf.equity_lc.index, pd.DatetimeIndex)


def test_prices_aligned_columns(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert list(pf.equity_lc.columns) == list(pf.fi_lc.columns)


def _empty_result(out_dir: Path) -> RunResult:
    empty = pd.DataFrame()
    empty_s = pd.Series(dtype=float)
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=empty_s)
    bbs = BetaBySegment(by_segment={"global": bf})
    return RunResult(
        config=EngineConfig(data_path=Path("data.xlsx"), output_dir=out_dir),
        universe=Universe(countries=empty, composites=empty),
        returns=ReturnsFrame(log_returns_3m=empty, daily_log_returns=empty),
        vol=VolFrame(ewma_sigma_annualized=empty),
        beta=bbs,
        regime=RegimeFrame(
            percentile_5y=empty,
            tercile=empty,
            quintile=empty,
            direction=empty,
            n_per_segment=empty,
            thin_cut_flag=empty,
            bootstrap_flag=empty,
        ),
        correlation=CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty),
        validation=ValidationFrame(
            rolling_corr_60d=empty, internal_consistency=empty, correlation_alerts=empty
        ),
        tripwire=bbs,
        alerts=AlertSet(
            bucket_transitions=empty, disagreement_events=empty, validation_degradation=empty
        ),
        warnings=["x"],
        data_fingerprint={"data_xlsx_sha256": "abc"},
        code_version={"git_sha": "def", "dirty": "false"},
    )


def test_write_run_atomic_and_round_trip(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    result = _empty_result(out_root)
    path = write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    assert path.exists()
    assert (path / "snapshot.json").exists()
    snapshot = json.loads((path / "snapshot.json").read_text())
    assert snapshot["methodology_version"] == "1.0.0"
    assert snapshot["data_fingerprint"]["data_xlsx_sha256"] == "abc"
    assert not (out_root / "2026-05-27.tmp").exists()


def test_write_run_force_overwrites(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    result = _empty_result(out_root)
    write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    with pytest.raises(FileExistsError):
        write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    write_run(
        result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26", force=True
    )


def _result_with_hmm(out_dir: Path) -> RunResult:
    base = _empty_result(out_dir)
    idx = pd.bdate_range("2026-05-20", periods=3)
    lab = pd.DataFrame({"global": ["Risk-off", "Transitional", "Risk-on"]}, index=idx)
    num = pd.DataFrame({"global": [0.2, 0.5, 0.8]}, index=idx)
    state = pd.DataFrame({"global": [0, 1, 2]}, index=idx)
    flag = pd.DataFrame({"global": [False, False, False]}, index=idx)
    hmm = HmmRegimeFrame(
        state=state, label=lab,
        prob_risk_off=num, prob_transitional=num, prob_risk_on=num,
        confidence=num, n_per_segment=pd.DataFrame({"global": [20, 20, 20]}, index=idx),
        thin_cut_flag=flag, cold_start_flag=flag,
        refit_dates={"global": [idx[0]]},
    )
    return replace(base, regime_hmm=hmm)


def test_write_run_emits_hmm_artifacts(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    result = _result_with_hmm(out_root)
    out = write_run(
        result, run_date="2026-06-03", out_dir=out_root, as_of_data_date="2026-06-03", force=True
    )
    assert (out / "regimes_hmm.csv").exists()
    assert (out / "hmm_refit_log.csv").exists()
    df = pd.read_csv(out / "regimes_hmm.csv")
    assert {"date", "segment", "label", "p_risk_off"}.issubset(df.columns)
    snap = json.loads((out / "snapshot.json").read_text())
    assert snap["regime_hmm"]["global"]["label"] == "Risk-on"  # last row


def test_write_run_no_hmm_artifacts_when_disabled(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs2"
    result = _empty_result(out_root)  # regime_hmm is None
    out = write_run(
        result, run_date="2026-06-03", out_dir=out_root, as_of_data_date="2026-06-03", force=True
    )
    assert not (out / "regimes_hmm.csv").exists()
    snap = json.loads((out / "snapshot.json").read_text())
    assert "regime_hmm" not in snap  # key omitted entirely when HMM disabled
