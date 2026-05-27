import json
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
