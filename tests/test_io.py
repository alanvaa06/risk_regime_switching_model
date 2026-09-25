import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.io import _write_regime_jm, load_panel, load_prices, write_run
from roro.types import (
    AlertSet,
    AttributionFrame,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    HmmRegimeFrame,
    JmRegimeFrame,
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


def test_write_run_force_replaces_content_and_leaves_no_aside_dir(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    first = _empty_result(out_root)
    path = write_run(first, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    (path / "stale.txt").write_text("from the previous run", encoding="utf-8")
    (out_root / "2026-05-27.old").mkdir()  # leftover of an earlier interrupted overwrite
    second = replace(first, warnings=["second run"])
    path = write_run(
        second, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26", force=True
    )
    assert json.loads((path / "snapshot.json").read_text())["warnings"] == ["second run"]
    assert not (path / "stale.txt").exists()
    assert sorted(p.name for p in out_root.iterdir()) == ["2026-05-27"]


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
    expected_cols = [
        "date", "segment", "state", "label", "p_risk_off", "p_transitional",
        "p_risk_on", "confidence", "cold_start", "thin_cut"
    ]
    assert list(df.columns) == expected_cols
    snap = json.loads((out / "snapshot.json").read_text())
    assert snap["regime_hmm"]["global"]["label"] == "Risk-on"  # last row


def test_write_run_no_hmm_artifacts_when_disabled(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs2"
    result = _empty_result(out_root)  # regime_hmm is None
    out = write_run(
        result, run_date="2026-06-03", out_dir=out_root, as_of_data_date="2026-06-03", force=True
    )
    assert not (out / "regimes_hmm.csv").exists()
    assert not (out / "hmm_refit_log.csv").exists()
    snap = json.loads((out / "snapshot.json").read_text())
    assert "regime_hmm" not in snap  # key omitted entirely when HMM disabled


def test_write_regime_jm_columns_and_rows(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    def wide(v: list) -> pd.DataFrame:  # type: ignore[type-arg]
        return pd.DataFrame({"global": v}, index=idx)

    jf = JmRegimeFrame(
        state=wide([0.0, 1.0, 2.0]), label=wide(["Risk-off", "Transitional", "Risk-on"]),
        prob_risk_off=wide([1.0, 0.0, 0.0]), prob_transitional=wide([0.0, 1.0, 0.0]),
        prob_risk_on=wide([0.0, 0.0, 1.0]), confidence=wide([1.0, 1.0, 1.0]),
        n_per_segment=wide([12, 12, 12]), thin_cut_flag=wide([False, False, False]),
        cold_start_flag=wide([False, False, False]), refit_dates={"global": [idx[1]]},
    )
    out = tmp_path / "regimes_jm.csv"
    _write_regime_jm(jf, out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "segment", "state", "label", "p_risk_off",
                                "p_transitional", "p_risk_on", "confidence",
                                "cold_start", "thin_cut"]
    assert len(df) == 3


def _result_with_attribution(out_dir: Path) -> RunResult:
    base = _empty_result(out_dir)
    d = pd.Timestamp("2026-05-26")
    level = pd.DataFrame(
        [
            {"date": d, "cut": "global", "weighting": "cap", "series": "A__Eq", "block": "DM_Eq",
             "latam": False, "vol": 0.2, "ret3m": 0.1, "weight": 0.5, "leverage": 1.0,
             "contribution": 0.1, "share": 0.5, "quadrant": "HI/+", "xbar": 0.15, "ybar": 0.05},
            {"date": d, "cut": "global", "weighting": "cap", "series": "B__FI", "block": "DM_FI",
             "latam": False, "vol": 0.1, "ret3m": -0.1, "weight": 0.5, "leverage": -1.0,
             "contribution": 0.1, "share": 0.5, "quadrant": "LO/-", "xbar": 0.15, "ybar": 0.05},
        ]
    )
    delta = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "horizon": "fixed",
          "anchor_date": d - pd.Timedelta(days=90), "label_anchor": "Risk-off",
          "label_t": "Risk-on", "beta_anchor": 0.05, "beta_t": 0.2, "series": "A__Eq",
          "block": "DM_Eq", "effect_return": 0.1, "effect_position": 0.03,
          "effect_interaction": 0.02, "effect_universe": 0.0, "delta_total": 0.15}]
    )
    rollup = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "group_kind": "block",
          "group": "DM_Eq", "contribution_sum": 0.1, "n": 1}]
    )
    conc = pd.DataFrame(
        [{"date": d, "cut": "global", "weighting": "cap", "n": 2, "beta": 0.2, "hhi": 0.5,
          "top1_series": "A__Eq", "top1_share": 0.5, "top5_share": 1.0, "beta_ex_top1": 0.1,
          "pct_today": 0.8, "pct_ex_top1": 0.5, "fragile_flag": True}]
    )
    pc1 = pd.DataFrame(
        [{"date": d, "cut": "global", "series": "A__Eq", "pc1_load_sq": 0.6, "var_share": 0.5,
          "decoupling": -0.1, "row_mean_corr": 0.3}]
    )
    af = AttributionFrame(level=level, delta=delta, rollup=rollup, concentration=conc, pc1=pc1,
                          anchors={"global": d - pd.Timedelta(days=90)})
    alerts = replace(
        base.alerts,
        concentration_alerts=pd.DataFrame(
            [{"date": d, "segment": "global", "weighting": "cap", "top1_series": "A__Eq",
              "top1_share": 0.5, "hhi": 0.5, "fragile_flag": True, "trigger": "fragile"}]
        ),
    )
    return replace(base, attribution=af, alerts=alerts)


def test_write_run_emits_attribution_artifacts(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    path = write_run(_result_with_attribution(out_root), run_date="2026-05-27",
                     out_dir=out_root, as_of_data_date="2026-05-26")
    for name in ("attribution.csv", "attribution_delta.csv", "attribution_rollup.csv",
                 "concentration.csv", "attribution_pc1.csv"):
        assert (path / name).exists(), name
    assert not (path / "attribution_history_global.csv").exists()
    level = pd.read_csv(path / "attribution.csv")
    assert list(level.columns)[:3] == ["date", "cut", "weighting"]
    assert len(level) == 2
    alerts = pd.read_csv(path / "alerts.csv")
    assert "concentration" in set(alerts["kind"])
    snap = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    g = snap["attribution"]["global"]
    assert g["anchor_date"] == "2026-02-25"
    assert g["top1_series"] == "A__Eq" and g["fragile_flag"] is True
    assert [x["series"] for x in g["top3"]] == ["A__Eq", "B__FI"]


def test_write_run_history_global_when_present(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    res = _result_with_attribution(out_root)
    assert res.attribution is not None
    hist = pd.DataFrame({"A__Eq": [0.1, 0.2], "B__FI": [0.0, -0.1]},
                        index=pd.DatetimeIndex(["2026-05-25", "2026-05-26"], name="date"))
    res = replace(res, attribution=replace(res.attribution, history_global=hist))
    path = write_run(res, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    back = pd.read_csv(path / "attribution_history_global.csv", index_col="date",
                       parse_dates=["date"])
    assert list(back.columns) == ["A__Eq", "B__FI"] and len(back) == 2


def test_write_run_no_attribution_artifacts_when_none(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    path = write_run(_empty_result(out_root), run_date="2026-05-27", out_dir=out_root,
                     as_of_data_date="2026-05-26")
    assert not (path / "attribution.csv").exists()
    snap = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    assert "attribution" not in snap
