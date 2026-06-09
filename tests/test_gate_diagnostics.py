"""Tests for the read-only acceptance-gate diagnostic harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.backtest import Event
from roro.gate_diagnostics import (
    classify_gate,
    repair_g3,
    sweep_external_corr,
    sweep_segmentation_lift,
)


def test_sweep_external_corr_is_monotone_nonincreasing() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    col = ("global", "VIXCLS")
    rho = pd.DataFrame({col: pd.Series(np.linspace(0.0, 1.0, len(idx)), index=idx)})
    rho.columns = pd.MultiIndex.from_tuples([col])
    out = sweep_external_corr(rho, series_id="VIXCLS", rho_grid=[0.2, 0.4, 0.6, 0.8])
    fr = out["fraction_above"].to_numpy()
    assert (np.diff(fr) <= 1e-12).all()  # higher rho_min -> fewer days clear it


def test_sweep_segmentation_lift_is_monotone_in_gap() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    terc = pd.DataFrame(
        {"DM_Eq": pd.Series("Risk-on", index=idx), "EM_Eq": pd.Series("Risk-off", index=idx)}
    )
    out = sweep_segmentation_lift(terc, gap_grid=[1, 2, 3])
    fr = out["fraction_with_gap_ge"].to_numpy()
    assert (np.diff(fr) <= 1e-12).all()
    # gap == 2 here (Risk-on=3 vs Risk-off=1) -> fraction 1.0 at gap<=2, 0.0 at gap 3
    assert out.set_index("gap").loc[2, "fraction_with_gap_ge"] == 1.0
    assert out.set_index("gap").loc[3, "fraction_with_gap_ge"] == 0.0


def test_classify_gate_tags() -> None:
    assert classify_gate(passed=True, percentile_value=1.0, hmm_value=1.0,
                         value=1.0, threshold=5.0, vacuous_reason="no DM mapping data",
                         out_of_range=[]) == "vacuous"
    assert classify_gate(passed=False, percentile_value=0.5, hmm_value=0.5,
                         value=0.5, threshold=0.8, vacuous_reason=None,
                         out_of_range=[]) == "shared"
    assert classify_gate(passed=False, percentile_value=7, hmm_value=6,
                         value=7, threshold=8, vacuous_reason=None,
                         out_of_range=["2008_lehman"]) == "bug"
    assert classify_gate(passed=False, percentile_value=0.183, hmm_value=0.161,
                         value=0.183, threshold=0.2, vacuous_reason=None,
                         out_of_range=[]) == "borderline"
    assert classify_gate(passed=False, percentile_value=18, hmm_value=9,
                         value=18, threshold=2, vacuous_reason=None,
                         out_of_range=[]) == "real"


def test_classify_gate_shared_on_passing_gate() -> None:
    # Equal percentile/hmm values tag "shared" even when the gate passed:
    # the tag is diagnostic metadata (methods agree), not a pass/fail verdict.
    assert classify_gate(passed=True, percentile_value=0.5, hmm_value=0.5,
                         value=0.5, threshold=0.8, vacuous_reason=None,
                         out_of_range=[]) == "shared"


def test_repair_g3_excludes_out_of_range_and_grades() -> None:
    idx = pd.bdate_range("2010-01-01", "2022-12-31")
    terc = pd.DataFrame({"global": pd.Series("Risk-off", index=idx)})
    events = (
        Event(name="2008_lehman", date="2008-10-10"),  # before start -> excluded
        Event(name="2020_COVID", date="2020-03-16"),
        Event(name="2018_q4", date="2018-12-24"),
    )
    out = repair_g3(terc, events, start="2010-01-01", end="2022-12-31")
    assert out["out_of_range"] == ["2008_lehman"]
    assert out["total_in_range"] == 2
    assert out["hits"] == 2
    assert out["graded"] == 1.0  # 2/2 in-range events caught
    assert out["passed_in_range"] is True
