from datetime import datetime

import numpy as np
import pandas as pd

from roro.types import FredFrame, RegimeFrame
from roro.validation import (
    COMPOSITE_MAPPING,
    compute_internal_consistency,
    compute_rolling_external_corr,
    detect_validation_degradation,
)


def _tercile(v: float) -> str:
    if v > 0.66:
        return "Risk-on"
    if v < 0.33:
        return "Risk-off"
    return "Transitional"


def _regime_frame(idx: pd.DatetimeIndex) -> RegimeFrame:
    pct = pd.DataFrame(
        {"global": np.linspace(0, 1, len(idx)), "DM": np.linspace(0, 1, len(idx))},
        index=idx,
    )
    empty = pd.DataFrame(False, index=idx, columns=["global", "DM"], dtype=bool)
    return RegimeFrame(
        percentile_5y=pct,
        tercile=pct.map(_tercile),
        quintile=pct,
        direction=pct,
        n_per_segment=pct,
        thin_cut_flag=empty,
        bootstrap_flag=empty,
    )


def test_rolling_external_corr_window_60() -> None:
    idx = pd.bdate_range("2024-01-01", periods=200)
    regime = _regime_frame(idx)
    ff = FredFrame(
        series={"VIXCLS": pd.Series(np.linspace(20, 30, 200), index=idx)},
        pulled_at=datetime(2026, 5, 27),
        series_hashes={"VIXCLS": "h"},
    )
    out = compute_rolling_external_corr(regime, ff, window_days=60)
    assert ("global", "VIXCLS") in out.columns
    assert out[("global", "VIXCLS")].iloc[-1] is not pd.NA


def test_detect_degradation_flags_low_correlation() -> None:
    idx = pd.bdate_range("2024-01-01", periods=120)
    df = pd.DataFrame({("global", "VIXCLS"): [0.1] * 120}, index=idx)
    alerts = detect_validation_degradation(df, threshold=0.3)
    assert alerts["below_threshold"].all()


def test_internal_consistency_runs_for_mapped_segments() -> None:
    idx = pd.bdate_range("2024-01-01", periods=120)
    pct = pd.DataFrame(
        {"DM": np.linspace(0, 1, 120), "EM": np.linspace(1, 0, 120)},
        index=idx,
    )
    regime = RegimeFrame(
        percentile_5y=pct,
        tercile=pct.map(_tercile),
        quintile=pct,
        direction=pct,
        n_per_segment=pct,
        thin_cut_flag=pct.copy().astype(bool),
        bootstrap_flag=pct.copy().astype(bool),
    )
    composite_eq = pd.DataFrame(
        {"MXWO": np.linspace(100, 130, 120), "MXEF": np.linspace(100, 120, 120)},
        index=idx,
    )
    composite_fi = pd.DataFrame(
        {"I35402US": np.linspace(50, 55, 120), "EMUSTRUU": np.linspace(50, 53, 120)},
        index=idx,
    )
    out = compute_internal_consistency(
        regime=regime,
        composite_eq_prices=composite_eq,
        composite_fi_prices=composite_fi,
        return_window_days=63,
    )
    assert "DM" in out.columns
    assert "EM" in out.columns


def test_composite_mapping_contains_documented_pairs() -> None:
    assert COMPOSITE_MAPPING["DM"] == ("MXWO", "I35402US")
    assert COMPOSITE_MAPPING["LatAm"] == ("MXLA", "H04338US")
