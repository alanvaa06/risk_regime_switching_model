import numpy as np
import pandas as pd

from roro.classify import (
    classify,
    direction_flag,
    quintile_label,
    rolling_percentile,
    tercile_label,
)
from roro.config import BucketScheme
from roro.types import BetaBySegment, BetaFrame


def _toy_betas(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2014-01-01", periods=len(values))
    return pd.Series(values, index=idx, name="beta")


def test_rolling_percentile_in_unit_interval() -> None:
    s = _toy_betas([float(i) for i in range(500)])
    p = rolling_percentile(s, window_days=252)
    p_valid = p.dropna()
    assert (p_valid >= 0).all()
    assert (p_valid <= 1).all()


def test_tercile_label_monotone() -> None:
    assert tercile_label(0.1) == "Risk-off"
    assert tercile_label(0.5) == "Transitional"
    assert tercile_label(0.9) == "Risk-on"


def test_quintile_label_five_buckets() -> None:
    assert quintile_label(0.05) == "Q1"
    assert quintile_label(0.25) == "Q2"
    assert quintile_label(0.45) == "Q3"
    assert quintile_label(0.65) == "Q4"
    assert quintile_label(0.95) == "Q5"


def test_direction_rising_when_beta_increases() -> None:
    s = _toy_betas([0.1 * i for i in range(10)])
    flag = direction_flag(s, lookback_days=5)
    assert flag.iloc[-1] == "rising"


def test_classify_full_pipeline_produces_all_frames() -> None:
    idx = pd.bdate_range("2010-01-01", periods=2000)
    cap = pd.DataFrame(
        {
            "beta": np.linspace(-1, 1, 2000),
            "r2": 0.5,
            "n": 32,
            "suppressed": False,
            "singular": False,
        },
        index=idx,
    )
    bf = BetaFrame(cap_wtd=cap, eq_wtd=cap, slope_spread=pd.Series(0.0, index=idx))
    bbs = BetaBySegment(by_segment={"global": bf, "DM": bf, "LatAm": bf})
    rf = classify(
        bbs,
        bucket_scheme=BucketScheme.TERCILE,
        percentile_window_days=1260,
        direction_lookback_days=5,
        bootstrap_min_days=252,
        thin_cuts=frozenset({"LatAm"}),
    )
    for frame in (rf.percentile_5y, rf.tercile, rf.quintile, rf.direction, rf.bootstrap_flag):
        assert set(frame.columns) >= {"global", "DM", "LatAm"}
    assert rf.thin_cut_flag.loc[idx[-1], "LatAm"]
    assert not rf.thin_cut_flag.loc[idx[-1], "global"]
