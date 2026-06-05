import dataclasses
from datetime import datetime

import pandas as pd
import pytest

from roro.types import (
    AlertSet,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    FredFrame,
    HmmRegimeFrame,
    PriceFrame,
    RegimeFrame,
    ReturnsFrame,
    Universe,
    ValidationFrame,
    VolFrame,
)


def test_universe_frozen() -> None:
    u = Universe(
        countries=pd.DataFrame({"Country": ["US"], "Segment": ["DM"]}),
        composites=pd.DataFrame({"Country": ["World"]}),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.countries = pd.DataFrame()  # type: ignore[misc]


def test_latam_default_membership() -> None:
    u = Universe(
        countries=pd.DataFrame({"Country": ["Brazil"]}),
        composites=pd.DataFrame({"Country": ["LatAm"]}),
    )
    assert set(u.latam_countries) == {"Brazil", "Mexico", "Chile", "Peru", "Colombia"}


def test_beta_by_segment_keys() -> None:
    empty = pd.DataFrame()
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=pd.Series(dtype=float))
    bbs = BetaBySegment(by_segment={"global": bf})
    assert "global" in bbs.by_segment


def test_fred_frame_carries_fingerprint() -> None:
    ff = FredFrame(
        series={"VIXCLS": pd.Series(dtype=float)},
        pulled_at=datetime(2026, 5, 27),
        series_hashes={"VIXCLS": "abc"},
    )
    assert ff.series_hashes["VIXCLS"] == "abc"


def test_hmm_regime_frame_constructs() -> None:
    idx = pd.bdate_range("2014-01-01", periods=3)
    df = pd.DataFrame({"global": [0, 1, 2]}, index=idx)
    f = HmmRegimeFrame(
        state=df, label=df.astype(str), prob_risk_off=df, prob_transitional=df,
        prob_risk_on=df, confidence=df, n_per_segment=df, thin_cut_flag=df,
        cold_start_flag=df, refit_dates={"global": [idx[0]]},
    )
    assert list(f.label.columns) == ["global"]
    assert f.refit_dates["global"] == [idx[0]]


def test_imports_available() -> None:
    # Sanity: every symbol imports without error
    assert PriceFrame is not None
    assert ReturnsFrame is not None
    assert VolFrame is not None
    assert RegimeFrame is not None
    assert CorrelationFrame is not None
    assert ValidationFrame is not None
    assert AlertSet is not None
