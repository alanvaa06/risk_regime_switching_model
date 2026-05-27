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


def test_imports_available() -> None:
    # Sanity: every symbol imports without error
    assert PriceFrame is not None
    assert ReturnsFrame is not None
    assert VolFrame is not None
    assert RegimeFrame is not None
    assert CorrelationFrame is not None
    assert ValidationFrame is not None
    assert AlertSet is not None
