"""Tests for the 1M-window tripwire fast-signal mirror."""

from __future__ import annotations

import pandas as pd

from roro.segments import partition
from roro.tripwire import compute_tripwire_signal
from roro.types import PriceFrame, Universe


def _tiny_universe() -> Universe:
    return Universe(
        countries=pd.DataFrame(
            {
                "Country": [f"C{i}" for i in range(20)],
                "Segment": ["DM"] * 10 + ["EM"] * 10,
                "Equity_Mkt_Cap_Val": list(range(1, 21)),
                "Fixed_Income_Mkt_Cap_Val": list(range(1, 21)),
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )


def test_tripwire_returns_betabysegment_same_keys() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    cols = [f"C{i}" for i in range(20)]
    eq = pd.DataFrame(1.0, index=dates, columns=cols).cumsum() + 100
    fi = pd.DataFrame(1.0, index=dates, columns=cols).cumsum() + 100
    pf = PriceFrame(equity_lc=eq, fi_lc=fi)
    bbs = compute_tripwire_signal(
        prices=pf,
        cuts=partition(_tiny_universe()),
        return_window_days=21,
        ewma_halflife_days=10,
        min_n=5,
    )
    expected = {"global", "DM", "EM", "Equity", "FI", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI", "LatAm"}
    assert set(bbs.by_segment) == expected
