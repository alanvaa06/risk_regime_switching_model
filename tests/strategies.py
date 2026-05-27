"""Hypothesis strategies shared across tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


@st.composite
def positive_price_series(draw: st.DrawFn, *, length: int = 200) -> pd.Series:
    """Geometric-Brownian-motion-like positive price series."""
    increments = draw(
        hnp.arrays(
            np.float64,
            shape=length,
            elements=st.floats(
                min_value=-0.05,
                max_value=0.05,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    log_prices = np.cumsum(increments) + 4.0  # start near exp(4) ~ 55
    return pd.Series(
        np.exp(log_prices),
        index=pd.bdate_range("2010-01-01", periods=length),
    )


@st.composite
def finite_return_series(draw: st.DrawFn, *, length: int = 200) -> pd.Series:
    arr = draw(
        hnp.arrays(
            np.float64,
            shape=length,
            elements=st.floats(
                min_value=-0.2,
                max_value=0.2,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return pd.Series(arr, index=pd.bdate_range("2010-01-01", periods=length))


@st.composite
def positive_weights(draw: st.DrawFn, *, n: int = 10) -> np.ndarray[Any, np.dtype[np.float64]]:
    raw = draw(
        hnp.arrays(
            np.float64,
            shape=n,
            elements=st.floats(
                min_value=0.1,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return raw
