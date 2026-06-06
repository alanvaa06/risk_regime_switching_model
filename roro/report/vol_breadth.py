"""Simple realized vol + expanding-min-5Y per-series volatility percentile."""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from roro.returns import TRADING_DAYS_PER_YEAR


def realized_vol(daily_returns: pd.DataFrame, *, window: int = 63) -> pd.DataFrame:
    """Simple annualized realized vol: rolling stdev of daily log returns × √252."""
    sigma = daily_returns.rolling(window).std()
    return sigma * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def _expanding_percentile(values: np.ndarray, min_history_days: int) -> np.ndarray:  # type: ignore[type-arg]
    """For each t with a valid value: fraction of prior+current valid values ≤ value[t].

    NaN until ≥ min_history_days valid readings seen. Exact; incremental sorted list.
    """
    hist: list[float] = []
    out = np.full(values.shape[0], np.nan)
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        bisect.insort(hist, float(v))
        if len(hist) >= min_history_days:
            cnt = bisect.bisect_right(hist, float(v))
            out[i] = cnt / len(hist)
    return out


def vol_percentile_matrix(
    vol_full: pd.DataFrame, *, min_history_days: int = 1260
) -> pd.DataFrame:
    """Per series: expanding percentile rank of vol(t) vs all prior valid vol ≤ t.

    Returns a date × series frame in [0, 1], NaN during the per-series warmup.
    """
    cols = {
        col: _expanding_percentile(vol_full[col].to_numpy(dtype=float), min_history_days)
        for col in vol_full.columns
    }
    return pd.DataFrame(cols, index=vol_full.index)
