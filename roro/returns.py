"""3M total log return + daily log return + EWMA vol kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return_3m(prices_lc: pd.DataFrame, *, window_days: int = 63) -> pd.DataFrame:
    """Trailing log return over `window_days`: R_t = ln(P_t / P_{t-W})."""
    ratio = prices_lc / prices_lc.shift(window_days)
    return pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)


def daily_log_returns(prices_lc: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}). First row is NaN."""
    ratio = prices_lc / prices_lc.shift(1)
    return pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)


def ewma_vol(daily_returns: pd.DataFrame, *, halflife: int) -> pd.DataFrame:
    """RiskMetrics-style EWMA vol, annualized by sqrt(252).

    sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_t with
    lambda = exp(-ln 2 / halflife).
    Implementation: pandas ewm(adjust=False).std() gives the same recursion family.
    """
    sigma: pd.DataFrame = daily_returns.ewm(halflife=halflife, adjust=False).std()
    return sigma * float(np.sqrt(TRADING_DAYS_PER_YEAR))
