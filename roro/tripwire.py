"""1M-window tripwire - parallel return + vol + cross-section beta only."""

from __future__ import annotations

import pandas as pd

from roro.regression import compute_beta_by_segment
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import SeriesId
from roro.types import BetaBySegment, PriceFrame


def compute_tripwire_signal(
    *,
    prices: PriceFrame,
    cuts: dict[str, list[SeriesId]],
    return_window_days: int,
    ewma_halflife_days: int,
    min_n: int,
) -> BetaBySegment:
    """Fast-signal mirror of the main regime engine using shorter windows.

    Computes a 1M-style cross-sectional beta by reusing the same kernels
    (`total_return_3m`, `ewma_vol`, `compute_beta_by_segment`) with caller-
    supplied short windows. Returns a `BetaBySegment` keyed by the same 10
    PRD segment cuts.
    """
    eq_ret = total_return_3m(prices.equity_lc, window_days=return_window_days)
    fi_ret = total_return_3m(prices.fi_lc, window_days=return_window_days)
    eq_daily = daily_log_returns(prices.equity_lc)
    fi_daily = daily_log_returns(prices.fi_lc)
    eq_vol = ewma_vol(eq_daily, halflife=ewma_halflife_days)
    fi_vol = ewma_vol(fi_daily, halflife=ewma_halflife_days)
    return compute_beta_by_segment(
        dates=pd.DatetimeIndex(prices.equity_lc.index),
        cuts=cuts,
        equity_returns_3m=eq_ret,
        fi_returns_3m=fi_ret,
        equity_vol=eq_vol,
        fi_vol=fi_vol,
        min_n=min_n,
    )
