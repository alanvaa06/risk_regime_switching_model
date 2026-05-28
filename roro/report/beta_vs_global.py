"""63d rolling OLS β per series vs cap-weighted global proxy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_beta_vs_global(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    *,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling OLS β of each column vs the cap-weighted global proxy return.

    Args:
        log_returns: date x series_id, daily log returns.
        weights: series_id -> non-negative weight (e.g., market cap).
        window: rolling window in business days (default 63).

    Returns:
        date x series_id frame of β values. NaN until `window` observations
        are available.
    """
    # Align weights to columns; missing -> 0
    aligned_weights = weights.reindex(log_returns.columns).fillna(0.0).astype(float)
    total = float(aligned_weights.sum())
    if total <= 0.0:
        raise ValueError("Sum of weights must be positive to build global proxy.")

    proxy = (log_returns * aligned_weights.to_numpy()).sum(axis=1) / total
    proxy_var = proxy.rolling(window).var()

    out: dict[str, pd.Series] = {}
    for col in log_returns.columns:
        cov = log_returns[col].rolling(window).cov(proxy)
        out[col] = cov / proxy_var

    result = pd.DataFrame(out, index=log_returns.index)
    # Replace inf (from 0 variance) with NaN, then re-NaN any pre-window row
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
