"""Avg pairwise correlation + PC1 variance share per segment, rolling window."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SeriesId
from roro.types import CorrelationFrame

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_MIN_OBS: int = 3  # minimum rows required to compute a correlation / covariance
_MIN_COLS: int = 2  # minimum columns required to form a pairwise statistic


def avg_pairwise_rolling(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Rolling average of the upper-triangle pairwise correlations across columns."""

    def _avg_corr(window_arr: FloatArray) -> float:
        if window_arr.shape[0] < _MIN_OBS or window_arr.shape[1] < _MIN_COLS:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(window_arr, rowvar=False)
        n = corr.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = corr[iu]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return float("nan")
        return float(finite.mean())

    return _rolling_matrix_reduce(df, window=window, reducer=_avg_corr)


def pc1_share_rolling(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Rolling share of total covariance variance explained by the first PC."""

    def _pc1(window_arr: FloatArray) -> float:
        if window_arr.shape[0] < _MIN_OBS or window_arr.shape[1] < _MIN_COLS:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.cov(window_arr, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return float("nan")
        eigvals = np.linalg.eigvalsh(cov)
        trace = float(eigvals.sum())
        if trace <= 0:
            return float("nan")
        return float(eigvals.max() / trace)

    return _rolling_matrix_reduce(df, window=window, reducer=_pc1)


def _rolling_matrix_reduce(
    df: pd.DataFrame, *, window: int, reducer: Callable[[FloatArray], float]
) -> pd.Series:
    out: list[float] = []
    arr: FloatArray = df.to_numpy(dtype=np.float64)
    for end in range(len(df)):
        start = end - window + 1
        if start < 0:
            out.append(float("nan"))
            continue
        window_arr = arr[start : end + 1]
        # Drop columns that are entirely NaN within the window.
        mask = ~np.all(np.isnan(window_arr), axis=0)
        out.append(reducer(window_arr[:, mask]))
    return pd.Series(out, index=df.index)


def compute_correlation_panel(
    *,
    daily_log_returns_eq: pd.DataFrame,
    daily_log_returns_fi: pd.DataFrame,
    cuts: dict[str, list[SeriesId]],
    window: int,
) -> CorrelationFrame:
    """Compute avg pairwise correlation and PC1 variance share per segment cut."""
    avg_frames: dict[str, pd.Series] = {}
    pc1_frames: dict[str, pd.Series] = {}
    for cut_name, series in cuts.items():
        cols_eq = [s.country for s in series if s.asset_class == ASSET_EQ]
        cols_fi = [s.country for s in series if s.asset_class == ASSET_FI]
        sub_eq = daily_log_returns_eq[[c for c in cols_eq if c in daily_log_returns_eq.columns]]
        sub_fi = daily_log_returns_fi[[c for c in cols_fi if c in daily_log_returns_fi.columns]]
        # Tag columns to keep them distinct in the merged frame.
        sub_eq = sub_eq.add_suffix("__Eq")
        sub_fi = sub_fi.add_suffix("__FI")
        merged = pd.concat([sub_eq, sub_fi], axis=1)
        avg_frames[cut_name] = avg_pairwise_rolling(merged, window=window)
        pc1_frames[cut_name] = pc1_share_rolling(merged, window=window)
    return CorrelationFrame(
        avg_pairwise_3m=pd.DataFrame(avg_frames),
        pc1_variance_share=pd.DataFrame(pc1_frames),
    )
