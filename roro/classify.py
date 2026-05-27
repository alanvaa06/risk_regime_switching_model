"""Rolling percentile classifier: tercile, quintile, direction, thin-cut flag."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from roro.config import BucketScheme
from roro.types import BetaBySegment, RegimeFrame

_TERCILE_LOW = 1.0 / 3.0
_TERCILE_HIGH = 2.0 / 3.0
_QUINTILE_EDGES = (0.2, 0.4, 0.6, 0.8)
_ASYM_LOW = 0.2
_ASYM_HIGH = 0.8
_MIN_DIRECTION_WINDOW = 2
_DIRECTION_NOISE_RATIO = 0.1


def rolling_percentile(beta: pd.Series, *, window_days: int) -> pd.Series:
    """For each t, fraction of trailing-window values <= beta[t]. NaN if empty window."""

    def _pct(window: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        if len(window) == 0:
            return float("nan")
        return float((window <= window[-1]).sum() - 1) / float(max(len(window) - 1, 1))

    return beta.rolling(window=window_days, min_periods=1).apply(_pct, raw=True)


def tercile_label(p: float) -> str:
    if np.isnan(p):
        return "Unknown"
    if p <= _TERCILE_LOW:
        return "Risk-off"
    if p >= _TERCILE_HIGH:
        return "Risk-on"
    return "Transitional"


def quintile_label(p: float) -> str:
    if np.isnan(p):
        return "Unknown"
    for i, edge in enumerate(_QUINTILE_EDGES, start=1):
        if p <= edge:
            return f"Q{i}"
    return "Q5"


def asym_label(p: float) -> str:
    """20/60/20 scheme."""
    if np.isnan(p):
        return "Unknown"
    if p <= _ASYM_LOW:
        return "Risk-off"
    if p >= _ASYM_HIGH:
        return "Risk-on"
    return "Transitional"


def _bucket(p: float, scheme: BucketScheme) -> str:
    if scheme is BucketScheme.TERCILE:
        return tercile_label(p)
    if scheme is BucketScheme.QUINTILE:
        return quintile_label(p)
    return asym_label(p)


def direction_flag(beta: pd.Series, *, lookback_days: int) -> pd.Series:
    def _flag(window: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        if len(window) < _MIN_DIRECTION_WINDOW or np.isnan(window).any():
            return float("nan")
        x = np.arange(len(window), dtype=np.float64)
        slope = float(np.polyfit(x, window, 1)[0])
        # Use the window stdev as a noise scale to discriminate
        scale = float(np.std(window))
        if scale == 0 or abs(slope) < _DIRECTION_NOISE_RATIO * scale:
            return 0.0
        return 1.0 if slope > 0 else -1.0

    raw = beta.rolling(window=lookback_days, min_periods=lookback_days).apply(_flag, raw=True)
    return raw.map({1.0: "rising", -1.0: "falling", 0.0: "stable"}).astype(object)


def classify(
    bbs: BetaBySegment,
    *,
    bucket_scheme: BucketScheme,
    percentile_window_days: int,
    direction_lookback_days: int,
    bootstrap_min_days: int,
    thin_cuts: frozenset[str],
) -> RegimeFrame:
    pct_frames: dict[str, pd.Series] = {}
    terc_frames: dict[str, pd.Series] = {}
    quin_frames: dict[str, pd.Series] = {}
    dir_frames: dict[str, pd.Series] = {}
    n_frames: dict[str, pd.Series] = {}
    boot_frames: dict[str, pd.Series] = {}
    thin_frames: dict[str, pd.Series] = {}

    bucket_fn = partial(_bucket, scheme=bucket_scheme)

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        pct = rolling_percentile(beta, window_days=percentile_window_days)
        # Bootstrap suppression
        valid_count = beta.expanding(min_periods=1).count()
        boot = valid_count < percentile_window_days
        pct = pct.where(valid_count >= bootstrap_min_days, other=np.nan)
        terc = pct.map(bucket_fn)
        quin = pct.map(quintile_label)
        dir_s = direction_flag(beta, lookback_days=direction_lookback_days)
        pct_frames[cut] = pct
        terc_frames[cut] = terc
        quin_frames[cut] = quin
        dir_frames[cut] = dir_s
        n_frames[cut] = bf.cap_wtd["n"]
        boot_frames[cut] = boot
        thin_frames[cut] = pd.Series(cut in thin_cuts, index=beta.index)

    return RegimeFrame(
        percentile_5y=pd.DataFrame(pct_frames),
        tercile=pd.DataFrame(terc_frames),
        quintile=pd.DataFrame(quin_frames),
        direction=pd.DataFrame(dir_frames),
        n_per_segment=pd.DataFrame(n_frames),
        thin_cut_flag=pd.DataFrame(thin_frames),
        bootstrap_flag=pd.DataFrame(boot_frames),
    )
