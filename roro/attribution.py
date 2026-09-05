"""Exact per-asset attribution of the cross-sectional regime slope.

The WLS slope of 3M return on EWMA vol (roro/regression.py) is linear in returns:

    beta = sum_i c_i,   c_i = h_i * y_i,   h_i = w_i (x_i - xbar) / D,
    xbar = sum_i w_i x_i,   D = sum_i w_i (x_i - xbar)^2,   sum_i w_i = 1.

Because sum_i w_i (x_i - xbar) = 0 the intercept drops out, so the decomposition
is exact, additive and residual-free. Everything here consumes the *identical*
DailyPanel the slope is fitted on; regression.py is never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from roro.regression import DailyPanel
from roro.segments import LATAM_COUNTRIES, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Weighting = Literal["cap", "eq"]

WEIGHTINGS: tuple[Weighting, ...] = ("cap", "eq")  # iterated by compute_attribution (Task 8)
_MIN_PANEL: int = 3  # below this a slope + leave-one-out are not meaningful
_D_REL_FLOOR: float = 8.0 * float(np.finfo(np.float64).eps)

LEVEL_COLUMNS: tuple[str, ...] = (
    "series", "block", "latam", "vol", "ret3m", "weight", "leverage",
    "contribution", "share", "quadrant", "xbar", "ybar",
)


def series_key(s: SeriesId) -> str:
    """Stable per-asset key: 'Country__Eq' / 'Country__FI' (matches correlation.py suffixes)."""
    return f"{s.country}__{s.asset_class}"


@dataclass(frozen=True)
class PanelContributions:
    """Array-level attribution of one panel under one weighting."""

    w: FloatArray  # normalized weights
    h: FloatArray  # leverage
    c: FloatArray  # contributions, sum == beta
    xbar: float
    ybar: float
    beta: float


@dataclass(frozen=True)
class AttributionRow:
    series: str
    block: str  # DM_Eq | EM_Eq | DM_FI | EM_FI
    latam: bool
    vol: float
    ret3m: float
    weight: float
    leverage: float
    contribution: float
    share: float  # contribution / beta; NaN when beta == 0
    quadrant: str  # HI/+ HI/- LO/+ LO/-
    xbar: float
    ybar: float


def _normalized_weights(panel: DailyPanel, weighting: Weighting) -> FloatArray:
    n = len(panel.returns)
    w = panel.weights if weighting == "cap" else np.ones(n, dtype=np.float64)
    return cast(FloatArray, w / w.sum())


def contributions(
    panel: DailyPanel, *, weighting: Weighting, min_n: int
) -> PanelContributions | None:
    """Exact contributions c_i with sum(c) == WLS slope. None if the panel is degenerate."""
    n = len(panel.returns)
    if n < max(min_n, _MIN_PANEL):
        return None
    w = _normalized_weights(panel, weighting)
    x = panel.vols
    y = panel.returns
    xbar = float(np.sum(w * x))
    d = float(np.sum(w * (x - xbar) ** 2))
    # D is the weighted variance of vol. Below ~8 ulp of the weighted second moment it is
    # floating-point noise, not dispersion: the slope is undefined and leverage explodes.
    scale = float(np.sum(w * x**2))
    if scale <= 0.0 or d <= _D_REL_FLOOR * scale:
        return None
    h = w * (x - xbar) / d
    c = h * y
    return PanelContributions(
        w=w, h=h, c=c, xbar=xbar, ybar=float(np.sum(w * y)), beta=float(c.sum())
    )


def _quadrant(vol: float, xbar: float, ret: float) -> str:
    return ("HI" if vol > xbar else "LO") + ("/+" if ret >= 0.0 else "/-")


def attribute_panel(
    panel: DailyPanel, *, weighting: Weighting, min_n: int
) -> tuple[list[AttributionRow], float]:
    """Per-asset rows (sorted by series key) + beta. ([], NaN) when degenerate."""
    pc = contributions(panel, weighting=weighting, min_n=min_n)
    if pc is None:
        return [], float("nan")
    rows = [
        AttributionRow(
            series=series_key(s),
            block=f"{s.segment}_{s.asset_class}",
            latam=s.country in LATAM_COUNTRIES,
            vol=float(panel.vols[i]),
            ret3m=float(panel.returns[i]),
            weight=float(pc.w[i]),
            leverage=float(pc.h[i]),
            contribution=float(pc.c[i]),
            share=float(pc.c[i] / pc.beta) if pc.beta != 0.0 else float("nan"),
            quadrant=_quadrant(float(panel.vols[i]), pc.xbar, float(panel.returns[i])),
            xbar=pc.xbar,
            ybar=pc.ybar,
        )
        for i, s in enumerate(panel.series)
    ]
    rows.sort(key=lambda r: r.series)
    return rows, pc.beta


def rows_to_frame(rows: list[AttributionRow]) -> pd.DataFrame:
    """Rows -> DataFrame with LEVEL_COLUMNS order (empty frame keeps the columns)."""
    return pd.DataFrame([asdict(r) for r in rows], columns=list(LEVEL_COLUMNS))
