"""Exact per-asset attribution of the cross-sectional regime slope.

The WLS slope of 3M return on EWMA vol (roro/regression.py) is linear in returns:

    beta = sum_i c_i,   c_i = h_i * y_i,   h_i = w_i (x_i - xbar) / D,
    xbar = sum_i w_i x_i,   D = sum_i w_i (x_i - xbar)^2,   sum_i w_i = 1.

Because sum_i w_i (x_i - xbar) = 0 the intercept drops out, so the decomposition
is exact, additive and residual-free. Everything here consumes the *identical*
DailyPanel the slope is fitted on; regression.py is never modified.
"""

from __future__ import annotations

import warnings
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


DELTA_COLUMNS: tuple[str, ...] = (
    "series", "block", "effect_return", "effect_position",
    "effect_interaction", "effect_universe", "delta_total",
)


def attribute_delta(
    rows_t: list[AttributionRow],
    rows_a: list[AttributionRow],
    *,
    beta_t: float,
    beta_a: float,
) -> pd.DataFrame:
    """Brinson-style split of beta_t - beta_a per asset (exact; sums to the delta).

    Common assets: return effect h_a*dy + position effect dh*y_a + interaction dh*dy.
    Entries (only at t) / exits (only at anchor): whole contribution in `effect_universe`.
    Empty frame when either beta is NaN (suppressed panel).
    """
    if not (np.isfinite(beta_t) and np.isfinite(beta_a)):
        return pd.DataFrame(columns=list(DELTA_COLUMNS))
    by_t = {r.series: r for r in rows_t}
    by_a = {r.series: r for r in rows_a}
    out: list[dict[str, object]] = []
    for s in sorted(set(by_t) | set(by_a)):
        rt, ra = by_t.get(s), by_a.get(s)
        eff_ret = eff_pos = eff_int = eff_uni = 0.0
        if rt is not None and ra is not None:
            dh = rt.leverage - ra.leverage
            dy = rt.ret3m - ra.ret3m
            eff_ret = ra.leverage * dy
            eff_pos = dh * ra.ret3m
            eff_int = dh * dy
            block = rt.block
        elif rt is not None:
            eff_uni = rt.contribution
            block = rt.block
        else:
            assert ra is not None
            eff_uni = -ra.contribution
            block = ra.block
        out.append(
            {
                "series": s, "block": block,
                "effect_return": eff_ret, "effect_position": eff_pos,
                "effect_interaction": eff_int, "effect_universe": eff_uni,
                "delta_total": eff_ret + eff_pos + eff_int + eff_uni,
            }
        )
    return pd.DataFrame(out, columns=list(DELTA_COLUMNS))


_TOP_N: int = 5


@dataclass(frozen=True)
class ConcentrationRow:
    hhi: float
    top1_series: str
    top1_share: float
    top5_share: float
    beta_ex_top1: float


_EMPTY_CONCENTRATION = ConcentrationRow(
    hhi=float("nan"), top1_series="", top1_share=float("nan"),
    top5_share=float("nan"), beta_ex_top1=float("nan"),
)


def _panel_without(panel: DailyPanel, index: int) -> DailyPanel:
    keep = [i for i in range(len(panel.series)) if i != index]
    return DailyPanel(
        date=panel.date,
        series=tuple(panel.series[i] for i in keep),
        returns=panel.returns[keep],
        vols=panel.vols[keep],
        weights=panel.weights[keep],
    )


def concentration(
    panel: DailyPanel, pc: PanelContributions, *, weighting: Weighting, min_n: int
) -> ConcentrationRow:
    """HHI of |c|, top-1/top-5 shares, and the slope re-estimated without the top-1 asset.

    Ties in |c| resolve by panel order (stable argsort) -> deterministic.
    """
    abs_c = np.abs(pc.c)
    total = float(abs_c.sum())
    if total <= 0.0:
        return _EMPTY_CONCENTRATION
    a = abs_c / total
    order = np.argsort(-abs_c, kind="stable")
    top1 = int(order[0])
    ex = contributions(_panel_without(panel, top1), weighting=weighting, min_n=min_n)
    return ConcentrationRow(
        hhi=float(np.sum(a**2)),
        top1_series=series_key(panel.series[top1]),
        top1_share=float(a[top1]),
        top5_share=float(a[order[:_TOP_N]].sum()),
        beta_ex_top1=ex.beta if ex is not None else float("nan"),
    )


def rank_against_prior(window: FloatArray, value: float) -> float:
    """Percentile of `value` against the window's prior values (window[-1] is today).

    Mirrors classify.rolling_percentile: count(prior <= value) / (len(window) - 1),
    NaN comparisons count as False. NaN when there is no prior history.
    Differs from rolling_percentile only when there is no prior history: NaN here, 0.0 there.
    """
    prior = window[:-1]
    if prior.size == 0:
        return float("nan")
    hits = int(np.sum(prior <= value))
    return float(hits) / float(prior.size)


PC1_COLUMNS: tuple[str, ...] = (
    "series", "pc1_load_sq", "var_share", "decoupling", "row_mean_corr",
)
_MIN_OBS_PC1: int = 3
_MIN_COLS_PC1: int = 2


def pc1_loadings(window_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset PC1 loading^2 (sums to 1), variance share (sums to 1), decoupling, row-mean corr.

    Same window/covariance as roro/correlation.py. Columns that are entirely NaN are
    dropped; any remaining NaN in the returns makes the covariance undefined -> empty
    frame. A zero-variance asset keeps its loading/variance share but has NaN
    row_mean_corr. Loadings assume a well-separated top eigenvalue; under near-ties the
    PC1 eigenvector is not unique.
    """
    empty = pd.DataFrame(columns=list(PC1_COLUMNS))
    arr = window_returns.to_numpy(dtype=np.float64)
    mask = ~np.all(np.isnan(arr), axis=0)
    arr = arr[:, mask]
    cols = [c for c, m in zip(window_returns.columns, mask, strict=True) if m]
    if arr.shape[0] < _MIN_OBS_PC1 or arr.shape[1] < _MIN_COLS_PC1:
        return empty
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        cov = np.cov(arr, rowvar=False)
        corr = np.corrcoef(arr, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return empty
    trace = float(np.trace(cov))
    if trace <= 0.0:
        return empty
    _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues; last column = PC1
    v1 = eigvecs[:, -1]
    load_sq = v1**2
    var_share = np.diag(cov) / trace
    off_diag = corr.copy()
    np.fill_diagonal(off_diag, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN mean
        row_mean = np.nanmean(off_diag, axis=1)
    df = pd.DataFrame(
        {
            "series": [str(c) for c in cols],
            "pc1_load_sq": load_sq,
            "var_share": var_share,
            "decoupling": var_share - load_sq,
            "row_mean_corr": row_mean,
        },
        columns=list(PC1_COLUMNS),
    )
    return df.sort_values("series").reset_index(drop=True)


_UNKNOWN_LABEL: str = "Unknown"


def find_anchor(
    labels: pd.Series, t: pd.Timestamp, *, max_lookback_days: int
) -> pd.Timestamp | None:
    """Anchor for the delta waterfall: the day BEFORE the most recent label transition <= t.

    Uses only rows <= t (causal). Transitions from NaN/'Unknown' do not count. The
    transition must be at most `max_lookback_days` rows before `t` (the row
    `max_lookback_days` back counts).
    """
    hist = labels.loc[:t]
    known = hist.notna() & (hist != _UNKNOWN_LABEL)
    prev = hist.shift(1)
    prev_known = prev.notna() & (prev != _UNKNOWN_LABEL)
    is_transition = known & prev_known & (hist != prev)
    if not bool(is_transition.any()):
        return None
    pos = int(np.flatnonzero(is_transition.to_numpy())[-1])
    if (len(hist) - 1 - pos) > max_lookback_days:
        return None
    return pd.Timestamp(hist.index[pos - 1])
