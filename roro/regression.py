"""Daily cross-sectional WLS regression of 3M return on EWMA vol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SeriesId

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _cell(df: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    """Read a scalar cell from a DataFrame; returns NaN if the column is missing."""
    if col not in df.columns:
        return float("nan")
    return float(cast(Any, df.at[date, col]))


@dataclass(frozen=True)
class DailyPanel:
    date: pd.Timestamp
    series: tuple[SeriesId, ...]
    returns: FloatArray  # shape (n,)
    vols: FloatArray  # shape (n,)
    weights: FloatArray  # shape (n,), unnormalized cap weights


@dataclass(frozen=True)
class CrossSectionResult:
    date: pd.Timestamp
    beta_cap: float
    beta_eq: float
    r2_cap: float
    r2_eq: float
    slope_spread: float
    n: int
    suppressed: bool
    singular: bool


def daily_panel(
    date: pd.Timestamp,
    series: list[SeriesId],
    *,
    equity_returns: pd.DataFrame,
    fi_returns: pd.DataFrame,
    equity_vol: pd.DataFrame,
    fi_vol: pd.DataFrame,
) -> DailyPanel:
    """Assemble the per-date cross-section panel from per-segment frames."""
    rets: list[float] = []
    vols: list[float] = []
    ws: list[float] = []
    used: list[SeriesId] = []
    for s in series:
        if s.asset_class == ASSET_EQ:
            r = _cell(equity_returns, date, s.country)
            v = _cell(equity_vol, date, s.country)
        elif s.asset_class == ASSET_FI:
            r = _cell(fi_returns, date, s.country)
            v = _cell(fi_vol, date, s.country)
        else:  # pragma: no cover - guarded by SeriesId construction
            raise ValueError(f"Unknown asset_class: {s.asset_class}")
        if not (np.isfinite(r) and np.isfinite(v)):
            continue
        rets.append(r)
        vols.append(v)
        ws.append(s.mcap)
        used.append(s)
    return DailyPanel(
        date=date,
        series=tuple(used),
        returns=np.asarray(rets, dtype=np.float64),
        vols=np.asarray(vols, dtype=np.float64),
        weights=np.asarray(ws, dtype=np.float64),
    )


def cross_section(panel: DailyPanel, *, min_n: int) -> CrossSectionResult:
    """WLS cap-weighted + OLS equal-weighted slope of returns on vols."""
    n = len(panel.returns)
    if n < min_n:
        return CrossSectionResult(
            date=panel.date,
            beta_cap=np.nan,
            beta_eq=np.nan,
            r2_cap=np.nan,
            r2_eq=np.nan,
            slope_spread=np.nan,
            n=n,
            suppressed=True,
            singular=False,
        )
    try:
        beta_cap, r2_cap = _wls_with_r2(panel.vols, panel.returns, panel.weights)
        beta_eq, r2_eq = _wls_with_r2(panel.vols, panel.returns, np.ones(n))
        singular = False
    except np.linalg.LinAlgError:
        return CrossSectionResult(
            date=panel.date,
            beta_cap=np.nan,
            beta_eq=np.nan,
            r2_cap=np.nan,
            r2_eq=np.nan,
            slope_spread=np.nan,
            n=n,
            suppressed=False,
            singular=True,
        )
    return CrossSectionResult(
        date=panel.date,
        beta_cap=beta_cap,
        beta_eq=beta_eq,
        r2_cap=r2_cap,
        r2_eq=r2_eq,
        slope_spread=beta_cap - beta_eq,
        n=n,
        suppressed=False,
        singular=singular,
    )


def _wls_slope(x: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """Return the WLS slope coefficient for y on x with weights w (intercept included)."""
    w_norm = w / w.sum()
    X = np.column_stack([np.ones_like(x), x])  # noqa: N806 - design matrix convention
    W = np.diag(w_norm)  # noqa: N806 - weight matrix convention
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(theta[1])


def _wls_with_r2(x: FloatArray, y: FloatArray, w: FloatArray) -> tuple[float, float]:
    """Return (slope, weighted-R^2) for y on x with weights w (intercept included)."""
    w_norm = w / w.sum()
    X = np.column_stack([np.ones_like(x), x])  # noqa: N806 - design matrix convention
    W = np.diag(w_norm)  # noqa: N806 - weight matrix convention
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    y_hat = X @ theta
    ss_res = float(np.sum(w_norm * (y - y_hat) ** 2))
    y_bar = float(np.sum(w_norm * y))
    ss_tot = float(np.sum(w_norm * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(theta[1]), r2
