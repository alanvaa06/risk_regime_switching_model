"""External (FRED) rolling correlation + internal composite-consistency check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.types import FredFrame, RegimeFrame

# Engine-segment -> (equity composite ticker, FI composite ticker). None = no pairing.
COMPOSITE_MAPPING: dict[str, tuple[str | None, str | None]] = {
    "global": ("MXWD", "LEGATRUU"),
    "DM": ("MXWO", "I35402US"),
    "EM": ("MXEF", "EMUSTRUU"),
    "Equity": ("MXWD", None),
    "FI": (None, "LEGATRUU"),
    "DM_Eq": ("MXWO", None),
    "EM_Eq": ("MXEF", None),
    "DM_FI": (None, "I35402US"),
    "EM_FI": (None, "EMUSTRUU"),
    "LatAm": ("MXLA", "H04338US"),
}

_TERC_LOW: float = 1.0 / 3.0
_TERC_HIGH: float = 2.0 / 3.0
_TERC_RISK_OFF: int = 1
_TERC_TRANSITIONAL: int = 2
_TERC_RISK_ON: int = 3
_FFILL_LIMIT: int = 2


def compute_rolling_external_corr(
    regime: RegimeFrame,
    fred: FredFrame,
    *,
    window_days: int,
) -> pd.DataFrame:
    """Rolling Pearson rho between each (segment percentile, FRED series).

    Returns a wide DataFrame whose columns are MultiIndex tuples of
    ``(segment, fred_series_id)``.
    """
    out: dict[tuple[str, str], pd.Series] = {}
    for segment in regime.percentile_5y.columns:
        seg_series = regime.percentile_5y[segment]
        for sid, fred_series in fred.series.items():
            aligned = pd.concat([seg_series, fred_series.rename(sid)], axis=1).ffill(
                limit=_FFILL_LIMIT
            )
            rho = aligned[segment].rolling(window=window_days, min_periods=window_days).corr(
                aligned[sid]
            )
            out[(segment, sid)] = rho
    return pd.DataFrame(out)


def detect_validation_degradation(
    rolling_corr: pd.DataFrame, *, threshold: float
) -> pd.DataFrame:
    """Stack the rolling-corr panel and flag entries with |rho| below ``threshold``."""
    below = rolling_corr.abs() < threshold
    stacked = below.stack(level=[0, 1], future_stack=True)
    series = pd.Series(stacked, name="below_threshold")
    return series.to_frame()


def compute_internal_consistency(
    *,
    regime: RegimeFrame,
    composite_eq_prices: pd.DataFrame,
    composite_fi_prices: pd.DataFrame,
    return_window_days: int,
) -> pd.DataFrame:
    """Engine-vs-composite tercile gap for each mapped segment.

    For each segment with a mapped composite, compare engine tercile to the
    composite's own rolling-return percentile tercile and emit the integer
    tercile gap (0 = aligned, 2 = inverted).
    """
    eq_returns = pd.DataFrame(
        np.log(composite_eq_prices / composite_eq_prices.shift(return_window_days)),
        index=composite_eq_prices.index,
        columns=composite_eq_prices.columns,
    )
    fi_returns = pd.DataFrame(
        np.log(composite_fi_prices / composite_fi_prices.shift(return_window_days)),
        index=composite_fi_prices.index,
        columns=composite_fi_prices.columns,
    )

    rows: dict[str, pd.Series] = {}
    for segment, (eq_ticker, fi_ticker) in COMPOSITE_MAPPING.items():
        if segment not in regime.tercile.columns:
            continue
        composite_pct = _blend_composite_returns(eq_returns, fi_returns, eq_ticker, fi_ticker)
        if composite_pct is None:
            continue
        comp_terc = composite_pct.map(_terc)
        engine_terc = regime.tercile[segment]
        gap = (_terc_to_int(engine_terc) - _terc_to_int(comp_terc)).abs()
        rows[segment] = gap
    return pd.DataFrame(rows)


def _blend_composite_returns(
    eq_returns: pd.DataFrame,
    fi_returns: pd.DataFrame,
    eq_ticker: str | None,
    fi_ticker: str | None,
) -> pd.Series | None:
    if eq_ticker and fi_ticker:
        eq = eq_returns[eq_ticker] if eq_ticker in eq_returns.columns else None
        fi = fi_returns[fi_ticker] if fi_ticker in fi_returns.columns else None
        if eq is None and fi is None:
            return None
        blended = pd.concat([eq, fi], axis=1).mean(axis=1)
    elif eq_ticker:
        if eq_ticker not in eq_returns.columns:
            return None
        blended = eq_returns[eq_ticker]
    elif fi_ticker:
        if fi_ticker not in fi_returns.columns:
            return None
        blended = fi_returns[fi_ticker]
    else:
        return None
    return blended.rank(pct=True)


def _terc(p: float) -> str:
    if pd.isna(p):
        return "Unknown"
    if p <= _TERC_LOW:
        return "Risk-off"
    if p >= _TERC_HIGH:
        return "Risk-on"
    return "Transitional"


def _terc_to_int(s: pd.Series) -> pd.Series:
    mapping = {
        "Risk-off": _TERC_RISK_OFF,
        "Transitional": _TERC_TRANSITIONAL,
        "Risk-on": _TERC_RISK_ON,
        "Unknown": _TERC_TRANSITIONAL,
    }
    return s.map(mapping).astype(float)
