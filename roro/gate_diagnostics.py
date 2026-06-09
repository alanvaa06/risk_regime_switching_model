"""Read-only diagnostic harness for the S9 acceptance gates.

Recomputes G1-G6 over a backtested RunResult, sweeps thresholds, repairs the
G3 out-of-range bug, and traces the causal G3<->G5 frontier. Reuses production
primitives verbatim so the evidence cannot drift from the real gates. Writes
no production artifacts and mutates nothing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roro.alerts import _bucket_transitions
from roro.backtest import (
    _TERCILE_ORDINAL,
    Event,
    _calm_quarters,
    _count_max_calm_transitions,
    _event_hits,
)
from roro.report.figures import _smooth_regime_hysteresis

_SHARED_EPS = 1e-9
_BORDERLINE_REL = 0.15


def sweep_external_corr(
    rolling_corr_60d: pd.DataFrame,
    *,
    series_id: str,
    rho_grid: list[float],
) -> pd.DataFrame:
    """Fraction of days with |rolling corr| >= rho_min, as rho_min varies."""
    key = ("global", series_id)
    rows: list[dict[str, float]] = []
    if key in rolling_corr_60d.columns:
        rho = rolling_corr_60d[key].abs().dropna()
    else:
        rho = pd.Series(dtype=float)
    for rho_min in rho_grid:
        frac = float((rho >= rho_min).mean()) if not rho.empty else 0.0
        rows.append({"rho_min": float(rho_min), "fraction_above": frac})
    return pd.DataFrame(rows)


def sweep_segmentation_lift(
    terc: pd.DataFrame,
    *,
    gap_grid: list[int],
) -> pd.DataFrame:
    """Fraction of days with |DM_Eq - EM_Eq| ordinal gap >= g, as g varies."""
    rows: list[dict[str, float]] = []
    if {"DM_Eq", "EM_Eq"}.issubset(terc.columns):
        a = terc["DM_Eq"].map(_TERCILE_ORDINAL).astype(float)
        b = terc["EM_Eq"].map(_TERCILE_ORDINAL).astype(float)
        diff = (a - b).abs().dropna()
    else:
        diff = pd.Series(dtype=float)
    for gap in gap_grid:
        frac = float((diff >= gap).mean()) if not diff.empty else 0.0
        rows.append({"gap": int(gap), "fraction_with_gap_ge": frac})
    return pd.DataFrame(rows)


def classify_gate(
    *,
    passed: bool,
    percentile_value: float,
    hmm_value: float,
    value: float,
    threshold: float,
    vacuous_reason: str | None,
    out_of_range: list[str],
) -> str:
    """Tag a gate: vacuous | bug | shared | borderline | real.

    Order matters: a vacuous pass outranks everything; an out-of-range event is a
    scorecard bug; identical percentile/hmm values mean the gate cannot
    discriminate method (shared); a near-miss is borderline; else a real failure.
    """
    if vacuous_reason is not None:
        return "vacuous"
    if out_of_range:
        return "bug"
    if abs(percentile_value - hmm_value) <= _SHARED_EPS:
        return "shared"
    if not passed and threshold != 0 and abs(value - threshold) / abs(threshold) <= _BORDERLINE_REL:
        return "borderline"
    return "real"


def repair_g3(
    labels: pd.DataFrame,
    events: tuple[Event, ...],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """G3 with out-of-range events excluded; graded k/total_in_range."""
    hits, total_in_range, out_of_range = _event_hits(labels, events, start=start, end=end)
    graded = float(hits / total_in_range) if total_in_range else 0.0
    return {
        "hits": hits,
        "total_in_range": total_in_range,
        "out_of_range": out_of_range,
        "graded": graded,
        "passed_in_range": bool(total_in_range > 0 and hits == total_in_range),
    }


def g3_g5_frontier(
    terc: pd.DataFrame,
    daily_log_returns: pd.DataFrame,
    *,
    events: tuple[Event, ...],
    start: str,
    end: str,
    confirm_grid: list[int],
    segment: str = "global",
) -> pd.DataFrame:
    """Trace (events caught, worst calm-quarter transitions) vs confirm-days.

    n == 0 is the raw (unsmoothed) percentile labels; n > 0 applies the causal
    hysteresis smoother. Transition counting and calm-quarter detection reuse
    production primitives verbatim.
    """
    labels = terc[segment]
    calm = _calm_quarters(daily_log_returns)
    rows: list[dict[str, float]] = []
    for n in confirm_grid:
        smoothed = labels if n == 0 else _smooth_regime_hysteresis(labels, n)
        smoothed_df = smoothed.to_frame(name=segment)
        transitions = _bucket_transitions(smoothed_df)
        worst = _count_max_calm_transitions(transitions, calm, segment=segment)
        hits, _total, _oor = _event_hits(smoothed_df, events, start=start, end=end)
        rows.append(
            {
                "confirm_days": int(n),
                "events_caught": int(hits),
                "max_calm_transitions": worst,
            }
        )
    return pd.DataFrame(rows)
