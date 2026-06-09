"""Read-only diagnostic harness for the S9 acceptance gates.

Recomputes G1-G6 over a backtested RunResult, sweeps thresholds, repairs the
G3 out-of-range bug, and traces the causal G3<->G5 frontier. Reuses production
primitives verbatim so the evidence cannot drift from the real gates. Writes
no production artifacts and mutates nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from roro.alerts import _bucket_transitions
from roro.backtest import (
    _TERCILE_ORDINAL,
    EVENTS,
    Event,
    _calm_quarters,
    _count_max_calm_transitions,
    _evaluate_gates,
    _event_hits,
)
from roro.report.figures import _smooth_regime_hysteresis
from roro.types import RunResult

_SHARED_EPS = 1e-9
_BORDERLINE_REL = 0.15
_RHO_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_GAP_GRID = [1, 2]
_CONFIRM_GRID = [0, 1, 2, 3, 5, 8, 13, 21, 34]


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


def diagnose(
    result: RunResult,
    *,
    start: str,
    end: str,
    baseline_compare_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute gates, sweep, repair G3, trace the frontier, classify each gate."""
    terc = result.regime.tercile
    hmm_label = result.regime_hmm.label if result.regime_hmm is not None else terc
    perc_gates = _evaluate_gates(
        result, labels=terc, transitions=result.alerts.bucket_transitions
    )
    hmm_gates = _evaluate_gates(
        result,
        labels=hmm_label,
        transitions=(
            result.alerts.hmm_bucket_transitions
            if result.regime_hmm is not None
            else result.alerts.bucket_transitions
        ),
    )

    g3 = repair_g3(terc, EVENTS, start=start, end=end)

    def _num(d: dict[str, Any]) -> float:
        for k in (
            "fraction_above",
            "fraction_with_gap_ge_2",
            "max_transitions_in_calm_quarter",
            "matched_events",
        ):
            if k in d:
                return float(d[k])
        return 0.0

    thresholds: dict[str, float] = {
        "G1_vix": 0.8,
        "G2_bbb": 0.8,
        "G3_events": float(len(EVENTS)),
        "G4_segmentation_lift": 0.2,
        "G5_stability": 2.0,
        "G6_internal": 5.0,
    }
    gates: dict[str, dict[str, Any]] = {}
    for name, pg in perc_gates.items():
        hg = hmm_gates[name]
        oor = g3["out_of_range"] if name == "G3_events" else []
        tag = classify_gate(
            passed=bool(pg.get("passed", False)),
            percentile_value=_num(pg),
            hmm_value=_num(hg),
            value=_num(pg),
            threshold=thresholds[name],
            vacuous_reason=pg.get("reason"),
            out_of_range=oor,
        )
        gates[name] = {
            "percentile": pg,
            "hmm": hg,
            "threshold": thresholds[name],
            "root_cause": tag,
            "shared": abs(_num(pg) - _num(hg)) <= _SHARED_EPS,
        }

    sweeps = {
        "G1_vix": sweep_external_corr(
            result.validation.rolling_corr_60d,
            series_id="VIXCLS",
            rho_grid=_RHO_GRID,
        ),
        "G2_bbb": sweep_external_corr(
            result.validation.rolling_corr_60d,
            series_id="BAMLC0A4CBBB",
            rho_grid=_RHO_GRID,
        ),
        "G4_segmentation_lift": sweep_segmentation_lift(terc, gap_grid=_GAP_GRID),
    }
    frontier = g3_g5_frontier(
        terc,
        result.returns.daily_log_returns,
        events=EVENTS,
        start=start,
        end=end,
        confirm_grid=_CONFIRM_GRID,
    )

    baseline_pinned: bool | None = None
    if baseline_compare_path is not None and baseline_compare_path.exists():
        ref: dict[str, Any] = json.loads(
            baseline_compare_path.read_text(encoding="utf-8")
        )
        baseline_pinned = all(
            ref[g]["percentile"].get("passed") == perc_gates[g].get("passed")
            for g in perc_gates
            if g in ref
        )

    return {
        "start": start,
        "end": end,
        "gates": gates,
        "g3_repair": g3,
        "sweeps": sweeps,
        "frontier": frontier,
        "baseline_pinned": baseline_pinned,
    }


def write_diagnostics(out_dir: Path, diag: dict[str, Any]) -> None:
    """Emit gate_diagnostics.json + g3_g5_frontier.csv (deterministic, no timestamps)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {
        "start": diag["start"],
        "end": diag["end"],
        "baseline_pinned": diag["baseline_pinned"],
        "gates": diag["gates"],
        "g3_repair": diag["g3_repair"],
        "sweeps": {
            k: v.to_dict(orient="records") for k, v in diag["sweeps"].items()
        },
    }
    (out_dir / "gate_diagnostics.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    diag["frontier"].to_csv(out_dir / "g3_g5_frontier.csv", index=False)
