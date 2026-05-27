"""Backtest harness: replay engine over a date range and evaluate PRD §10 gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FredClient
from roro.types import RunResult

# Gate thresholds (PRD §10).
_G1_VIX_RHO_MIN: float = 0.5
_G1_FRACTION_MIN: float = 0.8
_G2_BBB_RHO_MIN: float = 0.4
_G2_FRACTION_MIN: float = 0.8
_G4_SEG_GAP_MIN: int = 2
_G4_SEG_FRACTION_MIN: float = 0.2
_G5_MAX_CALM_TRANSITIONS: float = 2.0
_G6_GAP_BREACH_THRESHOLD: int = 1
_G6_MAX_BREACHES: float = 5.0
_G6_ROLLING_WINDOW: int = 30

# Event window: +/- 6 business days around the reference date.
_EVENT_WINDOW_DAYS: int = 6

# Realized-vol parameters (PRD §10 G5 calm-quarter detection).
_REALIZED_VOL_WINDOW: int = 63
_TRADING_DAYS_PER_YEAR: int = 252

# Tercile -> ordinal mapping (Risk-off=1, Transitional=2, Risk-on=3).
_TERCILE_ORDINAL: dict[str, int] = {
    "Risk-off": 1,
    "Transitional": 2,
    "Risk-on": 3,
    "Unknown": 2,
}


@dataclass(frozen=True)
class Event:
    """A historical risk-off episode used to score gate G3."""

    name: str
    date: str
    expected_buckets: tuple[str, ...] = ("Risk-off",)
    segments: tuple[str, ...] = ("global",)


EVENTS: tuple[Event, ...] = (
    Event(name="2010_greek_crisis", date="2010-05-06"),
    Event(name="2011_eurozone_us_downgrade", date="2011-08-08"),
    Event(name="2015_china_devaluation", date="2015-08-11"),
    Event(name="2018_q4_selloff", date="2018-12-24"),
    Event(name="2020_COVID", date="2020-03-16"),
    Event(name="2022_rate_shock_jan", date="2022-01-24"),
    Event(name="2022_rate_shock_sep", date="2022-09-26"),
    Event(name="2008_lehman", date="2008-10-10"),  # bootstrap calibration period
)


def run_backtest(
    cfg: EngineConfig,
    *,
    fred_client: FredClient,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Replay the engine and score PRD §10 acceptance gates.

    Writes ``acceptance_report.json``, ``event_recognition.csv``,
    ``validation_corr_history.csv``, and ``stability_metrics.csv`` to
    ``cfg.output_dir``. Returns the in-memory report dict.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result = engine_run(
        cfg,
        fred_client=fred_client,
        run_date=end,
        as_of_data_date=end,
        force=True,
    )
    gates = _evaluate_gates(result)
    report: dict[str, Any] = {
        "start": start,
        "end": end,
        "methodology_version": cfg.methodology_version,
        "gates": gates,
        "all_passed": all(bool(v.get("passed", False)) for v in gates.values()),
    }
    (cfg.output_dir / "acceptance_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_event_recognition(cfg.output_dir / "event_recognition.csv")
    _write_validation_history(result, cfg.output_dir / "validation_corr_history.csv")
    _write_stability_metrics(result, cfg.output_dir / "stability_metrics.csv")
    return report


def _evaluate_gates(result: RunResult) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    gates["G1_vix"] = _gate_external_corr(
        result,
        series_id="VIXCLS",
        rho_min=_G1_VIX_RHO_MIN,
        fraction_min=_G1_FRACTION_MIN,
    )
    gates["G2_bbb"] = _gate_external_corr(
        result,
        series_id="BAMLC0A4CBBB",
        rho_min=_G2_BBB_RHO_MIN,
        fraction_min=_G2_FRACTION_MIN,
    )
    gates["G3_events"] = _gate_events(result)
    gates["G4_segmentation_lift"] = _gate_segmentation_lift(result)
    gates["G5_stability"] = _gate_stability(result)
    gates["G6_internal"] = _gate_internal_consistency(result)
    return gates


def _gate_external_corr(
    result: RunResult,
    *,
    series_id: str,
    rho_min: float,
    fraction_min: float,
) -> dict[str, Any]:
    rolling = result.validation.rolling_corr_60d
    key = ("global", series_id)
    if key not in rolling.columns:
        return {"passed": False, "reason": "missing series"}
    rho = rolling[key].abs().dropna()
    if rho.empty:
        return {"passed": False, "fraction_above": 0.0}
    fraction = float((rho >= rho_min).mean())
    return {"passed": bool(fraction >= fraction_min), "fraction_above": fraction}


def _gate_events(result: RunResult) -> dict[str, Any]:
    terc = result.regime.tercile
    hits = 0
    details: list[dict[str, Any]] = []
    for event in EVENTS:
        ts = pd.Timestamp(event.date)
        window_start = ts - pd.Timedelta(days=_EVENT_WINDOW_DAYS)
        window_end = ts + pd.Timedelta(days=_EVENT_WINDOW_DAYS)
        local = terc.loc[(terc.index >= window_start) & (terc.index <= window_end)]
        ok = False
        for seg in event.segments:
            if seg in local.columns and local[seg].isin(event.expected_buckets).any():
                ok = True
                break
        details.append({"event": event.name, "matched": ok})
        if ok:
            hits += 1
    return {
        "passed": hits == len(EVENTS),
        "matched_events": hits,
        "total": len(EVENTS),
        "details": details,
    }


def _gate_segmentation_lift(result: RunResult) -> dict[str, Any]:
    terc = result.regime.tercile
    if not {"DM_Eq", "EM_Eq"}.issubset(terc.columns):
        return {"passed": False, "reason": "missing DM_Eq/EM_Eq"}
    a = terc["DM_Eq"].map(_TERCILE_ORDINAL).astype(float)
    b = terc["EM_Eq"].map(_TERCILE_ORDINAL).astype(float)
    diff = (a - b).abs().dropna()
    if diff.empty:
        return {"passed": False, "fraction_with_gap_ge_2": 0.0}
    fraction = float((diff >= _G4_SEG_GAP_MIN).mean())
    return {"passed": bool(fraction >= _G4_SEG_FRACTION_MIN), "fraction_with_gap_ge_2": fraction}


def _gate_stability(result: RunResult) -> dict[str, Any]:
    transitions = result.alerts.bucket_transitions
    if transitions.empty:
        return {"passed": True, "max_transitions_in_calm_quarter": 0.0}
    daily = result.returns.daily_log_returns
    if daily.empty:
        return {"passed": False, "reason": "missing returns"}
    proxy = daily.iloc[:, 0].dropna()
    if proxy.empty:
        return {"passed": True, "max_transitions_in_calm_quarter": 0.0}
    realized = proxy.rolling(_REALIZED_VOL_WINDOW).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
    by_quarter_vol = realized.groupby(pd.Grouper(freq="QE")).mean().dropna()
    if by_quarter_vol.empty:
        return {"passed": True, "max_transitions_in_calm_quarter": 0.0}
    median_vol = by_quarter_vol.median()
    calm_quarters = by_quarter_vol[by_quarter_vol < median_vol].index
    transitions = transitions.copy()
    transitions["quarter"] = (
        pd.PeriodIndex(transitions["date"], freq="Q").to_timestamp(how="end").normalize()
    )
    calm_index = pd.DatetimeIndex(calm_quarters).normalize()
    global_transitions = transitions[transitions["segment"] == "global"]
    by_quarter = global_transitions.groupby("quarter").size()
    calm_counts = by_quarter.reindex(calm_index, fill_value=0)
    worst = float(calm_counts.max()) if len(calm_counts) else 0.0
    return {
        "passed": bool(worst <= _G5_MAX_CALM_TRANSITIONS),
        "max_transitions_in_calm_quarter": worst,
    }


def _gate_internal_consistency(result: RunResult) -> dict[str, Any]:
    gap = result.validation.internal_consistency
    if "DM" not in gap.columns:
        return {"passed": True, "reason": "no DM mapping data"}
    breaches = (gap["DM"] > _G6_GAP_BREACH_THRESHOLD).astype(int)
    rolling = breaches.rolling(_G6_ROLLING_WINDOW, min_periods=1).sum()
    worst = float(rolling.max()) if len(rolling) else 0.0
    return {
        "passed": bool(worst <= _G6_MAX_BREACHES),
        "max_breaches_in_30d_window": worst,
    }


def _write_event_recognition(path: Path) -> None:
    pd.DataFrame([{"event": e.name, "date": e.date} for e in EVENTS]).to_csv(path, index=False)


def _write_validation_history(result: RunResult, path: Path) -> None:
    df = result.validation.rolling_corr_60d
    if df.empty:
        path.write_text("date,segment,fred_series,rho\n", encoding="utf-8")
        return
    stacked = df.stack(level=[0, 1], future_stack=True)
    out = pd.DataFrame(stacked).reset_index()
    out.columns = pd.Index(["date", "segment", "fred_series", "rho"])
    out.to_csv(path, index=False)


def _write_stability_metrics(result: RunResult, path: Path) -> None:
    transitions = result.alerts.bucket_transitions
    if transitions.empty:
        path.write_text("quarter,segment,transitions\n", encoding="utf-8")
        return
    transitions = transitions.copy()
    transitions["quarter"] = pd.PeriodIndex(transitions["date"], freq="Q").astype(str)
    out = transitions.groupby(["quarter", "segment"]).size().reset_index(name="transitions")
    out.to_csv(path, index=False)
