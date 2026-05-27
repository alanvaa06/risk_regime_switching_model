"""Alert detection: bucket transitions, disagreement events, validation degradation."""

from __future__ import annotations

import pandas as pd

from roro.types import AlertSet, CorrelationFrame, RegimeFrame, ValidationFrame

_DISAGREEMENT_CORR_THRESHOLD: float = 0.6


def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
) -> AlertSet:
    return AlertSet(
        bucket_transitions=_bucket_transitions(regime.tercile),
        disagreement_events=_disagreement_events(regime, correlation),
        validation_degradation=_validation_degradation(validation),
    )


def _bucket_transitions(tercile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in tercile.columns:
        s = tercile[segment]
        prev = s.shift(1)
        mask = (prev.notna()) & (s != prev)
        idx = s.index[mask]
        for ts, from_b, to_b in zip(idx, prev[mask].to_numpy(), s[mask].to_numpy(), strict=True):
            rows.append(
                {
                    "date": ts,
                    "segment": segment,
                    "from_bucket": from_b,
                    "to_bucket": to_b,
                }
            )
    return pd.DataFrame(rows, columns=["date", "segment", "from_bucket", "to_bucket"])


def _disagreement_events(
    regime: RegimeFrame, correlation: CorrelationFrame
) -> pd.DataFrame:
    if correlation.avg_pairwise_3m.empty or regime.tercile.empty:
        return pd.DataFrame(columns=["date", "segment", "vol_slope_bucket", "avg_pairwise"])
    rows: list[dict[str, object]] = []
    for segment in regime.tercile.columns:
        if segment not in correlation.avg_pairwise_3m.columns:
            continue
        terc = regime.tercile[segment]
        corr = correlation.avg_pairwise_3m[segment]
        mask = (terc == "Risk-on") & (corr > _DISAGREEMENT_CORR_THRESHOLD)
        idx = terc.index[mask]
        for ts, val in zip(idx, corr[mask].to_numpy(), strict=True):
            rows.append(
                {
                    "date": ts,
                    "segment": segment,
                    "vol_slope_bucket": "Risk-on",
                    "avg_pairwise": float(val),
                }
            )
    return pd.DataFrame(rows, columns=["date", "segment", "vol_slope_bucket", "avg_pairwise"])


def _validation_degradation(validation: ValidationFrame) -> pd.DataFrame:
    if validation.correlation_alerts.empty:
        return pd.DataFrame(columns=["date", "segment", "fred_series"])
    df = validation.correlation_alerts
    df = df[df["below_threshold"]].copy()
    if "level_0" in df.columns:
        df = df.rename(columns={"level_0": "date"})
    df = df.reset_index()
    rename_map = {"level_1": "segment", "level_2": "fred_series"}
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    keep = [c for c in ("date", "segment", "fred_series") if c in df.columns]
    return df[keep]
