"""Alert detection: bucket transitions, disagreement events, validation degradation."""

from __future__ import annotations

import pandas as pd

from roro.types import (
    CONCENTRATION_ALERT_COLUMNS,
    AlertSet,
    AttributionFrame,
    CorrelationFrame,
    HmmRegimeFrame,
    JmRegimeFrame,
    RegimeFrame,
    ValidationFrame,
)

_DISAGREEMENT_CORR_THRESHOLD: float = 0.6
_DEFAULT_TOP1_ALERT: float = 0.5


def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
    regime_hmm: HmmRegimeFrame | None = None,
    regime_jm: JmRegimeFrame | None = None,
    attribution: AttributionFrame | None = None,
    top1_alert: float = _DEFAULT_TOP1_ALERT,
) -> AlertSet:
    transitions = _bucket_transitions(regime.tercile)
    return AlertSet(
        bucket_transitions=transitions,
        disagreement_events=_disagreement_events(regime, correlation),
        validation_degradation=_validation_degradation(validation),
        hmm_bucket_transitions=(
            _bucket_transitions(regime_hmm.label)
            if regime_hmm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
        jm_bucket_transitions=(
            _bucket_transitions(regime_jm.label)
            if regime_jm is not None
            else pd.DataFrame(columns=["date", "segment", "from_bucket", "to_bucket"])
        ),
        concentration_alerts=(
            _concentration_alerts(attribution.concentration, transitions, top1_alert=top1_alert)
            if attribution is not None
            else pd.DataFrame(columns=list(CONCENTRATION_ALERT_COLUMNS))
        ),
    )


def _concentration_alerts(
    conc: pd.DataFrame, transitions: pd.DataFrame, *, top1_alert: float
) -> pd.DataFrame:
    """Cap-weighted rows where (top1_share > threshold on a bucket-transition day) or fragile."""
    if conc.empty:
        return pd.DataFrame(columns=list(CONCENTRATION_ALERT_COLUMNS))
    cap = conc[conc["weighting"] == "cap"]
    if transitions.empty:
        on_transition = pd.Series(False, index=cap.index)
    else:
        keys = set(zip(transitions["date"], transitions["segment"], strict=True))
        on_transition = pd.Series(
            [(d, s) in keys for d, s in zip(cap["date"], cap["cut"], strict=True)],
            index=cap.index,
        )
    concentrated = cap["top1_share"] > top1_alert
    fragile = cap["fragile_flag"].astype(bool)
    hit = cap[(on_transition & concentrated) | fragile].copy()
    if hit.empty:
        return pd.DataFrame(columns=list(CONCENTRATION_ALERT_COLUMNS))
    trig_transition = (on_transition & concentrated).loc[hit.index]
    hit["trigger"] = ["transition_day" if t else "fragile" for t in trig_transition]
    hit = hit.rename(columns={"cut": "segment"})
    return hit[list(CONCENTRATION_ALERT_COLUMNS)].sort_values(
        ["date", "segment"], kind="stable"
    ).reset_index(drop=True)


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
