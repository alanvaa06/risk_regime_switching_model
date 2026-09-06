"""Frozen-dataclass contracts that flow through the engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from roro.config import EngineConfig

#: Column order of ``AlertSet.concentration_alerts`` and of the ``concentration``
#: rows in ``alerts.csv``. Single source of truth, also used by ``roro.alerts``.
CONCENTRATION_ALERT_COLUMNS: tuple[str, ...] = (
    "date", "segment", "weighting", "top1_series", "top1_share", "hhi", "fragile_flag", "trigger",
)


@dataclass(frozen=True)
class PriceFrame:
    equity_lc: pd.DataFrame
    fi_lc: pd.DataFrame


@dataclass(frozen=True)
class Universe:
    countries: pd.DataFrame
    composites: pd.DataFrame
    latam_countries: tuple[str, ...] = ("Brazil", "Mexico", "Chile", "Peru", "Colombia")


@dataclass(frozen=True)
class FredFrame:
    series: dict[str, pd.Series]
    pulled_at: datetime
    series_hashes: dict[str, str]


@dataclass(frozen=True)
class ReturnsFrame:
    log_returns_3m: pd.DataFrame
    daily_log_returns: pd.DataFrame


@dataclass(frozen=True)
class VolFrame:
    ewma_sigma_annualized: pd.DataFrame


@dataclass(frozen=True)
class BetaFrame:
    cap_wtd: pd.DataFrame
    eq_wtd: pd.DataFrame
    slope_spread: pd.Series


@dataclass(frozen=True)
class BetaBySegment:
    by_segment: dict[str, BetaFrame]


@dataclass(frozen=True)
class RegimeFrame:
    percentile_5y: pd.DataFrame
    tercile: pd.DataFrame
    quintile: pd.DataFrame
    direction: pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    bootstrap_flag: pd.DataFrame


@dataclass(frozen=True)
class HmmRegimeFrame:
    state: pd.DataFrame
    label: pd.DataFrame
    prob_risk_off: pd.DataFrame
    prob_transitional: pd.DataFrame
    prob_risk_on: pd.DataFrame
    confidence: pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    cold_start_flag: pd.DataFrame
    refit_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)


@dataclass(frozen=True)
class JmRegimeFrame:
    state: pd.DataFrame
    label: pd.DataFrame
    prob_risk_off: pd.DataFrame
    prob_transitional: pd.DataFrame
    prob_risk_on: pd.DataFrame
    confidence: pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    cold_start_flag: pd.DataFrame
    refit_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    jump_penalty_used: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrelationFrame:
    avg_pairwise_3m: pd.DataFrame
    pc1_variance_share: pd.DataFrame


@dataclass(frozen=True)
class AttributionFrame:
    """Per-asset attribution of the regime slope (see roro/attribution.py).

    level / delta / rollup / pc1 are LAST-DATE snapshots (long format, all cuts x
    weightings); concentration is FULL HISTORY; history_global is the optional
    date x series matrix of contributions for the global cut (cap-weighted).
    """

    level: pd.DataFrame
    delta: pd.DataFrame
    rollup: pd.DataFrame
    concentration: pd.DataFrame
    pc1: pd.DataFrame
    anchors: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    history_global: pd.DataFrame | None = None


@dataclass(frozen=True)
class ValidationFrame:
    rolling_corr_60d: pd.DataFrame
    internal_consistency: pd.DataFrame
    correlation_alerts: pd.DataFrame


@dataclass(frozen=True)
class AlertSet:
    bucket_transitions: pd.DataFrame
    disagreement_events: pd.DataFrame
    validation_degradation: pd.DataFrame
    hmm_bucket_transitions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "segment", "from_bucket", "to_bucket"]
        )
    )
    jm_bucket_transitions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "segment", "from_bucket", "to_bucket"]
        )
    )
    concentration_alerts: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(CONCENTRATION_ALERT_COLUMNS))
    )


@dataclass(frozen=True)
class RunResult:
    config: EngineConfig
    universe: Universe
    returns: ReturnsFrame
    vol: VolFrame
    beta: BetaBySegment
    regime: RegimeFrame
    correlation: CorrelationFrame
    validation: ValidationFrame
    tripwire: BetaBySegment
    alerts: AlertSet
    regime_hmm: HmmRegimeFrame | None = None
    regime_jm: JmRegimeFrame | None = None
    attribution: AttributionFrame | None = None
    warnings: list[str] = field(default_factory=list)
    data_fingerprint: dict[str, str] = field(default_factory=dict)
    code_version: dict[str, str] = field(default_factory=dict)
