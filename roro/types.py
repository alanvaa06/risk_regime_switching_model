"""Frozen-dataclass contracts that flow through the engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from roro.config import EngineConfig


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
class CorrelationFrame:
    avg_pairwise_3m: pd.DataFrame
    pc1_variance_share: pd.DataFrame


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
    warnings: list[str] = field(default_factory=list)
    data_fingerprint: dict[str, str] = field(default_factory=dict)
    code_version: dict[str, str] = field(default_factory=dict)
