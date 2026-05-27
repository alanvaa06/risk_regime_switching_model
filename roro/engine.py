"""Engine orchestrator: wires the functional pipeline end-to-end."""

from __future__ import annotations

import pandas as pd

from roro.alerts import detect_alerts
from roro.classify import classify
from roro.config import EngineConfig
from roro.correlation import compute_correlation_panel
from roro.fred_client import FredClient
from roro.io import (
    code_version,
    compute_data_fingerprint,
    load_fred,
    load_panel,
    load_prices,
    write_run,
)
from roro.regression import compute_beta_by_segment
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import partition
from roro.tripwire import compute_tripwire_signal
from roro.types import ReturnsFrame, RunResult, ValidationFrame, VolFrame
from roro.validation import (
    compute_internal_consistency,
    compute_rolling_external_corr,
    detect_validation_degradation,
)
from roro.validators import validate_prices, validate_universe


def run(
    cfg: EngineConfig,
    *,
    fred_client: FredClient,
    run_date: str,
    as_of_data_date: str,
    force: bool = False,
) -> RunResult:
    """Execute the full RoRo pipeline and persist outputs under ``cfg.output_dir``.

    Steps: load + validate inputs -> FRED pull -> returns/vol kernels ->
    segment partition -> cross-sectional regression -> classification ->
    correlation panel -> external + internal validation -> tripwire ->
    alerts -> write run.
    """
    warnings: list[str] = []

    # 1) Load inputs and validate.
    universe = load_panel(cfg.data_path)
    warnings.extend(validate_universe(universe))
    prices = load_prices(cfg.data_path)
    warnings.extend(validate_prices(prices))

    # 2) External (FRED) pull aligned to the equity price window.
    eq_index = pd.DatetimeIndex(prices.equity_lc.index)
    fred = load_fred(
        fred_client,
        start=eq_index.min().to_pydatetime().date(),
        end=eq_index.max().to_pydatetime().date(),
    )

    # 3) Returns + EWMA vol kernels (3M window for the main regime engine).
    eq_ret = total_return_3m(prices.equity_lc, window_days=cfg.return_window_days)
    fi_ret = total_return_3m(prices.fi_lc, window_days=cfg.return_window_days)
    eq_daily = daily_log_returns(prices.equity_lc)
    fi_daily = daily_log_returns(prices.fi_lc)
    eq_vol = ewma_vol(eq_daily, halflife=cfg.ewma_halflife_days)
    fi_vol = ewma_vol(fi_daily, halflife=cfg.ewma_halflife_days)

    # 4) Segment partition + cross-sectional regression per cut.
    cuts = partition(universe)
    beta = compute_beta_by_segment(
        dates=eq_index,
        cuts=cuts,
        equity_returns_3m=eq_ret,
        fi_returns_3m=fi_ret,
        equity_vol=eq_vol,
        fi_vol=fi_vol,
        min_n=cfg.min_n_per_cut,
    )

    # 5) Classify percentile/tercile/quintile/direction per segment.
    regime = classify(
        beta,
        bucket_scheme=cfg.bucket_scheme,
        percentile_window_days=cfg.percentile_window_years * 252,
        direction_lookback_days=cfg.direction_lookback_days,
        bootstrap_min_days=cfg.bootstrap_min_days,
        thin_cuts=frozenset({"LatAm"}),
    )

    # 6) Cross-sectional correlation panel (avg pairwise + PC1 variance share).
    correlation = compute_correlation_panel(
        daily_log_returns_eq=eq_daily,
        daily_log_returns_fi=fi_daily,
        cuts=cuts,
        window=cfg.return_window_days,
    )

    # 7) Validation: external (FRED) rolling correlation + alerts.
    rolling_corr = compute_rolling_external_corr(
        regime, fred, window_days=cfg.external_corr_window_days
    )
    corr_alerts = detect_validation_degradation(
        rolling_corr, threshold=cfg.external_corr_alert_threshold
    )
    # Composites are loaded in Universe but not piped as price series yet (v1 limitation).
    # Internal consistency runs against an empty composite frame and returns an empty DataFrame.
    internal = compute_internal_consistency(
        regime=regime,
        composite_eq_prices=pd.DataFrame(),
        composite_fi_prices=pd.DataFrame(),
        return_window_days=cfg.return_window_days,
    )
    validation = ValidationFrame(
        rolling_corr_60d=rolling_corr,
        internal_consistency=internal,
        correlation_alerts=corr_alerts,
    )

    # 8) Tripwire (short-window mirror of the main engine).
    tripwire = compute_tripwire_signal(
        prices=prices,
        cuts=cuts,
        return_window_days=cfg.tripwire_window_days,
        ewma_halflife_days=cfg.tripwire_ewma_halflife_days,
        min_n=cfg.min_n_per_cut,
    )

    # 9) Alerts: bucket transitions, disagreement events, validation degradation.
    alerts = detect_alerts(regime=regime, correlation=correlation, validation=validation)

    # 10) Fingerprint inputs + assemble the RunResult.
    fingerprint = compute_data_fingerprint(cfg.data_path)
    fingerprint["fred_pulled_at"] = fred.pulled_at.isoformat()
    for k, v in fred.series_hashes.items():
        fingerprint[f"fred_{k}"] = v

    result = RunResult(
        config=cfg,
        universe=universe,
        returns=ReturnsFrame(log_returns_3m=eq_ret, daily_log_returns=eq_daily),
        vol=VolFrame(ewma_sigma_annualized=eq_vol),
        beta=beta,
        regime=regime,
        correlation=correlation,
        validation=validation,
        tripwire=tripwire,
        alerts=alerts,
        warnings=warnings,
        data_fingerprint=fingerprint,
        code_version=code_version(),
    )

    write_run(
        result,
        run_date=run_date,
        out_dir=cfg.output_dir,
        as_of_data_date=as_of_data_date,
        force=force,
    )
    return result
