# Memory

- decision: static cap weights (Panel snapshot 2026-05-26) applied across full history; time-varying deferred to v1.1.
- decision: composite rows (Panel 32–37) used only by `roro.validation`, never by regressions.
- decision: engine writes CSV + snapshot.json; no Postgres, no Parquet, no dashboard in this scope.
- decision: bucket schemes available are TERCILE (default), QUINTILE, ASYM_20_60_20.
- decision: bootstrap suppression below `bootstrap_min_days=252` of beta history.
- decision: tripwire mirrors the main pipeline with 1M return window + 10d EWMA halflife.
- decision: composite price series not yet wired into PriceFrame; internal consistency runs against empty composite frames in v1.0 — deferred to v1.1.
- decision: pandas .ewm(halflife, adjust=False).std() approximates RiskMetrics; mean-demeaned + sample-bias-corrected, numerically close to raw recursion for near-zero-mean daily returns.
- decision: viz layer is a pure pipeline parallel to engine; CSV-in / HTML-out; Plotly bundled via CDN.
- decision: per-series β vs cap-wtd global proxy lives in viz layer (roro/report/beta_vs_global.py), not engine.
- decision: viz v2 uses scipy.stats.linregress for OLS+stderr; CI ribbon = upper trace (no fill) + lower trace (fill=tonexty) per color group.
- decision: scatter dropdown exposes 7 segments (Full + 6 sub-segments); _TRACES_PER_SEGMENT=8 (markers + fit + ci_upper + ci_lower per DM and EM).
- decision: viz v3 — seg_beta/seg_tercile carry full history (no reindex to scatter window); beta_timeseries x-axis = seg_beta.index; scatters keep 252d slider.
- decision: viz v3 — scatter axis ranges are per-segment (computed via _segment_axis_range); segment dropdown atomically updates visibility + xaxis.range + yaxis.range + title.
- constraint: NEVER name Alan's employer / corporate affiliation anywhere (code, docs, commits, README). RoRo is Alan's own personal work; author credit is exactly "Alan Vazquez, CFA" and nothing more.
- decision: viz v4 — beta_timeseries shades a HYSTERESIS-SMOOTHED tercile (_smooth_regime_hysteresis, _REGIME_CONFIRM_DAYS=21); raw daily tercile is too noisy to shade. Cosmetic only — engine regime output unchanged.
