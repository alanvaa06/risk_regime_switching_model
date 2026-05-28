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
