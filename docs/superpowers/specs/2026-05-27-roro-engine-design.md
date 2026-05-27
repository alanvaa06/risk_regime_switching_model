# RoRo Engine — Design Spec

**Author:** Alan Vazquez, CFA
**Date:** 2026-05-27
**Status:** Design approved through brainstorming. Ready for implementation planning.
**Source PRD:** `docs/prd/PRD.md` v1.2
**Scope:** PRD sprints S0–S7 + S9. Sprint S8 (HTML dashboard) deferred to a separate visualization phase.

---

## 1. Goal

Build a reproducible, testable Python engine that ingests the proprietary `data.xlsx` and free FRED series and emits daily risk-regime classifications across one global cut and nine segmentation cuts, plus correlation structure, external validation, fast-signal tripwire, alerts, and an S9 backtest harness. Output is plain CSV plus a `snapshot.json` per run. No application layer, no dashboard, no database.

---

## 2. Locked decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Engine scope | Full S0–S7 + S9 | Defer S8 viz only |
| Package shape | Single `roro/` package with submodules | Solo build, refactor later if needed |
| Storage | CSV per artifact + `snapshot.json`, run-stamped dir | Engine, not app |
| Entry point | Library API + thin Click CLI | Cron-friendly + interactive |
| Config | YAML default + CLI overrides → frozen `EngineConfig` dataclass | Reproducible, version-stampable |
| FRED key | In hand; via `FRED_API_KEY` env | Real validation from day one |
| Testing | pytest + hypothesis + mypy `--strict` + ruff | Matches `docs/references/python_best_practices.md` |
| Architecture | Functional pipeline of pure stage functions over frozen dataclasses | Matches PRD reproducibility goal |
| Cap weights | Static — 2026-05-26 Panel snapshot applied across full history | Only snapshot available in dataset |
| Mcap columns | `Equity_Mkt_Cap_Val` + `Fixed_Income_Mkt_Cap_Val` (Panel) | Zero NaN, USD-normalized |
| Universe split | Panel rows 0–31 → regression universe; rows 32–37 → composites for validation only | Per dataset structure |

---

## 3. Architecture

Pure functional pipeline. Each stage is a pure module function `(input dataclass) -> output dataclass`. The orchestrator `roro.engine.run(config) -> RunResult` wires them. State flows through the dataclass chain; nothing mutates in place.

```
data.xlsx (Panel + Equity_LC + Fixed_Income_LC)        FRED API (5 series)
            │                                                  │
            ▼                                                  ▼
  roro.io.load_panel        roro.io.load_prices        roro.io.load_fred
            │                                                  │
            └─────────────────────┬────────────────────────────┘
                                  ▼
              PriceFrame · Universe · FredFrame
                                  ▼
              roro.returns.compute  →  ReturnsFrame, VolFrame
                                  ▼
              roro.regression.cross_section
                  (cap-wtd + eq-wtd + slope-spread, per segment cut)
                                  ▼
              roro.segments.partition
                  (global, DM, EM, Eq, FI, DM_Eq, EM_Eq, DM_FI, EM_FI, LatAm)
                                  ▼
                          BetaBySegment
                                  ▼
              roro.classify.percentile
                  (5Y rolling pct, tercile + underlying quintile, direction, N flag)
                                  ▼
              roro.correlation.compute
                  (avg pairwise, PC1 variance share, per segment)
                                  ▼
              roro.validation.compare
                  (rolling 60d ρ vs FRED, internal consistency vs composites)
                                  ▼
              roro.tripwire.fast_signal
                  (1M return + 10d EWMA mirror)
                                  ▼
              roro.alerts.detect
                                  ▼
              roro.io.write_run  →  outputs/YYYY-MM-DD/{*.csv, snapshot.json}

  roro.backtest.run (separate CLI command)
      → backtest/YYYY-MM-DD/{event_recognition.csv,
                              validation_corr_history.csv,
                              stability_metrics.csv,
                              acceptance_report.json}
```

**Design properties:**
- Full historical recompute every run. ~4540 days × 64 series in vectorized numpy is seconds. Zero incremental-state complexity.
- Static cap weights loaded once, broadcast across history.
- All public dataclasses `@dataclass(frozen=True)`. mypy `--strict` clean.
- `FredClient` Protocol → `FredApiClient` + `MockFredClient`.
- `EngineConfig` version-stamped into `snapshot.json` for reproducibility.

---

## 4. Modules

```
roro/
├── __init__.py
├── config.py          EngineConfig (frozen), YAML loader, CLI merge, version stamp
├── types.py           PriceFrame, Universe, ReturnsFrame, VolFrame,
│                      BetaFrame, BetaBySegment, RegimeFrame,
│                      CorrelationFrame, ValidationFrame, AlertSet, RunResult
├── io.py              load_panel, load_prices, load_fred, write_run, read_run
├── validators.py      schema, date continuity, NaN policy
├── returns.py         total_return_3m, ewma_vol
├── regression.py      cross_section (cap-wtd + eq-wtd + slope-spread)
├── segments.py        partition (global + 9 cuts), SegmentRegistry
├── classify.py        rolling_percentile, tercile_label, quintile_label, direction_flag
├── correlation.py     avg_pairwise, pc1_share
├── validation.py      rolling_corr_external, internal_consistency
├── tripwire.py        fast_signal (1M mirror of return + vol + β + classify)
├── alerts.py          bucket_transitions, disagreement_events, validation_degradation
├── backtest.py        run_backtest, event_recognition, stability_metrics
├── engine.py          run(config) -> RunResult   (pipeline orchestrator)
├── cli.py             `roro run` / `roro backtest` Click commands
└── fred_client.py     FredClient Protocol + FredApiClient + MockFredClient

tests/
├── conftest.py        fixtures: tiny_panel, tiny_prices, mock_fred
├── strategies.py      hypothesis strategy bank
├── golden/2024-Q1/    golden CSVs for integration regression
├── test_io.py
├── test_validators.py
├── test_returns.py
├── test_regression.py
├── test_segments.py
├── test_classify.py
├── test_correlation.py
├── test_validation.py
├── test_tripwire.py
├── test_alerts.py
├── test_engine.py
├── test_backtest.py
└── test_reproducibility.py

configs/
└── default.yaml       documented defaults for every EngineConfig field

outputs/                 (gitignored)
└── YYYY-MM-DD/        beta_series.csv, regimes.csv, correlation.csv,
                       external_validation.csv, alerts.csv, snapshot.json

backtest/                (gitignored)
└── YYYY-MM-DD/        acceptance_report.json + supporting CSVs

docs/superpowers/specs/2026-05-27-roro-engine-design.md   (this file)
docs/context/           todo.md / results.md / lessons.md / memory.md / sesion-log.md
                        updated per CLAUDE.md workflow
```

---

## 5. Public contracts

```python
class BucketScheme(Enum):
    TERCILE   = auto()
    QUINTILE  = auto()
    ASYM_20_60_20 = auto()

@dataclass(frozen=True)
class EngineConfig:
    data_path: Path
    output_dir: Path
    ewma_halflife_days: int = 30
    return_window_days: int = 63          # ~3M
    tripwire_window_days: int = 21        # 1M
    tripwire_ewma_halflife_days: int = 10
    percentile_window_years: int = 5
    bucket_scheme: BucketScheme = BucketScheme.TERCILE
    min_n_per_cut: int = 10
    direction_lookback_days: int = 5
    external_corr_window_days: int = 60
    external_corr_alert_threshold: float = 0.3
    bootstrap_min_days: int = 252         # below this → bootstrap_period flag
    methodology_version: str = "1.0.0"
    fred_api_key: str | None = None       # from env FRED_API_KEY

@dataclass(frozen=True)
class PriceFrame:
    equity_lc: pd.DataFrame    # date index, country columns
    fi_lc:     pd.DataFrame    # date index, country columns

@dataclass(frozen=True)
class Universe:
    countries: pd.DataFrame    # 32 rows: Country, Segment, mcap_eq_val, mcap_fi_val
    composites: pd.DataFrame   # 6 rows: validation-only
    latam_countries: tuple[str, ...] = ("Brazil","Mexico","Chile","Peru","Colombia")

@dataclass(frozen=True)
class FredFrame:
    series: dict[str, pd.Series]   # keys: VIXCLS, BAMLC0A4CBBB,
                                   #       BAMLEMCBPIOAS, BAMLH0A0HYM2, T10Y2Y
    pulled_at: datetime
    series_hashes: dict[str, str]

@dataclass(frozen=True)
class ReturnsFrame:
    log_returns_3m: pd.DataFrame   # date × series
    daily_log_returns: pd.DataFrame

@dataclass(frozen=True)
class VolFrame:
    ewma_sigma_annualized: pd.DataFrame   # date × series

@dataclass(frozen=True)
class BetaFrame:
    cap_wtd: pd.DataFrame          # date × {beta, r2, n, suppressed, singular}
    eq_wtd:  pd.DataFrame          # same schema
    slope_spread: pd.Series        # cap − eq

@dataclass(frozen=True)
class BetaBySegment:
    by_segment: dict[str, BetaFrame]   # keys: global, DM, EM, Equity, FI,
                                       #       DM_Eq, EM_Eq, DM_FI, EM_FI, LatAm

@dataclass(frozen=True)
class RegimeFrame:
    percentile_5y: pd.DataFrame    # date × segment
    tercile:       pd.DataFrame
    quintile:      pd.DataFrame
    direction:     pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    bootstrap_flag: pd.DataFrame

@dataclass(frozen=True)
class CorrelationFrame:
    avg_pairwise_3m: pd.DataFrame  # date × segment
    pc1_variance_share: pd.DataFrame

@dataclass(frozen=True)
class ValidationFrame:
    rolling_corr_60d: pd.DataFrame              # date × (segment, fred_series)
    internal_consistency: pd.DataFrame          # date × segment, gap_terciles
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
    tripwire: BetaBySegment            # parallel β from 1M window
    alerts: AlertSet
    warnings: list[str]
    data_fingerprint: dict[str, str]
    code_version: dict[str, str]

@runtime_checkable
class FredClient(Protocol):
    def fetch(self, series_id: str, start: date, end: date) -> pd.Series: ...
```

---

## 6. Compute kernels

### 6.1 Returns — `roro.returns`

- `total_return_3m(prices_lc, W=63)`: `R_t = ln(P_t / P_{t-W})`. Log returns (additive in vol). Vectorized: `np.log(prices).diff(W)`.
- `daily_log_returns(prices_lc)`: `r_t = ln(P_t / P_{t-1})`.
- `ewma_vol(daily_log_returns, halflife)`:
  - `λ = exp(-ln(2) / halflife)`
  - `σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t`
  - Annualize ×`√252`.
  - Implementation: `r.ewm(halflife=halflife, adjust=False).std() * sqrt(252)`.
  - First 60 days → NaN (insufficient prior).

### 6.2 Cross-sectional regression — `roro.regression.cross_section`

For each date `t` and each segment cut `S`:
1. **Inputs.** `R_i = total_return_3m_i(t)` (from §6.1), `σ_i = ewma_vol_i(t)` (annualized, from §6.1). Both are the t-indexed column of their respective frames.
2. Filter series in `S` with finite `R_i` and `σ_i`. Apply `min_n_per_cut`.
3. Cap-weighted WLS: `R_i = α + β·σ_i + ε_i`, `w_i = mcap_i / Σ_j mcap_j` (static, from Panel `*_Mkt_Cap_Val`).
4. Equal-weighted OLS: same with `w_i = 1/N`.
5. Closed-form: `θ = (X'WX)^-1 X'Wy` in numpy. No statsmodels per-call.
6. Emit `β, R², N, suppressed_flag, singular_flag` per scheme.
7. `slope_spread = β_cap − β_eq`.

### 6.3 Segmentation — `roro.segments`

| Cut | N | Filter |
|---|---:|---|
| `global` | 64 | All 32 countries × {Eq, FI} |
| `DM` | 34 | Panel Segment=='DM' × {Eq, FI} |
| `EM` | 30 | Panel Segment=='EM' × {Eq, FI} |
| `Equity` | 32 | Asset class==Eq |
| `FI` | 32 | Asset class==FI |
| `DM_Eq` | 17 | Segment=='DM' & Eq |
| `EM_Eq` | 15 | Segment=='EM' & Eq |
| `DM_FI` | 17 | Segment=='DM' & FI |
| `EM_FI` | 15 | Segment=='EM' & FI |
| `LatAm` | 10 | Country ∈ {Brazil, Mexico, Chile, Peru, Colombia} × {Eq, FI} (thin_cut flag) |

`SegmentRegistry` exposes the filter list as data, not hard-coded in the regression code, so adding a cut is one entry.

### 6.4 Percentile classifier — `roro.classify`

- Per segment, per date `t`:
  - Window: trailing `5Y × 252 ≈ 1260` days of β.
  - `pct_t = rank(β_t within window) / N_window`.
  - Tercile: `pct ≤ 1/3 → Risk-off`, `pct ≥ 2/3 → Risk-on`, else `Transitional`.
  - Quintile retained per PRD §F4.3.
- Bootstrap policy (per PRD §10):
  - When `cfg.bootstrap_min_days ≤ N_window < 5Y × 252`, emit percentile from the available trailing window, set `bootstrap_flag=True`, record `N_window`.
  - When `N_window < cfg.bootstrap_min_days = 252`, suppress percentile (NaN) and mark `bootstrap_flag=True, suppressed=True`. This covers the very first months of beta history at dataset start.

### 6.5 Direction flag

`slope_5d = OLS(β over last cfg.direction_lookback_days)`. Threshold ±0.1·σ(β over rolling window) → `rising` / `falling` / `stable`.

### 6.6 Correlation — `roro.correlation`

- Per segment, window = 63 daily log returns.
- `avg_pairwise_3m`: mean of upper triangle of the rolling 63d correlation matrix.
- `pc1_variance_share`: largest eigenvalue / trace of the same window's covariance matrix (`np.linalg.eigh`).

### 6.7 External validation — `roro.validation`

**External (FRED).**
- `FredApiClient.fetch(series_id, start, end)` for `VIXCLS, BAMLC0A4CBBB, BAMLEMCBPIOAS, BAMLH0A0HYM2, T10Y2Y`.
- Align to engine date index. Forward-fill up to 2 trading days; beyond → NaN.
- For each `(segment β, FRED series)` pair: rolling 60d Pearson ρ.
- Alert when `|ρ| < cfg.external_corr_alert_threshold` over the window.

**Internal consistency (composites from Panel rows 32–37).** For each engine segment, derive a 3M-return tercile on the paired composite series and flag any day where the engine's regime tercile and the composite's return tercile differ by more than 1.

| Engine segment | Equity composite | FI composite |
|---|---|---|
| `global`    | `MXWD` (World)  | `LEGATRUU` |
| `DM`        | `MXWO` (DM)     | `I35402US` |
| `EM`        | `MXEF` (EM)     | `EMUSTRUU` |
| `Equity`    | `MXWD` (World)  | — |
| `FI`        | —               | `LEGATRUU` |
| `DM_Eq`     | `MXWO`          | — |
| `EM_Eq`     | `MXEF`          | — |
| `DM_FI`     | —               | `I35402US` |
| `EM_FI`     | —               | `EMUSTRUU` |
| `LatAm`     | `MXLA`          | `H04338US` |

Europe (`MXEU` / `I20912US`) and Asia (`MXAS` / `H02503US`) composites are not paired to any v1.0 segment cut; they are loaded and persisted in `validation_frame` for ad-hoc inspection but no automated check fires on them.

### 6.8 Tripwire — `roro.tripwire`

Mirror of 6.1–6.4 with `return_window_days = 21`, `ewma_halflife_days = 10`. Parallel `BetaBySegment` output. No separate classification; consumers display alongside main β.

### 6.9 Backtest — `roro.backtest`

- Replays the engine over `[start, end]` with no leakage (only data ≤ t is used; achieved by slicing input frames before each call).
- Outputs:
  - `event_recognition.csv` — for each of the 8 PRD events, the assigned tercile within ±3 trading days at global + relevant segment cuts.
  - `validation_corr_history.csv` — rolling 60d ρ vs each FRED series, by segment.
  - `stability_metrics.csv` — bucket transitions per quarter, partitioned by calm/non-calm (MXWO realized vol vs its 10Y median).
  - `acceptance_report.json` — pass/fail per PRD §10 gate.

---

## 7. Errors & failure policy

| Failure | Detection | Action |
|---|---|---|
| `data.xlsx` missing / unreadable | `roro.io.load_*` | `DataSourceError`. Hard fail. |
| Schema drift | `validators.py` on load | Hard fail with diff vs expected schema. |
| Date continuity broken (>3 missing trading days) | Validator | Hard fail. |
| Per-series NaN mid-history | Validator | Warn + drop on that day; record `N_used`. |
| Composite row missing | Validator | Warn; skip internal-consistency for that composite. |
| FRED unreachable | `FredApiClient.fetch` | Retry 3× with 2s backoff. On final failure, write `external_validation.csv` empty + warning in snapshot. Engine run still succeeds. |
| FRED partial | Per-series try | Continue with remaining; flag in snapshot. |
| Segment N < `min_n_per_cut` | `regression.cross_section` | `suppressed=True` row, no β. |
| Singular weighted design matrix | numpy raises | Emit NaN + `singular=True`. |
| Output dir exists, no `--force` | `io.write_run` | Hard fail. With `--force`, write to `.tmp/` then atomic rename. |

No silent fallback. Every degraded state surfaces in `snapshot.json.warnings[]`.

---

## 8. Reproducibility contract

Every run writes `snapshot.json`:

```json
{
  "run_date": "2026-05-27",
  "as_of_data_date": "2026-05-26",
  "methodology_version": "1.0.0",
  "config_resolved": { /* full EngineConfig after YAML + CLI merge */ },
  "data_fingerprint": {
    "data_xlsx_sha256": "...",
    "data_xlsx_mtime": "...",
    "fred_pulled_at": "...",
    "fred_series_hashes": { "VIXCLS": "...", "...": "..." }
  },
  "code_version": { "git_sha": "...", "dirty": false },
  "warnings": [],
  "global":   { "bucket": "...", "percentile_5y": 0.52, ... },
  "segments": { "DM_Eq": { ... }, "EM_Eq": { ... }, ... },
  "correlation": { "avg_pairwise_3m": 0.41, "pc1_variance_share": 0.38 },
  "external_corr_60d": { "VIX": -0.55, "BBB_OAS": -0.48, ... },
  "internal_consistency": { "MXWO_aligned": true, "MXEF_aligned": false },
  "alerts": [ "..." ]
}
```

Reproducibility invariant: two runs with identical `data_fingerprint + config_resolved + code_version` MUST produce byte-identical CSVs. Enforced by `test_reproducibility.py`.

---

## 9. Run lifecycle

`roro run --date YYYY-MM-DD --config configs/default.yaml [--ewma-halflife N] [--out outputs/] [--force]`

1. Load + freeze `EngineConfig` (YAML + CLI merge).
2. Fingerprint inputs (sha256 of `data.xlsx`, FRED pull timestamps + hashes).
3. Load Panel → split `countries_df` (rows 0–31) + `composites_df` (rows 32–37).
4. Load `Equity_LC` + `Fixed_Income_LC` → `PriceFrame` (skip ticker row, use country-name row as header).
5. Pull FRED (cached in `~/.roro/fred_cache/` with daily TTL).
6. Run pipeline (returns → vol → β → segments → classify → corr → validation → tripwire → alerts).
7. Write to `outputs/<run_date>.tmp/`, atomic rename to `outputs/<run_date>/`.
8. Append one line to `outputs/_history.csv` (run index).
9. Exit 0 on success, 1 on hard fail.

`roro backtest --start 2010-01-01 --end 2024-12-31 [--assert-gates]` — same pipeline looped over the date range; writes to `backtest/` and asserts PRD §10 gates when `--assert-gates` is set.

---

## 10. Testing

### 10.1 Pyramid

| Layer | Scope | Tools |
|---|---|---|
| Unit | One pure function per test. Each numeric kernel. | `pytest` |
| Property | Mathematical invariants over generated inputs. | `hypothesis` |
| Integration | Engine end-to-end on a 90-day slice of real data with golden CSVs. | `pytest` |
| Acceptance | PRD §10 success metrics on full backtest. | `roro backtest --assert-gates` |

### 10.2 Property invariants

- `returns`: `len(R) == len(P)`; `R` finite where `P > 0`; log-additivity.
- `ewma_vol`: `σ ≥ 0`; monotone in `|r|`; converges to `σ_long` with constant input.
- `regression`: `Σ w_i = 1`; β finite when σ has variance; OLS == WLS with uniform weights.
- `percentile`: `pct ∈ [0,1]`; monotone in β within window; tercile(pct) consistent.
- `correlation`: `avg_pairwise ∈ [-1,1]`; `PC1_share ∈ [0,1]`; `PC1_share ≥ 1/N`.
- `classify`: direction stable when β flat; rising when β strictly increases.
- `io`: roundtrip `write_run + read_run` preserves all frames bit-for-bit.

### 10.3 Golden integration

`tests/golden/2024-Q1/` holds expected CSVs for the default config over `2024-01-01 → 2024-03-31`. Test asserts byte-equal output (after canonical column ordering). Regenerate via `pytest --regenerate-goldens`.

### 10.4 Backtest acceptance gates (S9)

```
G1  ρ(global β, VIXCLS)        rolling-60d |ρ| ≥ 0.5 in ≥ 80% of 2010-2024
G2  ρ(global β, BAMLC0A4CBBB)  rolling-60d |ρ| ≥ 0.4 in ≥ 80% of 2010-2024
G3  Event recognition          8/8 events classified Risk-off within ±3 days
G4  Segmentation lift          ≥ 20% of days with DM_Eq vs EM_Eq tercile gap ≥ 2
G5  Stability                  ≤ 2 spurious transitions per calm quarter
G6  Internal consistency       ≤ 5 days / 30d window with DM tercile vs MXWO gap > 1
```

S9 also sweeps `ewma_halflife ∈ {20,30,40}` × `bucket_scheme ∈ {tercile, quintile, asym}` and emits a selection report. Lock-in is a manual config edit, not auto.

### 10.5 Static analysis

- `mypy --strict roro/`.
- `ruff check roro/ tests/` (replaces flake8 + black).
- CI hook: `pytest && mypy --strict && ruff check`.

### 10.6 Hypothesis strategy bank

`tests/strategies.py` centralizes:
- `prices_strat` — positive monotone-ish series (geometric BM-like).
- `returns_strat` — finite floats with optional NaN injection.
- `weights_strat` — positive summing to arbitrary positive total.
- `panel_strat` — synthetic 5-country × 2-class universe for partition tests.

---

## 11. Dependencies

| Package | Use |
|---|---|
| `python 3.12` | Runtime |
| `numpy` | Vectorized math, linalg |
| `pandas` | Frames, IO, EWMA |
| `pyarrow` | Fast CSV (engine='pyarrow') |
| `openpyxl` | Excel read |
| `fredapi` | FRED ingest |
| `click` | Thin CLI |
| `pyyaml` | Config load |
| `pytest`, `hypothesis`, `mypy`, `ruff` | Dev |

Locked via `pyproject.toml` + `uv` (or `pip-tools`). No statsmodels, no scikit-learn — closed-form WLS in numpy and PCA via `np.linalg.eigh`.

---

## 12. Out of scope (this engine)

- HTML dashboard / JSON-API server (PRD §F7) — separate visualization phase.
- Postgres / TimescaleDB — engine writes plain CSV only.
- Predictive layer / regime-transition forecasting (PRD §15, v2.0).
- GFP integration (PRD §F6.5, v1.1).
- Time-varying mcap weights (v1.1+; static snapshot is the v1.0 contract).
- Intraday signals.

---

## 13. Open follow-ups for the implementation plan

- Concrete pyproject.toml + lockfile creation (uv vs pip-tools).
- Decision: cache FRED responses to disk vs always re-pull. Default: cache 24h TTL in `~/.roro/fred_cache/`.
- CI choice (GitHub Actions vs local pre-commit only). User can defer.
- `--regenerate-goldens` flag wiring and review process for golden updates.

These are tactical and belong in the implementation plan, not the design.
