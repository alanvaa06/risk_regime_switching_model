# RoRo Visualization Layer — Design Spec

**Version:** 1.0.0
**Date:** 2026-05-27
**Status:** Draft (pending implementation)
**Scope:** Two-chart interactive HTML report consuming engine run-dir artifacts. PRD §F7 Tier-1 subset.

---

## 1. Goal

Produce a single self-contained interactive HTML report per engine run that exposes:

1. **Risk-return scatter** — one marker per country-asset, x = EWMA annualized volatility, y = 3M total log return.
2. **Beta-return scatter** — one marker per country-asset, x = 63d rolling β vs cap-weighted global proxy, y = 3M total log return.
3. **Segment beta time-series** — selected segment's cap-weighted β over the last 252 business days, with tercile regime bands shaded as background.

All three figures live in the same HTML file. The scatter pair shares a date slider (last 252 business days) and a segment dropdown (Full / DM / EM / EM_Eq / EM_FI). The beta time-series exposes its own segment dropdown with the broader segment set (Full / DM / EM / EM_Eq / EM_FI / DM_Eq / DM_FI / LatAm) since engine output covers all cuts at segment level.

The report is the visual companion to the engine's CSV outputs and is regenerated per run.

---

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Delivery format | Interactive HTML (Plotly, CDN bundle) | Matches PRD §F7 "static-served HTML"; no server needed; selectors work offline |
| Risk-axis metric | EWMA annualized volatility (existing `VolFrame`) | Already computed by engine; standard risk measure |
| Return-axis metric | 3M total log return (existing `ReturnsFrame.total_return_3m`) | Already computed; pairs with EWMA vol on comparable horizon |
| Beta-axis metric | 63d rolling OLS β vs cap-weighted global proxy | New computation in viz layer; per-series granularity |
| Scatter point unit | One point per country-asset (e.g., `Brazil_Eq`, `Brazil_FI`) | Matches reference image cluster style |
| Trend line | OLS per DM/EM color group | Visual reference parity |
| Date selector range | Last 252 business days | Bundle size, load speed |
| Regime band scheme | Tercile (Risk-off / Transitional / Risk-on) | Engine default; user noted mapping may evolve in future versions |
| Approach | Approach A: `roro/report/` module + `roro report` CLI | Pure pipeline, CSV-in / HTML-out, headless testable |

---

## 3. Architecture

```
[outputs/<run>/*.csv]   [data.xlsx Equity_LC / FI_LC]
         │                          │
         ▼                          ▼
   roro.report.load        roro.report.beta_vs_global
         │                          │
         └─────────► DataBundle ◄───┘
                          │
                          ▼
                 roro.report.figures
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
   scatter_vol_return  scatter_beta_  beta_timeseries
                       return
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                 roro.report.html.assemble
                          │
                          ▼
                     report.html
```

Pipeline of pure functions over frozen dataclasses. No runtime mutation, no global state. Headless build path tested end-to-end against the golden 2024-Q1 fixture.

---

## 4. Modules

New package: `roro/report/`.

| File | Public surface | Responsibility |
|---|---|---|
| `roro/report/__init__.py` | `build_report(run_dir, xlsx_path, out_path) -> Path` | One-call orchestrator for CLI and library users |
| `roro/report/load.py` | `load_bundle(run_dir, xlsx_path, *, window=252) -> DataBundle` | Read run-dir CSVs + xlsx prices, slice last 252 business days, return typed bundle |
| `roro/report/beta_vs_global.py` | `compute_beta_vs_global(log_returns, weights, *, window=63) -> pd.DataFrame` | 63d rolling OLS β per country-asset vs cap-wtd global proxy |
| `roro/report/bundle.py` | `DataBundle` frozen dataclass | All inputs to figure builders |
| `roro/report/figures.py` | `scatter_vol_return(bundle) -> Figure`, `scatter_beta_return(bundle) -> Figure`, `beta_timeseries(bundle) -> Figure` | Three pure figure builders |
| `roro/report/html.py` | `assemble(figures, *, run_date, methodology_version) -> str` | Build single HTML doc with header, three figure divs, minimal CSS |
| `roro/cli.py` (existing, modified) | `cmd_report` Click subcommand | Wire CLI to `build_report` |

Tests live under `tests/report/` mirroring layout one-for-one.

---

## 5. Data contracts

### 5.1 `DataBundle`

```python
@dataclass(frozen=True)
class DataBundle:
    run_date: pd.Timestamp                  # from snapshot.json
    methodology_version: str                # from snapshot.json
    dates: pd.DatetimeIndex                 # last 252 business days available
    # Per-series panels: index = dates, columns = series_id (e.g., "Brazil_Eq")
    vol: pd.DataFrame                       # EWMA annualized vol
    ret_3m: pd.DataFrame                    # 3M cumulative log return
    beta_vs_global: pd.DataFrame            # 63d rolling OLS β vs cap-wtd global proxy
    # Series metadata: index = series_id, columns = country, asset (Eq|FI), segment (DM|EM), weight
    meta: pd.DataFrame
    # Segment-level panels: index = dates, columns = segment cuts
    seg_beta: pd.DataFrame                  # cap-wtd β from beta_series.csv
    seg_tercile: pd.DataFrame               # tercile label per (date, segment)
```

### 5.2 Stage signatures

| Function | Input | Output |
|---|---|---|
| `load.load_bundle` | `run_dir: Path`, `xlsx_path: Path`, `window: int = 252` | `DataBundle` |
| `beta_vs_global.compute_beta_vs_global` | `log_returns: pd.DataFrame` (date × series), `weights: pd.Series` (series → cap weight), `window: int` | `pd.DataFrame` (date × series, β values, NaN before window fills) |
| `figures.scatter_vol_return` | `DataBundle` | `plotly.graph_objects.Figure` with date slider + segment dropdown |
| `figures.scatter_beta_return` | `DataBundle` | same shape as above |
| `figures.beta_timeseries` | `DataBundle` | `Figure` with segment dropdown + tercile `add_vrect` shading |
| `html.assemble` | `figures: list[Figure]`, `run_date: pd.Timestamp`, `methodology_version: str` | HTML string |
| `build_report` | `run_dir: Path`, `xlsx_path: Path`, `out_path: Path` | `Path` (written file) |

### 5.3 Cap-weighted global proxy

```
proxy_return_t = Σ_i (w_i × log_return_i_t) / Σ_i w_i
```

Weights come from the Panel `Equity_Mkt_Cap_Val` + `Fixed_Income_Mkt_Cap_Val` snapshot (same static weights as engine). Sum is over all 64 country-asset series.

### 5.4 Per-series β computation

For each series `i`, each date `t` (where `t >= window + first_date`):

```
β_i_t = Cov(r_i, r_global) / Var(r_global)   over window of `window` prior business days
```

Window = 63 business days (≈3 months). Implementation: pandas `.rolling(63).cov()` and `.rolling(63).var()`, division. NaN until window fills.

---

## 6. Figure specs

### 6.1 Scatter: vol × 3M return (`scatter_vol_return`)

- **Markers:** one per `series_id` in the active segment filter, on the active selected date.
- **x:** `bundle.vol.loc[date, series_id]`
- **y:** `bundle.ret_3m.loc[date, series_id]`
- **Color:** DM = `#1f77b4` (blue), EM = `#ff7f0e` (orange) per `meta.segment`.
- **Hover:** `series_id`, country, asset, segment, vol value, return value, date.
- **Trend lines:** Plotly Express `trendline="ols"` per color group (DM and EM separate fits).
- **Selectors:**
  - Date slider: 252 frames (one per business day), default = `run_date` (last frame).
  - Segment dropdown via `updatemenus`: Full / DM / EM / EM_Eq / EM_FI. Default = Full.
- **Axes:** x label "EWMA annualized volatility", y label "3M total log return". Both linear. Axis ranges fixed across frames using global min/max with 5% padding so the camera does not jump as the date slider moves.
- **Title:** "Risk vs Return — &lt;date&gt; — &lt;segment&gt;" (updates with selectors).

### 6.2 Scatter: β × 3M return (`scatter_beta_return`)

Identical to §6.1 except:
- **x:** `bundle.beta_vs_global.loc[date, series_id]`
- **x label:** "β vs cap-weighted global"
- **Title prefix:** "Beta vs Return — …"

### 6.3 Beta time-series (`beta_timeseries`)

- **Trace:** line of `bundle.seg_beta[segment]` over `bundle.dates`.
- **Background shading:** consecutive runs of identical `bundle.seg_tercile[segment]` rendered as `add_vrect`:
  - Risk-off → `fillcolor="rgba(220, 50, 47, 0.12)"` (red)
  - Transitional → `fillcolor="rgba(128, 128, 128, 0.06)"` (gray)
  - Risk-on → `fillcolor="rgba(46, 160, 67, 0.12)"` (green)
  - `layer="below"`, `line_width=0`.
- **Selector:** segment dropdown via `updatemenus`: Full / DM / EM / DM_Eq / DM_FI / EM_Eq / EM_FI / LatAm. Default = Full (global).
- **Axes:** x = date, y = "Cap-weighted β".
- **Title:** "Segment β with regime bands — &lt;segment&gt;".

### 6.4 Rendering consolidation

A run-level helper bundles regime-run detection so the same code path produces vrects for every segment in the dropdown without per-segment branching.

---

## 7. HTML assembly (`html.assemble`)

Single static HTML document. Layout:

```
┌──────────────────────────────────────────────────────────┐
│ Header                                                   │
│  RoRo Risk-Regime Report                                 │
│  Run date: YYYY-MM-DD   |   Methodology: v1.0.0          │
├──────────────────────────────────────────────────────────┤
│ Section 1 — Risk vs Return                               │
│   [scatter_vol_return figure]                            │
├──────────────────────────────────────────────────────────┤
│ Section 2 — Beta vs Return                               │
│   [scatter_beta_return figure]                           │
├──────────────────────────────────────────────────────────┤
│ Section 3 — Segment β with regime bands                  │
│   [beta_timeseries figure]                               │
├──────────────────────────────────────────────────────────┤
│ Footer: generated <timestamp> · roro git_sha · v1.0.0    │
└──────────────────────────────────────────────────────────┘
```

- Use Plotly's `to_html(include_plotlyjs="cdn", full_html=False, div_id=<deterministic>)` per figure, then concatenate inside a hand-written shell.
- Inline CSS (≤ 50 lines): system font stack, max-width 1200px container, light theme.
- Deterministic `div_id` (e.g., `fig_scatter_vol`, `fig_scatter_beta`, `fig_beta_ts`) for reproducibility.
- No JavaScript beyond what Plotly ships.

---

## 8. CLI

New Click subcommand on existing `roro` group:

```bash
roro report --run-dir outputs/2026-05-27 --xlsx data.xlsx --out outputs/2026-05-27/report.html
```

Flags:
- `--run-dir PATH` (required): engine run directory containing CSVs + snapshot.json.
- `--xlsx PATH` (optional, default = `snapshot.json["config_resolved"]["data_path"]` from the supplied `--run-dir`): source xlsx for prices used by `beta_vs_global`. If the snapshot key is missing or the file does not exist, the CLI raises a `UsageError` instructing the user to pass `--xlsx` explicitly.
- `--out PATH` (optional, default = `<run-dir>/report.html`).

Exit code 0 on success, non-zero on any `ReportInputError` / `FileNotFoundError`.

---

## 9. Errors + edge cases

| Case | Behavior |
|---|---|
| Required CSV missing in `run_dir` (`regimes.csv`, `beta_series.csv`) | Raise `ReportInputError` listing the missing file. Fail fast. |
| `snapshot.json` missing or unparseable | Raise `ReportInputError`. |
| `--xlsx` path missing | Raise `FileNotFoundError` from existing `load_prices`. |
| Available history < 252 business days | Use what's available; `bundle.dates` = available subset; warn to stderr; date slider has fewer frames. |
| Per-series NaN vol or return on a date | Drop that marker on that frame; remaining series render. |
| Segment filter produces empty set | Render empty plot with `annotation` "no data for selection". |
| Tercile column missing for a segment | Skip shading for that segment; β line still drawn; annotate "no regime data". |
| `beta_vs_global` window incomplete on early dates | NaN — handled by frame-level marker drop. |
| Output path exists | Overwrite (single artifact, no `--force` flag). |
| Resulting HTML > 5 MB | Log warning to stderr; still write. |

**Validation:** All input frames re-checked for date monotonicity (`is_monotonic_increasing`) and numeric dtype before figure build. No silent coercion.

---

## 10. Reproducibility

Required guarantees:

1. Two `build_report` calls on identical inputs produce byte-identical HTML.
2. Deterministic `div_id` per figure.
3. No timestamp inside the figure JSON (Plotly default: none).
4. Footer "generated &lt;timestamp&gt;" replaced by `snapshot.json.run_date` (already deterministic from engine).
5. Plotly version pinned in `pyproject.toml`.

Tested via `tests/report/test_report_reproducibility.py`.

---

## 11. Testing

### 11.1 Unit (no IO)

- **`test_beta_vs_global.py`**
  - β of constant series → 0.
  - β of series identical to proxy → 1.
  - Property (hypothesis): β invariant to multiplicative scaling of input prices.
  - Property: β of two scaled copies of proxy returns NaN-tolerant.
  - Window edge: first valid β at index `window` not `window - 1`.

- **`test_figures.py`**
  - Each builder returns `plotly.graph_objects.Figure`.
  - Scatter frame count == `len(bundle.dates)`.
  - Trace count per segment-filter state == count of matching `series_id` (plus trend-line traces).
  - `beta_timeseries`: `add_vrect` count == number of regime runs in selected segment.
  - Hover template contains country, asset, segment.

- **`test_html.py`**
  - Assembled string contains 3 `<div class="plotly-graph-div">`.
  - Contains `run_date` and `methodology_version` in header.
  - Parses as valid HTML5 via `html.parser`.

- **`test_load.py`**
  - Loads golden 2024-Q1 run dir → bundle has expected columns, date count, no NaN in metadata.
  - Missing `regimes.csv` → `ReportInputError`.

### 11.2 Integration

- **`test_build_report_e2e.py`** — golden 2024-Q1 run-dir + tiny xlsx fixture → `build_report` writes `report.html`, file exists, size > 50 KB, parses as HTML.

### 11.3 Reproducibility

- **`test_report_reproducibility.py`** — two consecutive `build_report` calls on identical inputs produce byte-identical files.

### 11.4 Golden

Skip golden HTML byte diff (Plotly internals churn between versions). Golden test stores JSON of figure `to_dict()` minus non-deterministic fields (`uid`); regenerated via `--regenerate-goldens`.

### 11.5 Tooling

- `pytest` (existing).
- `hypothesis` (existing).
- `mypy --strict` on `roro/report/**`.
- `ruff` same rules as engine.

---

## 12. Dependencies

Additions to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...
    "plotly>=5.20,<6.0",
]
```

No other new runtime deps. CDN-loaded plotly.js means no bundled JS asset.

---

## 13. Out of scope

Deferred to follow-up specs:

- PRD §F7 Tier-2 8-card regime grid (one card per segment).
- Correlation panel chart (avg pairwise + PC1 variance share).
- External validation strip chart (rolling ρ vs FRED proxies).
- Alerts UI section (bucket transitions, disagreements, validation degradation).
- Streamlit / Dash interactive web app.
- LaTeX / PDF report export.
- Multi-run side-by-side comparison view.
- Authentication and Coolify / Vercel deploy automation.
- Backtest acceptance-gate visuals (G1–G6 from §10 of PRD).
- 4-band custom regime mapping (user flagged for future engine work, not viz work).

In scope for this spec: only the three figures, selectors, and single HTML artifact.

---

## 14. Acceptance criteria

A successful implementation of this spec:

1. `roro report --run-dir outputs/2026-05-27 --xlsx data.xlsx` writes `outputs/2026-05-27/report.html`.
2. Opening that HTML in a browser shows three Plotly figures with working date slider, segment dropdowns, and tercile-shaded β chart.
3. All tests in `tests/report/**` pass.
4. `mypy --strict roro/report` passes.
5. `ruff check roro/report tests/report` passes.
6. Two consecutive `roro report` invocations produce byte-identical HTML.
