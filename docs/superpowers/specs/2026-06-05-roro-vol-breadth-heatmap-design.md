# RoRo — Volatility-Percentile Breadth Heatmap

**Date:** 2026-06-05
**Status:** Design approved through brainstorming. Ready for implementation planning.
**Author:** Alan Vazquez, CFA
**Scope:** Add two volatility-breadth heatmaps to the `roro.report` HTML report — a sorted-rank breadth view and a per-asset identity view — answering "how many assets are trading at elevated volatility vs their own history." Built on branch `feat/hmm-report-viz` (merged together with the HMM viz).

---

## 1. Goal

For each asset, compute simple annualized realized volatility, then rank today's vol against the asset's own long history. Visualize the cross-section over time as heatmaps so an analyst can see breadth of volatility stress: how many assets sit at elevated vol-percentiles, and which ones.

Two complementary lenses:
1. **Breadth (sorted-rank):** each day, sort the cross-section of per-asset vol-percentiles descending. The thickness of the high-percentile band shows *how many* assets are stressed. Identity is dropped.
2. **Identity (asset-rows):** fixed row per asset, ordered by mean percentile. Shows *which* assets are stressed and lets you track one asset over time.

---

## 2. Locked decisions

| Fork | Choice |
|---|---|
| Vol metric | Simple realized: `rolling(63).std(daily_log_returns) × √252` (NOT EWMA) |
| Vol window | 63 trading days (matches engine `return_window_days`) |
| Percentile reference | Per series, **expanding** window, **min 5Y (1260d)** warmup, ranked vs the asset's own prior history |
| Heatmaps | Both: breadth (sorted-rank) + identity (asset-rows) |
| Class toggle | All / Equity / Fixed-Income on each heatmap (single Plotly dropdown, precomputed z per class) |
| Universe | All per-series (≈38 countries × {Eq, FI}) |
| Colorscale | Sequential white → red (low → high vol-percentile) |
| Branch | `feat/hmm-report-viz` (heatmap + HMM viz merge together) |

---

## 3. Two windows (kept distinct)

These are independent and must not be conflated:
- **Vol window (63d):** trailing days of daily returns feeding each day's vol *level*. `vol(t) = stdev(daily_returns[t-63 : t]) × √252`.
- **Percentile reference (expanding, min 5Y):** what `vol(t)` is *ranked against* — all of the asset's prior `vol` readings up to `t`, emitting a rank only once ≥1260 valid readings exist.

A short vol window makes spikes visible; the long ranking reference makes "elevated vs own history" meaningful.

---

## 4. Data / computation layer

New module `roro/report/vol_breadth.py`:

```python
def realized_vol(daily_returns: pd.DataFrame, *, window: int = 63) -> pd.DataFrame:
    """Simple annualized realized vol: rolling stdev of daily log returns × √252."""
    return daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def vol_percentile_matrix(vol_full: pd.DataFrame, *, min_history_days: int = 1260) -> pd.DataFrame:
    """Per series: expanding percentile rank of vol(t) vs all prior vol readings ≤ t.

    pct_t = (#{i ≤ t : vol_i ≤ vol_t}) / (#valid readings ≤ t), in [0, 1].
    NaN until the series has ≥ min_history_days valid (non-NaN) readings.
    Exact, computed via incremental searchsorted into a maintained-sorted history
    per series (O(n log n)) — not expanding().apply (O(n²)).
    """
```

`DataBundle` ([roro/report/bundle.py](../../../roro/report/bundle.py)) gains:
```python
vol_pct: pd.DataFrame | None = None   # date × series_id, full history, [0,1] or NaN
```

`load_bundle` ([roro/report/load.py](../../../roro/report/load.py)) computes it from FULL-history daily returns (the existing `_build_series_panels` already builds a full per-series price frame before slicing vol to the 252d window — reuse that frame's daily returns):
```python
daily_full = daily_log_returns(per_series_prices)        # full history, all series
vol_full = realized_vol(daily_full, window=63)
vol_pct = vol_percentile_matrix(vol_full, min_history_days=1260)
# restricted to series present in meta (same `common` filter used for the 252d panels)
```
`vol_pct` is **always** populated (needs only the xlsx). The existing 252d `vol`/`ret_3m`/`beta_vs_global` panels are unchanged. `meta` (already in the bundle) maps `series_id → asset (Eq/FI) → segment` for class subsetting and row ordering.

The percentile is **per-series vs its own history** — class-independent. The All/Eq/FI toggle only subsets which series feed each heatmap; the matrix is computed once.

---

## 5. Breadth heatmap (sorted-rank)

`vol_breadth_heatmap(bundle)` in [roro/report/figures.py](../../../roro/report/figures.py):
- For each class subset (All / Eq / FI), each date column: take that subset's per-series percentiles, drop NaN, **sort descending**, place into `z[rank, date]` (rank 1 = highest). Ragged columns (fewer non-NaN on a date) pad with NaN at the bottom; warmup dates are blank.
- One `go.Heatmap` trace: `z` = percentile (0–1), x = date, y = rank (1..N), colorscale white→red, `zmin=0, zmax=1`.
- **Class toggle:** one Plotly dropdown (All/Eq/FI). Each button restyles `z` (precomputed per class) + title. Single control → no JS.
- Reading: thickness of the red band at the top = count of elevated-vol assets that day.

---

## 6. Asset-rows heatmap (identity)

`vol_pct_asset_heatmap(bundle)` in figures.py:
- Rows = series (fixed), ordered by **mean percentile descending** within the active subset; x = date; `z` = `vol_pct` per series; same white→red scale, `zmin=0, zmax=1`.
- One `go.Heatmap` trace. Up to ~76 rows.
- **Class toggle** (All/Eq/FI): one Plotly dropdown; each button restyles `z` **and** `y` (row labels differ per subset, reordered by that subset's mean). Precomputed per class.
- Reading: which assets are persistently/currently stressed; track one asset left→right.

---

## 7. Orchestration & assembly

- **orchestrate** ([roro/report/orchestrate.py](../../../roro/report/orchestrate.py)): always append the two heatmaps after the existing figures (and after the HMM probability figure when present). Report = 3 base + (1 HMM if present) + 2 heatmaps = **5 or 6 figures**.
- **assemble refactor** ([roro/report/html.py](../../../roro/report/html.py)): the current positional `DIV_IDS`/`SECTION_TITLES` (indexed by figure position) break once figures are conditionally present — the HMM figure shifts the heatmaps' indices and mislabels them. Refactor `assemble` to take **explicit per-figure specs**:
  ```python
  @dataclass(frozen=True)
  class FigureSpec:
      figure: go.Figure
      div_id: str
      title: str

  def assemble(specs: list[FigureSpec], *, run_date, methodology_version,
               beta_div_id: str | None = None,
               beta_band_lookup: ... | None = None) -> str: ...
  ```
  The band toggle keys off `beta_div_id` (string) instead of a positional index. `orchestrate` builds the specs list with explicit div_ids/titles, conditionally including the HMM figure and always the two heatmaps. The module-level `DIV_IDS`/`SECTION_TITLES`/`_MIN_FIGURES`/`_MAX_FIGURES` positional constants are removed (specs carry id+title; bound the count via `len(specs)` if a guard is still wanted, e.g. ≥3).

---

## 8. Testing

New `tests/report/test_vol_breadth.py` + additions to existing report tests:
- **realized_vol** — known returns → expected annualized stdev; NaN for the first `window-1` rows.
- **vol_percentile_matrix** — a hand-built tiny series with a known expanding rank (e.g., monotone increasing vol → percentile → 1.0 at each new max; verify the fraction-≤ formula on a small case); assert NaN before `min_history_days` and valid [0,1] after; assert it matches a brute-force `expanding().apply` reference on a small input (correctness anchor for the fast algorithm).
- **vol_breadth_heatmap** — each date column of `z` is sorted descending (non-NaN portion); one heatmap trace; class dropdown has 3 buttons; `zmin/zmax` = 0/1.
- **vol_pct_asset_heatmap** — rows ordered by mean percentile descending; class toggle restyles both z and y; row count matches the subset.
- **load** — `vol_pct` populated, full-history (longer than the 252d window), warmup region NaN.
- **assemble (refactored)** — specs honored (each figure gets its given div_id + title); band toggle still injected and targets `beta_div_id`; 3-spec report (no HMM, no heatmaps in a minimal call) still valid.
- **orchestrate e2e** — built report contains both heatmap divs + their class dropdowns; with HMM present, all 6 figures appear; existing no-HMM report still builds.
- All existing `tests/report/` tests pass after the assemble refactor.

---

## 9. Non-goals

- No EWMA vol for this feature (simple realized only).
- No absolute-vol heatmap (percentile-of-own-history only — that's the whole point).
- No engine changes; this is report-only and reads from the xlsx + run dir.
- No cross-asset percentile (each series ranked vs ITSELF, never vs other series).
- No new CLI surface — heatmaps appear automatically in `roro report` output.
- No per-segment aggregation rows (per-series only; class toggle is the only grouping).

---

## 10. Files touched

- **Create** `roro/report/vol_breadth.py` — `realized_vol`, `vol_percentile_matrix`.
- **Modify** `roro/report/bundle.py` — `vol_pct` field.
- **Modify** `roro/report/load.py` — compute `vol_pct` (full-history).
- **Modify** `roro/report/figures.py` — `vol_breadth_heatmap`, `vol_pct_asset_heatmap`.
- **Modify** `roro/report/html.py` — `FigureSpec` + assemble refactor; band toggle keyed by div_id.
- **Modify** `roro/report/orchestrate.py` — build specs list; append heatmaps; conditional HMM.
- **Tests** — `tests/report/test_vol_breadth.py` + updates to `test_load.py`, `test_figures.py`, `test_html.py`, `test_orchestrate.py`.
