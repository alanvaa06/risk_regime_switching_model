# RoRo Visualization Layer v3 — Iteration Spec

**Version:** 3.0.0
**Date:** 2026-05-28
**Status:** Draft (pending implementation)
**Scope:** Iteration on the v2 viz layer (spec `2026-05-28-roro-viz-v2-design.md`). Four user-reported issues; no architectural changes.

---

## 1. Goal

Fix four user-reported issues observed in the v2 viz output:

1. Scatter axes are fixed to the global universe min/max — when a sub-segment is selected, points cluster in a tiny region of the plot. Range should retighten per active segment.
2. Date slider overlaps the x-axis title (`EWMA annualized volatility`) and the bottom-anchored legend.
3. β time-series only shows the last 252 business days; user wants the full historical span to read regime structure over years.
4. Hover tooltip on scatter markers should expose the country + asset name. (Already wired in v2 via `text` + `hovertemplate`; this iteration locks the behavior with a test.)

The v2 architectural model is unchanged: pure pipeline, CSV-in / HTML-out, `roro.report` package, three Plotly figures, same dependency surface. No new modules, no new CLI flags.

---

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| History window split | Per-series panels stay at `window=252` (scatter slider). `seg_beta` + `seg_tercile` carry full history from the engine run-dir CSVs (no reindex). | β time-series benefits from multi-year context; scatter HTML size stays under control (~27 MB) |
| Scatter axis-range strategy | Per-segment fixed range, computed from that segment's own data with 5% padding. Atomically updated via dropdown buttons. | Tight cluster view per filter; no per-frame camera jumps on the date slider |
| Legend placement on scatters | Top-left overlay on plot region, vertical orientation, semi-transparent white background, light border | Frees the bottom row for the slider; resolves overlap |
| Slider position | `y=-0.18`, bottom margin `b=140` | Pushes slider clear of x-axis title; vertical room for slider widget |
| Hover format | Unchanged — `%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>` with `text=["<country>_<asset>", ...]` | Already works; lock with an assertion test |
| New dependencies | None | All work uses existing libraries |

---

## 3. Changes — file-by-file

### 3.1 `roro/report/load.py`

Delete the two lines that reindex segment frames to the per-series window:

```python
# REMOVE these two lines from load_bundle:
seg_beta = seg_beta.reindex(dates)
seg_tercile = seg_tercile.reindex(dates)
```

`seg_beta` and `seg_tercile` are returned in the bundle with their natural full date index from the engine run-dir CSVs (`beta_series.csv` + `regimes.csv`). They are sorted ascending by date.

`bundle.dates` continues to reflect the per-series 252d window — that's used by scatters. β time-series uses `bundle.seg_beta.index` as its x-axis.

### 3.2 `roro/report/figures.py`

**New helper at module top (after existing constants):**

```python
def _segment_axis_range(
    panel: pd.DataFrame,
    segment: SegmentFilter,
    meta: pd.DataFrame,
) -> tuple[float, float]:
    """Compute (lo, hi) range for a panel restricted to one segment's series.

    Returns the panel's nanmin/nanmax over the segment's columns with 5% padding.
    Falls back to the panel's global min/max if the segment's slice is all NaN
    or empty.
    """
    series_ids = _series_for_segment(meta, segment)
    if not series_ids:
        sub = panel.to_numpy(dtype=float)
    else:
        present = [c for c in series_ids if c in panel.columns]
        sub = panel[present].to_numpy(dtype=float) if present else panel.to_numpy(dtype=float)
    if not np.isfinite(sub).any():
        sub = panel.to_numpy(dtype=float)
    lo = float(np.nanmin(sub))
    hi = float(np.nanmax(sub))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad
```

**`_build_scatter` changes:**

- Compute initial axis range from `_segment_axis_range(x_panel, "Full", bundle.meta)` and `_segment_axis_range(y_panel, "Full", bundle.meta)` instead of the current global panel min/max + padding block. The existing `x_min, x_max, x_pad, y_min, y_max, y_pad` derivations are deleted.
- Each segment button's layout-update dict (`args[1]`) gains `"xaxis.range"` and `"yaxis.range"` entries:

```python
segment_buttons = []
for seg in SCATTER_SEGMENTS:
    x_lo, x_hi = _segment_axis_range(x_panel, seg, bundle.meta)
    y_lo, y_hi = _segment_axis_range(y_panel, seg, bundle.meta)
    segment_buttons.append({
        "method": "update",
        "label": seg,
        "args": [
            {"visible": _visibility_mask(seg)},
            {
                "title": f"{title_prefix} — {dates[-1].date()} — {seg}",
                "xaxis.range": [x_lo, x_hi],
                "yaxis.range": [y_lo, y_hi],
            },
        ],
    })
```

- Legend layout updates from horizontal-below to vertical-top-left overlay:

```python
legend={
    "orientation": "v",
    "yanchor": "top",
    "y": 0.98,
    "xanchor": "left",
    "x": 0.02,
    "bgcolor": "rgba(255, 255, 255, 0.75)",
    "bordercolor": "#cccccc",
    "borderwidth": 1,
},
```

- Slider config gains explicit `"y": -0.18` and `"pad": {"t": 30, "b": 10}` for spacing.
- Layout margin updates: `margin={"l": 60, "r": 200, "t": 60, "b": 140}` (was `b=120`).

**`beta_timeseries` changes:**

- The initial line trace uses `bundle.seg_beta.index` as its `x`, not `bundle.dates`. Same for each button's `y`. The button bodies become:

```python
for seg in available:
    y = bundle.seg_beta[seg].to_numpy(dtype=float)
    x_ts = bundle.seg_beta.index  # full history, NOT bundle.dates
    shapes: list[dict[str, object]] = []
    # ... build shapes from _regime_runs(bundle.seg_tercile[seg]) — already iterates seg's own full series
    buttons.append({
        "method": "update",
        "label": seg,
        "args": [
            {"x": [x_ts], "y": [y], "name": [seg]},
            {"title": f"Segment β with regime bands — {seg}", "shapes": shapes},
        ],
    })
```

The initial scatter trace also reads from `bundle.seg_beta.index`. Tercile shading still uses `_regime_runs` on the full `seg_tercile` series.

No other figure-builder changes.

### 3.3 Tests

**`tests/report/test_load.py`:**

- New: `test_load_bundle_seg_beta_carries_full_history` — assert that `bundle.seg_beta.index` is longer than `len(bundle.dates)` (or equal when xlsx history equals the window) and matches the raw `beta_series.csv` date range. Similarly for `seg_tercile`.

**`tests/report/test_figures.py`:**

- New: `test_scatter_segment_buttons_carry_per_segment_axis_ranges` — for each of the 7 scatter buttons, `args[1]["xaxis.range"]` and `args[1]["yaxis.range"]` exist and are non-degenerate (`hi > lo`).
- Delete existing `test_scatter_legend_is_horizontal_below_plot` (legend is no longer horizontal-below).
- New: `test_scatter_legend_is_top_left_overlay` — `legend.orientation == "v"`, `legend.y == 0.98`, `legend.x == 0.02`, `legend.bgcolor` contains `"rgba"`.
- New: `test_scatter_slider_y_is_minus_0_18` — `fig.layout.sliders[0].y == -0.18`.
- New: `test_scatter_bottom_margin_is_140` — `fig.layout.margin.b == 140`.
- New: `test_scatter_marker_hover_template_includes_text` — for the initial Full segment, an active markers trace exists whose `hovertemplate` contains `"%{text}"` and whose `text` first element matches the pattern `r"^[A-Za-z ]+_(Eq|FI)$"`.
- New: `test_beta_timeseries_x_axis_uses_full_seg_beta_index` — invoke `beta_timeseries(bundle)`; assert that the initial line trace's `x` matches `bundle.seg_beta.index` (which by virtue of the load.py change carries full history).

### 3.4 No new dependencies

`pyproject.toml` is not modified.

---

## 4. Out of scope

- HTML file-size optimization (compression, frame culling) — deferred follow-up.
- Module split of `figures.py` into `_scatter.py` + `_beta_ts.py` — deferred follow-up.
- Tier-2 panels (correlation, alerts, etc.).
- Changing the date-slider granularity (currently one frame per business day).
- v2 carryover follow-ups (snapshot type coercion, `cap_wtd` magic literal).

---

## 5. Acceptance criteria

1. Selecting `DM_Eq` from the scatter dropdown retightens both axes around the DM_Eq cluster (not the full universe).
2. The date slider sits below the x-axis title without overlapping it; the legend sits inside the plot region in the upper-left corner.
3. Opening the β time-series tab shows the engine's full historical span (multi-year if available), not just the trailing 252 days.
4. Hovering a marker on the scatter pops a tooltip containing the country and asset (e.g., `United States_Eq`).
5. All previously passing tests still pass (127 → ~134 with the new assertions).
6. Two consecutive `roro report` invocations on identical inputs continue to produce byte-identical HTML.
7. mypy --strict + ruff sweep stay clean.
