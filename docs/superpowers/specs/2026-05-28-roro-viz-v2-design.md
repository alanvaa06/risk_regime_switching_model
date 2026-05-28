# RoRo Visualization Layer v2 — Iteration Spec

**Version:** 2.0.0
**Date:** 2026-05-28
**Status:** Draft (pending implementation)
**Scope:** Iteration on the v1 viz layer (spec `2026-05-27-roro-viz-design.md`). Six concrete changes; no architectural changes.

---

## 1. Goal

Address visual and ergonomic issues observed in the v1 viz output:

1. Segment dropdown is missing DM_Eq and DM_FI options.
2. Legend on scatters is overlapped by the segment dropdown.
3. Default Plotly aesthetic looks dated; user wants seaborn-like styling with confidence bands around trend lines (matching the palmerpenguins reference image).
4. EM color (`#ff7f0e`, orange) clashes with regime band colors and the reference image; switch to green.
5. Charts are too short; user wants taller layout.
6. Trend lines should carry 95% confidence intervals.

The architectural model from v1 (pure pipeline, CSV-in / HTML-out, `roro.report` package, three figures, Plotly) is unchanged. This iteration touches only `roro/report/figures.py`, `roro/report/html.py`, and `pyproject.toml`.

---

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Library | Plotly (unchanged) with seaborn-like aesthetic applied | Preserves the date slider + segment dropdown interactivity from v1 |
| Confidence band | 95% CI ribbon around each OLS trend line, computed from `scipy.stats.linregress` | Matches reference image; deterministic; cheap |
| Chart height | 700 px per figure | User-chosen |
| EM color | `#2ca02c` (green) | User-chosen; matches reference image's Gentoo green |
| DM color | `#1f77b4` (blue, unchanged) | Already correct |
| Scatter segment set | `("Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI")` — 7 options | User flagged the missing DM_Eq/DM_FI |
| Traces per segment | 8 (4 per color group: markers, fit line, CI upper, CI lower) | Required to render the CI ribbon |
| New dep | `scipy>=1.11,<2.0` | Standard library; provides `linregress` with slope + intercept + stderr_slope + stderr_intercept in one call |

---

## 3. Changes — file-by-file

### 3.1 `roro/report/figures.py`

**Constants:**

```python
SCATTER_SEGMENTS: tuple[SegmentFilter, ...] = (
    "Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI",
)

SegmentFilter = Literal["Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI"]

COLOR_DM: str = "#1f77b4"   # blue, unchanged
COLOR_EM: str = "#2ca02c"   # green; was "#ff7f0e"

_TRACES_PER_SEGMENT: int = 8  # 2 color groups × (markers + fit + ci_hi + ci_lo); was 4

_CI_Z: float = 1.96           # 95% CI multiplier
_MIN_OBS_FOR_CI: int = 3      # minimum points for linregress stderr
```

**`_series_for_segment` — extend the dispatch.** Add cases:

```python
if segment == "DM_Eq":
    return list(meta.index[(meta["segment"] == "DM") & (meta["asset"] == "Eq")])
if segment == "DM_FI":
    return list(meta.index[(meta["segment"] == "DM") & (meta["asset"] == "FI")])
```

(Plus the existing Full/DM/EM/EM_Eq/EM_FI branches.)

**New helper: `_ols_with_ci`.**

```python
def _ols_with_ci(
    x: FloatArray,
    y: FloatArray,
) -> tuple[float, float, float, float] | None:
    """Return (slope, intercept, stderr_slope, stderr_intercept) or None if
    fewer than _MIN_OBS_FOR_CI finite (x, y) pairs.

    Wraps scipy.stats.linregress, falling back to None on degenerate input
    (zero variance in x, all-NaN, etc).
    """
```

When `None` is returned, the markers trace is still emitted but the fit + CI traces are emitted as empty-array placeholders so the trace count stays at 4 per color group.

**New helper: `_ci_band_traces`.**

```python
def _ci_band_traces(
    slope: float,
    intercept: float,
    se_slope: float,
    se_intercept: float,
    mean_x: float,
    x_range: tuple[float, float],
    color: str,
    *,
    z: float = _CI_Z,
    n_points: int = 50,
) -> tuple[go.Scatter, go.Scatter]:
    """Two scatter traces forming a filled CI ribbon. Upper bound first
    (no fill), lower bound second (fill='tonexty' with rgba alpha 0.18).
    """
```

**Refactor `_scatter_traces_for_date`.** Each (date, segment, color_group) now emits 4 traces in fixed order:

1. Markers (existing).
2. Fit line (existing; uses 2-point endpoint pair).
3. CI upper trace (line, no fill).
4. CI lower trace (line, `fill="tonexty"` with translucent fillcolor).

When the partition has fewer than `_MIN_OBS_FOR_CI` finite points, emit empty-array placeholders for traces 2, 3, 4 (markers stay).

**Layout updates inside `_build_scatter`:**

```python
fig.update_layout(
    height=700,
    template="simple_white",
    font={"family": "system-ui, sans-serif", "size": 13},
    legend={
        "orientation": "h",
        "yanchor": "bottom", "y": -0.20,
        "xanchor": "center", "x": 0.5,
    },
    updatemenus=[{
        # existing dropdown config,
        "x": 1.12,         # was 1.02; push further right to clear legend
        "y": 1.0,
    }],
    margin={"l": 60, "r": 160, "t": 60, "b": 120},  # b=120 for legend
)
fig.update_xaxes(showgrid=True, gridcolor="#e6e6e6", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#e6e6e6", zeroline=False)
```

**Marker style update:**

```python
marker={"color": color, "size": 9, "opacity": 0.85,
        "line": {"color": "white", "width": 1}}
```

**`beta_timeseries` updates:**

- Set `height=700` and `template="simple_white"` on layout.
- Legend horizontal, anchored bottom.
- Dropdown at `x=1.12`.
- Margin `b=120`.
- Tercile band fill alphas unchanged (0.12 / 0.06 / 0.12) — they're semantic regime colors, not chart palette.

### 3.2 `roro/report/html.py`

CSS tweaks only — bump `.container` `max-width` from 1200 to 1280 to comfortably fit the dropdown that now sits further right. No structural changes.

### 3.3 `pyproject.toml`

Add to `[project].dependencies`:

```toml
"scipy>=1.11,<2.0",
```

Add to mypy override list:

```toml
[[tool.mypy.overrides]]
module = "scipy.*"
ignore_missing_imports = true
```

---

## 4. Math — confidence band

For OLS fit `y_hat = a + b*x` over `n` points with `mean_x = x̄`:

```
SE(y_hat at x) = sqrt( SE_intercept² + (x - x̄)² × SE_slope² )
CI(x) = z × SE(y_hat at x)        with z = 1.96 for 95%
```

`scipy.stats.linregress` returns `slope, intercept, rvalue, pvalue, stderr` (stderr of slope) and an `intercept_stderr` attribute. Compute the band over 50 evenly-spaced x values across the data's `[min, max]`.

When `n < 3` or `var(x) == 0`: skip the band entirely (emit empty-array placeholder traces).

---

## 5. Test impact

Update existing tests in `tests/report/test_figures.py`:

- `test_scatter_vol_return_traces_per_frame_invariant`: expected count becomes `7 × 8 = 56`.
- `test_scatter_vol_return_dropdown_toggles_visibility`: 7 buttons, each with a 56-bool visibility mask, 8 trues in the active block.
- `test_scatter_vol_return_initial_visibility_is_full_only`: first 8 traces visible, next 48 hidden.
- `test_scatter_vol_return_has_segment_dropdown`: expected labels list extends to all 7.

Add new tests:

- `test_scatter_has_ci_ribbon_when_enough_points` — feeds a synthetic bundle (≥3 points per group), asserts the CI traces are present with non-empty x/y arrays.
- `test_scatter_skips_ci_ribbon_when_too_few_points` — feeds a partition with 2 points, asserts the CI traces have empty arrays.
- `test_scatter_em_color_is_green` — inspects an EM marker trace, confirms `marker.color == "#2ca02c"`.
- `test_scatter_figure_height_is_700` — `fig.layout.height == 700`.
- `test_scatter_legend_is_horizontal_below` — `fig.layout.legend.orientation == "h"` and `legend.y < 0`.
- `test_beta_timeseries_figure_height_is_700` — `fig.layout.height == 700`.

The byte-level reproducibility test in `test_orchestrate.py` should continue to pass because `scipy.stats.linregress` is deterministic on identical input.

---

## 6. Out of scope

- Switching to a different charting library (seaborn/matplotlib/altair).
- Adding more figure types (correlation panel, alerts strip, etc.).
- New CLI flags.
- Customizing the regime band colors.
- Themed dark/light toggle.
- v1 carryover follow-ups (snapshot type coercion, `cap_wtd` magic literal, slider title drift).

---

## 7. Acceptance criteria

1. `roro report --run-dir <dir>` produces an HTML where both scatter dropdowns expose all 7 segments and the legend is fully visible below each plot.
2. EM markers are green (`#2ca02c`); DM markers are blue (`#1f77b4`).
3. Each segment-color group with ≥3 finite points shows a translucent CI ribbon around its OLS line.
4. All three figures render at `height=700`.
5. Existing scatter dropdown tests pass with the updated trace-count math (56 traces × 7 buttons × 8-trace blocks).
6. Two `roro report` invocations on identical inputs produce byte-identical HTML.
7. mypy --strict + ruff sweep stay clean.
