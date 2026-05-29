# RoRo Visualization Layer v5 — Asset Split Within DM / EM Scatter

**Version:** 5.0.0
**Date:** 2026-05-28
**Status:** Draft (pending implementation)
**Scope:** When the scatter segment dropdown selects DM or EM, split the points into equity vs fixed-income sub-groups, each with a distinct color, marker shape, and its own OLS trend line + 95% CI band. Confined to `roro/report/figures.py`. No engine change.

---

## 1. Goal

The scatter figures (`scatter_vol_return`, `scatter_beta_return`) currently group points by region only — DM (blue) and EM (green) — with one trend line + CI per region. Selecting DM shows a single blue cloud with one fit; the user cannot see how equities behave versus bonds inside the block.

This iteration splits each region into its two asset classes when DM or EM is selected, so the user can read equities vs fixed income within DM (and within EM) as distinctly-colored point sets, each with its own trend line and confidence band. The Full view is unchanged.

---

## 2. Locked decisions

| Decision | Value |
|---|---|
| Distinguish assets by | **Color** (primary) **+ marker shape** (redundant cue) |
| DM_Eq | blue `#1f77b4`, circle |
| DM_FI | orange `#ff7f0e`, diamond |
| EM_Eq | green `#2ca02c`, circle |
| EM_FI | red `#d62728`, diamond |
| Full view | **unchanged** — 2 groups: DM (blue circle), EM (green circle), one fit+CI each |
| DM / EM views | 2 sub-groups (Eq, FI), each its own color + shape + OLS fit + 95% CI |
| Leaf views (DM_Eq, EM_Eq, DM_FI, EM_FI) | single sub-group with its color + shape + one fit |
| `_TRACES_PER_SEGMENT` | **stays 8** (≤2 sub-groups × 4 traces) — no architecture change |
| New dependencies | none |

---

## 3. Sub-group table

Each scatter segment maps to one or two sub-groups. A sub-group's member series come from the existing `_series_for_segment(meta, key)`; it carries a display label, a color, and a marker symbol.

| Segment | Sub-group A (label, series key, color, symbol) | Sub-group B |
|---|---|---|
| `Full` | "DM", `DM`, `#1f77b4`, circle | "EM", `EM`, `#2ca02c`, circle |
| `DM` | "DM Eq", `DM_Eq`, `#1f77b4`, circle | "DM FI", `DM_FI`, `#ff7f0e`, diamond |
| `EM` | "EM Eq", `EM_Eq`, `#2ca02c`, circle | "EM FI", `EM_FI`, `#d62728`, diamond |
| `DM_Eq` | "DM Eq", `DM_Eq`, `#1f77b4`, circle | — |
| `EM_Eq` | "EM Eq", `EM_Eq`, `#2ca02c`, circle | — |
| `DM_FI` | "DM FI", `DM_FI`, `#ff7f0e`, diamond | — |
| `EM_FI` | "EM FI", `EM_FI`, `#d62728`, diamond | — |

When a segment has only one sub-group, the second sub-group's four traces are emitted as empty-array placeholders so the per-segment trace count stays invariant at 8.

---

## 4. Changes — `roro/report/figures.py`

### 4.1 Color constants

Keep `COLOR_DM = "#1f77b4"` and `COLOR_EM = "#2ca02c"`. Add:

```python
COLOR_DM_FI: str = "#ff7f0e"  # orange
COLOR_EM_FI: str = "#d62728"  # red
```

(DM_Eq reuses `COLOR_DM`; EM_Eq reuses `COLOR_EM`.)

### 4.2 Sub-group descriptor + table

Add a small frozen descriptor and the segment→sub-groups mapping near the scatter helpers:

```python
@dataclass(frozen=True)
class _SubGroup:
    label: str       # legend text, e.g. "DM Eq"
    series_key: SegmentFilter  # key into _series_for_segment
    color: str
    symbol: str      # plotly marker symbol, e.g. "circle" | "diamond"


_SEGMENT_SUBGROUPS: dict[SegmentFilter, tuple[_SubGroup, ...]] = {
    "Full": (
        _SubGroup("DM", "DM", COLOR_DM, "circle"),
        _SubGroup("EM", "EM", COLOR_EM, "circle"),
    ),
    "DM": (
        _SubGroup("DM Eq", "DM_Eq", COLOR_DM, "circle"),
        _SubGroup("DM FI", "DM_FI", COLOR_DM_FI, "diamond"),
    ),
    "EM": (
        _SubGroup("EM Eq", "EM_Eq", COLOR_EM, "circle"),
        _SubGroup("EM FI", "EM_FI", COLOR_EM_FI, "diamond"),
    ),
    "DM_Eq": (_SubGroup("DM Eq", "DM_Eq", COLOR_DM, "circle"),),
    "EM_Eq": (_SubGroup("EM Eq", "EM_Eq", COLOR_EM, "circle"),),
    "DM_FI": (_SubGroup("DM FI", "DM_FI", COLOR_DM_FI, "diamond"),),
    "EM_FI": (_SubGroup("EM FI", "EM_FI", COLOR_EM_FI, "diamond"),),
}

_SUBGROUPS_PER_SEGMENT: int = 2  # max sub-groups in any segment; pad shorter ones
```

`dataclass` is already imported in the engine layer; add the import to `figures.py` if absent.

### 4.3 Refactor `_scatter_traces_for_date`

Replace the hardcoded `for color_label, color in (("DM", COLOR_DM), ("EM", COLOR_EM)):` loop. The new body iterates the segment's sub-groups (padding to `_SUBGROUPS_PER_SEGMENT` with empty placeholders), and for each sub-group:

- Member series = `_series_for_segment(bundle.meta, subgroup.series_key)`, intersected with the panel columns.
- Markers trace: `marker={"color": subgroup.color, "symbol": subgroup.symbol, "size": 9, "opacity": 0.85, "line": {"color": "white", "width": 1}}`, `name=subgroup.label`, hover text `"<country>_<asset>"` (unchanged hovertemplate).
- Fit + CI traces via the existing `_ols_with_ci` / `_ci_band_traces`, colored `subgroup.color`.
- Empty/degenerate partitions emit empty-array placeholders (markers + fit + ci_upper + ci_lower) so each sub-group always contributes exactly 4 traces.

`_TRACES_PER_SEGMENT` stays `_SUBGROUPS_PER_SEGMENT * 4 = 8`. The existing empty-placeholder helpers are reused/extended to accept a color and symbol.

### 4.4 Unchanged

- `_visibility_mask`, `_segment_axis_range`, `_build_scatter` layout (legend, slider, dropdown, height, margins), `scatter_vol_return`, `scatter_beta_return`, `beta_timeseries`, regime bands.
- `SCATTER_SEGMENTS` (still 7), dropdown wiring, per-segment axis ranges.

---

## 5. Tests — `tests/report/test_figures.py`

Update existing name-dependent tests, then add new ones. Use a hand-crafted bundle with both DM and EM, both assets, and cross-sectional spread (the shared `tiny_xlsx` fixture has 2 DM + 2 EM countries, each with Eq + FI — sufficient for membership/color/symbol assertions; a varied bundle is only needed where a non-degenerate fit must exist).

- `test_scatter_dm_view_splits_eq_fi` — in the `DM` button's trace block, the two marker traces are named "DM Eq" and "DM FI", colored `#1f77b4` and `#ff7f0e`, with symbols "circle" and "diamond".
- `test_scatter_em_view_splits_eq_fi` — `EM` block marker traces named "EM Eq"/"EM FI", colored `#2ca02c`/`#d62728`, symbols circle/diamond.
- `test_scatter_full_view_unchanged_region_groups` — `Full` block marker traces named "DM"/"EM", colored `#1f77b4`/`#2ca02c`, both symbol "circle".
- `test_scatter_traces_per_segment_still_8` — `_TRACES_PER_SEGMENT == 8`; every frame has `7 * 8 == 56` traces.
- `test_scatter_dm_fi_diamond_symbol` — the FI marker trace uses `marker.symbol == "diamond"`.
- Update `test_scatter_em_marker_trace_uses_green` (currently asserts a trace named `"Full:EM"`): the Full EM marker trace is now named `"EM"` and colored `#2ca02c`.
- The existing dropdown-visibility / initial-visibility / per-segment-range / hover tests continue to pass (trace count and structure unchanged; only marker styling + names differ). Update any that hard-code the old `"Full:DM"` / `"{segment}:{label}"` names.

Reproducibility (`test_build_report_reproducible`) holds — the mapping is static and deterministic.

---

## 6. Out of scope

- Splitting the Full view into 4 groups (explicitly kept as 2).
- Per-asset regime bands on the β time-series (bands stay segment-level).
- Configurable palette via CLI/config (module constants suffice).
- LatAm asset split (LatAm is not a scatter segment).

---

## 7. Acceptance criteria

1. Selecting DM shows two distinctly-colored, distinctly-shaped point clouds — DM Eq (blue ●) and DM FI (orange ◆) — each with its own dashed trend line and 95% CI ribbon.
2. Selecting EM shows EM Eq (green ●) and EM FI (red ◆) likewise.
3. The Full view is visually unchanged (DM blue ●, EM green ●, one fit each).
4. The legend names read "DM Eq" / "DM FI" / "EM Eq" / "EM FI" (no more redundant "DM:DM").
5. `_TRACES_PER_SEGMENT` is still 8; total scatter traces still 56.
6. Two consecutive `roro report` runs produce byte-identical HTML.
7. mypy --strict + ruff stay clean.
