# Volatility-Percentile Breadth Heatmap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two volatility-breadth heatmaps to the report — a sorted-rank breadth view and a per-asset identity view — of 63-day realized-vol percentiles ranked against each asset's expanding ≥5Y history, with an All/Eq/FI class toggle.

**Architecture:** New `roro/report/vol_breadth.py` computes simple realized vol + an expanding-min-5Y per-series percentile matrix. `load_bundle` adds a full-history `vol_pct` to `DataBundle`. Two new `figures.py` heatmap builders shape per-class z-matrices. `assemble` is refactored to explicit `FigureSpec`s (positional title/id constants break once figures are conditional), and `orchestrate` appends the two heatmaps.

**Tech Stack:** Python 3.12 (uv `.venv`), pandas, numpy, Plotly, pytest. mypy strict, ruff, `filterwarnings=["error"]`.

**Spec:** `docs/superpowers/specs/2026-06-05-roro-vol-breadth-heatmap-design.md`

---

## ENVIRONMENT (every task)
- Canonical env = uv-managed `.venv` at Python 3.12. The machine default `python` is 3.14 — DO NOT use it. Run ALL python/pytest/mypy/ruff via `.venv\Scripts\python.exe`.
- Branch `feat/hmm-report-viz` is already checked out (heatmap merges together with the HMM viz). Do NOT switch branches.

## Task order rationale
Tasks 5 (assemble refactor) and 6 (orchestrate appends heatmaps) are split so each leaves the suite green: Task 5 refactors `assemble` AND updates `orchestrate` to build specs for the *existing* figures only (report visually unchanged); Task 6 then appends the two heatmap specs.

---

## Task 1: vol_breadth.py — realized_vol + vol_percentile_matrix

**Files:**
- Create: `roro/report/vol_breadth.py`
- Test: `tests/report/test_vol_breadth.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/report/test_vol_breadth.py`:
```python
import numpy as np
import pandas as pd

from roro.report.vol_breadth import realized_vol, vol_percentile_matrix


def test_realized_vol_annualizes() -> None:
    idx = pd.bdate_range("2010-01-01", periods=10)
    # constant daily return → zero stdev → zero vol
    r = pd.DataFrame({"A": [0.01] * 10}, index=idx)
    v = realized_vol(r, window=5)
    assert v["A"].iloc[:4].isna().all()      # window-1 warmup NaN
    assert abs(float(v["A"].iloc[5])) < 1e-9  # zero variance → zero vol


def test_realized_vol_matches_manual() -> None:
    idx = pd.bdate_range("2010-01-01", periods=6)
    r = pd.DataFrame({"A": [0.0, 0.02, -0.02, 0.02, -0.02, 0.0]}, index=idx)
    v = realized_vol(r, window=3)
    # window of last 3 returns at t=2: [0.0, 0.02, -0.02], sample std * sqrt(252)
    expected = float(np.std([0.0, 0.02, -0.02], ddof=1) * np.sqrt(252))
    assert abs(float(v["A"].iloc[2]) - expected) < 1e-9


def test_vol_percentile_matrix_expanding_and_min_history() -> None:
    idx = pd.bdate_range("2010-01-01", periods=10)
    # strictly increasing vol → each new point is the max → percentile 1.0
    vol = pd.DataFrame({"A": np.arange(1.0, 11.0)}, index=idx)
    pct = vol_percentile_matrix(vol, min_history_days=4)
    assert pct["A"].iloc[:3].isna().all()           # < min_history → NaN
    assert (pct["A"].iloc[3:] == 1.0).all()         # each new max → pct 1.0
    assert (pct.dropna().to_numpy() >= 0).all() and (pct.dropna().to_numpy() <= 1).all()


def test_vol_percentile_matrix_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=300)
    vol = pd.DataFrame({"A": rng.normal(0.2, 0.05, 300)}, index=idx)
    fast = vol_percentile_matrix(vol, min_history_days=50)

    def brute(s: pd.Series) -> pd.Series:
        out = []
        for t in range(len(s)):
            hist = s.iloc[: t + 1].dropna()
            if len(hist) < 50:
                out.append(np.nan)
            else:
                out.append(float((hist <= s.iloc[t]).sum()) / len(hist))
        return pd.Series(out, index=s.index)

    expected = brute(vol["A"])
    pd.testing.assert_series_equal(fast["A"], expected, check_names=False)
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_vol_breadth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roro.report.vol_breadth'`.

- [ ] **Step 3: Implement `roro/report/vol_breadth.py`**

```python
"""Simple realized vol + expanding-min-5Y per-series volatility percentile."""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from roro.returns import TRADING_DAYS_PER_YEAR


def realized_vol(daily_returns: pd.DataFrame, *, window: int = 63) -> pd.DataFrame:
    """Simple annualized realized vol: rolling stdev of daily log returns × √252."""
    sigma = daily_returns.rolling(window).std()
    return sigma * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def _expanding_percentile(values: np.ndarray, min_history_days: int) -> np.ndarray:  # type: ignore[type-arg]
    """For each t with a valid value: fraction of prior+current valid values ≤ value[t].

    NaN until ≥ min_history_days valid readings have been seen. Exact; uses an
    incrementally maintained sorted list (bisect) — C-level inserts keep this fast
    for the offline report build.
    """
    hist: list[float] = []
    out = np.full(values.shape[0], np.nan)
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        bisect.insort(hist, float(v))
        if len(hist) >= min_history_days:
            cnt = bisect.bisect_right(hist, float(v))  # # of values ≤ v (incl. itself)
            out[i] = cnt / len(hist)
    return out


def vol_percentile_matrix(
    vol_full: pd.DataFrame, *, min_history_days: int = 1260
) -> pd.DataFrame:
    """Per series: expanding percentile rank of vol(t) vs all prior valid vol ≤ t.

    Returns a date × series frame in [0, 1], NaN during the per-series warmup.
    """
    cols = {
        col: _expanding_percentile(vol_full[col].to_numpy(dtype=float), min_history_days)
        for col in vol_full.columns
    }
    return pd.DataFrame(cols, index=vol_full.index)
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_vol_breadth.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/vol_breadth.py` and `.venv\Scripts\python.exe -m ruff check roro/report/vol_breadth.py tests/report/test_vol_breadth.py`
Then:
```bash
git add roro/report/vol_breadth.py tests/report/test_vol_breadth.py
git commit -m "feat(report): realized_vol + expanding-min-5Y vol_percentile_matrix"
```

---

## Task 2: DataBundle.vol_pct + load computes it

**Files:**
- Modify: `roro/report/bundle.py`
- Modify: `roro/report/load.py`
- Test: `tests/report/test_load.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_load.py`:
```python
def test_load_bundle_vol_pct_full_history_with_warmup(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.vol_pct is not None
    # full history (longer than the 252d scatter window)
    assert len(bundle.vol_pct.index) > len(bundle.dates)
    # values are percentiles in [0,1] where present
    vals = bundle.vol_pct.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert (finite >= 0).all() and (finite <= 1).all()
```
(Add `import numpy as np` to the test file imports if not present.)

NOTE: the `tiny_xlsx` fixture spans 2020-01-02..2024-12-31 (~1300 bdays). With `min_history_days=1260` the warmup nearly fills the range, so `vol_pct` may be mostly NaN on tiny data — that is fine; the test only checks the frame exists, spans full history, and is in-range where finite. Do NOT assert non-NaN values on the tiny fixture.

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py::test_load_bundle_vol_pct_full_history_with_warmup -v`
Expected: FAIL — `AttributeError: 'DataBundle' object has no attribute 'vol_pct'`.

- [ ] **Step 3: Add the DataBundle field**

In `roro/report/bundle.py`, append to `DataBundle` (after the `seg_hmm_*` fields):
```python
    vol_pct: pd.DataFrame | None = None
```

- [ ] **Step 4: Compute vol_pct in load_bundle**

In `roro/report/load.py`:
(a) Add imports at top:
```python
from roro.report.vol_breadth import realized_vol, vol_percentile_matrix
```
(`daily_log_returns` is already imported from `roro.returns`.)

(b) `_build_series_panels` currently computes `daily = daily_log_returns(per_series_prices)` then slices vol to `window`. We need the FULL-history percentile. The cleanest minimal change: have `load_bundle` compute `vol_pct` from the full per-series price frame. Refactor `_build_series_panels` to ALSO return the full-history `vol_pct`:

In `_build_series_panels`, after `daily = daily_log_returns(per_series_prices)` and the `common` column filter (where `daily = daily[common]` etc.), add:
```python
    vol_full = realized_vol(daily, window=63)
    vol_pct = vol_percentile_matrix(vol_full, min_history_days=1260)
```
Change the function's return to append `vol_pct` (full history, NOT sliced to `tail_dates`):
```python
    return (
        vol.loc[tail_dates],
        ret_3m.loc[tail_dates],
        beta.loc[tail_dates],
        meta.loc[common],
        vol_pct,                # full history
    )
```
Update the unpacking in `load_bundle`:
```python
    vol, ret_3m, beta_vs_global, meta, vol_pct = _build_series_panels(xlsx_path, window=window)
```
And add `vol_pct=vol_pct` to the `return DataBundle(...)` call.

- [ ] **Step 5: Run tests, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py -v`
Expected: PASS (new + existing).

- [ ] **Step 6: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/bundle.py roro/report/load.py` and `.venv\Scripts\python.exe -m ruff check roro/report/bundle.py roro/report/load.py tests/report/test_load.py`
Then:
```bash
git add roro/report/bundle.py roro/report/load.py tests/report/test_load.py
git commit -m "feat(report): compute full-history vol_pct in load_bundle"
```

---

## Task 3: vol_breadth_heatmap (sorted-rank) figure

**Files:**
- Modify: `roro/report/figures.py`
- Test: `tests/report/test_figures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_figures.py` (imports already include pandas/DataBundle; add `import numpy as np` if absent, and `from roro.report.figures import vol_breadth_heatmap`):
```python
def _bundle_with_vol_pct() -> DataBundle:
    idx = pd.bdate_range("2014-01-02", periods=8)
    # 3 series, 2 Eq + 1 FI; distinct percentiles per day
    vp = pd.DataFrame(
        {"A_Eq": [0.9] * 8, "B_Eq": [0.5] * 8, "C_FI": [0.1] * 8}, index=idx
    )
    meta = pd.DataFrame(
        {"country": ["A", "B", "C"], "asset": ["Eq", "Eq", "FI"],
         "segment": ["DM", "EM", "DM"], "weight": [1.0, 1.0, 1.0]},
        index=["A_Eq", "B_Eq", "C_FI"],
    )
    empty = pd.DataFrame(index=idx)
    return DataBundle(
        run_date=idx[-1], methodology_version="1.0.0", dates=pd.DatetimeIndex(idx),
        vol=empty, ret_3m=empty, beta_vs_global=empty, meta=meta,
        seg_beta=empty, seg_tercile=empty, vol_pct=vp,
    )


def test_vol_breadth_heatmap_columns_sorted_descending() -> None:
    fig = vol_breadth_heatmap(_bundle_with_vol_pct())
    assert len(fig.data) == 1
    z = np.asarray(fig.data[0].z, dtype=float)
    # "All" subset (3 series); each column sorted descending: 0.9, 0.5, 0.1
    col0 = z[:, 0]
    assert col0[0] >= col0[1] >= col0[2]
    assert abs(col0[0] - 0.9) < 1e-9 and abs(col0[2] - 0.1) < 1e-9
    # class dropdown: All / Eq / FI
    menus = fig.layout.updatemenus
    assert len(menus) == 1
    assert [b.label for b in menus[0].buttons] == ["All", "Eq", "FI"]
    assert fig.data[0].zmin == 0.0 and fig.data[0].zmax == 1.0
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_vol_breadth_heatmap_columns_sorted_descending -v`
Expected: FAIL — `ImportError: cannot import name 'vol_breadth_heatmap'`.

- [ ] **Step 3: Implement**

Add to `roro/report/figures.py` (near the other builders). Add `from roro.segments import ASSET_EQ, ASSET_FI` to imports if not present:
```python
VOL_COLORSCALE: list[list[object]] = [[0.0, "#ffffff"], [0.5, "#f4a582"], [1.0, "#b2182b"]]


def _class_subsets(bundle: DataBundle) -> dict[str, list[str]]:
    """series_id lists for All / Eq / FI, restricted to vol_pct columns."""
    assert bundle.vol_pct is not None
    cols = [c for c in bundle.vol_pct.columns if c in bundle.meta.index]
    eq = [c for c in cols if bundle.meta.loc[c, "asset"] == ASSET_EQ]
    fi = [c for c in cols if bundle.meta.loc[c, "asset"] == ASSET_FI]
    return {"All": cols, "Eq": eq, "FI": fi}


def _breadth_z(vol_pct: pd.DataFrame, series_ids: list[str]) -> np.ndarray:  # type: ignore[type-arg]
    """(rank, date) matrix: each date column = that day's percentiles sorted descending."""
    arr = vol_pct[series_ids].to_numpy(dtype=float)  # (T, M)
    n_dates, n_series = arr.shape
    z = np.full((n_series, n_dates), np.nan)
    for j in range(n_dates):
        valid = arr[j][~np.isnan(arr[j])]
        valid_desc = np.sort(valid)[::-1]
        z[: valid_desc.shape[0], j] = valid_desc
    return z


def vol_breadth_heatmap(bundle: DataBundle) -> go.Figure:
    """Sorted-rank breadth heatmap of vol percentiles, with an All/Eq/FI class toggle."""
    assert bundle.vol_pct is not None
    subsets = _class_subsets(bundle)
    x = bundle.vol_pct.index
    z_by_class = {k: _breadth_z(bundle.vol_pct, ids) for k, ids in subsets.items()}
    default = "All"
    z0 = z_by_class[default]

    trace = go.Heatmap(
        z=z0,
        x=x,
        y=list(range(1, z0.shape[0] + 1)),
        zmin=0.0,
        zmax=1.0,
        colorscale=VOL_COLORSCALE,
        colorbar={"title": "vol pctile"},
        hovertemplate="%{x|%Y-%m-%d}<br>rank %{y}<br>pctile=%{z:.2f}<extra></extra>",
    )

    buttons = [
        {
            "method": "update",
            "label": k,
            "args": [
                {"z": [z_by_class[k]], "y": [list(range(1, z_by_class[k].shape[0] + 1))]},
                {"title": f"Volatility breadth (sorted percentile) — {k}"},
            ],
        }
        for k in ("All", "Eq", "FI")
    ]

    return go.Figure(
        data=[trace],
        layout=go.Layout(
            title=f"Volatility breadth (sorted percentile) — {default}",
            height=700,
            template="simple_white",
            font={"family": "system-ui, -apple-system, sans-serif", "size": 13},
            margin={"l": 60, "r": 200, "t": 60, "b": 120},
            xaxis={"title": "Date"},
            yaxis={"title": "Asset rank (1 = highest vol pctile)", "autorange": "reversed"},
            updatemenus=[
                {"type": "dropdown", "showactive": True, "buttons": buttons,
                 "x": 1.12, "y": 1.0, "xanchor": "left", "yanchor": "top"}
            ],
        ),
    )
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_vol_breadth_heatmap_columns_sorted_descending -v`

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/figures.py` and `.venv\Scripts\python.exe -m ruff check roro/report/figures.py tests/report/test_figures.py`
Then:
```bash
git add roro/report/figures.py tests/report/test_figures.py
git commit -m "feat(report): vol_breadth_heatmap (sorted-rank) with class toggle"
```

---

## Task 4: vol_pct_asset_heatmap (identity) figure

**Files:**
- Modify: `roro/report/figures.py`
- Test: `tests/report/test_figures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_figures.py` (reuses `_bundle_with_vol_pct` from Task 3; add `from roro.report.figures import vol_pct_asset_heatmap`):
```python
def test_vol_pct_asset_heatmap_rows_ordered_by_mean_desc() -> None:
    fig = vol_pct_asset_heatmap(_bundle_with_vol_pct())
    assert len(fig.data) == 1
    # All subset, ordered by mean percentile descending: A_Eq(0.9), B_Eq(0.5), C_FI(0.1)
    assert list(fig.data[0].y) == ["A_Eq", "B_Eq", "C_FI"]
    menus = fig.layout.updatemenus
    assert [b.label for b in menus[0].buttons] == ["All", "Eq", "FI"]
    assert fig.data[0].zmin == 0.0 and fig.data[0].zmax == 1.0
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_vol_pct_asset_heatmap_rows_ordered_by_mean_desc -v`
Expected: FAIL — `ImportError: cannot import name 'vol_pct_asset_heatmap'`.

- [ ] **Step 3: Implement**

Add to `roro/report/figures.py` (after `vol_breadth_heatmap`; reuses `_class_subsets`, `VOL_COLORSCALE`):
```python
def _asset_z_y(vol_pct: pd.DataFrame, series_ids: list[str]) -> tuple[np.ndarray, list[str]]:  # type: ignore[type-arg]
    """(series, date) matrix + row labels, rows ordered by mean percentile descending."""
    sub = vol_pct[series_ids]
    order = list(sub.mean().sort_values(ascending=False).index)
    z = sub[order].to_numpy(dtype=float).T  # (series, date)
    return z, order


def vol_pct_asset_heatmap(bundle: DataBundle) -> go.Figure:
    """Per-asset vol-percentile heatmap (rows = series, mean-ordered) with class toggle."""
    assert bundle.vol_pct is not None
    subsets = _class_subsets(bundle)
    x = bundle.vol_pct.index
    zy_by_class = {k: _asset_z_y(bundle.vol_pct, ids) for k, ids in subsets.items()}
    default = "All"
    z0, y0 = zy_by_class[default]

    trace = go.Heatmap(
        z=z0,
        x=x,
        y=y0,
        zmin=0.0,
        zmax=1.0,
        colorscale=VOL_COLORSCALE,
        colorbar={"title": "vol pctile"},
        hovertemplate="%{x|%Y-%m-%d}<br>%{y}<br>pctile=%{z:.2f}<extra></extra>",
    )

    buttons = []
    for k in ("All", "Eq", "FI"):
        zk, yk = zy_by_class[k]
        buttons.append(
            {
                "method": "update",
                "label": k,
                "args": [
                    {"z": [zk], "y": [yk]},
                    {"title": f"Volatility percentile by asset — {k}"},
                ],
            }
        )

    return go.Figure(
        data=[trace],
        layout=go.Layout(
            title=f"Volatility percentile by asset — {default}",
            height=700,
            template="simple_white",
            font={"family": "system-ui, -apple-system, sans-serif", "size": 13},
            margin={"l": 120, "r": 200, "t": 60, "b": 120},
            xaxis={"title": "Date"},
            yaxis={"title": "Asset", "autorange": "reversed"},
            updatemenus=[
                {"type": "dropdown", "showactive": True, "buttons": buttons,
                 "x": 1.12, "y": 1.0, "xanchor": "left", "yanchor": "top"}
            ],
        ),
    )
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py -v`
Expected: PASS (this + all prior figures tests).

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/figures.py` and `.venv\Scripts\python.exe -m ruff check roro/report/figures.py tests/report/test_figures.py`
Then:
```bash
git add roro/report/figures.py tests/report/test_figures.py
git commit -m "feat(report): vol_pct_asset_heatmap (identity) with class toggle"
```

---

## Task 5: assemble → FigureSpec refactor (+ orchestrate to specs, no heatmaps yet)

**Files:**
- Modify: `roro/report/html.py`
- Modify: `roro/report/orchestrate.py`
- Test: `tests/report/test_html.py`

- [ ] **Step 1: Rewrite the html tests for the FigureSpec API**

In `tests/report/test_html.py`, add a spec helper near the top (after imports):
```python
from roro.report.html import FigureSpec


def _spec(fig: go.Figure, i: int = 0) -> FigureSpec:
    return FigureSpec(figure=fig, div_id=f"fig_{i}", title=f"Section {i}")


def _specs(n: int) -> list[FigureSpec]:
    return [_spec(_dummy_fig(), i) for i in range(n)]
```
Then update EVERY `assemble(...)` call in the file to pass a `list[FigureSpec]` instead of a `list[go.Figure]`:
- Replace `assemble([_dummy_fig(), _dummy_fig(), _dummy_fig()], run_date=..., methodology_version=...)` with `assemble(_specs(3), run_date=..., methodology_version=...)` in `test_assemble_returns_string`, `test_assemble_contains_three_plotly_divs`, `test_assemble_contains_run_date_and_version`, `test_assemble_is_valid_html5`, `test_assemble_container_max_width_is_1280`.
- `test_assemble_contains_three_plotly_divs`: keep `assert out.count('class="plotly-graph-div"') == 3`.
- Delete `test_assemble_requires_exactly_three_figures` (the count guard is removed) and replace with:
  ```python
  def test_assemble_rejects_empty() -> None:
      import pytest  # noqa: PLC0415
      with pytest.raises(ValueError, match="at least one figure"):
          assemble([], run_date=pd.Timestamp("2024-03-29"), methodology_version="1.0.0")
  ```
- Update `test_assemble_accepts_four_figures_with_hmm_toggle`: build 4 specs where the 3rd is the beta chart with a known div_id, and pass `beta_div_id`:
  ```python
  def test_assemble_accepts_four_figures_with_hmm_toggle() -> None:
      specs = [
          _spec(_dummy_fig(), 0), _spec(_dummy_fig(), 1),
          FigureSpec(_dummy_fig(), "fig_beta_ts", "Segment β with regime bands"),
          FigureSpec(_dummy_fig(), "fig_hmm_probs", "HMM regime probabilities"),
      ]
      lookup = {"global": {"percentile": [], "hmm": [{"type": "rect", "x0": "2020-01-02",
                "x1": "2020-02-01", "y0": 0, "y1": 1, "fillcolor": "rgba(1,1,1,0.5)",
                "line": {"width": 0}, "layer": "below"}]}}
      html = assemble(specs, run_date=pd.Timestamp("2024-12-31"), methodology_version="1.0.0",
                      beta_div_id="fig_beta_ts", beta_band_lookup=lookup)
      assert "HMM regime probabilities" in html
      assert 'id="hmm-band-method"' in html
      assert "plotly_relayout" in html
      assert "fig_beta_ts" in html
      assert html.index('id="hmm-band-method"') < html.index("Segment β with regime bands")
      assert html.index("Segment β with regime bands") < html.index("HMM regime probabilities")
  ```
- Update `test_assemble_three_figures_unchanged_without_lookup`:
  ```python
  def test_assemble_three_figures_unchanged_without_lookup() -> None:
      html = assemble(_specs(3), run_date=pd.Timestamp("2024-12-31"), methodology_version="1.0.0")
      assert "hmm-band-method" not in html
      assert "HMM regime probabilities" not in html
  ```
Remove the now-unused `_three_figs` helper if present.

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_html.py -v`
Expected: FAIL — `ImportError: cannot import name 'FigureSpec'`.

- [ ] **Step 3: Refactor `assemble` in `roro/report/html.py`**

Add a `FigureSpec` dataclass at module top (after imports; `from dataclasses import dataclass` — add if absent):
```python
@dataclass(frozen=True)
class FigureSpec:
    figure: go.Figure
    div_id: str
    title: str
```
Remove the positional `DIV_IDS`, `SECTION_TITLES`, `_MIN_FIGURES`, `_MAX_FIGURES` constants. Replace the `assemble` signature + body:
```python
def assemble(
    specs: list[FigureSpec],
    *,
    run_date: pd.Timestamp,
    methodology_version: str,
    beta_div_id: str | None = None,
    beta_band_lookup: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> str:
    """Assemble figure specs into one self-contained HTML report page."""
    if not specs:
        raise ValueError("assemble requires at least one figure")

    fig_html_blocks: list[str] = []
    for idx, spec in enumerate(specs):
        include: str | bool = "cdn" if idx == 0 else False
        block = spec.figure.to_html(
            include_plotlyjs=include,
            full_html=False,
            div_id=spec.div_id,
            config={"displaylogo": False},
        )
        fig_html_blocks.append(f'<section><h2>{spec.title}</h2>{block}</section>')

    if beta_band_lookup is not None and beta_div_id is not None:
        toggle = _band_toggle_markup(beta_div_id, beta_band_lookup)
        # insert directly above the beta-timeseries section it controls
        beta_pos = next((i for i, s in enumerate(specs) if s.div_id == beta_div_id), len(fig_html_blocks))
        fig_html_blocks.insert(beta_pos, toggle)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RoRo Risk-Regime Report — {run_date.date()}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<header>
<h1>RoRo Risk-Regime Report</h1>
<div class="meta">Run date: {run_date.date()} &nbsp;|&nbsp;
Methodology: v{methodology_version}</div>
</header>
{''.join(fig_html_blocks)}
<footer>Generated by RoRo · methodology v{methodology_version}</footer>
</div>
</body>
</html>
"""
```
Keep `_band_toggle_markup` unchanged (it already takes a `beta_div_id` string). Update the `assemble` docstring's args accordingly.

- [ ] **Step 4: Update `orchestrate.build_report` to build specs (existing figures only — NO heatmaps yet)**

In `roro/report/orchestrate.py`, change the import to also bring `FigureSpec`:
```python
from roro.report.html import FigureSpec, assemble
```
Replace the figures/assemble block:
```python
    bundle = load_bundle(run_dir, xlsx_path, window=window)
    specs = [
        FigureSpec(scatter_vol_return(bundle), "fig_scatter_vol", "Risk vs Return"),
        FigureSpec(scatter_beta_return(bundle), "fig_scatter_beta", "Beta vs Return"),
        FigureSpec(beta_timeseries(bundle), "fig_beta_ts", "Segment β with regime bands"),
    ]
    lookup = None
    if bundle.seg_hmm_label is not None:
        specs.append(FigureSpec(
            regime_probability_area(bundle), "fig_hmm_probs", "HMM regime probabilities"
        ))
        lookup = beta_band_lookup(bundle)
    html = assemble(
        specs,
        run_date=bundle.run_date,
        methodology_version=bundle.methodology_version,
        beta_div_id="fig_beta_ts",
        beta_band_lookup=lookup,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
```

- [ ] **Step 5: Run report suite, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report -q`
Expected: PASS (test_html rewritten; test_orchestrate/test_e2e still green — output structurally identical, just specs-driven).

- [ ] **Step 6: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/html.py roro/report/orchestrate.py` and `.venv\Scripts\python.exe -m ruff check roro/report/html.py roro/report/orchestrate.py tests/report/test_html.py`
Then:
```bash
git add roro/report/html.py roro/report/orchestrate.py tests/report/test_html.py
git commit -m "refactor(report): assemble takes explicit FigureSpec list (div_id+title)"
```

---

## Task 6: orchestrate appends the two heatmaps

**Files:**
- Modify: `roro/report/orchestrate.py`
- Test: `tests/report/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_orchestrate.py`:
```python
def test_build_report_includes_vol_heatmaps(minimal_run_dir, tiny_xlsx, tmp_path) -> None:
    out = build_report(minimal_run_dir, tiny_xlsx, tmp_path / "r.html", window=21)
    html = out.read_text(encoding="utf-8")
    assert "fig_vol_breadth" in html
    assert "fig_vol_assets" in html
    assert "Volatility breadth (sorted percentile)" in html
    assert "Volatility percentile by asset" in html
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_orchestrate.py::test_build_report_includes_vol_heatmaps -v`
Expected: FAIL — heatmap divs absent.

- [ ] **Step 3: Append heatmap specs in build_report**

In `roro/report/orchestrate.py`, update the figures import to add the heatmap builders:
```python
from roro.report.figures import (
    beta_band_lookup,
    beta_timeseries,
    regime_probability_area,
    scatter_beta_return,
    scatter_vol_return,
    vol_breadth_heatmap,
    vol_pct_asset_heatmap,
)
```
After the HMM `if` block and BEFORE the `assemble(...)` call, append:
```python
    specs.append(FigureSpec(
        vol_breadth_heatmap(bundle), "fig_vol_breadth",
        "Volatility breadth (sorted percentile)",
    ))
    specs.append(FigureSpec(
        vol_pct_asset_heatmap(bundle), "fig_vol_assets",
        "Volatility percentile by asset",
    ))
```
(`bundle.vol_pct` is always populated by load, so the heatmaps always build.)

- [ ] **Step 4: Run tests, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_orchestrate.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/orchestrate.py` and `.venv\Scripts\python.exe -m ruff check roro/report/orchestrate.py tests/report/test_orchestrate.py`
Then:
```bash
git add roro/report/orchestrate.py tests/report/test_orchestrate.py
git commit -m "feat(report): build_report appends vol-breadth + asset heatmaps"
```

---

## Task 7: Full report-suite verification + real e2e

**Files:** none (verification)

- [ ] **Step 1: Full report suite + fast suite**

Run: `.venv\Scripts\python.exe -m pytest tests/report -v` then `.venv\Scripts\python.exe -m pytest -m "not slow" -q`
Expected: all PASS.

- [ ] **Step 2: Type + lint the report package**

Run: `.venv\Scripts\python.exe -m mypy roro/report` and `.venv\Scripts\python.exe -m ruff check roro/report tests/report`
Expected: clean.

- [ ] **Step 3: Build a real report from the HMM eval run and verify the heatmaps**

Run:
```bash
.venv\Scripts\roro.exe report --run-dir outputs/hmm_eval/2026-05-26 --xlsx data.xlsx --out outputs/hmm_eval/report_hmm.html
```
Then:
```bash
grep -c "fig_vol_breadth" outputs/hmm_eval/report_hmm.html
grep -c "fig_vol_assets" outputs/hmm_eval/report_hmm.html
```
Expected: both ≥ 1. Also confirm the report still contains `fig_hmm_probs` and `hmm-band-method` (HMM viz intact). (`outputs/` is gitignored — artifact not committed.) On the real data (history from 2008), the vol_pct warmup (min 1260 ≈ 5Y) means the heatmaps populate from ~2014 onward — expected.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test(report): verify vol-breadth heatmaps end-to-end on real run"
```
(Skip if no changes were needed.)

---

## Self-Review

**Spec coverage:** §4 data/compute → Tasks 1 (vol_breadth.py) + 2 (load/bundle); §5 breadth heatmap → Task 3; §6 asset heatmap → Task 4; §7 assemble FigureSpec refactor → Task 5, orchestrate heatmaps → Task 6; §8 testing → embedded per task + Task 7; two-windows distinction (§3) → Task 1 (`realized_vol` window=63 vs `vol_percentile_matrix` min_history_days=1260). All covered.

**Placeholder scan:** No TBD/TODO. The expanding-percentile algorithm is concrete (bisect.insort + bisect_right) with a brute-force correctness anchor test. The test-rewrite in Task 5 enumerates exactly which `assemble(...)` calls change and how. Heatmap z/y construction is fully shown.

**Type consistency:** `vol_pct` (DataBundle field, Tasks 2-6) consistent. `realized_vol(window=63)` / `vol_percentile_matrix(min_history_days=1260)` signatures match between Task 1 and Task 2's call. `_class_subsets` / `_breadth_z` / `_asset_z_y` defined in Task 3-4 and reused consistently. `FigureSpec(figure, div_id, title)` identical across html.py (Task 5), orchestrate (Tasks 5-6), and test_html (Task 5). `beta_div_id="fig_beta_ts"` (string) replaces the old positional `beta_div_index=2`; the β spec uses div_id `"fig_beta_ts"`, so the toggle inserts above it. Heatmap div_ids `fig_vol_breadth`/`fig_vol_assets` consistent between Task 6 and Task 7's grep. `ASSET_EQ`/`ASSET_FI` imported from `roro.segments` (matches load.py usage).
