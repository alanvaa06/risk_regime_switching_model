# HMM Regime Visualization in the Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two HMM-native views to the `roro.report` HTML report — a stacked filtered-probability area figure and a percentile↔HMM band toggle on the β time-series — additive and run-dependent (non-HMM runs render unchanged).

**Architecture:** `load_bundle` optionally reads `regimes_hmm.csv` into four new `DataBundle` fields. A new `regime_probability_area` figure renders the three stacked probs. A `beta_band_lookup` helper precomputes percentile + HMM band shapes per segment; `assemble` injects an HTML method `<select>` + a small JS handler that swaps the β chart's `shapes` on segment or method change. Everything keys off `bundle.seg_hmm_label is not None`.

**Tech Stack:** Python 3.12 (uv-managed `.venv`), pandas, Plotly, pytest. mypy strict, ruff, `filterwarnings=["error"]`.

**Spec:** `docs/superpowers/specs/2026-06-05-roro-hmm-report-viz-design.md`

---

## ENVIRONMENT (every task)
- Canonical env = uv-managed `.venv` at Python 3.12. The machine default `python` is 3.14 — DO NOT use it.
- Run ALL python/pytest/mypy/ruff via `.venv\Scripts\python.exe` (e.g. `.venv\Scripts\python.exe -m pytest tests/report/test_load.py -v`).
- Branch `feat/hmm-report-viz` is already checked out. Do NOT switch branches.

---

## File Structure
- **Modify** `roro/report/bundle.py` — 4 optional HMM fields on `DataBundle`.
- **Modify** `roro/report/load.py` — read + pivot `regimes_hmm.csv` when present.
- **Modify** `roro/report/figures.py` — `regime_probability_area(bundle)` + `beta_band_lookup(bundle)`.
- **Modify** `roro/report/html.py` — variable figure count + method `<select>` + JS handler.
- **Modify** `roro/report/orchestrate.py` — conditional 4th figure + lookup wiring.
- **Modify** `tests/report/conftest.py` — helper to add `regimes_hmm.csv` to a run dir.
- **Tests** — `tests/report/test_load.py`, `test_figures.py`, `test_html.py`, `test_orchestrate.py`.

---

## Task 1: DataBundle HMM fields + load_bundle reads regimes_hmm.csv

**Files:**
- Modify: `roro/report/bundle.py`
- Modify: `roro/report/load.py`
- Modify: `tests/report/conftest.py`
- Test: `tests/report/test_load.py`

- [ ] **Step 1: Add an HMM-csv fixture helper to conftest**

In `tests/report/conftest.py`, after `_write_minimal_run_dir`, add:
```python
def write_hmm_csv(run_dir: Path) -> Path:
    """Add a regimes_hmm.csv to an existing run dir (segments match the minimal fixture)."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rows = []
    labels = ("Risk-off", "Transitional", "Risk-on")
    for i, d in enumerate(dates):
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
            lab = labels[i % 3]
            p = {"Risk-off": (0.7, 0.2, 0.1), "Transitional": (0.2, 0.6, 0.2),
                 "Risk-on": (0.1, 0.2, 0.7)}[lab]
            rows.append({
                "date": d, "segment": seg, "state": labels.index(lab), "label": lab,
                "p_risk_off": p[0], "p_transitional": p[1], "p_risk_on": p[2],
                "confidence": max(p), "cold_start": False, "thin_cut": seg == "LatAm",
            })
    path = run_dir / "regimes_hmm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
```

- [ ] **Step 2: Write the failing test**

Add to `tests/report/test_load.py`:
```python
from tests.report.conftest import write_hmm_csv


def test_load_bundle_reads_hmm_when_present(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    write_hmm_csv(minimal_run_dir)
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.seg_hmm_label is not None
    assert bundle.seg_hmm_p_off is not None
    assert "global" in bundle.seg_hmm_label.columns
    # three prob frames present and aligned to the label frame
    assert bundle.seg_hmm_p_off.shape == bundle.seg_hmm_label.shape
    assert bundle.seg_hmm_p_tr is not None and bundle.seg_hmm_p_on is not None


def test_load_bundle_hmm_none_when_absent(minimal_run_dir: Path, tiny_xlsx: Path) -> None:
    bundle = load_bundle(minimal_run_dir, tiny_xlsx, window=21)
    assert bundle.seg_hmm_label is None
    assert bundle.seg_hmm_p_off is None
    assert bundle.seg_hmm_p_tr is None
    assert bundle.seg_hmm_p_on is None
```

- [ ] **Step 3: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py::test_load_bundle_reads_hmm_when_present -v`
Expected: FAIL — `AttributeError: 'DataBundle' object has no attribute 'seg_hmm_label'` (or TypeError on construction).

- [ ] **Step 4: Add the DataBundle fields**

In `roro/report/bundle.py`, append to the `DataBundle` dataclass (after `seg_tercile`):
```python
    seg_hmm_label: pd.DataFrame | None = None
    seg_hmm_p_off: pd.DataFrame | None = None
    seg_hmm_p_tr: pd.DataFrame | None = None
    seg_hmm_p_on: pd.DataFrame | None = None
```

- [ ] **Step 5: Read regimes_hmm.csv in load_bundle**

In `roro/report/load.py`, inside `load_bundle`, after `seg_tercile = _pivot_segment(...)` and before the `return DataBundle(...)`:
```python
    seg_hmm_label = seg_hmm_p_off = seg_hmm_p_tr = seg_hmm_p_on = None
    hmm_path = run_dir / "regimes_hmm.csv"
    if hmm_path.exists():
        hmm = pd.read_csv(hmm_path, parse_dates=["date"])
        seg_hmm_label = hmm.pivot(index="date", columns="segment", values="label").sort_index()
        seg_hmm_p_off = hmm.pivot(index="date", columns="segment", values="p_risk_off").sort_index()
        seg_hmm_p_tr = hmm.pivot(index="date", columns="segment", values="p_transitional").sort_index()
        seg_hmm_p_on = hmm.pivot(index="date", columns="segment", values="p_risk_on").sort_index()
```
Then add to the `return DataBundle(...)` call:
```python
        seg_hmm_label=seg_hmm_label,
        seg_hmm_p_off=seg_hmm_p_off,
        seg_hmm_p_tr=seg_hmm_p_tr,
        seg_hmm_p_on=seg_hmm_p_on,
```

- [ ] **Step 6: Run tests, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_load.py -v`
Expected: PASS (new + existing).

- [ ] **Step 7: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/bundle.py roro/report/load.py` and `.venv\Scripts\python.exe -m ruff check roro/report/bundle.py roro/report/load.py tests/report/conftest.py tests/report/test_load.py`
Then:
```bash
git add roro/report/bundle.py roro/report/load.py tests/report/conftest.py tests/report/test_load.py
git commit -m "feat(report): load regimes_hmm.csv into DataBundle (optional)"
```

---

## Task 2: regime_probability_area figure

**Files:**
- Modify: `roro/report/figures.py`
- Test: `tests/report/test_figures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_figures.py` (imports: `import pandas as pd`, `from roro.report.bundle import DataBundle`, `from roro.report.figures import regime_probability_area`):
```python
def _bundle_with_hmm() -> DataBundle:
    idx = pd.bdate_range("2020-01-02", periods=10)
    label = pd.DataFrame({"global": ["Risk-on"] * 10, "DM": ["Risk-off"] * 10}, index=idx)
    p_off = pd.DataFrame({"global": [0.1] * 10, "DM": [0.7] * 10}, index=idx)
    p_tr = pd.DataFrame({"global": [0.2] * 10, "DM": [0.2] * 10}, index=idx)
    p_on = pd.DataFrame({"global": [0.7] * 10, "DM": [0.1] * 10}, index=idx)
    empty = pd.DataFrame(index=idx)
    return DataBundle(
        run_date=idx[-1], methodology_version="1.0.0", dates=pd.DatetimeIndex(idx),
        vol=empty, ret_3m=empty, beta_vs_global=empty, meta=pd.DataFrame(),
        seg_beta=empty, seg_tercile=empty,
        seg_hmm_label=label, seg_hmm_p_off=p_off, seg_hmm_p_tr=p_tr, seg_hmm_p_on=p_on,
    )


def test_regime_probability_area_three_stacked_traces() -> None:
    fig = regime_probability_area(_bundle_with_hmm())
    assert len(fig.data) == 3
    # all three traces share a stackgroup (stacked area)
    assert all(getattr(tr, "stackgroup", None) for tr in fig.data)
    # on a given day the three probs sum to ~1
    ysum = sum(float(tr.y[0]) for tr in fig.data)
    assert abs(ysum - 1.0) < 1e-9
    # one dropdown with a button per available segment (global, DM)
    menus = fig.layout.updatemenus
    assert len(menus) == 1
    assert len(menus[0].buttons) == 2
    assert fig.layout.yaxis.range == (0.0, 1.0) or list(fig.layout.yaxis.range) == [0.0, 1.0]
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_regime_probability_area_three_stacked_traces -v`
Expected: FAIL — `ImportError: cannot import name 'regime_probability_area'`.

- [ ] **Step 3: Implement the figure**

In `roro/report/figures.py`, add (after `beta_timeseries`). Reuses the existing `REGIME_COLORS` and `BETA_TS_SEGMENTS` module constants:
```python
_PROB_TRACE_ORDER: tuple[tuple[str, str], ...] = (
    ("Risk-off", "seg_hmm_p_off"),
    ("Transitional", "seg_hmm_p_tr"),
    ("Risk-on", "seg_hmm_p_on"),
)


def regime_probability_area(bundle: DataBundle) -> go.Figure:
    """Stacked filtered-probability area (3 probs → 1.0) per segment, HMM only.

    Assumes bundle.seg_hmm_* are not None (caller guards on seg_hmm_label).
    """
    assert bundle.seg_hmm_label is not None  # caller guards
    p_off = bundle.seg_hmm_p_off
    p_tr = bundle.seg_hmm_p_tr
    p_on = bundle.seg_hmm_p_on
    assert p_off is not None and p_tr is not None and p_on is not None
    panels = {"seg_hmm_p_off": p_off, "seg_hmm_p_tr": p_tr, "seg_hmm_p_on": p_on}

    available = [s for s in BETA_TS_SEGMENTS if s in bundle.seg_hmm_label.columns]
    default = "global" if "global" in available else available[0]
    x = p_off.index

    traces: list[go.Scatter] = []
    for label, attr in _PROB_TRACE_ORDER:
        traces.append(
            go.Scatter(
                x=x,
                y=panels[attr][default].to_numpy(dtype=float),
                mode="lines",
                line={"width": 0.5, "color": REGIME_COLORS[label]},
                fillcolor=REGIME_COLORS[label],
                stackgroup="p",
                name=label,
                hovertemplate="%{x|%Y-%m-%d}<br>" + label + "=%{y:.2f}<extra></extra>",
            )
        )

    buttons = []
    for seg in available:
        buttons.append(
            {
                "method": "update",
                "label": seg,
                "args": [
                    {"y": [panels[attr][seg].to_numpy(dtype=float) for _, attr in _PROB_TRACE_ORDER]},
                    {"title": f"HMM regime probabilities — {seg}"},
                ],
            }
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"HMM regime probabilities — {default}",
            height=700,
            template="simple_white",
            font={"family": "system-ui, -apple-system, sans-serif", "size": 13},
            margin={"l": 60, "r": 200, "t": 60, "b": 120},
            xaxis={"title": "Date", "showgrid": True, "gridcolor": "#e6e6e6", "zeroline": False},
            yaxis={
                "title": "Filtered P(state)",
                "range": [0.0, 1.0],
                "showgrid": True,
                "gridcolor": "#e6e6e6",
                "zeroline": False,
            },
            updatemenus=[
                {
                    "type": "dropdown",
                    "showactive": True,
                    "buttons": buttons,
                    "x": 1.12,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ],
        ),
    )
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_regime_probability_area_three_stacked_traces -v`
Expected: PASS.

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/figures.py` and `.venv\Scripts\python.exe -m ruff check roro/report/figures.py tests/report/test_figures.py`
Then:
```bash
git add roro/report/figures.py tests/report/test_figures.py
git commit -m "feat(report): regime_probability_area stacked-area figure"
```

---

## Task 3: beta_band_lookup helper (percentile + HMM shapes per segment)

**Files:**
- Modify: `roro/report/figures.py`
- Test: `tests/report/test_figures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/report/test_figures.py`:
```python
from roro.report.figures import beta_band_lookup


def test_beta_band_lookup_has_both_methods_per_segment() -> None:
    bundle = _bundle_with_hmm()
    # give seg_tercile some labels so percentile shapes are non-empty
    idx = bundle.seg_hmm_label.index
    object.__setattr__(
        bundle, "seg_tercile",
        pd.DataFrame({"global": ["Risk-on"] * len(idx), "DM": ["Risk-off"] * len(idx)}, index=idx),
    )
    lookup = beta_band_lookup(bundle)
    assert set(lookup["global"].keys()) == {"percentile", "hmm"}
    # shapes are JSON-serializable: x0/x1 are date strings, not Timestamps
    shp = lookup["global"]["hmm"][0]
    assert isinstance(shp["x0"], str)
    assert shp["type"] == "rect"
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py::test_beta_band_lookup_has_both_methods_per_segment -v`
Expected: FAIL — `ImportError: cannot import name 'beta_band_lookup'`.

- [ ] **Step 3: Implement the helper**

In `roro/report/figures.py`, add (after `regime_probability_area`). Reuses `_regime_runs`, `_smooth_regime_hysteresis`, `_REGIME_CONFIRM_DAYS`, `REGIME_COLORS`, `BETA_TS_SEGMENTS`:
```python
def _band_shapes(labels: pd.Series, *, smooth: bool) -> list[dict[str, object]]:
    """Regime-run rectangles as JSON-serializable shape dicts (x0/x1 = ISO date strings)."""
    series = _smooth_regime_hysteresis(labels, _REGIME_CONFIRM_DAYS) if smooth else labels
    shapes: list[dict[str, object]] = []
    for start, end, label in _regime_runs(series):
        color = REGIME_COLORS.get(label)
        if color is None:
            continue
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": str(pd.Timestamp(start).date()),
                "x1": str(pd.Timestamp(end).date()),
                "y0": 0,
                "y1": 1,
                "fillcolor": color,
                "line": {"width": 0},
                "layer": "below",
            }
        )
    return shapes


def beta_band_lookup(bundle: DataBundle) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Per-segment precomputed band shapes for both methods, keyed for the JS toggle.

    Percentile = hysteresis-smoothed tercile runs (matches beta_timeseries).
    HMM = raw label runs (no smoothing — HMM is persistent by construction).
    Only segments present in BOTH seg_beta and the relevant label frame are included.
    """
    assert bundle.seg_hmm_label is not None  # caller guards
    out: dict[str, dict[str, list[dict[str, object]]]] = {}
    for seg in BETA_TS_SEGMENTS:
        if seg not in bundle.seg_beta.columns:
            continue
        percentile: list[dict[str, object]] = []
        if seg in bundle.seg_tercile.columns:
            percentile = _band_shapes(bundle.seg_tercile[seg], smooth=True)
        hmm: list[dict[str, object]] = []
        if seg in bundle.seg_hmm_label.columns:
            hmm = _band_shapes(bundle.seg_hmm_label[seg], smooth=False)
        out[seg] = {"percentile": percentile, "hmm": hmm}
    return out
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_figures.py -v`
Expected: PASS (this + Task 2 test).

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/figures.py` and `.venv\Scripts\python.exe -m ruff check roro/report/figures.py tests/report/test_figures.py`
Then:
```bash
git add roro/report/figures.py tests/report/test_figures.py
git commit -m "feat(report): beta_band_lookup precomputes percentile+HMM band shapes"
```

---

## Task 4: assemble — variable figure count + method select + JS handler

**Files:**
- Modify: `roro/report/html.py`
- Test: `tests/report/test_html.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/report/test_html.py` (it already imports `assemble`; add `import plotly.graph_objects as go`, `import pandas as pd` if missing):
```python
def _three_figs() -> list[go.Figure]:
    return [go.Figure(), go.Figure(), go.Figure()]


def test_assemble_accepts_four_figures_with_hmm_toggle() -> None:
    figs = _three_figs() + [go.Figure()]
    lookup = {"global": {"percentile": [], "hmm": [{"type": "rect", "x0": "2020-01-02",
              "x1": "2020-02-01", "y0": 0, "y1": 1, "fillcolor": "rgba(1,1,1,0.5)",
              "line": {"width": 0}, "layer": "below"}]}}
    html = assemble(figs, run_date=pd.Timestamp("2024-12-31"), methodology_version="1.0.0",
                    beta_div_index=2, beta_band_lookup=lookup)
    assert "HMM regime probabilities" in html      # 4th section title
    assert 'id="hmm-band-method"' in html          # method select present
    assert "plotly_relayout" in html               # JS handler present
    assert "fig_beta_ts" in html                   # targets the beta chart div


def test_assemble_three_figures_unchanged_without_lookup() -> None:
    html = assemble(_three_figs(), run_date=pd.Timestamp("2024-12-31"),
                    methodology_version="1.0.0")
    assert "hmm-band-method" not in html
    assert "HMM regime probabilities" not in html
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_html.py::test_assemble_accepts_four_figures_with_hmm_toggle -v`
Expected: FAIL — `TypeError` (assemble has no `beta_band_lookup` kwarg) or the 3-figure assertion in assemble raises.

- [ ] **Step 3: Extend DIV_IDS / SECTION_TITLES and relax the count check**

In `roro/report/html.py`, change the module constants and `_REQUIRED_FIGURES`:
```python
DIV_IDS: tuple[str, ...] = ("fig_scatter_vol", "fig_scatter_beta", "fig_beta_ts", "fig_hmm_probs")
SECTION_TITLES: tuple[str, ...] = (
    "Risk vs Return",
    "Beta vs Return",
    "Segment β with regime bands",
    "HMM regime probabilities",
)
_MIN_FIGURES: int = 3
_MAX_FIGURES: int = 4
```
(Delete the old `_REQUIRED_FIGURES = 3` line.)

- [ ] **Step 4: Update the `assemble` signature, count guard, zip, and JS injection**

Replace the `assemble` function body’s guard, loop, and return. New signature + guard:
```python
def assemble(
    figures: list[go.Figure],
    *,
    run_date: pd.Timestamp,
    methodology_version: str,
    beta_div_index: int | None = None,
    beta_band_lookup: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> str:
    if not _MIN_FIGURES <= len(figures) <= _MAX_FIGURES:
        raise ValueError(
            f"assemble requires {_MIN_FIGURES}-{_MAX_FIGURES} figures, got {len(figures)}"
        )
```
Change the loop to index into the (longer) constants instead of `strict=True` zip:
```python
    fig_html_blocks: list[str] = []
    for idx, fig in enumerate(figures):
        div_id = DIV_IDS[idx]
        title = SECTION_TITLES[idx]
        include: str | bool = "cdn" if idx == 0 else False
        block = fig.to_html(
            include_plotlyjs=include,
            full_html=False,
            div_id=div_id,
            config={"displaylogo": False},
        )
        fig_html_blocks.append(f'<section><h2>{title}</h2>{block}</section>')
```
Build the optional toggle markup (place before the return):
```python
    toggle_html = ""
    if beta_band_lookup is not None and beta_div_index is not None:
        toggle_html = _band_toggle_markup(DIV_IDS[beta_div_index], beta_band_lookup)
```
Insert `{toggle_html}` into the returned HTML right after `{''.join(fig_html_blocks)}`:
```python
{''.join(fig_html_blocks)}
{toggle_html}
<footer>Generated by RoRo · methodology v{methodology_version}</footer>
```

- [ ] **Step 5: Add the toggle-markup builder**

Add to `roro/report/html.py` (above `assemble`). Needs `import json` at top of the file (add if absent):
```python
def _band_toggle_markup(
    beta_div_id: str,
    lookup: dict[str, dict[str, list[dict[str, object]]]],
) -> str:
    """HTML <select> + JS that swaps the beta chart's regime bands by method.

    Segment changes are detected via plotly_relayout (the segment dropdown sets a
    `title` containing the segment name); the method <select> sets the method.
    On either, the beta div is relayouted with lookup[segment][method] shapes.
    """
    payload = json.dumps(lookup)
    return f"""
<div style="max-width:1280px;margin:0 auto;padding:0 24px 12px;">
  <label for="hmm-band-method" style="font-size:13px;color:#444;">Regime bands:</label>
  <select id="hmm-band-method" style="font-size:13px;margin-left:6px;">
    <option value="percentile">Percentile</option>
    <option value="hmm">HMM</option>
  </select>
</div>
<script>
(function() {{
  var lookup = {payload};
  var betaId = "{beta_div_id}";
  var seg = "global";
  var method = "percentile";
  function apply() {{
    var div = document.getElementById(betaId);
    if (!div || !window.Plotly) return;
    var byMethod = lookup[seg];
    if (!byMethod) return;
    window.Plotly.relayout(div, {{shapes: byMethod[method] || []}});
  }}
  function wire() {{
    var div = document.getElementById(betaId);
    var sel = document.getElementById("hmm-band-method");
    if (!div || !sel || !window.Plotly) {{ setTimeout(wire, 100); return; }}
    sel.addEventListener("change", function() {{ method = sel.value; apply(); }});
    div.on("plotly_relayout", function(e) {{
      var t = e && (e["title"] || (e["title.text"]));
      if (typeof t === "string" && t.indexOf("—") !== -1) {{
        seg = t.split("—").pop().trim();
        apply();
      }}
    }});
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", wire);
  }} else {{ wire(); }}
}})();
</script>
"""
```

- [ ] **Step 6: Run tests, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_html.py -v`
Expected: PASS (new + existing). Existing `assemble` callers pass 3 figures with no lookup → no toggle markup, unchanged.

- [ ] **Step 7: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/html.py` and `.venv\Scripts\python.exe -m ruff check roro/report/html.py tests/report/test_html.py`
Then:
```bash
git add roro/report/html.py tests/report/test_html.py
git commit -m "feat(report): assemble supports 4th HMM figure + band-toggle JS"
```

---

## Task 5: build_report wiring

**Files:**
- Modify: `roro/report/orchestrate.py`
- Test: `tests/report/test_orchestrate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/report/test_orchestrate.py` (it already imports `build_report`; add `from tests.report.conftest import write_hmm_csv` and `import pandas as pd` if missing):
```python
def test_build_report_includes_hmm_when_present(minimal_run_dir, tiny_xlsx, tmp_path) -> None:
    write_hmm_csv(minimal_run_dir)
    out = build_report(minimal_run_dir, tiny_xlsx, tmp_path / "r.html", window=21)
    html = out.read_text(encoding="utf-8")
    assert "HMM regime probabilities" in html
    assert 'id="hmm-band-method"' in html
    assert "fig_hmm_probs" in html


def test_build_report_no_hmm_unchanged(minimal_run_dir, tiny_xlsx, tmp_path) -> None:
    out = build_report(minimal_run_dir, tiny_xlsx, tmp_path / "r.html", window=21)
    html = out.read_text(encoding="utf-8")
    assert "hmm-band-method" not in html
    assert "fig_hmm_probs" not in html
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_orchestrate.py::test_build_report_includes_hmm_when_present -v`
Expected: FAIL — `"HMM regime probabilities"` not in html (4th figure not built).

- [ ] **Step 3: Wire build_report**

In `roro/report/orchestrate.py`, update the import and body. Imports:
```python
from roro.report.figures import (
    beta_band_lookup,
    beta_timeseries,
    regime_probability_area,
    scatter_beta_return,
    scatter_vol_return,
)
```
Replace the figures/assemble block:
```python
    bundle = load_bundle(run_dir, xlsx_path, window=window)
    figures = [
        scatter_vol_return(bundle),
        scatter_beta_return(bundle),
        beta_timeseries(bundle),
    ]
    lookup = None
    if bundle.seg_hmm_label is not None:
        figures.append(regime_probability_area(bundle))
        lookup = beta_band_lookup(bundle)
    html = assemble(
        figures,
        run_date=bundle.run_date,
        methodology_version=bundle.methodology_version,
        beta_div_index=2,
        beta_band_lookup=lookup,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/report/test_orchestrate.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: mypy + ruff + commit**

Run: `.venv\Scripts\python.exe -m mypy roro/report/orchestrate.py` and `.venv\Scripts\python.exe -m ruff check roro/report/orchestrate.py tests/report/test_orchestrate.py`
Then:
```bash
git add roro/report/orchestrate.py tests/report/test_orchestrate.py
git commit -m "feat(report): build_report adds HMM figure + band toggle when present"
```

---

## Task 6: Full report-suite verification + real e2e

**Files:** none (verification)

- [ ] **Step 1: Run the whole report suite + the rest of the fast suite**

Run: `.venv\Scripts\python.exe -m pytest tests/report -v` then `.venv\Scripts\python.exe -m pytest -m "not slow" -q`
Expected: all PASS. Existing report tests (`test_e2e.py`, `test_html.py`, `test_package.py`, etc.) must be unaffected on the no-HMM path.

- [ ] **Step 2: Type + lint the whole report package**

Run: `.venv\Scripts\python.exe -m mypy roro/report` and `.venv\Scripts\python.exe -m ruff check roro/report tests/report`
Expected: clean.

- [ ] **Step 3: Build a real report from the HMM eval run and eyeball the markup**

Run:
```bash
.venv\Scripts\python.exe -m roro.cli report --run-dir outputs/hmm_eval/2026-05-26 --xlsx data.xlsx --out outputs/hmm_eval/report_hmm.html
```
(If `roro.cli` is not directly invokable, use `.venv\Scripts\roro.exe report --run-dir outputs/hmm_eval/2026-05-26 --xlsx data.xlsx --out outputs/hmm_eval/report_hmm.html`.)
Then confirm the output contains the HMM pieces:
```bash
grep -c "fig_hmm_probs" outputs/hmm_eval/report_hmm.html
grep -c "hmm-band-method" outputs/hmm_eval/report_hmm.html
```
Expected: both ≥ 1. (`outputs/` is gitignored — this artifact is not committed.)

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test(report): verify HMM report end-to-end on real eval run"
```
(If no changes were needed, skip the commit.)

---

## Self-Review

**Spec coverage:** §3 data layer → Task 1; §4 probability-area figure → Task 2; §5 β-chart band toggle (beta_band_lookup + JS handler, no-hysteresis HMM bands, default percentile) → Tasks 3 & 4; §6 orchestration/assembly (variable figure count, conditional 4th figure, lookup wiring) → Tasks 4 & 5; §7 testing → embedded per task + Task 6; degradation (no regimes_hmm.csv → unchanged) → Tasks 1, 5, 6. All covered.

**Placeholder scan:** No TBD/TODO. The JS handler is concrete (uses `plotly_relayout` + title-parsing, the spec's robust path — no reliance on the uncertain `plotly_buttonclicked`). Test helper `write_hmm_csv` is fully written, not referenced as "similar to".

**Type consistency:** `seg_hmm_label/seg_hmm_p_off/seg_hmm_p_tr/seg_hmm_p_on` identical across bundle.py (Task 1), figures.py (Tasks 2-3), orchestrate.py (Task 5). `beta_band_lookup` return type `dict[str, dict[str, list[dict[str, object]]]]` matches the `assemble` `beta_band_lookup` param (Task 4) and `_band_toggle_markup` (Task 4). `regime_probability_area` and `beta_band_lookup` signatures consumed consistently in orchestrate. `beta_div_index=2` ↔ `DIV_IDS[2] = "fig_beta_ts"`.
