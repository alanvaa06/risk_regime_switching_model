# RoRo — HMM Regime Visualization in the Report

**Date:** 2026-06-05
**Status:** Design approved through brainstorming. Ready for implementation planning.
**Author:** Alan Vazquez, CFA
**Scope:** Add HMM regime visualization to the `roro.report` HTML report — a stacked filtered-probability area figure and a percentile↔HMM band toggle on the existing β time-series. Additive and run-dependent; non-HMM runs render unchanged.

---

## 1. Goal

The HMM regime classifier (shipped off-by-default) writes `regimes_hmm.csv` per run when `hmm_enabled=True`: hard `label` plus the three filtered probabilities (`p_risk_off`, `p_transitional`, `p_risk_on`) and `confidence` per segment per day. The report currently visualizes only the percentile classifier. This adds two HMM-native views so the soft regime signal and its persistence are visible:

1. **Stacked probability area** — the three filtered probs over full history per segment, summing to 1.0. Shows the soft regime and confidence (a thin dominant band = uncertain) — something the percentile hard labels cannot express.
2. **Percentile↔HMM band toggle** on the existing β time-series — swap which method's regime bands shade behind the β line, on the same axis, to compare directly.

HMM viz is purely additive: when a run has no `regimes_hmm.csv`, the report builds the existing three figures byte-identically.

---

## 2. Locked decisions

| Fork | Choice |
|---|---|
| HMM views | Both: stacked probability area (new figure) + hard bands toggle on the β chart |
| Bands placement | Toggle on the **existing** β time-series (same β line, swap band source) |
| Toggle mechanism | **C1** — two controls + a small JS handler doing `Plotly.relayout(shapes=...)` from a precomputed lookup |
| HMM bands smoothing | **None** — `_regime_runs(seg_hmm_label[seg])` raw; HMM is persistent by construction (contrast with hysteresis-smoothed percentile) |
| Default band method | Percentile (preserves current behavior); toggle appears only when HMM data present |
| Degradation | No `regimes_hmm.csv` → all HMM fields None → original 3-figure report unchanged |

---

## 3. Data layer

`DataBundle` ([roro/report/bundle.py](../../../roro/report/bundle.py)) gains four optional fields, default `None`:

```python
seg_hmm_label: pd.DataFrame | None = None   # date × segment, hard argmax label
seg_hmm_p_off: pd.DataFrame | None = None   # date × segment, P(Risk-off)
seg_hmm_p_tr:  pd.DataFrame | None = None   # date × segment, P(Transitional)
seg_hmm_p_on:  pd.DataFrame | None = None   # date × segment, P(Risk-on)
```

`load_bundle` ([roro/report/load.py](../../../roro/report/load.py)), after the required CSVs:

```python
hmm_path = run_dir / "regimes_hmm.csv"
if hmm_path.exists():
    hmm = pd.read_csv(hmm_path, parse_dates=["date"])
    seg_hmm_label = hmm.pivot(index="date", columns="segment", values="label").sort_index()
    seg_hmm_p_off = hmm.pivot(index="date", columns="segment", values="p_risk_off").sort_index()
    seg_hmm_p_tr  = hmm.pivot(index="date", columns="segment", values="p_transitional").sort_index()
    seg_hmm_p_on  = hmm.pivot(index="date", columns="segment", values="p_risk_on").sort_index()
# else: all stay None
```

- `regimes_hmm.csv` is the long-form artifact the engine's io layer writes (columns: date, segment, state, label, p_risk_off, p_transitional, p_risk_on, confidence, cold_start, thin_cut). Pivot mirrors the existing `seg_tercile` handling.
- Full history is carried (like `seg_beta`/`seg_tercile`), not the 252-day scatter window. Both new views span full history.

---

## 4. Probability-area figure

New pure builder `regime_probability_area(bundle)` in [roro/report/figures.py](../../../roro/report/figures.py):

- **Traces:** three `go.Scatter(mode="lines", stackgroup="p", fill=...)` stacked bottom→top in risk order (Risk-off, Transitional, Risk-on), colored from `REGIME_COLORS` (opaque fills). Band height = probability; thin dominant band = low confidence (read visually — no extra trace, YAGNI).
- **Segment dropdown:** one control (no composition issue). Reuses `BETA_TS_SEGMENTS ∩ seg_hmm_label.columns`. Each button restyles the three traces' `y` to that segment's probs + updates the title.
- **Cold-start days** (NaN probs before `min_history`) render as a left-side gap — honest about where HMM has no verdict.
- **Axes:** y fixed 0–1; x = `seg_hmm_p_off.index` (full history). Height/template/font match the existing figures.
- Called only when `bundle.seg_hmm_label is not None`.

---

## 5. β-chart band toggle (C1)

The existing `beta_timeseries` shades hysteresis-smoothed **percentile** bands as layout `shapes`, driven by a segment dropdown. Two controls (segment × method) jointly determining `shapes` cannot be expressed by stateless Plotly `updatemenu` buttons alone, so the band swap is done in JS.

- **HMM bands** = raw `_regime_runs(seg_hmm_label[seg])` — no hysteresis (HMM is already persistent; the chart visibly contrasts this with the debounced percentile bands).
- **Default** = percentile bands (current behavior). Toggle only rendered when HMM data present.
- **New helper** `beta_band_lookup(bundle) -> dict` in figures.py precomputes both shape-sets per segment:
  ```python
  {segment: {"percentile": [shape, ...], "hmm": [shape, ...]}}
  ```
  Percentile shapes = the current hysteresis-smoothed runs; HMM shapes = raw runs. Shapes use the same dict structure `beta_timeseries` already builds.
- **`beta_timeseries` output is unchanged** (segment dropdown still drives the β line + initial percentile shapes). The JS handler only swaps `shapes`.
- **JS handler (in html.py):** when a `beta_band_lookup` is provided, `assemble` injects an HTML `<select>` (Percentile / HMM) near the β figure and a script that, on segment-dropdown click *or* method-select change, computes `shapes = lookup[currentSegment][currentMethod]` and calls `Plotly.relayout(betaDiv, {shapes})`.
- **Implementation risk + fallback:** the handler needs to know the current segment. The plan must first verify Plotly emits a usable event for the in-plot segment dropdown (`plotly_buttonclicked` / equivalent). **If it does not**, render the segment control *also* as an HTML `<select>` (both controls outside the plot, fully JS-driven) — same lookup, robust composition. This fallback is the safe default if the event is unreliable.

---

## 6. Orchestration & assembly

`assemble` ([roro/report/html.py](../../../roro/report/html.py)):
- Relax the hard `_REQUIRED_FIGURES = 3` to accept a variable-length list (3 or 4) with matching section titles.
- New optional params: `beta_div_index` and `beta_band_lookup`. When `beta_band_lookup` is provided, inject the method `<select>` + JS handler targeting the β figure's div; otherwise emit nothing extra (existing output preserved).

`build_report` ([roro/report/orchestrate.py](../../../roro/report/orchestrate.py)):
```python
figures = [scatter_vol_return(b), scatter_beta_return(b), beta_timeseries(b)]
lookup = None
if b.seg_hmm_label is not None:
    figures.append(regime_probability_area(b))
    lookup = beta_band_lookup(b)
html = assemble(figures, run_date=b.run_date,
                methodology_version=b.methodology_version,
                beta_div_index=2, beta_band_lookup=lookup)
```

**Section titles** extend to include "HMM regime probabilities" for the 4th figure when present.

---

## 7. Testing

Mirrors the existing `tests/report/` discipline:

- **load_bundle** — fixture run dir with `regimes_hmm.csv` → four HMM fields populated and correctly pivoted; without it → all four None.
- **regime_probability_area** — three stacked traces; on non-cold days the three probs sum to ≈1.0; dropdown button count == available segments; y-axis range [0, 1].
- **beta_band_lookup** — both `"percentile"` and `"hmm"` keys present for each available segment; on the eval data, HMM run-count ≤ smoothed-percentile run-count per segment (sanity: HMM more persistent).
- **assemble** — 4 figures + lookup → HTML contains the method `<select>` and the JS handler and 4 figure divs; 3 figures + `lookup=None` → no HMM markup, existing report output unchanged.
- **e2e (orchestrate)** — build from a run dir containing `regimes_hmm.csv` → HTML has the probability-area div + toggle; build from a non-HMM run dir → original three figures, no HMM markup.
- All existing `tests/report/` tests must still pass (no-HMM path unchanged).

---

## 8. Non-goals

- No percentile-vs-HMM agreement/disagreement analytics figure (the toggle already lets the eye compare).
- No confidence as a separate trace/line (encoded in the stacked-area band heights).
- No changes to the engine, `regimes_hmm.csv` schema, or any non-report module.
- No new CLI surface — `roro report --run-dir <dir>` picks up HMM viz automatically when the run dir has `regimes_hmm.csv`.
- No HMM viz in the scatter figures (regime is a time-series concept here).

---

## 9. Files touched

- **Modify** `roro/report/bundle.py` — four optional HMM fields on `DataBundle`.
- **Modify** `roro/report/load.py` — read + pivot `regimes_hmm.csv` when present.
- **Modify** `roro/report/figures.py` — `regime_probability_area` + `beta_band_lookup`.
- **Modify** `roro/report/html.py` — variable figure count; method `<select>` + JS handler injection.
- **Modify** `roro/report/orchestrate.py` — conditional 4th figure + lookup wiring.
- **Tests** under `tests/report/` — new cases per §7.
