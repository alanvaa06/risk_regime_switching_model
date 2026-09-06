---
title: "RoRo — Regime Attribution Spec (per-asset slope decomposition)"
owner: Alan Vazquez, CFA
status: Design approved 2026-09-05 (options a/a) — ready for implementation planning; NO code written yet
date: 2026-09-05
target repo: C:\Proyectos\RoRo (alanvaa06/risk_regime_switching_model)
drop-in location: docs/superpowers/specs/2026-09-05-roro-attribution-design.md
depends on: RoRo v1.1 (percentile default, HMM/JM overlays, vol-breadth heatmap)
wiki anchors:
  - "[[Risk-On Risk-Off Regime Classifier (Alan 2026)]]"
  - "[[Brinson Attribution and Return Decomposition]]"
  - "[[Switching Risk Off — FX Correlations and Risk Premia (Beber Brandt Cen 2013)]]"
  - "[[CFA L3 Performance Evaluation]]"
---

# RoRo — Regime Attribution Spec

## 1. Goal

Answer, for any date and any segment cut: **"the regime label is risk-on / risk-off — which assets is that due to?"**

RoRo's regime label is a monotone function of one number per cut: the cross-sectional WLS slope `β̂` of 3M return on EWMA vol (`roro/regression.py`). The percentile classifier, the HMM overlay and the JM overlay are all univariate on that same `β̂`. Therefore **attributing the regime = attributing `β̂`**, and because WLS is linear in returns, the attribution is **exact, additive and residual-free** — the same fact [[Brinson Attribution and Return Decomposition]] records via Lu-Kane ("a Brinson allocation effect is a cross-sectional regression").

Deliverable: a pure, deterministic module `roro/attribution.py` producing (1) per-asset contributions to today's `β̂`, (2) a Brinson-style waterfall of `Δβ̂` since the last regime transition, (3) roll-ups by block, (4) concentration / fragility metrics with alerts, (5) per-asset loadings on the correlation pillar, plus report figures.

## 2. Motivation — the 2026-05-26 snapshot

Throwaway computation on the last panel date (script outside the repo). Global cut, cap-weighted, label **Risk-on (P74, Q4)**, `β̂ = 0.429`:

| Asset | Quadrant | Vol | Ret 3M | h (leverage) | c | Share of β̂ |
|---|---|---|---|---|---|---|
| South Korea Eq | HI/+ | 55% | +27% | 1.13 | +0.308 | **72%** |
| Taiwan Eq | HI/+ | 29% | +22% | 0.43 | +0.095 | 22% |
| China Eq | HI/− | 19% | −5% | 0.56 | −0.029 | −7% |
| US FI | LO/− | 4% | −1.5% | −1.84 | +0.027 | 6% |
| US Eq | HI/+ | 13% | +8% | 0.09 | +0.007 | 2% |

HHI(|c|) = 0.33; top-5 = 84% of Σ|c|. Block roll-up inside the global regression: EM_Eq +0.357, DM_Eq +0.046, DM_FI +0.043, EM_FI −0.017, LatAm −0.005. Equal-weighted: `β̂ = 0.195`, Korea = 107% of it; EM ex-Korea/Taiwan nets negative (HI/− quadrant = classic risk-off).

Reading: the "risk-on" label is a two-asset semis rally, not breadth. US Eq (35% of weight) contributes ~0 because its vol sits at the weighted mean (leverage ≈ 0). A single vol outlier dominates under **both** weighting schemes. None of this is visible from `β̂` alone. ==That is the product gap this spec closes.==

## 3. Locked decisions (2026-09-05)

| Fork | Choice | Why |
|---|---|---|
| Method | **Exact contribution `c_i = h_i · y_i`** (option a) | Uses the existing WLS unchanged; additive; no residual; Brinson ≡ regression |
| Jackknife / leave-one-out | Only as a **validation check** (sign of `Δβ̂_{-i}` vs sign of `c_i`) and for the top-1 fragility flag — not the primary method | Influence ≠ contribution; N regressions/day; does not sum |
| Δβ̂ horizon | **Anchored to the last bucket transition of the cut** + fixed 63d window in parallel (option a) | Answers "what caused *this* transition"; 63d matches `return_window_days` |
| Estimator | **Unchanged.** Concentration is *reported*, `β̂` is not robustified (option a) | Keeps `β̂` that already went through G1–G6; robust/winsorized variant is a separate decision (§10 D3) |
| Granularity | Per series (`country__Eq` / `country__FI`), roll-up to DM/EM × Eq/FI blocks and LatAm | Mirrors `SeriesId` and the 10 cuts |
| Weighting | Cap-weighted **and** equal-weighted, side by side | Same discipline as `BetaFrame` |
| Language of docs | English (repo convention) | — |

## 4. Mathematics

### 4.1 Level attribution (exact)

WLS with intercept, normalized weights $\tilde w_i$, $x_i$ = EWMA vol, $y_i$ = 3M return, $\bar x = \sum_i \tilde w_i x_i$, $D = \sum_i \tilde w_i (x_i-\bar x)^2$:

$$\hat\beta_t = \sum_i c_{i,t}, \qquad c_{i,t} = h_{i,t}\, y_{i,t}, \qquad h_{i,t} = \frac{\tilde w_i\,(x_i-\bar x)}{D}$$

Exact because $\sum_i \tilde w_i (x_i - \bar x) = 0$ removes the intercept. $h_i$ is the asset's **leverage** on the slope: far from the weighted-mean vol and heavy → moves `β̂`. Equal-weighted: $\tilde w_i = 1/n$. Shares: $s_i = c_i / \hat\beta$ (undefined when $\hat\beta = 0$; emit NaN).

### 4.2 Quadrants

| Vol vs $\bar x$ | Ret 3M | Sign of $c_i$ | Mechanism |
|---|---|---|---|
| HI | + | > 0 | Classic risk-on: beta paid |
| HI | − | < 0 | Classic risk-off: high-vol punished |
| LO | + | < 0 | Flight-to-quality: havens bid |
| LO | − | > 0 | Havens sold (rate-shock type, 2022) |

Two different risk-offs (high-vol punished vs havens bid) and two different risk-ons are indistinguishable in `β̂`; the quadrant roll-up separates them. Store `quad ∈ {HI/+, HI/-, LO/+, LO/-}` per asset.

### 4.3 Change attribution — Brinson-style waterfall

For horizon $k$ (anchor date $t-k$), with $I_t$ / $I_{t-k}$ the series present in each panel:

- $i \in I_t \cap I_{t-k}$: $\Delta c_i = \underbrace{h_{i,t-k}\,\Delta y_i}_{\text{return effect}} + \underbrace{\Delta h_i\, y_{i,t-k}}_{\text{position effect}} + \underbrace{\Delta h_i\,\Delta y_i}_{\text{interaction}}$
- $i \in I_t \setminus I_{t-k}$ (entry): $\Delta c_i = c_{i,t}$, tagged `universe`
- $i \in I_{t-k} \setminus I_t$ (exit): $\Delta c_i = -c_{i,t-k}$, tagged `universe`

$\sum_i \Delta c_i = \hat\beta_t - \hat\beta_{t-k}$ exactly. **Position effect** = the asset's vol or weight moved relative to the cross-section (and, through $\bar x$ and $D$, everybody else's did). **Interaction** is reported, not folded — per the Brinson article: a large interaction means "the asset that repriced is also the one whose leverage jumped", i.e. one bet, not two. Fold into return effect only in the report's compact view, and say so.

Anchor: the most recent date $a < t$ where the cut's tercile label changed (percentile method by default; `label_source` selectable across percentile / HMM / JM, same pattern as the backtest scorer). If no transition exists within the trailing 5Y, the anchored waterfall is NaN and only the fixed-63d one is emitted.

### 4.4 Roll-ups

Within a **single** regression (e.g. `global`), group sums are exact: $C_g = \sum_{i \in g} c_i$ for $g \in$ {DM_Eq, EM_Eq, DM_FI, EM_FI, LatAm}. This answers "is the global regime driven by EM_Eq or DM_FI". Note: the segment cuts run their **own** regressions with their own $\bar x$, $D$, so segment `β̂`s do not sum to the global `β̂` — roll-ups are always computed inside the regression being attributed, never across regressions.

### 4.5 Concentration and fragility

Per date × cut × weighting:

- $HHI = \sum_i a_i^2$, $a_i = |c_i| / \sum_j |c_j|$
- `top1_share`, `top5_share` of $\sum |c|$; `top1_series`
- $\hat\beta_{-top1}$: slope re-estimated dropping the top-1 contributor (one extra WLS, not N)
- `fragile_flag`: True if $\hat\beta_{-top1}$ falls in a different tercile than $\hat\beta_t$ against the **same** trailing-5Y thresholds the classifier used that day (reuse `roro/classify.py` thresholds; no re-fit)

Alert rule (F-A5): `top1_share > attribution_top1_alert` (default 0.50) on a bucket-transition day, or `fragile_flag` True → row in `alerts.csv` type `concentration`.

### 4.6 Correlation pillar (PC1) attribution

On the same 3M window of daily log returns used by `roro/correlation.py`, covariance $\Sigma$, first eigenpair $(\lambda_1, v_1)$ with $\|v_1\|=1$:

- Contribution of asset $i$ to PC1: $v_{1,i}^2$ (sums to 1)
- Share of total variance: $\Sigma_{ii} / \mathrm{tr}(\Sigma)$
- **Decoupling score** $= \Sigma_{ii}/\mathrm{tr}(\Sigma) - v_{1,i}^2$: positive and large → the asset carries variance that PC1 does not explain → safe-haven decoupling in the [[Switching Risk Off — FX Correlations and Risk Premia (Beber Brandt Cen 2013)]] sense (polarization, not average correlation)
- Row-mean pairwise correlation per asset (excluding self)

Sign of $v_1$ is arbitrary; squares are used, so no canonicalization needed.

## 5. Module design

### 5.1 `roro/attribution.py` (new, pure, no I/O)

```python
@dataclass(frozen=True)
class AttributionRow:            # one asset, one date, one cut, one weighting
    series: str                  # "Country__Eq" | "Country__FI"
    block: str                   # DM_Eq | EM_Eq | DM_FI | EM_FI
    latam: bool
    vol: float
    ret3m: float
    weight: float                # normalized
    leverage: float              # h_i
    contribution: float          # c_i
    share: float                 # c_i / beta (NaN if beta == 0)
    quadrant: str                # HI/+ HI/- LO/+ LO/-

def attribute_panel(panel: DailyPanel, *, weighting: Literal["cap","eq"]) -> tuple[list[AttributionRow], float]
    # returns rows + beta; asserts |sum(c) - wls_slope| < 1e-10

def attribute_delta(rows_t, rows_a, *, beta_t, beta_a) -> pd.DataFrame
    # columns: series, effect_return, effect_position, effect_interaction, effect_universe, delta_total
    # asserts sum(delta_total) == beta_t - beta_a to 1e-10

def concentration(rows, *, panel: DailyPanel, weighting) -> ConcentrationRow
    # hhi, top1_series, top1_share, top5_share, beta_ex_top1

def pc1_loadings(window_returns: pd.DataFrame) -> pd.DataFrame
    # series, pc1_load_sq, var_share, decoupling, row_mean_corr

def find_anchor(labels: pd.Series, t: pd.Timestamp, *, max_lookback_days: int) -> pd.Timestamp | None
    # last label change strictly before t; None if none in lookback

def compute_attribution(bbs: BetaBySegment, panels_by_date, regime: RegimeFrame, cfg: EngineConfig, ...) -> AttributionFrame
    # orchestrates per cut × weighting; history for concentration, snapshot for level/delta/pc1
```

Reuse `daily_panel` from `roro/regression.py` unchanged (attribution consumes the identical panel the slope is fitted on — this is what makes the exactness assertion meaningful). Do **not** modify `cross_section`.

### 5.2 Types (`roro/types.py`)

```python
@dataclass(frozen=True)
class AttributionFrame:
    level: pd.DataFrame            # last date: cut × weighting × series (long) — §4.1/4.2
    delta_anchor: pd.DataFrame     # last date vs anchor, per cut × weighting — §4.3
    delta_fixed: pd.DataFrame      # last date vs t-63d
    rollup: pd.DataFrame           # cut × weighting × block → C_g, plus quadrant sums
    concentration: pd.DataFrame    # FULL HISTORY: date × cut × weighting → hhi, top1_*, top5_share, beta_ex_top1, fragile_flag
    pc1: pd.DataFrame              # last date: cut × series → pc1_load_sq, var_share, decoupling, row_mean_corr
    anchors: dict[str, pd.Timestamp | None]   # per cut
    history_global: pd.DataFrame | None       # optional wide c-matrix, global cut only (D2)
```

`RunResult` gains `attribution: AttributionFrame | None`. `AlertSet` gains `concentration_alerts: pd.DataFrame`.

### 5.3 Config (`roro/config.py`, frozen fields, all flow into `snapshot.json`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `attribution_enabled` | bool | `True` (D1) | Master switch |
| `attribution_label_source` | str | `"percentile"` | Anchor source: percentile / hmm / jm |
| `attribution_anchor_lookback_days` | int | `1260` | Max lookback for a transition anchor |
| `attribution_fixed_horizon_days` | int | `63` | Parallel fixed window |
| `attribution_top1_alert` | float | `0.50` | Concentration alert threshold |
| `attribution_history_global` | bool | `False` | Persist wide c-matrix for the global cut (D2) |

### 5.4 Engine wiring (`roro/engine.py`)

After step 5 (classify) and before alerts: build panels for the last date and the anchor/fixed dates per cut, call `compute_attribution`, pass to `detect_alerts` and `RunResult`. Concentration history requires per-date panels for every cut — compute inside the existing `compute_beta_by_segment` date loop **or** re-run `daily_panel` per date (O(n) each, cheap; prefer re-run to keep `regression.py` untouched — D4).

### 5.5 Outputs (`roro/io.py`, atomic writes, deterministic column order)

| File | Content | Size |
|---|---|---|
| `attribution.csv` | level rows, last date, all cuts × both weightings | ~1.3k rows |
| `attribution_delta.csv` | anchored + fixed waterfalls, tagged `horizon ∈ {anchor, fixed63}` | ~2.6k rows |
| `attribution_rollup.csv` | block + quadrant sums | small |
| `concentration.csv` | full history metrics | dates × 10 × 2 |
| `attribution_pc1.csv` | PC1 loadings, last date | ~640 rows |
| `attribution_history_global.csv` | optional (D2) | dates × 64 |
| `snapshot.json` | new block `attribution`: per cut top-3 series + shares, hhi, top1_share, fragile_flag, anchor date, config echo | — |

### 5.6 Alerts (`roro/alerts.py`)

New `concentration_alerts` rows: `date, segment, weighting, top1_series, top1_share, hhi, fragile_flag, trigger ∈ {transition_day, fragile}`. Appended to `alerts.csv` with `type = concentration`.

### 5.7 Report (`roro/report/`)

| Figure | Content | Controls |
|---|---|---|
| `fig_attrib_bars` | Horizontal bars, top-N by \|c\|, colored by quadrant; annotation "β̂ = … ; top-1 = … %" | Cut dropdown; cap ↔ eq toggle |
| `fig_attrib_waterfall` | Δβ̂ from anchor to today: bars per block (DM_Eq, EM_Eq, DM_FI, EM_FI) stacked into return / position / interaction / universe; total bar; anchor date + labels in title | Cut dropdown; anchor ↔ fixed63 toggle |
| `fig_attrib_scatter` | Vol (x) vs ret 3M (y), marker size ∝ \|c\|, color by quadrant, dashed vertical at $\bar x$, fitted line = the regime slope | Cut dropdown; cap ↔ eq |
| `fig_concentration_ts` | top-1 share and HHI over the 5Y window with regime bands (reuse band toggle Percentile / HMM / JM) | Cut dropdown |
| `fig_pc1_loadings` | Bars: `pc1_load_sq` vs `var_share` per asset, sorted by decoupling | Cut dropdown |

Deterministic `div_id`s, no timestamps — byte-identical HTML test extended.

## 6. Tests (one module per code module; `hypothesis` for kernels)

- **Exactness (property):** random panels (n ∈ [10, 80], weights > 0, vols > 0) → $|\sum c_i - \text{slope}_{WLS}| < 10^{-10}$ for cap and eq.
- **Delta exactness (property):** random pairs of panels with random entries/exits → $|\sum \Delta c_i - (\hat\beta_t - \hat\beta_a)| < 10^{-10}$; universe term equals sum of entries − exits.
- **Quadrant coverage:** every row has a quadrant; signs of $c$ match the §4.2 table.
- **Leverage identity:** $\sum_i h_i = 0$ and $\sum_i h_i x_i = 1$ (both exact for WLS with intercept).
- **Concentration bounds:** $1/n \le HHI \le 1$; `top1_share ≤ top5_share ≤ 1`.
- **Jackknife sign check (slow, real data):** on ≥ 95% of date × cut, $\mathrm{sign}(c_{top1}) = \mathrm{sign}(\hat\beta - \hat\beta_{-top1})$; log exceptions.
- **PC1:** $\sum_i v_{1,i}^2 = 1$; decoupling sums to 0; invariance to eigenvector sign flip.
- **Anchor causality:** anchor strictly < t; appending future data does not change anchor or any attribution row for date t.
- **Determinism / golden:** two runs → byte-identical CSVs; golden fixture under `tests/golden/`.
- **Off switch:** `attribution_enabled=False` → all pre-existing outputs byte-identical to today.

## 7. Acceptance criteria

- **AA-1** Exactness properties pass (level + delta).
- **AA-2** With `attribution_enabled=False`, existing artifacts byte-identical (regression + golden).
- **AA-3** With it on, byte-identical attribution CSVs across two runs.
- **AA-4** Report renders the five figures; byte-identical HTML test green.
- **AA-5** Quality bars: `mypy --strict`, `ruff` (E,F,I,N,UP,B,SIM,PL), `pytest` with `filterwarnings=["error"]`.
- **AA-6** Real-data run 2008–2026: reproduce the 2026-05-26 snapshot (§2) to 4 decimals; concentration history written; count of days with `top1_share > 0.5` reported in `docs/context/results.md`.
- **AA-7 (product bar):** for the 7 in-range G3 events, the anchored waterfall names the driving block and top-3 assets; PM review confirms they are economically sensible (memo in `docs/analysis/`).

## 8. Milestones

| Sprint | Deliverable | Est |
|---|---|---|
| A1 | `attribution.py` kernels: `attribute_panel`, leverage/quadrant, `concentration`; property tests | 1 wk |
| A2 | `attribute_delta` + `find_anchor` (label_source percentile/HMM/JM); tests incl. entries/exits | 0.5 wk |
| A3 | `pc1_loadings`; tests | 0.5 wk |
| A4 | Types, config, engine wiring, io writers, snapshot block, alerts; golden + off-switch tests | 1 wk |
| A5 | Report figures (5) + HTML byte-identical test | 1 wk |
| A6 | Real-data run + results memo (AA-6/AA-7): concentration history, top-1 > 50% day count, event review | 0.5 wk |
| **Total** | | **~4.5 wks** |

## 9. Non-goals

- No change to `β̂`, weights, vol estimator or classifier thresholds.
- No predictive use of attribution (stays diagnostic, RoRo v1.x).
- No per-asset regime labels (asset-level "risk-on" is not defined here; only contribution to the cut's slope).
- No cross-regression reconciliation (segment `β̂`s → global `β̂`); see §4.4.
- No Shapley / full leave-one-out tables.

## 10. Open decisions

| ID | Decision | Options | Recommended |
|---|---|---|---|
| D1 | Default state | On by default (adds files only) / off like the overlays | **On** — pure additive diagnostic, no numeric change to existing files; regenerate golden once |
| D2 | Persist full history of per-asset c for the global cut | Yes (dates × 64 CSV, ~1.5 MB) / no, concentration metrics only | **No for v1**; expose `attribution_history_global` flag |
| D3 | Robust slope (Huber / vol winsorized P95) as a *parallel* series after concentration evidence | Do it / don't | **Defer.** Trigger: if concentration history shows top-1 > 50% on > 20% of days, open a separate spec; changes `β̂` → requires re-running G1–G6 |
| D4 | Where per-date panels for concentration history come from | Re-run `daily_panel` per date in attribution / extend `compute_beta_by_segment` to yield panels | **Re-run** — keeps `regression.py` untouched; cost is O(dates × cuts × n), trivial |
| D5 | Interaction term in compact report view | Show as its own bar / fold into return effect | **Show** in full view; fold in compact view with explicit label ("return incl. interaction") |

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Contributions "contaminate" each other through $\bar x$, $D$ | An asset's $c_i$ moves when others' vols move | Inherent to any regression attribution; the position effect in §4.3 isolates it; document in report tooltip |
| PMs read `share > 100%` as an error | Confusion | Shares can exceed 1 when others net negative (Korea 107% eq-wtd). Label as "share of β̂ (can exceed 100%)" |
| Golden regeneration masks a numeric regression elsewhere | Silent drift | Regenerate golden in its own commit with diff limited to new files + snapshot block; CI asserts old files unchanged |
| Thin LatAm cut (n=10) → unstable $D$ | Noisy attribution | Inherit `thin_cut_flag`; suppress fragility alert on thin cuts |
| Anchor label source disagreement (percentile vs JM transitions) | Different waterfalls | `attribution_label_source` explicit in snapshot; default percentile (production) |

## 12. References

- [[Risk-On Risk-Off Regime Classifier (Alan 2026)]] — design doc the repo implements (wiki is at v1.0; repo at v1.1).
- [[Brinson Attribution and Return Decomposition]] — Brinson ≡ cross-sectional regression (Lu-Kane); report interaction before folding; weight drift vs decision dates.
- [[CFA L3 Performance Evaluation]] — macro (allocation) vs micro (selection) → block vs asset.
- [[Switching Risk Off — FX Correlations and Risk Premia (Beber Brandt Cen 2013)]] — polarization / PC1 variance share → decoupling score.
- [[Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)]], [[jumpmodels (Python Library)]] — JM overlay whose labels can serve as anchor source.
- Repo contracts reused: `roro/regression.py::daily_panel` (identical panel), `roro/classify.py` thresholds (fragility), `roro/correlation.py` window, `roro/alerts.py` / `roro/io.py` / `roro/report` patterns, `tests/golden/` discipline.
- `output/roro/RoRo Risk-Regime Model — PRD.md`, `output/roro/PRD-RoRo-Jump-Model-Regime-Layer.md` — sibling docs.
