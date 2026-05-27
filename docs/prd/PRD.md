# RoRo Risk-Regime Classification Model — PRD v1.2

**Author:** Alan Vazquez, CFA
**Status:** Design document, v1.2 (revised after data inventory + design review + external-data corrections)
**Date:** May 27, 2026

---

## 1. TL;DR

Daily, cross-asset risk-regime classifier over **64 country-level index series** (32 equity + 32 fixed income, both in local currency) across DM and EM, weighted by USD market cap for the cross-sectional regression. Outputs (a) a single global regime label per day, (b) segmented regime labels per sub-block (DM/EM, Equity/FI, 4-way, LatAm), and (c) a parallel correlation-structure signal (PC1 variance share, avg pairwise correlation).

**v1.0 is diagnostic only.** Predictive layer deferred until the diagnostic validates against external risk-cycle proxies (VIX, US BBB OAS, ICE BofA EM Corporate OAS, HY OAS, 2s10s curve — all sourced free from FRED) and against six composite aggregates (DM, EM, Europe, Asia, World, LatAm) which serve as native internal consistency checks.

Three pillars, each academically anchored: cross-sectional return-on-vol slope (realized SML analog), cross-sectional correlation structure (Beber et al. RSDC simplified), segmentation (Berkman & Malloch partial-integration + regional pricing).

---

## 2. Problem

Single-proxy risk-regime measures (VIX level, HY OAS, VIX term structure) fail in three ways:

- **Lagged or coincident, not leading.** They confirm a regime, not anticipate transitions.
- **No segmentation.** A global signal hides DM/EM bifurcations and equity/FI decoupling.
- **No cross-validation.** Single proxies fail silently when their underlying mechanism breaks (e.g., VIX understates risk during a low→high FX correlation transition — Beber et al. 2013).

Common practice relies on ad-hoc reading of VIX + HY spreads + USD index. No formal classification, no segmentation, no academic backing, no auditable reproducibility.

---

## 3. Users

| User | Use case | Cadence |
|---|---|---|
| PM / Allocator (Alan) | Tilt SAA vs IPS bands; size risk overlays | Daily check; weekly review |
| Research / Strategist | Macro narrative input; CME refresh | Monthly |
| Risk team | Regime-conditional VaR / stress overlays | Daily / on alert |
| Client / IC use cases | Quarterly RoRo regime as narrative anchor | Quarterly |
| External (future) | LatAm-specific regime overlay for institutional fund-of-funds | TBD |

---

## 4. Goals / Non-Goals

### Goals (v1.0)

- Single daily regime label (risk-on / transitional / risk-off — terciles for headline) for the global universe.
- Segmented regime labels for DM, EM, Equity, FI, the four DM/EM × Eq/FI intersections, and the LatAm bloc (5 countries × 2 asset classes = 10 series).
- Parallel correlation-structure signal (PC1 variance share + avg pairwise correlation) plotted alongside vol-slope beta.
- External validation panel showing beta-regime vs VIX, US BBB OAS, ICE BofA EM Corporate OAS, HY OAS, 2s10s — plus internal consistency check vs the 6 composite aggregates (DM, EM, Europe, Asia, World, LatAm) already in the dataset.
- Reproducible nightly batch job → published dashboard.

### Non-Goals (v1.0)

- Next-regime forecast (deferred to v2 — based on Beber-style transition persistence, not realized-return forecasts).
- Position sizing / portfolio construction layer (downstream consumer).
- Intraday signal. Daily close only.
- Option-implied EMRP path (Berkman & Malloch) — infeasible across FI + LatAm universe.
- Asset-class-specific regimes (e.g., credit regime, EM-FX regime in isolation). Aggregate cross-asset only.

---

## 5. Scope

### In-scope universe — actual inventory from proprietary dataset

The data is delivered as a single proprietary Excel file (`data.xlsx`) containing six sheets. The investable universe consists of **32 countries**, each represented by an equity index and a fixed-income index, for **64 country-level series total** — plus 6 composite aggregates used for cross-validation.

#### DM block — 17 countries

United States, Japan, United Kingdom, Canada, France, Switzerland, Germany, Australia, Netherlands, Sweden, Spain, Hong Kong, Italy, Finland, Belgium, Israel, Norway.

#### EM block — 15 countries

China, India, Taiwan, South Korea, Brazil, Singapore, Mexico, South Africa, Indonesia, Thailand, Malaysia, Poland, Chile, Peru, Colombia.

#### LatAm bloc cut — 5 countries × 2 asset classes = 10 series

Brazil, Mexico, Chile, Peru, Colombia (both equity and FI). Argentina, Venezuela, and Russia are **not in the universe** — design decisions already resolved by the data inventory.

#### Aggregate composites (validation only, not regression inputs)

DM (MXWO), EM (MXEF), Europe (MXEU), Asia (MXAS), World (MXWD), LatAm (MXLA) — equity. Matching FI aggregates: I35402US, EMUSTRUU, H02503US, I20912US, LEGATRUU, H04338US.

#### Capitalization profile (as of 2026-05-26 reference date)

| Block | Eq mcap ($B) | FI mcap ($B) | Notes |
|---|---:|---:|---|
| DM total | 96,145 | 54,592 | US dominates: ~70% of DM eq mcap |
| EM total | 27,095 | 13,330 | China dominates: ~46% of EM eq mcap |
| LatAm | 1,626 | 695 | Brazil + Mexico = ~78% of LatAm eq mcap |

**Implication for design:** The cap-weighting concern raised in the design review is materially confirmed by the data. US equity alone is ~54% of total DM+EM equity mcap. The equal-weighted parallel regression (F2.2) and slope-spread auxiliary (F2.4) are **mandatory**, not optional.

### Data history

- Daily price observations, **2008-12-31 to 2026-05-26** (~17.4 years, ~4,540 trading days).
- No missing values across all 64 country series and 6 aggregates. Verified.
- Local-currency price levels for both equity (`Equity_LC` sheet) and fixed income (`Fixed_Income_LC` sheet). USD-quoted bond indices have been pre-converted to local currency upstream — confirmed by spot-check (e.g., Japan FI index ~12,660 in JPY vs ~79 in USD, May 2026).

### Out-of-scope (v1.0)

- Commodities, FX as primary axis (FX risk is embedded via local-currency vol).
- Single-name / single-issuer regimes.
- Crypto.

---

## 6. Functional Requirements

### F1 — Return + vol estimators

- F1.1 — Rolling **3M total return**, local currency, daily.
- F1.2 — **EWMA vol** with half-life parameterized; default 30d, calibrated empirically against 20 / 40 in S9 backtest.
- F1.3 — **1M parallel tripwire window** — faster signal for transition detection.
- F1.4 — All windows updated on T+1 close.
- F1.5 — **Data gap handling: drop-missing.** Given the dataset has no missing values, this is a defensive rule for future data refreshes. If a single asset is missing on date t, exclude it from that day's cross-section and document N actually used.

### F2 — Cross-sectional regression

- F2.1 — Daily weighted regression of return on vol.
- F2.2 — Two weighting schemes in parallel: **USD market-cap** AND **equal weight**.
- F2.3 — Output: slope β + R² + residual stats per day per scheme.
- F2.4 — **Slope spread** (cap-weighted − equal-weighted) as auxiliary signal — isolates US dominance.

### F3 — Segmentation

- F3.1 — Repeat F2 on DM (17 countries × 2 asset classes = 34 series), EM (15 × 2 = 30 series), Equity (32 series), FI (32 series).
- F3.2 — 4-way intersections: DM-Eq (17), EM-Eq (15), DM-FI (17), EM-FI (15).
- F3.3 — **LatAm bloc**: standalone regression on 10 series (5 countries × 2 asset classes).
- F3.4 — Output all slopes as parallel time series.
- F3.5 — **Minimum-N threshold per cut:** N≥10 series required to compute a slope; flag and suppress otherwise. LatAm at N=10 sits exactly at the threshold — accept but flag as "thin cut."

### F4 — Regime classification

- F4.1 — Per series, compute **percentile of current beta vs trailing 5Y rolling distribution**.
- F4.2 — **Headline classification: terciles** (revised from quintiles per design review):
  - T3 (top) → **Risk-on**
  - T2 → **Transitional**
  - T1 (bottom) → **Risk-off**
- F4.3 — **Underlying quintile percentile available** for users who want finer granularity, but not used as the headline label. This rebalances daily decision granularity against the "≤2 spurious transitions per quarter in calm regimes" success metric.
- F4.4 — Bucket thresholds parameterized — tercile is default, quintile and asymmetric (e.g., 20/60/20) selectable.
- F4.5 — **Direction flag** (rising / falling / stable) based on N-day slope of beta itself.

### F5 — Correlation-structure signal

- F5.1 — On same 3M window, compute **avg pairwise correlation** across the 64 country series.
- F5.2 — Compute **PC1 variance share** (PCA on covariance matrix of returns).
- F5.3 — Compute same on each segmentation cut.
- F5.4 — Display alongside beta. Flag **disagreement events** (vol-slope says risk-on, correlation says risk-off) — informative per Beber et al.

### F6 — External validation panel

- F6.1 — Pull and align via **FRED API only** (free, no licensing):
  - **VIX:** FRED `VIXCLS` (daily, CBOE-sourced, 1990 → present)
  - **US BBB OAS:** FRED `BAMLC0A4CBBB` (daily, ICE BofA, 1996 → present)
  - **ICE BofA EM Corporate OAS:** FRED `BAMLEMCBPIOAS` (daily, 2003 → present) — EM credit / spread proxy
  - **ICE BofA US HY OAS:** FRED `BAMLH0A0HYM2` (daily, 1996 → present) — secondary credit-cycle proxy
  - **10Y-2Y Treasury spread:** FRED `T10Y2Y` (daily, 1976 → present) — curve / recession proxy
- F6.2 — **Internal consistency: compare the model's regime classifications against the 6 composite aggregates** (DM, EM, Europe, Asia, World, LatAm) already in the dataset. If the model labels the DM cut "risk-off" while MXWO is making new highs, that's a flag — either segmentation noise or a real divergence worth surfacing.
- F6.3 — Rolling 60d correlation of beta-regime series vs each external proxy.
- F6.4 — **Alert** when correlation drops below threshold (default |ρ| < 0.3 over 60d) — diagnostic of signal degradation.
- F6.5 — **GFP (Miranda-Agrippino & Rey) deferred.** Monthly cadence, scrape friction, and the fact that the five FRED proxies + internal aggregates already provide adequate validation means GFP is not on the critical path. Add in v1.1 if validation gaps appear. Note: GFP is freely available on the author's website, no licensing concern.

### F7 — Dashboard / output API

- F7.1 — Daily HTML report with **two-tier UX** (revised per design review):
  - **Tier 1 (above the fold):** Global regime card + LatAm card + the segment showing the largest divergence today (auto-surfaced). Beta time-series chart with bucket bands shaded.
  - **Tier 2 (one click):** Full 8-card grid (DM, EM, Eq, FI, DM-Eq, EM-Eq, DM-FI, EM-FI) + correlation panel + external validation strip + alerts.
- F7.2 — JSON daily snapshot for programmatic consumption.
- F7.3 — Historical time-series exports (CSV / Parquet).
- F7.4 — Per-cut "regime card": current bucket, percentile, direction, vs prior 30/90/365 days.

---

## 7. Data Requirements

### Primary data source — single Excel file

| Sheet | Content | Use |
|---|---|---|
| `Panel` | Country, segment (DM/EM), index tickers, local currency, FX pair, market caps (latest snapshot) | Universe definition + USD market-cap weighting |
| `Equity_LC` | Daily equity index prices in local currency, 2008-12-31 → 2026-05-26 | Equity return + vol inputs |
| `Fixed_Income_LC` | Daily FI index prices converted to local currency (USD-quoted bond indices pre-converted upstream) | FI return + vol inputs |
| `Equity` | Equity index prices in native quotation currency | Reference only |
| `Fixed_Income` | FI index prices in native USD quotation | Reference only |
| `Currency` | FX time series | Available if needed for future FX overlay |

**Loading convention:** Both `Equity_LC` and `Fixed_Income_LC` sheets have row 0 = ticker IDs, row 1 = country names, row 2+ = daily price data. Load skipping row 0 (tickers), using row 1 (country names) as the column header.

### External validation data — all FRED, all free

| Dataset | FRED ticker | Frequency | History | Use |
|---|---|---|---|---|
| VIX | `VIXCLS` | Daily | 1990 → present | Primary daily external proxy |
| US BBB OAS | `BAMLC0A4CBBB` | Daily | 1996 → present | Credit-cycle proxy |
| ICE BofA EM Corporate OAS | `BAMLEMCBPIOAS` | Daily | 2003 → present | EM credit / spread proxy |
| ICE BofA US HY OAS | `BAMLH0A0HYM2` | Daily | 1996 → present | Secondary credit-cycle proxy |
| 10Y-2Y Treasury spread | `T10Y2Y` | Daily | 1976 → present | Curve / recession proxy |
| GFP (Miranda-Agrippino & Rey) | silviamirandaagrippino.com | Monthly | 1980 → latest | **Deferred to v1.1** — free, public |

### Storage architecture

Single proprietary Excel file is the source of truth. For v1.0:

- Nightly: read Excel via pandas → write to Parquet snapshots (`data_YYYYMMDD.parquet`).
- Append-only journal of historical Parquet snapshots → enables reproducibility and version auditing.
- Optional: load into Postgres + TimescaleDB if downstream consumers need SQL access.

**No vendor seat negotiations required, no paid licenses.**

---

## 8. Architecture

```
[Proprietary Excel: data.xlsx]   [FRED API — free]
            ↓                            ↓
   [Nightly ETL — pandas + fredapi]
            ↓
   [Parquet snapshot + Postgres optional]
            ↓
  [Compute engine — Python: numpy, pandas, statsmodels, scikit-learn]
            ↓
┌──────────┬──────────────┬──────────────────┬────────────────┐
↓          ↓              ↓                  ↓                ↓
[β est.] [Corr/PC1]  [External val.]   [Segmentation]   [Aggregate check]
└──────────┴──────────────┴────────────┬─────┴────────────────┘
                                       ↓
                  [Percentile classifier — 5Y rolling, terciles]
                                       ↓
                  [HTML dashboard (2-tier) + JSON API + Parquet]
                                       ↓
                       [Downstream: PM, Risk, IC]
```

**Tech stack:**

- Python 3.12 + numpy, pandas, statsmodels, scikit-learn (PCA), numba (hot loops), fredapi (FRED ingest).
- Storage: Parquet for snapshots; Postgres + TimescaleDB optional for downstream.
- Orchestration: cron → Python job. Claude Code for prototyping.
- Dashboard: HTML/SVG, static-served via Coolify on alanvaa.cloud (default) or Vercel.

---

## 9. Output / UX

### 9.1 Daily HTML report — two-tier

**Tier 1 (default landing view):**

- Top banner: Global regime — current tercile (Risk-on / Transitional / Risk-off), direction, 5/30/90d trajectory.
- LatAm card.
- "Largest divergence today" card — auto-surfaced segment whose bucket differs most from the global signal.
- β series chart (cap-weighted + equal-weighted) over 5Y, tercile bands shaded.

**Tier 2 (one click — full diagnostic):**

- Full 8-card grid: DM, EM, Equity, FI, DM-Eq, EM-Eq, DM-FI, EM-FI.
- Correlation panel: avg pairwise + PC1 variance share, same 5Y window.
- Internal consistency strip: model regimes vs the 6 composite aggregates (DM/EM/Europe/Asia/World/LatAm).
- External validation strip: rolling 60d correlation vs VIX / BBB OAS / EM Corp OAS / HY OAS / 2s10s.
- Alerts section: bucket transitions in last 5d; disagreement events; validation degradation.

### 9.2 JSON snapshot (per day)

```json
{
  "date": "2026-05-27",
  "methodology_version": "1.0.0",
  "global": {
    "bucket": "Transitional",
    "percentile_5y": 0.52,
    "quintile_underneath": "Q3",
    "direction": "falling"
  },
  "segments": {
    "DM_Eq":   {"bucket": "Risk-on",      "percentile_5y": 0.71, "direction": "stable",  "N": 17},
    "EM_Eq":   {"bucket": "Risk-off",     "percentile_5y": 0.18, "direction": "falling", "N": 15},
    "DM_FI":   {"bucket": "Transitional", "percentile_5y": 0.45, "direction": "stable",  "N": 17},
    "EM_FI":   {"bucket": "Risk-off",     "percentile_5y": 0.08, "direction": "falling", "N": 15},
    "LatAm":   {"bucket": "Risk-off",     "percentile_5y": 0.21, "direction": "falling", "N": 10, "flag": "thin_cut"}
  },
  "correlation": {"avg_pairwise_3m": 0.41, "pc1_variance_share": 0.38},
  "external_corr_60d": {
    "VIX": -0.55,
    "BBB_OAS": -0.48,
    "EM_Corp_OAS": -0.51,
    "HY_OAS": -0.49,
    "2s10s": 0.22
  },
  "internal_consistency": {"MXWO_aligned": true, "MXEF_aligned": false},
  "alerts": [
    "EM_FI moved Transitional → Risk-off on 2026-05-25",
    "Internal: MXEF aggregate aligned with EM_Eq cut, diverges from EM_FI cut"
  ]
}
```

---

## 10. Success Metrics

### Diagnostic quality (v1.0 acceptance gates)

- **External validation correlation:** rolling 60d |ρ| of global regime vs **VIX** ≥ 0.5, and vs **US BBB OAS** ≥ 0.4, in ≥ 80% of periods over the 2010-2024 backtest. (Signs expected negative — high VIX / wide spreads = risk-off bucket.)
- **Known-events recognition:** model classifies Risk-off bucket within ±3 trading days of each of:
  - 2010 Greek crisis (May)
  - 2011 Eurozone periphery + US downgrade (Aug)
  - 2015 China devaluation (Aug)
  - 2018 Q4 sell-off (Oct-Dec)
  - 2020 COVID (Feb-Mar)
  - 2022 rate shock (multiple windows)
  - 2008 Lehman is the earliest valid event but requires reduced trailing-5Y percentile (use 18-month bootstrap calibration for pre-2014 dates).
- **Segmentation lift:** ≥ 20% of days show DM-Eq and EM-Eq bucket labels **≥ 2 terciles apart** (i.e., DM = Risk-on and EM = Risk-off, or vice versa). Bar calibrated post-backtest if empirical distribution warrants adjustment.
- **Stability:** ≤ 2 spurious bucket transitions per quarter in calm regimes (defined as quarters where realized vol of MXWO < its 10Y median).
- **Internal consistency:** in any rolling 30d window, no more than 5 days where the DM regime label disagrees with the MXWO aggregate's own 3M return percentile (>1 tercile gap).

### Predictive (deferred — v2 only, not v1.0 gate)

- Post-transition realized drawdown in risky-asset proxy (MXWO or EM-Eq cap-weighted basket) consistent with Beber et al. magnitudes (~500 bps over 60 days post low→high transition in their FX-carry-equivalent measure).

### Operational

- Nightly batch completes by 6:00 local; dashboard published by 7:00.
- Zero missed days over rolling 90d window.

---

## 11. Milestones

| Sprint | Deliverable | Source phase | Est |
|---|---|---|---|
| S0 | **Pre-sprint:** obtain FRED API key (free, instantaneous at fred.stlouisfed.org); confirm Excel data refresh cadence; environment setup | — | 0.5 wk |
| S1 | Excel ETL → Parquet snapshots; loader + validator; FRED ingest | — | 1 wk |
| S2 | **Baseline β series** (EWMA vol + 3M return, cap-wtd + equal-wtd, full 64-series universe) | Phase 1 | 2 wks |
| S3 | Segmented β (DM/EM, Eq/FI, 4-way, LatAm) | Phase 2 | 1 wk |
| S4 | Percentile classifier, 5Y rolling, terciles + underlying quintiles | Phase 3 | 1 wk |
| S5 | **Correlation + PC1 panel** | Phase 4 | 1 wk |
| S6 | External validation panel (5 FRED series) + internal aggregate consistency | Phase 5 | 2 wks |
| S7 | 1M tripwire window | Phase 6 | 0.5 wk |
| S8 | Dashboard (HTML 2-tier) + JSON API | — | 2 wks |
| S9 | **Backtest + parameter selection + acceptance checks against known events** | — | 3 wks |
| **v1.0 release** | Diagnostic system live | | **~14 wks total** |
| v1.1+ | GFP integration; HMM / Markov-switching on validated β | Phase 7 | TBD |
| v1.2+ | Secondary risk axes (drawdown, semivariance, skew) | Phase 8 | TBD |
| v2.0 | Predictive layer (transition persistence) | Phase 9 | TBD |

---

## 12. Open Decisions

### Decisions resolved by data inventory

- ✅ Universe scope: 32 countries × 2 asset classes = 64 series. No Argentina, Venezuela, Russia.
- ✅ LatAm bloc composition: Brazil, Mexico, Chile, Peru, Colombia.
- ✅ Data history: 2008-12-31 onward. 2008 GFC included intact; no truncation or winsorization in primary backtest. Sensitivity check in S9 excluding 2008-2009.
- ✅ Local-currency conversion for FI already done upstream.
- ✅ No vendor seat negotiations needed for price data.
- ✅ No paid licenses needed for external validation — all five proxies are FRED-free.

### Decisions deferred to S9 empirical selection (not user-vote)

- EWMA half-life: 20 vs 30 vs 40 days — select on max validation-correlation × stability tradeoff.
- Percentile bucket structure: tercile (default) vs quintile vs asymmetric 20/60/20 — select on stability metric.
- Cap-weighted vs equal-weighted as headline: empirical, post-backtest. (Both produced and displayed regardless.)

### Decisions requiring user input before S0

| Decision | Options | Recommended default |
|---|---|---|
| Dashboard hosting | Coolify on alanvaa.cloud / Vercel / S3+CloudFront | Coolify (consistent with Provex AI stack, no marginal cost) |
| Postgres + TimescaleDB or Parquet-only | Both / Parquet only | Parquet only for v1.0; Postgres deferred until downstream consumers exist |
| GFP integration timing | v1.0 / v1.1 / never | v1.1 — not on critical path |
| FRED API key | Action: register free at fred.stlouisfed.org | **Action before S0** |

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Lagged signal still detects transitions late | Diagnostic value diminished | 1M tripwire window (F1.3) + correlation/PC1 panel (F5) as parallel fast signals |
| **Cap-weighted slope dominated by US** (~54% of total mcap) → "global signal" is US 60/40 in disguise | False sense of universality | Mandatory equal-weighted parallel (F2.2) + slope-spread auxiliary (F2.4); both displayed always |
| Realized returns are noisy expected-return proxy (Berkman & Malloch) | Model captures past, not forward-looking | v1.0 explicitly diagnostic; predictive layer deferred to v2 with transition-persistence basis |
| **Single Excel file as data source — single point of failure / no versioning** | Operational fragility | Parquet snapshot on every load + journaled append-only history; validator checks shape, date continuity, no-NaN invariant; alert on schema drift |
| Excel file update cadence unclear | Stale data risk | Document refresh SLA with data provider; alert if file mtime > T+2 |
| FRED API downtime or schema changes | External validation gap | Cache latest FRED pulls locally; alert on stale pull > T+2; FRED is highly stable historically but mitigation should exist |
| Regime classification becomes self-fulfilling (PMs trade on it → causes the regime) | Reflexivity | Internal-only release for v1.0; treat as one input among several in IC framework, not standalone bucket label; document reflexivity caveat in model card |
| Survives backtest, fails in next regime | Standard out-of-sample risk | Lock 18-month holdout pre-launch (2024-11 → 2026-05); document acceptance metrics; recalibration policy |
| **Methodology version drift confuses IC** ("why does Q1 2025 show a different regime now than it did last quarter?") | Trust erosion | Methodology version locked in every JSON snapshot; migration policy documented; historical β series recomputed only on explicit version bumps with changelog |
| **LatAm at N=10 is thin** — single-name moves can swing the cross-section | Noisy LatAm signal | Flag LatAm as "thin cut" in output; require both equity and FI confirmation for LatAm regime change alerts |

---

## 14. Dependencies

- **Data:** None for primary universe (proprietary Excel in hand). For external validation: free FRED API key only.
- **Engineering:** 1 quant FTE × ~14 weeks; 0.5 data eng × ~2 weeks; 0.5 frontend × ~2 weeks.
- **Decisions from user (§12) before S0 starts.**

---

## 15. Out-of-Scope Future Work (v2+)

- **Predictive layer** (Phase 9) based on observed regime-transition persistence (Beber et al. template: ~200 bps/20d, ~500 bps/60d post low→high in FX-carry-equivalent multi-asset measure). Not realized-return forecasting.
- **RSDC on full correlation matrix** (Beber et al. native implementation) — Phase 7-ish.
- **Statistical Jump Models** (Cortese et al. 2026, FTIC criterion) as alternative regime engine — sparser than HMM, scales to high-dim feature spaces.
- **Option-implied EMRP overlay** for the markets where Berkman & Malloch validated tightness — if vendor licensing economic.
- Drawdown / downside-semivariance / skew as parallel risk axes (Phase 8).
- LLM-narrated daily commentary on regime card.
- **GFP integration** as additional validation series (v1.1).

---

## 16. Model Card (Appendix)

**Model name:** RoRo Risk-Regime Classifier v1.0
**Owner:** Alan Vazquez, CFA
**Inputs:** 64 local-currency price series (32 countries × Eq+FI); USD market caps; external validation series from FRED (VIX, BBB OAS, EM Corp OAS, HY OAS, 2s10s).
**Outputs:** Daily tercile regime label per cut (global + 8 segments + LatAm); underlying percentile; direction flag; correlation/PC1 panel; validation correlation panel; alerts.

**What this model does NOT claim:**
- It is **not** a return forecast.
- It is **not** a market-timing signal.
- It is **not** a substitute for IC judgment — it is one input among several.
- It is **not** calibrated outside the 2009-2026 sample period.

**Known failure modes:**
- Lagged regime detection during fast transitions (mitigated by 1M tripwire + correlation panel, not eliminated).
- Cap-weighted slope dominated by US — read alongside equal-weighted always.
- LatAm cut is N=10 — interpret with caution.
- Methodology recalibration may shift historical regime labels; version-locked.

**Refresh cadence:** Nightly batch. Methodology version bumps quarterly at most, with explicit changelog.

**Human in the loop:** PM reviews daily; IC reviews quarterly. Bucket transitions in the last 5 days are surfaced as alerts but do not trigger automated action.

---

## Source / References

- raw/RoRo Risk Regime.md — design doc, 2026-05-27
- wiki/Finance/Risk-On Risk-Off Regime Classifier (Alan 2026).md
- wiki/Finance/Switching Risk Off — FX Correlations and Risk Premia (Beber Brandt Cen 2013).md
- wiki/Finance/Expected Market Risk Premiums International Cross-Section (Berkman & Malloch 2024).md
- wiki/Finance/Generalized Information Criteria for Jump Models (Cortese 2026).md
- `data.xlsx` — proprietary universe + price file, refreshed 2026-05-26
- FRED — Federal Reserve Bank of St. Louis, fred.stlouisfed.org (free)