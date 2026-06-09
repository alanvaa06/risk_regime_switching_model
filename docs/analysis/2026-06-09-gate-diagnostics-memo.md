# RoRo Gate-Diagnostics Findings Memo

**Date:** 2026-06-09
**Branch:** feat/jump-model
**Author:** Alan Vazquez, CFA
**Harness:** `roro gate-diagnostics` — `outputs/gate_diag/`
**Period:** 2008-12-31 → 2026-05-26

---

## 1. Headline

"RoRo fails 5/6 gates" collapses to **RoRo fails G5 (stability); the rest of the scorecard is mis-specified** — two gates are method-shared pipeline-health checks (G1/G2), one is a scorecard bug that counts an out-of-range event (G3), one is borderline (G4), and one is vacuous by construction (G6).

---

## 2. Per-Gate Table

| Gate | Bar | Percentile | HMM | Root-cause tag | Shared / Discriminating |
|---|---|---|---|---|---|
| **G1** `vix` | frac ≥ 0.80 | 0.507 | **0.507** (identical) | `shared` | Shared — pipeline-health check on β-vs-VIX correlation; output is independent of the classifier |
| **G2** `bbb` | frac ≥ 0.80 | 0.686 | **0.686** (identical) | `shared` | Shared — same logic applied to BBB spread; cannot discriminate method |
| **G3** `events` | 8/8 | 7/8 | 6/8 | `bug` | Discriminating — but the bar is unfillable by construction (see §3) |
| **G4** `seg_lift` | frac ≥ 0.20 | 0.183 | 0.161 | `borderline` | Discriminating — a hair below the bar for both methods |
| **G5** `stability` | max ≤ 2 | **18** | **9** | `real` | Discriminating — the one honest, binding method failure; HMM halved it |
| **G6** `internal` | max ≤ 5 | pass | pass | `vacuous` | Shared — composite price unwired → "no DM mapping data"; auto-passes by absence of data |

All values sourced from `outputs/gate_diag/gate_diagnostics.json`. Confirmed against `outputs/hmm_eval/acceptance_compare.json` (baseline pin verified — see §3).

---

## 3. G3 Repair — Scorecard Bug

The `g3_repair` block from the harness:

```
out_of_range : ["2008_lehman"]   (event date 2008-10-10; backtest start 2008-12-31)
total_in_range   : 7
hits (in-range)  : 7
graded (in-range): 1.0
passed_in_range  : true
```

The 8/8 binary bar is guaranteed to fail by construction. The only miss is `2008_lehman` (event date 2008-10-10), which falls **62 days before** the backtest start (2008-12-31). Its ±6-day detection window contains zero rows and can never match. The production harness does not exclude it from the denominator, so `7/8 < 8/8` is structurally baked in.

**`baseline_pinned = true`**: the harness reproduced the recorded on-disk verdicts (`acceptance_compare.json`) exactly before any sweep ran. This proves the diagnostic scores the real production gates.

**In-range G3 = 7/7 (percentile), 6/7 (HMM).** On the corrected event set, percentile is 7/7 (passes a fair bar); HMM misses the 2010 Greek crisis.

---

## 4. Sweeps — Achievable Ranges

### G1 (`vix`) and G2 (`bbb`) — fraction-vs-rho_min

Both gates sweep the same `rho_min` knob (minimum correlation threshold to count a quarter as "above"). Because G1/G2 read from the percentile-derived external-validation CSV regardless of which classifier is active, **both methods produce identical fractions at every rho_min level** — confirming the shared / pipeline-health classification.

| rho_min | G1 frac_above | G2 frac_above |
|---|---|---|
| 0.3 | 0.727 | 0.743 |
| 0.4 | 0.622 | 0.686 |
| 0.5 | 0.507 | 0.604 |
| 0.6 | 0.377 | 0.535 |
| 0.7 | 0.202 | 0.393 |
| 0.8 | 0.062 | 0.257 |

**Conclusion:** No defensible `rho_min` threshold achieves `frac_above ≥ 0.80`. The fraction at `rho_min=0.3` (the most permissive) is 0.727 / 0.743 — still below bar. These gates measure a property of the market regime itself (how often VIX / BBB are meaningfully correlated with the portfolio's β), not the quality of the classifier. **They should be re-tiered as pipeline-health diagnostics, not pass/fail classifier gates.**

### G4 (`seg_lift`) — fraction-vs-gap

| gap_threshold | frac_with_gap_ge (percentile) |
|---|---|
| ≥ 1 | 0.577 |
| ≥ 2 | 0.183 |

At gap ≥ 1 the percentile classifier satisfies 57.7% of quarters. At the scored gap ≥ 2 bar it is 18.3%, just below the 20% threshold. No alternative threshold assessed here passes without a design-level decision on what "lift" means (see §6). **Borderline — a calibration-split recalibration could plausibly pass this gate.**

---

## 5. G3↔G5 Frontier (Crown Jewel)

The frontier sweeps a causal hysteresis `confirm_days` knob over the percentile classifier: labels must persist for `n` consecutive days before flipping. This is the same `_smooth_regime_hysteresis` primitive used in production viz. All arithmetic is forward-looking only — the no-lookahead invariant is tested and guaranteed.

### Frontier Table

| confirm_days | events_caught (in-range) | max_calm_transitions |
|---|---|---|
| 0 | 7 | 18.0 |
| 1 | 7 | 10.0 |
| 2 | 7 | 10.0 |
| 3 | 7 | 8.0 |
| 5 | 6 | 6.0 |
| 8 | 5 | 4.0 |
| 13 | 6 | 3.0 |
| 21 | 6 | 2.0 |
| 34 | 4 | 1.0 |

*Source: `outputs/gate_diag/g3_g5_frontier.csv`. events_caught = in-range events (7 possible; 2008_lehman excluded).*

### Feasibility Conclusion

**No point on this frontier achieves both 7/7 in-range events AND ≤ 2 calm-quarter transitions simultaneously.**

- At `confirm_days = 3`: 7/7 events caught, but 8 calm transitions — still 4× the G5 bar.
- At `confirm_days = 21`: transitions reach 2 (G5 passes), but events drop to 6/7 (G3 misses).
- The curve exhibits a **hard tradeoff**: persistence sufficient to suppress the noisy micro-flickers that drive G5 also introduces lag that costs detection of a real stress event.

### The JM's Quantitative Success Target

This frontier is simultaneously the gate diagnosis and the Statistical Jump Model's **minimum bar**. A causal hysteresis filter on the percentile classifier cannot jointly satisfy G3 and G5 at any `confirm_days` setting. The Statistical Jump Model — whose jump penalty provides regime-persistence natively, without the lag of a trailing confirmation window — must land **above and to the left of this curve**: 7/7 events caught AND ≤ 2 calm transitions in any calm quarter. If the JM clears both simultaneously, it has demonstrated something a simple persistence post-filter structurally cannot.

---

## 6. Recalibration Options (Non-Committal Appendix)

These are inputs to a follow-up gate-recalibration decision. No threshold is locked here.

### 6.1 Gate Tiering: Pipeline-Health vs. Discriminating

The six gates fall naturally into two tiers:

| Tier | Gates | Rationale |
|---|---|---|
| **Pipeline-health** | G1 (vix), G2 (bbb), G6 (internal) | Method-shared or vacuous; measure data quality / wiring, not classifier quality |
| **Discriminating** | G3 (events), G4 (seg_lift), G5 (stability) | Method-specific outcomes; appropriate for classifier bake-off |

Under this tiering the acceptance scorecard becomes a 3-gate discriminating test (G3/G4/G5) with G1/G2/G6 as separate health monitors. The S9 "fail 5/6" headline disappears; the real story is "fails 1/3 discriminating gates (G5) and is borderline on a second (G4)."

### 6.2 G3 Scoring Fix

Drop `2008_lehman` from the event registry, or relocate the event reference date to post-2008-12-31 (the actual Lehman aftermath market dislocation extended into Q1/Q2 2009). Either change converts the binary 8/8 bar to a fair 7/7 bar — which percentile already passes.

Alternatively: keep the event in the registry but exclude it from the denominator when its window predates `backtest_start`. This is the minimal no-schema-change fix.

### 6.3 G4 Threshold Candidate

Fraction-with-gap ≥ 1 is 57.7% for percentile. A bar of ≥ 0.50 at gap ≥ 1 (rather than the current ≥ 0.20 at gap ≥ 2) would pass percentile comfortably and provide a more meaningful segmentation-quality check. Alternatively, calibrate the ≥ 0.20 at gap ≥ 2 bar against an in-sample (pre-2018) / out-of-sample (post-2018) split to avoid overfitting the threshold to full-period data.

### 6.4 G5 Threshold Candidate

The ≤ 2 bar for max calm-quarter transitions is extremely tight — only achievable with `confirm_days ≥ 21`, at which point G3 already breaks. A calibration-split approach would ask: what is the 90th-percentile max calm-quarter transition count for a "good" classifier over the in-sample period, and use that as the bar? The HMM value of 9 (vs. percentile's 18) suggests the bar should be in the range of ≤ 5–10 for a meaningful discriminating test at current data length.

### 6.5 Frontier as the Gate Standard

Rather than two separate G3 and G5 pass/fail gates, the frontier itself could become the acceptance criterion: a new classifier must land strictly inside (above-left of) the G3↔G5 curve established by the percentile baseline under optimal causal hysteresis. This is a Pareto-dominance standard that is both more rigorous and more interpretable than two independent binary bars.

---

*Harness commit: see `docs/context/sesion-log.md`. Artifacts: `outputs/gate_diag/gate_diagnostics.json`, `outputs/gate_diag/g3_g5_frontier.csv`. Design: `docs/superpowers/specs/2026-06-08-gate-diagnostics-design.md`.*
