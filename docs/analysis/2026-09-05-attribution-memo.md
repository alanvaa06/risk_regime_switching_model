# Attribution real-data memo (2026-09-05)

Author: Alan Vazquez, CFA

## Run

```
uv run roro run --config configs/default.yaml --date 2026-09-05 \
  --as-of-data-date 2026-05-26 --out outputs/attribution_run --force
```

| Field | Value |
| --- | --- |
| Run date | 2026-09-05 |
| As-of data date | 2026-05-26 |
| Methodology version | 1.0.0 |
| Git SHA (`snapshot.json.code_version`) | `99856c98e91b6e85ea3d52a5cfdd2bb99586cb4e` (clean) |
| Engine runtime | ~63 s total (as reported by the run controller; not re-timed for this memo) |
| Config | `configs/default.yaml`: `min_n_per_cut=10`, `percentile_window_years=5`, `attribution_top1_alert=0.5`, `attribution_fixed_horizon_days=63`, `attribution_label_source=percentile`, `attribution_anchor_lookback_days=1260`, `attribution_history_global=False` |
| Data fingerprint | `data.xlsx` sha256 `2df7f078…f72afa6`; FRED pulled 2026-09-05T18:58:28 |
| Snapshot warnings | 4 FI local-currency NaN-column notices (`Taiwan`, `Chile`, `DM`, `Asia`) — pre-existing, unrelated to attribution |

Artifact coverage: `attribution.csv` and `attribution_rollup.csv` carry the last date only
(2026-05-26); `concentration.csv`, `beta_series.csv`, `regimes.csv` and `alerts.csv` carry the
full history (2009-03-30 to 2026-05-26 for concentration; 4,477 business days per cut).

## Snapshot reproduction

Global cut, 2026-05-26. Top 6 by |contribution|, cap-weighted:

| Series | Block | Quadrant | Contribution | Share |
| --- | --- | --- | ---: | ---: |
| South Korea__Eq | EM_Eq | HI/+ | 0.3084 | 0.7187 |
| Taiwan__Eq | EM_Eq | HI/+ | 0.0954 | 0.2223 |
| China__Eq | EM_Eq | HI/- | -0.0294 | -0.0686 |
| United States__FI | DM_FI | LO/- | 0.0274 | 0.0638 |
| Netherlands__Eq | DM_Eq | HI/+ | 0.0182 | 0.0425 |
| China__FI | EM_FI | LO/+ | -0.0161 | -0.0375 |

Equal-weighted, same date and cut:

| Series | Block | Quadrant | Contribution | Share |
| --- | --- | --- | ---: | ---: |
| South Korea__Eq | EM_Eq | HI/+ | 0.2075 | 1.0654 |
| Indonesia__Eq | EM_Eq | HI/- | -0.0687 | -0.3529 |
| Taiwan__Eq | EM_Eq | HI/+ | 0.0616 | 0.3163 |
| Netherlands__Eq | DM_Eq | HI/+ | 0.0309 | 0.1586 |
| Peru__Eq | EM_Eq | HI/- | -0.0302 | -0.1549 |
| South Africa__Eq | EM_Eq | HI/- | -0.0257 | -0.1319 |

Verification against the spec section 2 snapshot:

| # | Quantity | Spec | Observed | Verdict |
| --- | --- | ---: | ---: | --- |
| 1 | global cap beta | 0.4290 | 0.4290 | PASS |
| 2 | South Korea__Eq contribution | 0.3084 | 0.3084 | PASS |
| 3 | South Korea__Eq share | 0.7187 | 0.7187 | PASS |
| 4 | South Korea__Eq quadrant | HI/+ | HI/+ | PASS |
| 5 | Taiwan__Eq contribution | 0.0954 | 0.0954 | PASS |
| 6 | China__Eq contribution | -0.0294 | -0.0294 | PASS |
| 7 | United States__FI contribution | 0.0274 | 0.0274 | PASS |
| 8 | global cap HHI | 0.33 | 0.3296 | PASS |
| 9 | global cap top-5 share | 84% | 84.10% | PASS |
| 10 | rollup block EM_Eq | +0.357 | +0.3567 | PASS |
| 11 | rollup block DM_Eq | +0.046 | +0.0458 | PASS |
| 12 | rollup block DM_FI | +0.043 | +0.0434 | PASS |
| 13 | rollup block EM_FI | -0.017 | -0.0168 | PASS |
| 14 | global eq beta | 0.195 | 0.1947 | PASS |
| 15 | South Korea__Eq eq share | 107% | 106.54% | PASS |

**15/15 PASS.** The `snapshot.json.attribution.global` block agrees with `attribution.csv`
and `concentration.csv` to full float precision (top1_share 0.541630, HHI 0.329572,
`fragile_flag=true`, anchor 2026-05-01).

Latest-date concentration detail, global cut:

| Weighting | n | beta | HHI | top1 | top1 share | top5 share | beta_ex_top1 | fragile |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| cap | 64 | 0.4290 | 0.3296 | South Korea__Eq | 0.5416 | 0.8410 | 0.2990 | True |
| eq | 64 | 0.1947 | 0.1495 | South Korea__Eq | 0.3405 | 0.6545 | -0.0121 | False |

Note the equal-weighted row: dropping Korea moves the slope from +0.195 to -0.012, i.e. a
sign flip, yet `fragile_flag` is False. That is by construction, not a defect —
`roro/attribution.py:446` computes `pct_ex_top1` and the fragility test for `weighting ==
"cap"` only, so every `eq` row carries `pct_ex_top1 = NaN` and `fragile_flag = False`. See
Open issues.

## Concentration statistics

Full history, per cut and weighting (4,477 business days; LatAm 3,390 because the cut only
clears `min_n_per_cut=10` from 2013 onward):

| Cut | Weighting | Days | top1_share > 0.5 | Fragile | median top1_share | NaN beta_ex_top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| global | cap | 4477 | 19.25% | 27.74% | 0.3490 | 0.00% |
| DM | cap | 4477 | 37.97% | 25.89% | 0.4391 | 0.00% |
| DM_Eq | cap | 4477 | 23.79% | 37.70% | 0.3891 | 0.00% |
| DM_FI | cap | 4477 | 27.72% | 38.33% | 0.4103 | 0.00% |
| EM | cap | 4477 | 59.41% | 35.96% | 0.5391 | 0.00% |
| EM_Eq | cap | 4477 | 42.01% | 38.95% | 0.4607 | 0.00% |
| EM_FI | cap | 4477 | 24.08% | 42.04% | 0.3982 | 0.00% |
| Equity | cap | 4477 | 25.73% | 36.54% | 0.4072 | 0.00% |
| FI | cap | 4477 | 7.04% | 37.59% | 0.3149 | 0.00% |
| LatAm | cap | 3390 | 49.65% | 29.14% | 0.4985 | 0.00% |
| global | eq | 4477 | 0.13% | n/a | 0.1312 | 0.00% |
| DM | eq | 4477 | 0.74% | n/a | 0.1720 | 0.00% |
| DM_Eq | eq | 4477 | 4.31% | n/a | 0.2387 | 0.00% |
| DM_FI | eq | 4477 | 9.45% | n/a | 0.2725 | 0.00% |
| EM | eq | 4477 | 1.07% | n/a | 0.2015 | 0.00% |
| EM_Eq | eq | 4477 | 4.85% | n/a | 0.2625 | 0.00% |
| EM_FI | eq | 4477 | 5.65% | n/a | 0.3034 | 0.00% |
| Equity | eq | 4477 | 1.34% | n/a | 0.1784 | 0.00% |
| FI | eq | 4477 | 0.58% | n/a | 0.2330 | 0.00% |
| LatAm | eq | 3390 | 15.34% | n/a | 0.3640 | 0.00% |

`n/a` for the eq fragility column: the flag is only defined on cap rows (see above), so the
literal 0.00% there carries no information.

**LatAm probe check.** `beta_ex_top1` is finite on 3,390 / 3,390 LatAm days for both
weightings (0.00% NaN), with `n` at the floor of 10 throughout. The 3-asset floor in the
ex-top1 probe is doing its job — this was the previously reported gap and it is closed.

Top-8 `top1_series` frequency, global cut:

| Rank | cap | days | share | eq | days | share |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 1 | China__Eq | 1833 | 40.94% | Peru__Eq | 771 | 17.22% |
| 2 | United States__Eq | 1743 | 38.93% | China__Eq | 656 | 14.65% |
| 3 | Japan__Eq | 382 | 8.53% | Brazil__Eq | 485 | 10.83% |
| 4 | United States__FI | 318 | 7.10% | Colombia__Eq | 321 | 7.17% |
| 5 | South Korea__Eq | 155 | 3.46% | Spain__Eq | 283 | 6.32% |
| 6 | China__FI | 27 | 0.60% | Italy__Eq | 281 | 6.28% |
| 7 | Taiwan__Eq | 16 | 0.36% | Indonesia__Eq | 208 | 4.65% |
| 8 | Spain__Eq | 1 | 0.02% | Japan__Eq | 189 | 4.22% |

Cap-weighted, the global slope is a two-name story: China and US equity together own the
top-1 slot on 79.9% of days, and the top 5 names cover 98.96%. Equal weighting disperses
this completely (top 1 name 17.2%), which is the expected mechanical consequence of removing
market-cap leverage from `c_i = h_i * y_i`.

### D3 verdict

Global cap `top1_share > 0.5` on **19.25%** of days (862 / 4,477), against a 20% threshold.

**D3 not triggered.** No change made. Flagging that this clears by 0.75pp — well inside the
margin that a data refresh or a universe change could move — so treat the verdict as
provisional rather than settled. At the cut level the picture is much worse: EM (59.41%),
LatAm (49.65%) and EM_Eq (42.01%) are all far past the same threshold; if the D3 trigger is
ever restated per-cut rather than global-only, it fires immediately.

### Jackknife note

The exact sign test from spec section 6 — share of days where `sign(beta - beta_ex_top1)`
equals the sign of the top-1 asset's own contribution — **could not be computed from this
run**. `attribution.csv` holds the last date only, so historic per-asset contributions are not
available; producing them requires `attribution_history_global`, which is `False` in
`configs/default.yaml` and confirmed off in `snapshot.json.config_resolved`. This has since
been resolved with a dedicated re-run — see "Jackknife sign check (spec §6)" below.

What can be measured on the 4,477 valid global-cap days:

| Statistic | Value |
| --- | ---: |
| `beta - beta_ex_top1`, median | 0.0486 |
| `beta - beta_ex_top1`, p10 | -0.2417 |
| `beta - beta_ex_top1`, p90 | 0.2239 |
| `abs(beta - beta_ex_top1) / abs(beta)`, median | 0.3773 |
| `abs(beta - beta_ex_top1) / abs(beta)`, p10 | 0.0664 |
| `abs(beta - beta_ex_top1) / abs(beta)`, p90 | 2.2012 |
| Days where dropping top-1 flips the sign of beta | 17.15% (768 / 4477) |
| Days where `abs(beta - beta_ex_top1) > abs(beta)` | 21.62% |

Reading: on a median day the top-1 name accounts for ~38% of the magnitude of the global
cap-weighted slope, and on roughly one day in six it accounts for more than all of it — the
slope changes sign when the single largest name is removed. That is a stronger robustness
concern than the 19.25% headline suggests, because it is a statement about the *sign* of the
regime read, not just its size.

### Jackknife sign check (spec §6)

Re-ran the engine with `configs/attribution-history.yaml` (= `configs/default.yaml` plus
`attribution_history_global: true`):

```
uv run roro run --config configs/attribution-history.yaml --date 2026-09-05 \
  --as-of-data-date 2026-05-26 --out outputs/attribution_history_run --force
```

This writes `attribution_history_global.csv`, the wide per-asset cap-weighted global
contribution history (4,477 dates x 64 series, 6.4 MB). Running the exact spec section 6 sign
test against it, over the same 4,477 valid global-cap days:

- `top1_series` in `concentration.csv` equals `argmax_i |c_i|` computed from the history matrix
  on **100.0%** of days — the two code paths (the daily concentration diagnostic and the full
  contribution history) agree exactly.
- `sign(beta - beta_ex_top1) == sign(c_top1)` on **85.7%** of days.
- Row sums of the history matrix equal `beta` to a maximum absolute error of **3.0e-15**,
  confirming the decomposition is exact over the full history, not just on the last date.

**Interpretation.** Removing the top-1 asset changes the slope through two channels: its own
contribution `c_top1`, and the re-estimated leverage of every other asset (`xbar` and `D` both
move once the top-1 name is dropped). Exact sign agreement is therefore not an identity —
85.7% says the direct channel dominates, but the cross-channel effect is material, which is
consistent with the 17.15% sign-flip rate reported above. Spec section 6 is closed by this run;
see the Open issues section for status.

## Concentration alerts

`alerts.csv` holds 57,344 rows across four kinds: `validation_degradation` 36,158,
`concentration` 15,749, `bucket_transition` 5,204, `disagreement` 233. All concentration
alerts are cap-weighted, as specified, spanning 2010-03-16 to 2026-05-26.

By segment and trigger:

| Segment | fragile | transition_day | Total |
| --- | ---: | ---: | ---: |
| global | 1208 | 56 | 1264 |
| DM | 1067 | 160 | 1227 |
| DM_Eq | 1616 | 111 | 1727 |
| DM_FI | 1592 | 171 | 1763 |
| EM | 1515 | 166 | 1681 |
| EM_Eq | 1691 | 93 | 1784 |
| EM_FI | 1815 | 91 | 1906 |
| Equity | 1581 | 87 | 1668 |
| FI | 1635 | 63 | 1698 |
| LatAm | 912 | 119 | 1031 |
| **Total** | **14632** | **1117** | **15749** |

Most-alerted names: China__Eq (3,971), United States__Eq (2,939), China__FI (2,476),
United States__FI (2,135), Japan__Eq (1,035), Brazil__Eq (678), Italy__FI (412),
Mexico__FI (396).

Ten most recent:

| Date | Segment | top1_series | top1_share | HHI | Fragile | Trigger |
| --- | --- | --- | ---: | ---: | --- | --- |
| 2026-05-25 | EM | South Korea__Eq | 0.7400 | 0.5747 | True | fragile |
| 2026-05-25 | EM_FI | China__FI | 0.5016 | 0.2891 | True | fragile |
| 2026-05-25 | Equity | South Korea__Eq | 0.5494 | 0.3731 | True | fragile |
| 2026-05-25 | global | South Korea__Eq | 0.5623 | 0.3479 | True | fragile |
| 2026-05-26 | Equity | South Korea__Eq | 0.5041 | 0.3464 | True | fragile |
| 2026-05-26 | EM | South Korea__Eq | 0.7158 | 0.5460 | True | fragile |
| 2026-05-26 | FI | China__FI | 0.2605 | 0.1659 | True | fragile |
| 2026-05-26 | DM | United States__Eq | 0.4071 | 0.2195 | True | fragile |
| 2026-05-26 | DM_Eq | United States__Eq | 0.6099 | 0.4148 | True | fragile |
| 2026-05-26 | global | South Korea__Eq | 0.5416 | 0.3296 | True | fragile |

The `fragile` trigger dominates 13:1 over `transition_day`. On global cap, fragility fires on
27.74% of days — roughly one day in four the regime bucket would change if the single largest
name were dropped. As a standalone alert that firing rate is too high to act on; it is better
read as a persistent property of the cap-weighted construction than as an event.

## Event review (AA-7)

G3 event list from `roro/backtest.py:56-64`. Seven of the eight events are in range;
`2008_lehman` (2008-10-10) pre-dates the backtest window (concentration history starts
2009-03-30) and is excluded. Transitions are global-segment tercile changes from
`regimes.csv`, excluding transitions into or out of `Unknown`, matched to the nearest
transition within +/-15 business days of the reference date.

Block moves are cap-weighted (`scheme=cap_wtd`) own betas from `beta_series.csv` on the
transition date versus the preceding row. Historic block rollups are not available
(`attribution_rollup.csv` is last-date only), so "block with largest own-beta move" is a
proxy for the block-level driver, not a decomposition of the global slope.

| Event | Transition date | From -> To | global top1 | top1 share | HHI | Fragile | Largest own-beta move | Block top1s (DM_Eq / EM_Eq / DM_FI / EM_FI) | Sensible? |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2010 Greek crisis (2010-05-06) | 2010-05-06 (+0d) | Transitional -> Risk-off | United States__Eq | 0.3931 | 0.1933 | False | EM_FI +0.1445 | Spain__Eq / China__Eq / United States__FI / China__FI | Partial |
| 2011 Eurozone + US downgrade (2011-08-08) | none in window (ref day 2011-08-08) | already Risk-off, all 31 bdays | United States__Eq | 0.5822 | 0.3528 | False | DM_Eq +0.5825 | United States__Eq / South Korea__Eq / United States__FI / Mexico__FI | Yes |
| 2015 China devaluation (2015-08-11) | none in window (ref day 2015-08-11) | already Risk-off, all 31 bdays | China__Eq | 0.7622 | 0.5838 | False | EM_FI +1.0929 | United States__Eq / China__Eq / Canada__FI / Colombia__FI | Yes |
| 2018 Q4 selloff (2018-12-24) | none in window (ref day 2018-12-24) | already Risk-off, all 31 bdays | United States__Eq | 0.6455 | 0.4364 | False | DM_Eq -0.1611 | United States__Eq / China__Eq / United States__FI / China__FI | Yes |
| 2020 COVID (2020-03-16) | 2020-02-25 (-20d) | Transitional -> Risk-off | United States__FI | 0.3387 | 0.1595 | False | DM_Eq +0.1991 | Japan__Eq / South Korea__Eq / United States__FI / China__FI | Partial |
| 2022 rate shock, Jan (2022-01-24) | 2022-01-14 (-10d) | Transitional -> Risk-off | China__Eq | 0.4689 | 0.2747 | **True** | DM_Eq -0.2633 | Netherlands__Eq / China__Eq / United States__FI / China__FI | Partial |
| 2022 rate shock, Sep (2022-09-26) | 2022-09-27 (+1d) | Risk-off -> Transitional | China__Eq | 0.3536 | 0.2212 | False | DM_FI -0.2223 | United States__Eq / China__Eq / United Kingdom__FI / China__FI | Yes |

One-line judgments:

- **2010 Greek crisis — Partial.** The transition lands on the exact reference date and DM_Eq's
  top-1 name is Spain__Eq, which is the right periphery transmission channel, but the largest
  own-beta move sits in EM_FI (+0.14), which has no natural link to a Greek sovereign event.
- **2011 Eurozone + US downgrade — Yes.** No transition is needed because global was already
  Risk-off across the whole window with beta -1.046 (high-vol punished), and both DM_Eq and
  DM_FI are led by US names days after a US sovereign downgrade.
- **2015 China devaluation — Yes.** The global slope is driven by China__Eq at a 76.2% share
  and HHI 0.584 on the devaluation date itself; the named driver is exactly the event, though
  the EM_FI +1.09 beta jump on a low top-1 share (Colombia__FI, 0.191) looks like small-n noise
  rather than signal.
- **2018 Q4 selloff — Yes.** United States__Eq owns 64.6% of the global slope with beta -1.182,
  and DM_Eq is the largest mover led by the same name — a textbook US-led equity de-risking.
- **2020 COVID — Partial.** The regime flips to Risk-off on 2020-02-25, the right date for the
  start of the drawdown, but the named global driver is United States__FI (the Treasury bid),
  not punished high-vol equities, and DM_Eq's beta moved *up* +0.199 on the transition day.
- **2022 rate shock, Jan — Partial.** DM_Eq taking the biggest hit (-0.263) led by
  Netherlands__Eq is precisely the long-duration-growth story of January 2022, but the global
  read is fragile: `beta = -0.125` versus `beta_ex_top1 = +0.214`, so removing China__Eq flips
  the sign of the slope and the global-level driver cannot be trusted that day.
- **2022 rate shock, Sep — Yes.** The largest own-beta move is DM_FI (-0.222) with top-1
  United Kingdom__FI at a 45.2% share — the gilt/LDI crisis named correctly — although the
  headline transition direction (Risk-off -> Transitional) reads backwards and sits inside a
  churn cluster of 8 global transitions in 31 business days.

Scorecard: 4 Yes, 3 Partial, 0 No. The three "no transition in window" events are not misses:
in all three the global segment was already sitting in Risk-off for the entire +/-31 business
day window, which is the correct label and is what the G3 gate actually tests (expected bucket
present in the window, not a transition inside it).

## Report

`outputs/attribution_report.html` exists and was built by the controller. Size: report HTML
~56.5 MB, of which attribution figures add ~4.2 MB (+8%); the remaining ~52 MB pre-exist
(vol heatmaps).

- `class="plotly-graph-div"` occurrences: **10** (expected 10) — PASS.
- All five attribution section titles present — PASS:
  - Slope attribution
  - Δβ̂ waterfall
  - Vol vs return, sized by |contribution|
  - Concentration of the slope
  - PC1 loadings vs variance share

The five pre-existing figures (Risk vs Return, Beta vs Return, Segment β with regime bands,
Volatility breadth, Volatility percentile by asset) are also intact.

## Open issues

- **Exact jackknife sign test — closed.** Re-run with `configs/attribution-history.yaml`
  (`attribution_history_global=True`) on the full 4,477-day global cap history: `top1_series`
  matches `argmax_i |c_i|` from the history matrix on 100.0% of days, `sign(beta -
  beta_ex_top1) == sign(c_top1)` on 85.7% of days, and history-matrix row sums equal `beta` to
  max abs error 3.0e-15. See "Jackknife sign check (spec §6)" above. Spec section 6 satisfied.
- **D3 clears by 0.75pp only.** 19.25% versus a 20% threshold on global cap. Provisional. If
  the trigger is ever evaluated per-cut, EM (59.41%), LatAm (49.65%) and EM_Eq (42.01%) fire
  immediately, so the global-only framing is doing a lot of work.
- **Sign instability is the sharper robustness finding.** Dropping the top-1 name flips the sign
  of the global cap slope on 17.15% of days, and its magnitude exceeds the full slope on 21.62%.
  This is a better argument for a robust-slope spec than the top1_share threshold is, and it
  should be the headline input if D3 is opened.
- **`fragile_flag` and `pct_ex_top1` are cap-only by construction** (`roro/attribution.py:446`).
  Every `eq` row therefore reads `fragile_flag=False` with `pct_ex_top1=NaN`, including the
  2026-05-26 global eq row where dropping Korea moves beta from +0.195 to -0.012 — a sign flip
  that the flag does not see. This matches the spec (the concentration alert is cap-only), but
  any downstream consumer that reads `fragile_flag` across weightings will silently
  under-report. Worth an explicit note in the artifact schema docs.
- **Fragility alert rate is too high to be actionable.** 27.74% of global cap days and 14,632
  of 15,749 concentration alerts come from the `fragile` trigger. As a daily alert this is
  noise; consider either a persistence requirement (n consecutive days) or promoting it to a
  standing metadata field rather than an alert. Feeds gate recalibration.
- **Global tercile label churn.** 410 global transitions over 4,540 business days (~22.8 per
  year), with a visible cluster of 8 transitions in the 31 business days around 2022-09-26.
  The Sep-2022 event matched a `Risk-off -> Transitional` transition on the day the gilt crisis
  peaked, which is directionally backwards and is a symptom of that churn. This should feed the
  gate-recalibration work alongside the fragility rate.
- **`attribution_rollup.csv` is last-date only**, so historic block-level attribution could not
  be used in the event review; own-block betas from `beta_series.csv` were substituted. If the
  event review is meant to be a recurring check, the rollup needs a history mode.
- **Small-n block betas can move implausibly.** EM_FI cap beta moved +1.09 in one day around the
  2015 devaluation with a top-1 share of only 0.191, and EM_Eq cap beta reads -2.28 in Sep-2022.
  Not wrong per se, but these are the kind of values that make block-level narratives fragile.
- **LatAm ex-top1 probe: closed.** 0.00% NaN `beta_ex_top1` on all 3,390 LatAm days, both
  weightings. No further action.
