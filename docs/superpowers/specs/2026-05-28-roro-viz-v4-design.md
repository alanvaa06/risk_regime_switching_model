# RoRo Visualization Layer v4 — Regime Band Visibility

**Version:** 4.0.0
**Date:** 2026-05-28
**Status:** Draft (pending implementation)
**Scope:** Make the β time-series regime bands legible. Two changes, both confined to `roro/report/figures.py`. Cosmetic only — the engine's `regimes.csv` and the headline tercile label are untouched.

---

## 1. Goal

The β time-series chart shades its background by tercile regime, but the bands are nearly invisible. Two compounding causes:

1. **Opacity too low.** `REGIME_COLORS` uses alpha 0.06 (Transitional) / 0.12 (Risk-off, Risk-on). On a white background that tint is barely perceptible.
2. **Daily tercile flicker.** Bands are drawn as one `vrect` per consecutive run of identical tercile label. The headline tercile flips day-to-day, producing hundreds of 1–3 day stripes that interleave red/grey/green into a faint wash rather than broad regime blocks.

This iteration (A) raises the band opacity and (B) debounces the daily tercile with a hysteresis filter before banding, so regimes render as broad blocks.

---

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Smoothing method | Hysteresis / N-day confirm | A band switch is shown only after the new label is confirmed; most "regime-like", crisp block edges, no future peeking |
| Confirm window `N` | 21 business days (~1 month) | Month-scale blocks without erasing genuine 1–2 month swings |
| Opacity (Risk-off, Risk-on) | 0.28 | Clearly visible, still a background tint |
| Opacity (Transitional) | 0.15 | Visible grey without drowning the line |
| Semantics | **Cosmetic only** — shading of the β chart | Engine `regimes.csv`, headline tercile, classifier output all unchanged |
| New dependencies | None | — |

---

## 3. Changes — `roro/report/figures.py`

### 3.1 Opacity (change A)

Replace the existing `REGIME_COLORS`:

```python
REGIME_COLORS: dict[str, str] = {
    "Risk-off": "rgba(220, 50, 47, 0.28)",
    "Transitional": "rgba(128, 128, 128, 0.15)",
    "Risk-on": "rgba(46, 160, 67, 0.28)",
}
```

(Was 0.12 / 0.06 / 0.12.)

### 3.2 Hysteresis smoothing helper (change B)

Add a module constant and a new helper near `_regime_runs`:

```python
_REGIME_CONFIRM_DAYS: int = 21


def _smooth_regime_hysteresis(labels: pd.Series, n: int) -> pd.Series:
    """Debounce a daily regime-label series for background shading.

    The "confirmed" band label switches to a new value only after that value
    has persisted for `n` consecutive observations. Flicker shorter than `n`
    is absorbed into the prevailing regime. NaN values are dropped first; an
    empty or all-NaN input returns an empty Series.

    This affects ONLY the chart's background bands — it does not alter the
    engine's regime labels.
    """
    s = labels.dropna()
    if s.empty:
        return s
    confirmed = s.iloc[0]
    candidate = confirmed
    streak = 0
    out: list[str] = []
    for label in s:
        if label == confirmed:
            candidate, streak = confirmed, 0
        elif label == candidate:
            streak += 1
            if streak >= n:
                confirmed, streak = candidate, 0
        else:
            candidate, streak = label, 1
        out.append(str(confirmed))
    return pd.Series(out, index=s.index)
```

Algorithm notes:
- The first observation seeds `confirmed` immediately (no warm-up window).
- A run of fewer than `n` days that differs from `confirmed` never commits — it is absorbed.
- A reversion to `confirmed` mid-candidate resets the candidate streak (genuine flicker rejection).

### 3.3 Wire smoothing into `beta_timeseries`

Both the initial-shapes block and each dropdown button's shapes currently iterate `_regime_runs(bundle.seg_tercile[seg])`. Change both call-sites to smooth first:

```python
# initial shapes
runs = _regime_runs(
    _smooth_regime_hysteresis(bundle.seg_tercile[default_segment], _REGIME_CONFIRM_DAYS)
)
for start, end, label in runs:
    ...

# per-button shapes
runs = _regime_runs(
    _smooth_regime_hysteresis(bundle.seg_tercile[seg], _REGIME_CONFIRM_DAYS)
)
for start, end, label in runs:
    ...
```

No other change to `beta_timeseries` (line trace, layout, dropdown wiring unchanged).

---

## 4. Tests — `tests/report/test_figures.py`

**Hysteresis helper:**

- `test_smooth_regime_hysteresis_absorbs_short_blip` — a 5-day "Risk-on" blip embedded in a long "Risk-off" run (with `n=21`) yields all "Risk-off".
- `test_smooth_regime_hysteresis_commits_sustained_switch` — `["Risk-off"]*30 + ["Risk-on"]*25` with `n=21` ends in "Risk-on". The band flips on the 21st consecutive "Risk-on" day; since there are 25 "Risk-on" inputs, the smoothed series has exactly 5 trailing "Risk-on" entries (the first 20 "Risk-on" inputs are still shaded "Risk-off" while the switch is unconfirmed).
- `test_smooth_regime_hysteresis_empty_returns_empty` — empty input → empty.
- `test_smooth_regime_hysteresis_all_nan_returns_empty` — all-NaN input → empty.

**Band rendering:**

- `test_regime_colors_opacity_raised` — `REGIME_COLORS["Risk-off"]` and `["Risk-on"]` contain `"0.28"`, `["Transitional"]` contains `"0.15"`.
- `test_beta_timeseries_smoothing_reduces_rect_count` — build a hand-crafted bundle whose `seg_tercile["global"]` alternates label every few days (heavy flicker). Assert the number of `rect` shapes in `beta_timeseries(bundle)` is **strictly fewer** than the number of runs from raw `_regime_runs` on the same unsmoothed series (smoothing collapses slivers). The hand-crafted bundle is required because the shared `tiny_xlsx`-backed fixture has a constant tercile (all "Transitional") and would produce a single run either way.
- The existing `test_beta_timeseries_has_shading_shapes` continues to pass (bands still present).

Reproducibility is unaffected (the helper is deterministic), so `test_build_report_reproducible` continues to pass.

---

## 5. Out of scope

- Changing the engine's regime classifier or `regimes.csv`.
- Making `N` or the opacities configurable via CLI/config (module constants suffice).
- Smoothing the scatter figures (no bands there).
- Regenerating the README PNG assets (kaleido deadlocks on the author's Windows box; the chart image is refreshed opportunistically where kaleido works, or by saving a fresh screenshot).

---

## 6. Acceptance criteria

1. The β time-series shows broad, clearly-tinted regime blocks instead of a faint shredded wash.
2. A regime block appears only after its label has persisted ≥ 21 business days.
3. Engine output (`regimes.csv`, headline tercile) is byte-identical to before this change.
4. All prior tests pass plus the new hysteresis + band tests.
5. Two consecutive `roro report` invocations on identical inputs still produce byte-identical HTML.
6. mypy --strict + ruff stay clean.
