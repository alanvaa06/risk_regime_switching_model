# JM Integration Contract — reference for `roro/regime_jm.py`

> Working reference compiled from a read-only audit of the HMM implementation. The JM mirrors this surface exactly. Cited by the JM spec + plan. Facts only — no design decisions (those are locked in the spec from the research brief).

## A. Module contract to mirror (`roro/regime_hmm.py`)

**`classify_hmm(bbs: BetaBySegment, *, cfg, thin_cuts: frozenset[str]) -> HmmRegimeFrame`**
- Iterates `bbs.by_segment.items()` → `(cut, bf)`. Beta series = `bf.cap_wtd["beta"]`; counts = `bf.cap_wtd["n"]`.
- `thin_cut_flag[cut] = pd.Series(cut in thin_cuts, index=beta.index)` (broadcast bool; thin cuts are NOT filtered, just flagged).
- Consumes `cfg.hmm_refit_interval_days` (21), `cfg.hmm_min_history_days` (252), `cfg.hmm_switching_variance` (True).

**`walk_forward(beta, *, refit_interval_days, min_history_days, switching_variance) -> dict[str, object]`** — the JM's `walk_forward` must return the **same dict keys**:
| Key | Type | Notes |
|---|---|---|
| `state` | `Series[float]` | 0/1/2 or NaN (cold/NaN-beta); float (np.where upcasts) |
| `label` | `Series[str]` | `Risk-off/Transitional/Risk-on`, `Unknown` for NaN |
| `prob_risk_off` | `Series[float]` | NaN where cold/NaN-beta |
| `prob_transitional` | `Series[float]` | |
| `prob_risk_on` | `Series[float]` | |
| `confidence` | `Series[float]` | argmax prob (discrete JM → 1.0 one-hot) |
| `cold_start` | `Series[bool]` | reindex `fill_value=True` (NaN-beta → True) |
| `refit_dates` | `list[pd.Timestamp]` | the day each new param block takes effect, converged refits only |

- Loop: `clean = beta.dropna()`; cursor `r` starts at `min_history_days`; while `r < n`: `block_end = min(r+refit_interval, n)`; fit on `clean.iloc[:r]` (expanding); if converged → `last_good=fit`, append refit date; infer over `clean.iloc[:block_end]`, write rows `[r:block_end]`, `cold[r:block_end]=False`; `r=block_end`. Rows `< min_history_days` stay NaN/cold.
- `_series(values) = pd.Series(values, index=clean.index).reindex(full_index)`; labels `.fillna("Unknown")`.

**State ordering / label-switching defense:** `perm = np.argsort(means)` ascending → `perm[0]`=lowest=Risk-off; re-applied every refit. JM: sort the 3 fitted centroids by mean.

**Convergence/degenerate:** fit failure → return sentinel with `converged=False`; caller falls back to `last_good`.

**DETERMINISM — the key divergence:** HMM has NO explicit seed (relies on statsmodels deterministic-given-data EM). **The JM MUST seed every stochastic step** (k-means++ init, restarts) from `cfg.jm_random_seed` — this is new work the HMM didn't need. Wrap fit in `warnings.catch_warnings()` like the HMM.

Constants: `_K_REGIMES=3`, `_ORDERED_LABELS=("Risk-off","Transitional","Risk-on")`, `_UNKNOWN="Unknown"`.

## B. Engine/IO/config/alerts/backtest wiring (each = minimal parallel of HMM)

| Touchpoint | File:line | JM parallel |
|---|---|---|
| Config fields | `config.py:35-39` | Add `jm_enabled=False`, `jm_*` fields. `to_dict()`/`load_config` are generic over `fields()` → **zero change** (new keys auto-allowed + auto-serialized to snapshot). |
| Classifier block | `engine.py:96-100` | `regime_jm = classify_jm(beta, cfg=cfg, thin_cuts=frozenset({"LatAm"})) if cfg.jm_enabled else None` |
| Alerts call | `engine.py:141-143` | add `regime_jm=regime_jm` kwarg |
| RunResult | `engine.py:158`, `types.py:120` | add `regime_jm: JmRegimeFrame | None = None` after `regime_hmm` |
| Type | `types.py:69-81` | new `JmRegimeFrame` (clone `HmmRegimeFrame`: state,label,prob_*,confidence,n_per_segment,thin_cut_flag,cold_start_flag,refit_dates) |
| AlertSet | `types.py:101-105` | add `jm_bucket_transitions` w/ same empty-DF default |
| detect_alerts | `alerts.py:12-28` | add `regime_jm` kwarg → `_bucket_transitions(regime_jm.label)` gated on not-None. `_bucket_transitions` is model-agnostic — reuse. |
| CSV writers | `io.py:131-133, 259-292` | `_write_regime_jm` → `regimes_jm.csv` (cols: `date,segment,state,label,p_risk_off,p_transitional,p_risk_on,confidence,cold_start,thin_cut`), `_write_jm_refit_log` → `jm_refit_log.csv`. Inside existing `.tmp/` atomic boundary. |
| snapshot.json | `io.py:315-328` | add `snapshot["regime_jm"]` last-row block |
| Backtest | `backtest.py:107-125` | parallel block: `_evaluate_gates(result, labels=regime_jm.label, transitions=jm_bucket_transitions)` → `acceptance_report_jm.json`; extend `acceptance_compare.json` to 3-way `{percentile,hmm,jm}` |

Note: G1/G2/G6 are method-shared (don't read labels) → identical across methods, as the gate-diagnostics memo documents.

## C. Report viz (3-way band toggle + JM probability figure)

CSV contract for `regimes_jm.csv` is **identical** to `regimes_hmm.csv` (`date,segment,label,p_risk_off,p_transitional,p_risk_on`).

| Touchpoint | File:line | JM parallel |
|---|---|---|
| Bundle | `report/bundle.py:29-33` | add 4 `seg_jm_*` optional fields |
| Load | `report/load.py:150-159` | parallel `if (run_dir/"regimes_jm.csv").exists():` pivot block → `DataBundle` |
| Prob figure | `report/figures.py:862-948` | clone `regime_probability_area` → `regime_probability_area_jm` (swap `seg_hmm_*`→`seg_jm_*`, title `"JM regime probabilities — {seg}"`); stacked-area, `REGIME_COLORS`, yaxis [0,1] |
| Band toggle | `report/figures.py:684-703` (`beta_band_lookup`) | add `"jm": _band_shapes(seg_jm_label[seg], smooth=False)` (raw runs, like HMM — JM is persistent) |
| Toggle UI | `report/html.py:53-55` | add `<option value="jm">Jump Model</option>` (third). `apply()` already does `byMethod[method]` → works. |
| Orchestrate | `report/orchestrate.py:48-52` | gate block: if `seg_jm_label is not None` append `FigureSpec(regime_probability_area_jm(bundle), "fig_jm_probs", "JM regime probabilities")`; build lookup w/ JM shapes |
| Tests | `tests/report/conftest.py:66-83` | `write_jm_csv()` + `test_build_report_includes_jm_when_present` |

Determinism in report: div_ids are hard-coded literals (`"fig_jm_probs"`), `run_date` from snapshot (no `datetime.now()`), band shapes use ISO date strings → byte-identical HTML test (`test_orchestrate.py`) must stay green.

Band colors (`figures.py:475`): Risk-off `rgba(220,50,47,0.55)`, Transitional `rgba(128,128,128,0.32)`, Risk-on `rgba(46,160,67,0.55)`.
