---
title: "RoRo — Incremental Historic Runs (checkpoint resume + one-click runner)"
owner: Alan Vazquez, CFA
status: Design approved 2026-09-25 (brainstorming Q1-Q7, approach A, sections 1-3) — ready for implementation planning; NO code written yet
date: 2026-09-25
target repo: C:\Proyectos\RoRo
drop-in location: docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md
depends on: RoRo v1.1 (percentile default, HMM/JM overlays, regime attribution)
---

# RoRo — Incremental Historic Runs Spec

## 1. Goal

A full run with percentile + HMM + JM takes **about 1 hour** on the author's machine, and longer on colleagues' work PCs. Every refresh of `data.xlsx` currently reprocesses 2008 → today from scratch.

Deliverables:

1. **Centralized history**: every run lands in `outputs/historic/<config-name>/results_<last-data-date>/`.
2. **Incremental update**: the engine detects the last processed date from the newest checkpoint, then processes only the dates not yet covered. The new folder holds the **complete** history (old rows + new rows), and it is **identical to what a full rerun would produce**.
3. **CLI**: `roro update`.
4. **One-click runner** for colleagues: `run_roro.py` (editable params block) + `run_roro.bat` (double-click, uses the project `.venv`).

Non-goals: Excel output (CSV only; colleagues convert manually), automatic deletion of old folders, running several configs per click, persisting fitted model objects.

## 2. Where the hour goes (and why resume is exact)

Verified in code:

- `roro/regime_hmm.py::walk_forward` and `roro/regime_jm.py::walk_forward` cut the cleaned beta series into blocks of `refit_interval_days` valid days, starting at `min_history_days`. At each block start `r` they fit on `beta[:r]` (expanding window), then infer the rows `[r, block_end)` causally with frozen params.
- The full 2008-2026 run ≈ 70 blocks × 10 segments ≈ 700 fits per overlay, i.e. ≈ 1,400 fits for HMM + JM. In addition, each block's inference re-filters from index 0 to `block_end` (`regime_hmm.py` `_filtered_probs(clean.iloc[:block_end])`, `regime_jm.py` `inf_lo = 0` when expanding), so total cost grows roughly with the square of history length.
- **Causality property:** row `t`'s output depends only on `beta[:block_end(t)]` and on fits over `beta[:r]` with `r <= t`. Appending new dates never changes rows in closed blocks. Rows in the open (last, partial) block also keep their values, because the block's fit is re-estimated on the same `beta[:r_open]` and inference is causal. Both fits are deterministic: statsmodels EM runs from fixed starting values, and the JM uses a seeded `SeedSequence`.

**Assumption (to be confirmed by the real-data bench, §8 test 7):** HMM/JM dominate the runtime. Everything else (returns, EWMA vol, regression, percentile, correlation, attribution, validation, tripwire, alerts) was measured at ~63 s total including FRED (`docs/context/results.md`, 2026-09-05).

## 3. Approach (A — resume only the slow step)

- **Cheap parts** are recomputed on the full data on every run. That keeps them exact and adds no new logic.
- **HMM/JM** resume from the checkpoint: rows before the open block are copied, and the loop restarts at the open block.
- Rejected: **B** incremental-everywhere. It touches ~8 modules, and EWMA's infinite memory means warmup-truncated values differ in the last decimals, which breaks the determinism invariant, all to save ~1 min. Also rejected: **C** pickled model params. They break across statsmodels/numpy versions, and re-fitting from known data is already cheap and exact.

## 4. Layout

```
outputs/historic/
  default/results_2026-09-25/      <- all run CSVs + snapshot.json + report.html
  eval/results_2026-09-25/
  jm-eval/results_2026-09-25/
  jm-only/...
  attribution-history/...
```

- Folder name = config file stem (`configs/jm-eval.yaml` → `jm-eval`).
- `results_<YYYY-MM-DD>` = **last data date processed** (after `data_until` cut), not the wall-clock run date. The run timestamp lives in `snapshot.json`.
- `run_update()` ignores the config's `output_dir`; `roro run` keeps its current behaviour unchanged.
- All folders are kept. No automatic deletion.
- Pre-existing folders (`outputs/jm_run/`, `outputs/jm_only/`, …) are **not** used as checkpoints: they predate attribution and use a different layout. The first `update` per config is FULL.

## 5. Components

| Unit | Responsibility | Interface |
|---|---|---|
| `roro/historic.py` (new) | Checkpoint discovery, run planning, orchestration | `find_checkpoint(historic_dir) -> Checkpoint \| None`; `plan_update(...) -> UpdatePlan` (pure); `run_update(config_path, *, type_run, data_until, build_report, fred_client) -> UpdateOutcome` |
| `roro/regime_hmm.py`, `roro/regime_jm.py` | Optional resume of `walk_forward` / `classify_*` | new kwarg `prior: PriorRegime \| None = None`; `None` = today's code path, byte-identical |
| `roro/engine.py` | Data cut + resume plumbing + revision check | `run(..., data_until: str \| None = None, resume: ResumeState \| None = None)`; raises `HistoryRevisedError` |
| `roro/io.py` | Read checkpoint state | `read_resume_state(run_dir) -> ResumeState` (beta_series, HMM/JM regime rows, refit logs, snapshot) |
| `roro/types.py` | New frozen dataclasses | `Checkpoint`, `UpdatePlan`, `UpdateMode` (Enum: FULL, RESUME, UP_TO_DATE), `ResumeState`, `PriorRegime`, `UpdateOutcome` |
| `roro/cli.py` | New subcommand | `roro update --config PATH [--full] [--data-until DATE] [--no-report] [--fred-key KEY]` |
| `run_roro.py` (repo root) | Colleague entry point | params block → `run_update()` |
| `run_roro.bat` (repo root) | Double-click launcher | `"%~dp0.venv\Scripts\python.exe" "%~dp0run_roro.py"` then `pause` |

The CLI and the script both call `run_update()`, so they cannot drift apart.

## 6. Data flow

1. Resolve config → `name = config_path.stem`, `historic_dir = outputs/historic/<name>`.
2. `find_checkpoint(historic_dir)`: newest `results_YYYY-MM-DD` folder **by the date in its name**, which contains `snapshot.json`. Ignores `*.tmp` folders and non-matching names.
3. Load prices and cut at `data_until` if given → `data_last` = last date available.
4. `plan_update()` evaluates, first match wins:

   | # | Condition | Mode | Reason printed |
   |---|---|---|---|
   | 1 | `type_run == "all"` | FULL | "full history requested" |
   | 2 | no checkpoint | FULL | "no checkpoint in outputs/historic/<name>" |
   | 3 | checkpoint `config_resolved` ≠ current config (ignoring `output_dir`), incl. `methodology_version` | FULL | "config changed since <checkpoint>: <keys>" |
   | 4 | checkpoint date ≥ `data_last` | UP_TO_DATE | "already processed through <date>" (+ hint to use `type_run='all'` when `data_until` < checkpoint date) |
   | 5 | otherwise | RESUME | "resuming from <checkpoint>, N new dates" |

5. UP_TO_DATE → print and exit. Nothing is recomputed (a missing `report.html` is rebuilt); the outcome's `run_dir` is the checkpoint folder.
6. RESUME → `read_resume_state(checkpoint)` → `engine.run(..., resume=state)`.
   - After betas are computed, compare them with the checkpoint's `beta_series.csv` on all dates ≤ the **last copied date**: same date set, same (segment, scheme) keys, `|Δ| <= 1e-12`, NaN == NaN. On mismatch, raise `HistoryRevisedError`. `run_update` catches it, prints "history revised in data.xlsx (first mismatch <date> <segment>) -> full rerun", and reruns FULL.
   - Last copied date = the MAX, over enabled overlays (HMM/JM) and segments with a prior, of `clean.index[r_open - 1]` (`copied_through`, §7). Rows from each open block onward are recomputed, so later dates may differ: a restated last checkpoint row (provisional close) still resumes. MAX, not MIN, because one comparison date covers every segment and must reach the longest copied range. Nothing copied (no overlay enabled or reusable) → no comparison: RESUME is then a full recompute.
7. Write via the existing `write_run(out_dir=historic_dir, run_date=f"results_{data_last}", force=True)`. It already builds `<name>.tmp`, removes stale tmp dirs, and renames atomically. `force=True` is safe: a same-named folder is overwritten only when it is not the checkpoint being resumed — it has no or a corrupt `snapshot.json` (skipped by `find_checkpoint`), the config changed since it was written (FULL), or `type_run='all'` asked for it. Outside `type_run='all'`, `run_update` first prints `[warn] overwriting existing results_<date>`.
8. `snapshot.json` gets a new `update` block: `{mode, reason, checkpoint, dates_added, first_new_date}`.
9. If `build_report`: `build_report(run_dir, data_path, run_dir / "report.html")`.
10. Console summary (ASCII only): mode, reason, dates added, elapsed seconds, output path.

## 7. HMM/JM resume mechanics (per segment)

Inputs: current cleaned beta `clean` (length `n`), prior rows for this segment (state, label, probs, confidence, cold_start) up to checkpoint date, prior refit log (converged refit dates).

1. `n_old` = number of `clean` rows with date ≤ checkpoint date. `n_old == n` cannot happen in RESUME mode.
2. Block starts: `r_k = min_history + k * refit_interval` for `r_k < n`. `r_open` = the block start whose block contains position `n_old` (the first new row). If `n_old < min_history`, there is nothing to reuse → run today's loop unchanged.
3. Rows `[0, r_open)` are copied from the prior. Refit dates `<= clean.index[r_open - 1]` (the last copied row; none if `r_open == 0`) are copied from the prior log. Not `< date(r_open)`: a checkpoint row after the last copied one may now be NaN or gone (e.g. a restated last row that opened a block), and its refit date would be copied although a full rerun never produces it.
4. Run today's loop starting at `r = r_open`. The fit at `r_open` uses `clean[:r_open]` → same params as the original run.
5. **Fallback:** today's code uses `last_good` when a fit fails to converge. In resume, `last_good` is initialised lazily. Only if the fit at `r_open` does not converge do we re-fit at the latest converged refit date `< date(r_open)` from the prior log (none → `None`, same as the original run).
6. **Contract:** `walk_forward(beta, prior=P) == walk_forward(beta)` exactly (all output series and `refit_dates`), whenever `beta` on dates ≤ `clean.index[r_open - 1]` equals the prior's beta (the copied rows and every reused fit, including the fallback, use only those dates).

Typical weekly run: ~5 new days → 1 fit per segment per overlay (~20 fits vs ~1,400).

## 8. Error handling

| Failure | Behaviour |
|---|---|
| Missing `FRED_API_KEY` | "Add FRED_API_KEY to .env (see .env.example)" |
| `data.xlsx` locked by Excel (`PermissionError`) | "Close data.xlsx in Excel and retry" |
| Invalid `CONFIG` | Lists the `configs/*.yaml` stems |
| Invalid `TYPE_RUN` | "TYPE_RUN must be 'new_data' or 'all'" |
| Invalid `DATA_UNTIL` | Accepts only `DD-MM-YYYY` or `YYYY-MM-DD` (position of the 4-digit year disambiguates); otherwise shows both examples |
| Crash / Ctrl-C mid-run | `.tmp` left behind; the checkpoint is untouched; the next run's `write_run` removes the stale tmp |
| History revised | Not an error: auto FULL with reason (§6 step 6) |
| Anything else | Full traceback printed; `.bat` pauses so the window stays open |

## 9. One-click runner

`run_roro.py` params block:

```python
CONFIG       = "jm-eval"    # default | eval | jm-only | jm-eval | attribution-history
TYPE_RUN     = "new_data"   # "new_data" = only unprocessed dates | "all" = full history
DATA_UNTIL   = None         # None = latest in data.xlsx; or "25-09-2026" / "2026-09-25"
BUILD_REPORT = True
```

- Resolves `configs/<CONFIG>.yaml` relative to the script file (works from any cwd).
- Loads `.env` for `FRED_API_KEY`.
- All console output is ASCII only (Windows cp1252 rule).

## 10. Testing (TDD)

1. **Resume == full**, for HMM, JM discrete and JM continuous, on synthetic beta. Split point inside a block, exactly on a block boundary, before `min_history`, and a forced non-converged fit at `r_open` (fallback path). Hypothesis property over split points.
2. `plan_update()`: one test per row of the §6 table, plus the `data_until < checkpoint` hint.
3. `find_checkpoint()`: newest by name date (not mtime); ignores `.tmp` and folders without `snapshot.json`; `None` when empty/missing.
4. **Engine-level equality:** FULL to T0 → RESUME to T1 vs FULL to T1, using `MockFredClient`, with HMM + JM enabled. Every CSV byte-identical (`snapshot.json` excluded: it holds timestamps). Note: the existing `tiny_xlsx` fixture has 4 countries, below `min_n_per_cut=10`, so every beta would be suppressed and HMM/JM never exercised. The test must use a fixture (new or overridden config) with non-suppressed betas and at least `min_history + 2` refit blocks of history, with T0 inside a block.
5. Revision: tamper one old price → `HistoryRevisedError` → `run_update` falls back to FULL, reason recorded.
6. `roro update` CLI smoke test. Existing `roro run` goldens unchanged (proves `prior=None` / `resume=None` / `data_until=None` are no-ops).
7. **Real-data bench** (`@pytest.mark.slow` or a manual script): time FULL vs RESUME on `configs/jm-eval.yaml`, and record it in `docs/context/results.md`. This confirms or refutes the §2 assumption.

`mypy --strict` + `ruff` clean; full non-slow suite green.

## 11. Docs

README usage section (`roro update`, `run_roro.bat`, historic layout); `docs/context/todo.md`, `results.md`, `memory.md`, `sesion-log.md`.

## 12. Acceptance

- AI-1: a RESUME output folder is byte-identical (all CSVs) to a FULL run over the same data.
- AI-2: `roro run` goldens unchanged.
- AI-3: UP_TO_DATE writes nothing.
- AI-4: revised history forces FULL with the reason printed and recorded in the snapshot.
- AI-5: a colleague can double-click `run_roro.bat` on a machine with `.venv` + `.env` and get `outputs/historic/<config>/results_<date>/report.html`.
- AI-6: real-data RESUME on `jm-eval` is materially faster than FULL (number recorded, not a hard threshold).
