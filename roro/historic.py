"""Incremental historic runs: checkpoint discovery, run planning, orchestration.

Layout: <historic_root>/<config-stem>/results_<YYYY-MM-DD>/, where the date is the
last data date processed. A RESUME run copies closed HMM/JM refit blocks from the
newest checkpoint; everything else recomputes on the full data. Spec:
docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from roro.config import EngineConfig, load_config
from roro.config import to_dict as config_to_dict
from roro.engine import run as engine_run
from roro.fred_client import FredClient
from roro.io import code_version, cut_prices, load_prices, read_resume_state
from roro.resume import HistoryRevisedError
from roro.types import ResumeState

DEFAULT_HISTORIC_ROOT: Path = Path("outputs") / "historic"

_RESULTS_DIR = re.compile(r"^results_(\d{4}-\d{2}-\d{2})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DMY_DATE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
#: Config keys that do not change results (paths, secrets): never force a FULL run.
_CONFIG_KEYS_IGNORED: frozenset[str] = frozenset({"output_dir", "data_path", "fred_api_key"})


class TypeRun(StrEnum):
    NEW_DATA = "new_data"
    ALL = "all"


class UpdateMode(StrEnum):
    FULL = "full"
    RESUME = "resume"
    UP_TO_DATE = "up_to_date"


@dataclass(frozen=True)
class Checkpoint:
    run_dir: Path
    last_date: pd.Timestamp
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class UpdatePlan:
    mode: UpdateMode
    reason: str


@dataclass(frozen=True)
class UpdateOutcome:
    mode: UpdateMode
    reason: str
    run_dir: Path | None
    dates_added: int
    elapsed_seconds: float


def parse_user_date(text: str) -> pd.Timestamp:
    """Parse DD-MM-YYYY or YYYY-MM-DD (the 4-digit year's position disambiguates)."""
    s = text.strip()
    message = f"Invalid date {text!r}: use DD-MM-YYYY (25-09-2026) or YYYY-MM-DD (2026-09-25)"
    if _ISO_DATE.match(s):
        fmt = "%Y-%m-%d"
    elif _DMY_DATE.match(s):
        fmt = "%d-%m-%Y"
    else:
        raise ValueError(message)
    try:
        return pd.Timestamp(datetime.strptime(s, fmt))
    except ValueError:
        raise ValueError(message) from None


def available_configs(configs_dir: Path) -> dict[str, Path]:
    """Config name (file stem) -> path, sorted by name."""
    return {p.stem: p for p in sorted(configs_dir.glob("*.yaml"))}


def find_checkpoint(historic_dir: Path) -> Checkpoint | None:
    """Newest complete ``results_YYYY-MM-DD`` folder, by the date in its name.

    Skips ``.tmp`` folders (interrupted writes), folders without snapshot.json, and
    folders whose date is calendar-invalid (e.g. ``results_2026-13-45``).
    """
    if not historic_dir.is_dir():
        return None
    found: list[tuple[pd.Timestamp, Path]] = []
    for p in historic_dir.iterdir():
        m = _RESULTS_DIR.match(p.name)
        if m and p.is_dir() and (p / "snapshot.json").is_file():
            try:
                ts = pd.Timestamp(m.group(1))
            except ValueError:
                continue
            found.append((ts, p))
    if not found:
        return None
    last_date, run_dir = max(found)
    snapshot = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    return Checkpoint(run_dir=run_dir, last_date=last_date, snapshot=snapshot)


def config_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Sorted config keys whose value differs (paths and secrets ignored).

    Precondition: both mappings must be JSON-round-tripped (as snapshot.json stores
    them); run_update passes ``_json_config(cfg)``.
    """
    keys = (set(old) | set(new)) - _CONFIG_KEYS_IGNORED
    return sorted(k for k in keys if old.get(k) != new.get(k))


def plan_update(
    *,
    type_run: TypeRun,
    checkpoint: Checkpoint | None,
    config_now: Mapping[str, Any],
    data_last: pd.Timestamp,
    data_until: pd.Timestamp | None,
) -> UpdatePlan:
    """Decide FULL / RESUME / UP_TO_DATE. First matching rule wins (spec section 6)."""
    if type_run is TypeRun.ALL:
        return UpdatePlan(UpdateMode.FULL, "full history requested")
    if checkpoint is None:
        return UpdatePlan(UpdateMode.FULL, "no checkpoint yet")
    changed = config_changes(checkpoint.snapshot.get("config_resolved", {}), config_now)
    if changed:
        return UpdatePlan(
            UpdateMode.FULL,
            f"config changed since {checkpoint.run_dir.name}: {', '.join(changed)}",
        )
    if checkpoint.last_date >= data_last:
        reason = f"already processed through {checkpoint.last_date.date()}"
        if data_until is not None and data_until < checkpoint.last_date:
            reason += "; use type_run='all' to rebuild as of an earlier date"
        return UpdatePlan(UpdateMode.UP_TO_DATE, reason)
    return UpdatePlan(UpdateMode.RESUME, f"resuming from {checkpoint.run_dir.name}")


def friendly_error(exc: BaseException) -> str | None:
    """Plain-English message for errors a non-developer can fix; None otherwise."""
    if isinstance(exc, PermissionError):
        name = exc.filename or "the file"
        return f"Cannot open {name}: close it in Excel (or any other program) and retry"
    if isinstance(exc, FileNotFoundError):
        where = Path(str(exc.filename)).resolve() if exc.filename else "the file"
        return f"File not found: {where} (run from the RoRo folder, or fix data_path in the config)"
    return None


def run_update(
    config_path: Path,
    *,
    type_run: TypeRun,
    data_until: pd.Timestamp | None,
    build_report: bool,
    fred_client: FredClient,
    historic_root: Path = DEFAULT_HISTORIC_ROOT,
    echo: Callable[[str], None] = print,
) -> UpdateOutcome:
    """Run one config into <historic_root>/<config-stem>/results_<last-data-date>/.

    RESUME reuses the newest checkpoint; a revised history falls back to FULL.
    UP_TO_DATE writes nothing, except a missing report.html for the checkpoint.
    All console output is ASCII.
    """
    started = time.perf_counter()
    name = config_path.stem
    historic_dir = historic_root / name
    cfg = load_config(config_path, overrides={"output_dir": historic_dir})
    dates = _data_dates(cfg.data_path, data_until)
    data_last = pd.Timestamp(dates.max())
    checkpoint = find_checkpoint(historic_dir)
    plan = plan_update(
        type_run=type_run,
        checkpoint=checkpoint,
        config_now=_json_config(cfg),
        data_last=data_last,
        data_until=data_until,
    )
    mode, reason = plan.mode, plan.reason
    if checkpoint is None and type_run is TypeRun.NEW_DATA:
        reason += f" in {historic_dir.resolve()}"
    echo(f"[{mode.value.upper()}] {name}: {reason}")
    if mode is UpdateMode.UP_TO_DATE:
        report_dir: Path | None = None
        # plan_update only returns UP_TO_DATE with a checkpoint; re-check for mypy.
        if build_report and checkpoint and not (checkpoint.run_dir / "report.html").exists():
            echo("report.html missing -> building it")
            _write_report(checkpoint.run_dir, cfg.data_path, checkpoint.last_date, echo)
            report_dir = checkpoint.run_dir
        return UpdateOutcome(mode, reason, report_dir, 0, time.perf_counter() - started)

    resume: ResumeState | None = None
    rejected: str | None = None
    if mode is UpdateMode.RESUME and checkpoint is not None:
        _warn_code_changed(checkpoint, echo)
        try:
            resume = read_resume_state(checkpoint.run_dir, checkpoint.last_date)
        except (FileNotFoundError, ValueError, KeyError) as exc:  # pandas errors are ValueError
            mode = UpdateMode.FULL
            reason = f"checkpoint {checkpoint.run_dir.name} unreadable ({exc}) -> full rerun"
            rejected = checkpoint.run_dir.name
            echo(f"[{mode.value.upper()}] {name}: {reason}")
    try:
        run_dir = _run_engine(cfg, fred_client, data_last, resume)
    except HistoryRevisedError as exc:
        mode = UpdateMode.FULL
        reason = f"history revised in {cfg.data_path.name} ({exc}) -> full rerun"
        resume = None
        rejected = checkpoint.run_dir.name if checkpoint is not None else None
        echo(f"[{mode.value.upper()}] {name}: {reason}")
        run_dir = _run_engine(cfg, fred_client, data_last, None)

    since = checkpoint.last_date if resume is not None and checkpoint is not None else None
    new_dates = dates[dates > since] if since is not None else dates
    _stamp_update(
        run_dir,
        {
            "mode": mode.value,
            "reason": reason,
            "checkpoint": checkpoint.run_dir.name if since is not None and checkpoint else None,
            "checkpoint_rejected": rejected,
            "dates_added": len(new_dates),
            "first_new_date": f"{new_dates.min():%Y-%m-%d}" if len(new_dates) else None,
        },
    )
    if build_report:
        echo("building report.html ...")
        _write_report(run_dir, cfg.data_path, data_last, echo)
    elapsed = time.perf_counter() - started
    what = "new dates" if mode is UpdateMode.RESUME else "dates processed"
    echo(f"[ok] {run_dir.resolve()} ({len(new_dates)} {what}, {elapsed:.0f}s)")
    return UpdateOutcome(mode, reason, run_dir, len(new_dates), elapsed)


def _warn_code_changed(checkpoint: Checkpoint, echo: Callable[[str], None]) -> None:
    """Warn (no behaviour change) when the checkpoint was written by other code."""
    old = str(checkpoint.snapshot.get("code_version", {}).get("git_sha", "unknown"))
    new = code_version()["git_sha"]
    if "unknown" in (old, new) or old == new:
        return
    echo(
        f"[warn] code changed since {checkpoint.run_dir.name} ({old[:7]} -> {new[:7]}); "
        "if regime math changed, rerun with type_run='all'"
    )


def _write_report(
    run_dir: Path, data_path: Path, data_until: pd.Timestamp, echo: Callable[[str], None]
) -> None:
    """report.html as of ``data_until``; on failure say where the results are, then re-raise."""
    # Lazy import: the report stack pulls plotly and is only needed here.
    from roro.report import build_report  # noqa: PLC0415

    try:
        build_report(run_dir, data_path, run_dir / "report.html", data_until=data_until)
    except Exception as exc:
        echo(
            f"[x] report failed: {exc}. Results are saved in {run_dir.resolve()}; "
            "the next run will retry the report."
        )
        raise


def _data_dates(data_path: Path, data_until: pd.Timestamp | None) -> pd.DatetimeIndex:
    prices = load_prices(data_path)
    if data_until is not None:
        prices = cut_prices(prices, data_until)
    dates = pd.DatetimeIndex(prices.equity_lc.index)
    if dates.empty:
        cut = f" on or before {data_until.date()}" if data_until is not None else ""
        raise ValueError(f"no data in {data_path}{cut}")
    return dates


def _json_config(cfg: EngineConfig) -> dict[str, Any]:
    """Config as snapshot.json stores it (JSON round trip), for like-for-like comparison."""
    loaded: dict[str, Any] = json.loads(json.dumps(config_to_dict(cfg), default=str))
    return loaded


def _run_engine(
    cfg: EngineConfig,
    fred_client: FredClient,
    data_last: pd.Timestamp,
    resume: ResumeState | None,
) -> Path:
    """One engine run into cfg.output_dir/results_<data_last>/ (atomic tmp + rename)."""
    stamp = f"{data_last:%Y-%m-%d}"
    engine_run(
        cfg,
        fred_client=fred_client,
        run_date=f"results_{stamp}",
        as_of_data_date=stamp,
        force=True,  # a same-named folder without snapshot.json is not a valid checkpoint
        data_until=stamp,
        resume=resume,
    )
    return cfg.output_dir / f"results_{stamp}"


def _stamp_update(run_dir: Path, info: dict[str, Any]) -> None:
    """Record how this folder was produced in snapshot.json["update"]."""
    path = run_dir / "snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["update"] = info
    tmp = path.with_name("snapshot.json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
