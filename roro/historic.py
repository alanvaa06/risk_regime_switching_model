"""Incremental historic runs: checkpoint discovery, run planning, orchestration.

Layout: <historic_root>/<config-stem>/results_<YYYY-MM-DD>/, where the date is the
last data date processed. A RESUME run copies closed HMM/JM refit blocks from the
newest checkpoint; everything else recomputes on the full data. Spec:
docs/superpowers/specs/2026-09-25-incremental-historic-runs-design.md
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

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

    Skips ``.tmp`` folders (interrupted writes) and folders without snapshot.json.
    """
    if not historic_dir.is_dir():
        return None
    found: list[tuple[pd.Timestamp, Path]] = []
    for p in historic_dir.iterdir():
        m = _RESULTS_DIR.match(p.name)
        if m and p.is_dir() and (p / "snapshot.json").is_file():
            found.append((pd.Timestamp(m.group(1)), p))
    if not found:
        return None
    last_date, run_dir = max(found)
    snapshot = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    return Checkpoint(run_dir=run_dir, last_date=last_date, snapshot=snapshot)


def config_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Sorted config keys whose value differs (paths and secrets ignored)."""
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
        return f"Cannot open {exc.filename}: close it in Excel (or any other program) and retry"
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {exc.filename}"
    return None
