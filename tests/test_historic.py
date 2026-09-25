"""Pure historic-run helpers: date parsing, checkpoint discovery, run planning."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from roro.historic import (
    Checkpoint,
    NoDataError,
    TypeRun,
    UpdateMode,
    UpdatePlan,
    available_configs,
    config_changes,
    find_checkpoint,
    friendly_error,
    parse_user_date,
    plan_update,
)


@pytest.mark.parametrize("text", ["2026-09-25", "25-09-2026", " 25-09-2026 "])
def test_parse_user_date_accepts_both_formats(text: str) -> None:
    assert parse_user_date(text) == pd.Timestamp("2026-09-25")


@pytest.mark.parametrize("text", ["09/25/2026", "2026-13-01", "31-02-2026", "25-9-2026", ""])
def test_parse_user_date_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="DD-MM-YYYY"):
        parse_user_date(text)


def test_available_configs_sorted_yaml_stems(tmp_path: Path) -> None:
    for name in ("b.yaml", "a.yaml", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert list(available_configs(tmp_path)) == ["a", "b"]


def _mk(root: Path, name: str, *, snapshot: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if snapshot:
        (d / "snapshot.json").write_text(json.dumps({"config_resolved": {}}), encoding="utf-8")
    return d


def test_find_checkpoint_newest_by_name_date(tmp_path: Path) -> None:
    older = _mk(tmp_path, "results_2026-09-18")
    newest = _mk(tmp_path, "results_2026-09-25")
    _mk(tmp_path, "results_2026-10-01.tmp")               # interrupted write
    _mk(tmp_path, "results_2026-10-02", snapshot=False)   # incomplete folder
    _mk(tmp_path, "results_2026-10-03.old")               # write_run's rename-aside leftover
    _mk(tmp_path, "scratch")
    (older / "touched.txt").write_text("x", encoding="utf-8")  # newer mtime must not matter
    cp = find_checkpoint(tmp_path)
    assert cp is not None
    assert cp.run_dir == newest
    assert cp.last_date == pd.Timestamp("2026-09-25")
    assert cp.snapshot == {"config_resolved": {}}


def test_find_checkpoint_missing_or_empty(tmp_path: Path) -> None:
    assert find_checkpoint(tmp_path / "nope") is None
    assert find_checkpoint(tmp_path) is None


def test_find_checkpoint_skips_calendar_invalid_date(tmp_path: Path) -> None:
    valid = _mk(tmp_path, "results_2026-09-18")
    _mk(tmp_path, "results_2026-13-45")  # matches the regex but is not a real date
    cp = find_checkpoint(tmp_path)
    assert cp is not None
    assert cp.run_dir == valid


@pytest.mark.parametrize("garbage", ["{not json", "[1, 2]", ""])
def test_find_checkpoint_skips_corrupt_snapshot(tmp_path: Path, garbage: str) -> None:
    older = _mk(tmp_path, "results_2026-09-18")
    newest = _mk(tmp_path, "results_2026-09-25")
    (newest / "snapshot.json").write_text(garbage, encoding="utf-8")
    cp = find_checkpoint(tmp_path)
    assert cp is not None
    assert cp.run_dir == older


def test_find_checkpoint_all_corrupt_is_none(tmp_path: Path) -> None:
    only = _mk(tmp_path, "results_2026-09-18")
    (only / "snapshot.json").write_text("{not json", encoding="utf-8")
    assert find_checkpoint(tmp_path) is None


def test_config_changes_ignores_paths_and_key() -> None:
    old = {"a": 1, "output_dir": "x", "data_path": "d1", "fred_api_key": None, "b": 2}
    new = {"a": 1, "output_dir": "y", "data_path": "d2", "fred_api_key": "k", "b": 3, "c": 0}
    assert config_changes(old, new) == ["b", "c"]


_CFG: dict[str, Any] = {"jm_enabled": True, "methodology_version": "1.0.0"}


def _cp(last: str = "2026-09-18", cfg: dict[str, Any] | None = None) -> Checkpoint:
    return Checkpoint(
        run_dir=Path(f"results_{last}"),
        last_date=pd.Timestamp(last),
        snapshot={"config_resolved": _CFG if cfg is None else cfg},
    )


def _plan(**over: Any) -> UpdatePlan:
    kw: dict[str, Any] = dict(
        type_run=TypeRun.NEW_DATA, checkpoint=_cp(), config_now=_CFG,
        data_last=pd.Timestamp("2026-09-25"), data_until=None,
    )
    kw.update(over)
    return plan_update(**kw)


def test_plan_all_forces_full() -> None:
    plan = _plan(type_run=TypeRun.ALL)
    assert plan.mode is UpdateMode.FULL
    assert "requested" in plan.reason


def test_plan_without_checkpoint_is_full() -> None:
    plan = _plan(checkpoint=None)
    assert plan.mode is UpdateMode.FULL
    assert "no checkpoint" in plan.reason


def test_plan_config_change_is_full_and_names_keys() -> None:
    plan = _plan(checkpoint=_cp(cfg={**_CFG, "methodology_version": "0.9.0"}))
    assert plan.mode is UpdateMode.FULL
    assert "methodology_version" in plan.reason


def test_plan_up_to_date() -> None:
    plan = _plan(checkpoint=_cp("2026-09-25"))
    assert plan.mode is UpdateMode.UP_TO_DATE
    assert "type_run='all'" not in plan.reason


def test_plan_up_to_date_hints_when_data_until_is_earlier() -> None:
    plan = _plan(data_last=pd.Timestamp("2026-09-10"), data_until=pd.Timestamp("2026-09-10"))
    assert plan.mode is UpdateMode.UP_TO_DATE
    assert "type_run='all' (or --full)" in plan.reason


def test_plan_resume() -> None:
    plan = _plan()
    assert plan.mode is UpdateMode.RESUME
    assert "results_2026-09-18" in plan.reason


def test_friendly_error_locked_file() -> None:
    msg = friendly_error(PermissionError(13, "Permission denied", "data.xlsx"))
    assert msg is not None
    assert "data.xlsx" in msg and "Excel" in msg


def test_friendly_error_missing_file_names_resolved_path() -> None:
    msg = friendly_error(FileNotFoundError(2, "No such file", "data.xlsx"))
    assert msg is not None
    assert str(Path("data.xlsx").resolve()) in msg
    assert "RoRo folder" in msg and "data_path" in msg


def test_friendly_error_missing_file_without_name() -> None:
    msg = friendly_error(FileNotFoundError("gone"))
    assert msg is not None
    assert "None" not in msg


def test_friendly_error_unknown_is_none() -> None:
    assert friendly_error(ValueError("boom")) is None


def test_friendly_error_no_filename_omits_none() -> None:
    msg = friendly_error(PermissionError("denied"))
    assert msg is not None
    assert "None" not in msg


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("proxy refused"),
        ConnectionError("reset by peer"),
        ConnectionRefusedError(10061, "refused"),
        TimeoutError("timed out"),
    ],
)
def test_friendly_error_network(exc: BaseException) -> None:
    msg = friendly_error(exc)
    assert msg is not None
    assert msg.startswith("Cannot reach FRED")


def test_friendly_error_no_data() -> None:
    exc = NoDataError("no data in data.xlsx on or before 2001-01-01")
    assert isinstance(exc, ValueError)  # callers catching ValueError still work
    assert friendly_error(exc) == "no data in data.xlsx on or before 2001-01-01"
