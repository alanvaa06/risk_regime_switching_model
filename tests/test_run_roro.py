"""run_roro.py parameter validation and dispatch (run_update is stubbed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from roro.historic import TypeRun, UpdateMode, UpdateOutcome

REPO = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_roro", REPO / "run_roro.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.chdir(tmp_path)  # main() chdirs to the repo; monkeypatch restores cwd
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    return _load()


def test_bad_config_lists_valid(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "CONFIG", "nope")
    assert runner.main() == 2
    out = capsys.readouterr().out
    assert "not found" in out and "default" in out


def test_bad_type_run(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                      capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "TYPE_RUN", "everything")
    assert runner.main() == 2
    assert "TYPE_RUN" in capsys.readouterr().out


def test_bad_date(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "DATA_UNTIL", "2026/09/25")
    assert runner.main() == 2
    assert "DD-MM-YYYY" in capsys.readouterr().out


def test_dispatches_to_run_update(runner: ModuleType,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run_update(config_path: Path, **kwargs: Any) -> UpdateOutcome:
        seen["config_path"] = config_path
        seen.update(kwargs)
        return UpdateOutcome(UpdateMode.UP_TO_DATE, "stub", None, 0, 0.0)

    monkeypatch.setattr(runner, "run_update", fake_run_update)
    monkeypatch.setattr(runner, "FredApiClient", lambda api_key: object())
    monkeypatch.setattr(runner, "CONFIG", "default")
    monkeypatch.setattr(runner, "TYPE_RUN", "all")
    monkeypatch.setattr(runner, "DATA_UNTIL", "25-09-2026")
    assert runner.main() == 0
    assert seen["config_path"] == REPO / "configs" / "default.yaml"
    assert seen["type_run"] is TypeRun.ALL
    assert str(seen["data_until"].date()) == "2026-09-25"


def test_known_error_prints_friendly_message(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def locked(config_path: Path, **kwargs: Any) -> UpdateOutcome:
        raise PermissionError(13, "Permission denied", "data.xlsx")

    monkeypatch.setattr(runner, "run_update", locked)
    monkeypatch.setattr(runner, "FredApiClient", lambda api_key: object())
    assert runner.main() == 1
    assert "close it in Excel" in capsys.readouterr().out
