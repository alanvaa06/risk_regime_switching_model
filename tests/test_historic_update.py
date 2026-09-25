"""run_update end to end: RESUME output is byte-identical to a FULL rerun (AI-1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import roro.historic as historic_mod
import roro.regime_jm as jm_mod
import roro.report as report_mod
from roro import jump_model
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.historic import TypeRun, UpdateMode, UpdateOutcome, run_update
from tests.conftest import RW_DATES, RW_SEED, build_xlsx, random_walk_prices

T0 = pd.Timestamp("2023-06-15")  # checkpoint date, inside a JM refit block


def _fred() -> MockFredClient:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _config(tmp_path: Path, xlsx: Path, *, hmm: bool = False, jm: bool = True,
            penalty: float = 50.0) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"data_path: {xlsx.as_posix()}\n"
        "output_dir: ignored\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "percentile_window_years: 1\n"
        "min_n_per_cut: 2\n"
        "bootstrap_min_days: 10\n"
        f"hmm_enabled: {str(hmm).lower()}\n"
        "hmm_min_history_days: 250\n"
        "hmm_refit_interval_days: 250\n"
        f"jm_enabled: {str(jm).lower()}\n"
        "jm_continuous: true\n"
        f"jm_jump_penalty: {penalty}\n"
        "jm_min_history_days: 250\n"
        "jm_refit_interval_days: 120\n"
        "jm_n_init: 4\n",
        encoding="utf-8",
    )
    return path


def _update(cfg: Path, root: Path, type_run: TypeRun,
            until: pd.Timestamp | None = None, *, report: bool = False,
            log: list[str] | None = None) -> UpdateOutcome:
    echo = log.append if log is not None else (lambda _m: None)
    return run_update(cfg, type_run=type_run, data_until=until, build_report=report,
                      fred_client=_fred(), historic_root=root, echo=echo)


def _snapshot(run_dir: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    return loaded


def _assert_same_csvs(a: Path, b: Path) -> None:
    files_a = sorted(p.name for p in a.glob("*.csv"))
    files_b = sorted(p.name for p in b.glob("*.csv"))
    assert files_a == files_b
    for name in files_a:
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} differs"


def test_resume_equals_full_rerun(rw_xlsx: Path, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    first = _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    assert first.mode is UpdateMode.FULL
    assert first.run_dir is not None and first.run_dir.name == "results_2023-06-15"

    calls = {"n": 0}
    real = jump_model.fit_jump_model

    def counting(x: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(x, **kwargs)

    monkeypatch.setattr(jm_mod, "fit_jump_model", counting)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    resume_fits = calls["n"]
    calls["n"] = 0
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    full_fits = calls["n"]

    assert resumed.mode is UpdateMode.RESUME
    assert resumed.run_dir is not None and full.run_dir is not None
    assert resumed.run_dir.name == full.run_dir.name == "results_2024-12-31"
    _assert_same_csvs(resumed.run_dir, full.run_dir)
    assert resume_fits < full_fits  # the resume really skipped closed blocks
    assert resumed.dates_added == int((RW_DATES > T0).sum())

    update = _snapshot(resumed.run_dir)["update"]
    assert update["mode"] == "resume"
    assert update["checkpoint"] == "results_2023-06-15"
    assert update["dates_added"] == resumed.dates_added


def test_up_to_date_writes_nothing(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    _update(cfg, tmp_path / "a", TypeRun.ALL)
    before = sorted(p.name for p in (tmp_path / "a" / "cfg").iterdir())
    again = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    assert again.mode is UpdateMode.UP_TO_DATE
    assert again.run_dir == tmp_path / "a" / "cfg" / "results_2024-12-31"
    assert sorted(p.name for p in (tmp_path / "a" / "cfg").iterdir()) == before


def test_config_change_forces_full(rw_xlsx: Path, tmp_path: Path) -> None:
    _update(_config(tmp_path, rw_xlsx), tmp_path / "a", TypeRun.ALL, until=T0)
    out = _update(_config(tmp_path, rw_xlsx, penalty=30.0), tmp_path / "a", TypeRun.NEW_DATA)
    assert out.mode is UpdateMode.FULL
    assert "jm_jump_penalty" in out.reason


def test_config_change_warns_before_overwriting_same_date(rw_xlsx: Path,
                                                         tmp_path: Path) -> None:
    root = tmp_path / "h"
    log: list[str] = []
    _update(_config(tmp_path, rw_xlsx, jm=False), root, TypeRun.ALL, log=log)
    assert not any(m.startswith("[warn]") for m in log)  # --full never warns
    log.clear()
    out = _update(_config(tmp_path, rw_xlsx, jm=False, penalty=30.0), root,
                  TypeRun.NEW_DATA, log=log)
    assert out.mode is UpdateMode.FULL
    assert "[warn] overwriting results_2024-12-31 (config changed since it was written)" in log


def test_restated_last_row_still_resumes(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    eq.loc[[T0], "Brazil"] *= 1.01  # the checkpoint's last close was provisional
    build_xlsx(rw_xlsx, eq, fi)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    assert resumed.mode is UpdateMode.RESUME, resumed.reason
    assert resumed.run_dir is not None and full.run_dir is not None
    _assert_same_csvs(resumed.run_dir, full.run_dir)


def test_revised_history_without_overlays_resumes_exactly(rw_xlsx: Path,
                                                          tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx, jm=False)  # no HMM/JM: nothing is copied
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    eq.iloc[300:301, 1] *= 1.05  # old revision: harmless when nothing is reused
    build_xlsx(rw_xlsx, eq, fi)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    assert resumed.mode is UpdateMode.RESUME, resumed.reason
    assert resumed.run_dir is not None and full.run_dir is not None
    _assert_same_csvs(resumed.run_dir, full.run_dir)


def test_revised_history_falls_back_to_full(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx)
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    eq.iloc[300:301, 1] *= 1.05  # vendor revises one old Brazil equity print
    build_xlsx(rw_xlsx, eq, fi)
    out = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    assert out.mode is UpdateMode.FULL
    assert "history revised" in out.reason
    assert out.run_dir is not None
    assert out.dates_added == len(RW_DATES)
    update = _snapshot(out.run_dir)["update"]
    assert update["mode"] == "full"
    assert update["checkpoint"] is None
    assert update["checkpoint_rejected"] == "results_2023-06-15"


def test_missing_report_is_rebuilt_and_failure_is_reported(
    rw_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, rw_xlsx, jm=False)  # fast: no HMM/JM
    root = tmp_path / "h"
    log: list[str] = []
    first = _update(cfg, root, TypeRun.NEW_DATA, until=T0, log=log)
    assert first.mode is UpdateMode.FULL and first.run_dir is not None
    assert f"no checkpoint yet in {(root / 'cfg').resolve()}" in log[0]
    assert f"[ok] {first.run_dir.resolve()} (" in log[-1]
    assert "dates processed" in log[-1]

    def failing(*_a: Any, **_k: Any) -> Path:
        raise RuntimeError("plotly exploded")

    monkeypatch.setattr(report_mod, "build_report", failing)
    log.clear()
    with pytest.raises(RuntimeError, match="plotly exploded"):
        _update(cfg, root, TypeRun.NEW_DATA, until=T0, report=True, log=log)
    assert "report.html missing -> building it" in log
    assert log[-1].startswith("[x] report failed: plotly exploded. Results are saved in ")
    assert str(first.run_dir.resolve()) in log[-1]
    assert not (first.run_dir / "report.html").exists()

    seen: dict[str, Any] = {}

    def stub(run_dir: Path, xlsx: Path, out: Path, **kw: Any) -> Path:
        seen.update(run_dir=run_dir, **kw)
        out.write_text("<html></html>", encoding="utf-8")
        return out

    monkeypatch.setattr(report_mod, "build_report", stub)
    retry = _update(cfg, root, TypeRun.NEW_DATA, until=T0, report=True)
    assert retry.mode is UpdateMode.UP_TO_DATE
    assert retry.run_dir == first.run_dir
    assert retry.dates_added == 0
    assert (first.run_dir / "report.html").is_file()
    assert seen == {"run_dir": first.run_dir, "data_until": T0}


def test_resume_warns_when_code_changed(
    rw_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, rw_xlsx, jm=False)
    first = _update(cfg, tmp_path / "h", TypeRun.ALL, until=T0)
    assert first.run_dir is not None
    snap = _snapshot(first.run_dir)
    snap["code_version"] = {"git_sha": "a" * 40, "dirty": "false"}
    (first.run_dir / "snapshot.json").write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setattr(historic_mod, "code_version",
                        lambda: {"git_sha": "b" * 40, "dirty": "false"})
    log: list[str] = []
    out = _update(cfg, tmp_path / "h", TypeRun.NEW_DATA, log=log)
    assert out.mode is UpdateMode.RESUME
    assert ("[warn] code changed since results_2023-06-15 (aaaaaaa -> bbbbbbb); "
            "if regime math changed, rerun with type_run='all' (or --full)") in log
    assert "new dates" in log[-1]


def test_unreadable_checkpoint_falls_back_to_full(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx, jm=False)
    first = _update(cfg, tmp_path / "h", TypeRun.ALL, until=T0)
    assert first.run_dir is not None
    (first.run_dir / "beta_series.csv").unlink()
    log: list[str] = []
    out = _update(cfg, tmp_path / "h", TypeRun.NEW_DATA, log=log)
    assert out.mode is UpdateMode.FULL
    assert out.reason.startswith("checkpoint results_2023-06-15 unreadable (")
    assert out.reason.endswith(") -> full rerun")
    assert any(m.startswith("[FULL] cfg: checkpoint results_2023-06-15 unreadable") for m in log)
    assert out.run_dir is not None
    update = _snapshot(out.run_dir)["update"]
    assert update["checkpoint"] is None
    assert update["checkpoint_rejected"] == "results_2023-06-15"


@pytest.mark.slow
def test_hmm_resume_equals_full_rerun(rw_xlsx: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, rw_xlsx, hmm=True, jm=False)
    _update(cfg, tmp_path / "a", TypeRun.ALL, until=T0)
    resumed = _update(cfg, tmp_path / "a", TypeRun.NEW_DATA)
    full = _update(cfg, tmp_path / "b", TypeRun.ALL)
    assert resumed.mode is UpdateMode.RESUME
    assert resumed.run_dir is not None and full.run_dir is not None
    _assert_same_csvs(resumed.run_dir, full.run_dir)
