"""CLI tests for `roro report`."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from roro.cli import main


def test_report_subcommand_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--help"])
    assert result.exit_code == 0
    assert "--run-dir" in result.output
    assert "--xlsx" in result.output
    assert "--out" in result.output


def test_report_runs_end_to_end(
    tmp_path: Path, minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    out = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            "--run-dir",
            str(minimal_run_dir),
            "--xlsx",
            str(tiny_xlsx),
            "--out",
            str(out),
            "--window",
            "21",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_report_defaults_xlsx_from_snapshot(
    tmp_path: Path, minimal_run_dir: Path
) -> None:
    # snapshot already records data_path = tiny_xlsx, so --xlsx omitted should still work
    out = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["report", "--run-dir", str(minimal_run_dir), "--out", str(out), "--window", "21"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_report_missing_xlsx_in_snapshot_errors(
    tmp_path: Path, minimal_run_dir: Path
) -> None:
    snap = minimal_run_dir / "snapshot.json"
    data = json.loads(snap.read_text(encoding="utf-8"))
    data["config_resolved"]["data_path"] = "/nonexistent/data.xlsx"
    snap.write_text(json.dumps(data), encoding="utf-8")

    out = tmp_path / "report.html"
    runner = CliRunner()
    result = runner.invoke(
        main, ["report", "--run-dir", str(minimal_run_dir), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "xlsx" in result.output.lower() or "data_path" in result.output.lower()
