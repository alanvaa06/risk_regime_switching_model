"""build_report orchestrator tests."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_build_report_writes_html(
    tmp_path: Path, minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    from roro.report import build_report  # noqa: PLC0415

    out = tmp_path / "report.html"
    result_path = build_report(minimal_run_dir, tiny_xlsx, out, window=21)

    assert result_path == out
    assert out.exists()
    assert out.stat().st_size > 20_000  # CDN-loaded plotly.js; tighten when running full 252d
    html_text = out.read_text(encoding="utf-8")
    assert html_text.startswith("<!DOCTYPE html>")


def test_build_report_reproducible(
    tmp_path: Path, minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    from roro.report import build_report  # noqa: PLC0415

    out_a = tmp_path / "a.html"
    out_b = tmp_path / "b.html"

    build_report(minimal_run_dir, tiny_xlsx, out_a, window=21)
    build_report(minimal_run_dir, tiny_xlsx, out_b, window=21)

    assert out_a.read_bytes() == out_b.read_bytes()


def test_build_report_missing_run_dir_csv_raises(
    tmp_path: Path, minimal_run_dir: Path, tiny_xlsx: Path
) -> None:
    from roro.report import build_report  # noqa: PLC0415
    from roro.report.errors import ReportInputError  # noqa: PLC0415

    (minimal_run_dir / "beta_series.csv").unlink()
    out = tmp_path / "report.html"

    with pytest.raises(ReportInputError, match="beta_series.csv"):
        build_report(minimal_run_dir, tiny_xlsx, out, window=21)
