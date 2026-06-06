"""build_report orchestrator tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.report.conftest import write_hmm_csv


def test_build_report_includes_hmm_when_present(
    minimal_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    from roro.report import build_report  # noqa: PLC0415

    write_hmm_csv(minimal_run_dir)
    out = build_report(minimal_run_dir, tiny_xlsx, tmp_path / "r.html", window=21)
    html = out.read_text(encoding="utf-8")
    assert "HMM regime probabilities" in html
    assert 'id="hmm-band-method"' in html
    assert "fig_hmm_probs" in html


def test_build_report_no_hmm_unchanged(
    minimal_run_dir: Path, tiny_xlsx: Path, tmp_path: Path
) -> None:
    from roro.report import build_report  # noqa: PLC0415

    out = build_report(minimal_run_dir, tiny_xlsx, tmp_path / "r.html", window=21)
    html = out.read_text(encoding="utf-8")
    assert "hmm-band-method" not in html
    assert "fig_hmm_probs" not in html


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
