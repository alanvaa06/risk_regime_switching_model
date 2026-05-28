"""End-to-end integration: golden 2024-Q1 run dir -> report.html."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from roro.report import build_report

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "2024-Q1"


@pytest.mark.skipif(not GOLDEN_DIR.exists(), reason="Golden 2024-Q1 fixture not present")
def test_build_report_from_golden_2024q1(tmp_path: Path, tiny_xlsx: Path) -> None:
    """Use the engine golden 2024-Q1 run dir + tiny xlsx fixture to build a real report."""
    # Stage golden CSVs + synthesize snapshot.json (golden intentionally excludes it).
    run_dir = tmp_path / "run"
    shutil.copytree(GOLDEN_DIR, run_dir)
    (run_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "run_date": "2024-03-31",
                "as_of_data_date": "2024-03-31",
                "methodology_version": "1.0.0",
                "config_resolved": {"data_path": str(tiny_xlsx)},
                "data_fingerprint": {},
                "code_version": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "report.html"
    build_report(run_dir, tiny_xlsx, out, window=21)

    assert out.exists()
    size_kb = out.stat().st_size / 1024
    assert size_kb > 50, f"Report too small: {size_kb:.1f} KB"
    assert size_kb < 5_000, f"Report unexpectedly large: {size_kb:.1f} KB"

    html_text = out.read_text(encoding="utf-8")
    assert html_text.startswith("<!DOCTYPE html>")
    # 3 figures rendered
    assert html_text.count('class="plotly-graph-div"') == 3
    # Section titles present
    assert "Risk vs Return" in html_text
    assert "Beta vs Return" in html_text
    assert "regime bands" in html_text
