"""Top-level build_report orchestrator: run_dir + xlsx -> report.html."""
from __future__ import annotations

from pathlib import Path

from roro.report.figures import (
    beta_timeseries,
    scatter_beta_return,
    scatter_vol_return,
)
from roro.report.html import assemble
from roro.report.load import DEFAULT_WINDOW, load_bundle


def build_report(
    run_dir: Path,
    xlsx_path: Path,
    out_path: Path,
    *,
    window: int = DEFAULT_WINDOW,
) -> Path:
    """Build a single self-contained HTML report from an engine run dir.

    Args:
        run_dir: engine run directory (snapshot.json + CSVs).
        xlsx_path: source xlsx with Equity_LC + Fixed_Income_LC + Panel.
        out_path: destination HTML file.
        window: trailing business-day window exposed in the bundle.

    Returns:
        Path that was written (== out_path).

    Raises:
        ReportInputError: required inputs missing or invalid.
        FileNotFoundError: xlsx_path does not exist.
    """
    bundle = load_bundle(run_dir, xlsx_path, window=window)
    figures = [
        scatter_vol_return(bundle),
        scatter_beta_return(bundle),
        beta_timeseries(bundle),
    ]
    html = assemble(
        figures,
        run_date=bundle.run_date,
        methodology_version=bundle.methodology_version,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
