"""Thin Click CLI: `roro run`, `roro backtest`, `roro report`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from roro.config import load_config
from roro.engine import run as engine_run
from roro.fred_client import FredApiClient, FredClient

load_dotenv()


def _build_fred_client(api_key: str | None) -> FredClient:
    if not api_key:
        raise click.UsageError(
            "FRED_API_KEY env var (or --fred-key) is required for `roro run`."
        )
    return FredApiClient(api_key=api_key)


@click.group()
def main() -> None:
    """RoRo Risk-Regime engine CLI."""


@main.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--date", "run_date", required=True)
@click.option("--as-of-data-date", required=True)
@click.option("--ewma-halflife", type=int, default=None)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
@click.option("--force", is_flag=True)
def cmd_run(
    config_path: Path,
    run_date: str,
    as_of_data_date: str,
    ewma_halflife: int | None,
    out_dir: Path | None,
    fred_key: str | None,
    force: bool,
) -> None:
    overrides: dict[str, Any] = {}
    if ewma_halflife is not None:
        overrides["ewma_halflife_days"] = ewma_halflife
    if out_dir is not None:
        overrides["output_dir"] = out_dir
    cfg = load_config(config_path, overrides=overrides)
    api_key = fred_key or os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    engine_run(
        cfg,
        fred_client=client,
        run_date=run_date,
        as_of_data_date=as_of_data_date,
        force=force,
    )
    click.echo(f"OK: {cfg.output_dir / run_date}")


@main.command("backtest")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--assert-gates", is_flag=True)
def cmd_backtest(config_path: Path, start: str, end: str, assert_gates: bool) -> None:
    # Lazy import: backtest module is heavy and only needed for this command.
    from roro.backtest import run_backtest  # noqa: PLC0415

    cfg = load_config(config_path)
    api_key = os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    report = run_backtest(cfg, fred_client=client, start=start, end=end)
    if assert_gates and not report["all_passed"]:
        raise click.ClickException(
            "Acceptance gates failed; see backtest/acceptance_report.json"
        )
    click.echo("OK")


@main.command("report")
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--xlsx",
    "xlsx_path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Source xlsx. Defaults to snapshot.json's config_resolved.data_path.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output HTML path. Defaults to <run-dir>/report.html.",
)
@click.option("--window", type=int, default=252)
def cmd_report(
    run_dir: Path,
    xlsx_path: Path | None,
    out_path: Path | None,
    window: int,
) -> None:
    """Build interactive HTML report from an engine run directory."""
    # Lazy import: report module pulls plotly, only needed here.
    import json  # noqa: PLC0415

    from roro.report import build_report  # noqa: PLC0415

    snapshot_path = run_dir / "snapshot.json"
    if xlsx_path is None:
        if not snapshot_path.exists():
            raise click.UsageError(
                f"--xlsx not given and {snapshot_path} missing; cannot infer xlsx path."
            )
        snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
        candidate = snap.get("config_resolved", {}).get("data_path")
        if not candidate:
            raise click.UsageError(
                "--xlsx not given and snapshot.json has no config_resolved.data_path."
            )
        xlsx_path = Path(candidate)

    if not xlsx_path.exists():
        raise click.UsageError(f"xlsx file does not exist: {xlsx_path}")

    if out_path is None:
        out_path = run_dir / "report.html"

    result = build_report(run_dir, xlsx_path, out_path, window=window)
    click.echo(f"OK: {result}")
