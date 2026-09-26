"""Thin Click CLI: `roro run`, `roro backtest`, `roro report`, `roro update`."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from roro.config import load_config
from roro.engine import run as engine_run
from roro.fred_client import FredApiClient, FredClient
from roro.layout import DEFAULT_OUTPUTS_ROOT, OutputKind, output_dir

load_dotenv()


def _build_fred_client(api_key: str | None) -> FredClient:
    if not api_key:
        raise click.UsageError(
            "FRED_API_KEY env var (or --fred-key) is required."
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
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Outputs root (default: the config's output_dir).",
)
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
    cfg = load_config(config_path, overrides=overrides)
    target = output_dir(
        out_dir or cfg.output_dir, as_of_data_date, OutputKind.RUN, config_path.stem
    )
    api_key = fred_key or os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    engine_run(
        replace(cfg, output_dir=target.parent),
        fred_client=client,
        run_date=run_date,
        as_of_data_date=as_of_data_date,
        force=force,
        run_name=target.name,
    )
    click.echo(f"OK: {target}")


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
    target = output_dir(cfg.output_dir, end, OutputKind.BACKTEST, config_path.stem)
    api_key = os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    report = run_backtest(
        replace(cfg, output_dir=target), fred_client=client, start=start, end=end
    )
    if assert_gates and not report["all_passed"]:
        raise click.ClickException(
            f"Acceptance gates failed; see {target / 'acceptance_report.json'}"
        )
    click.echo(f"OK: {target}")


@main.command("gate-diagnostics")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="acceptance_compare.json to pin the baseline against.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Folder for the diagnostics files (default: <output_dir>/<end>/gate_diag/<config>).",
)
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
def cmd_gate_diagnostics(
    config_path: Path,
    start: str,
    end: str,
    baseline_path: Path | None,
    out_dir: Path | None,
    fred_key: str | None,
) -> None:
    """Read-only diagnostic over the S9 acceptance gates."""
    # Lazy imports: engine + diagnostics pull heavy deps.
    from roro.gate_diagnostics import diagnose, write_diagnostics  # noqa: PLC0415

    cfg = load_config(config_path)
    target = out_dir or output_dir(cfg.output_dir, end, OutputKind.GATE_DIAG, config_path.stem)
    api_key = fred_key or os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    # Re-run the engine for a fresh causal RunResult (returns are not persisted).
    result = engine_run(replace(cfg, output_dir=target), fred_client=client, run_date=end,
                        as_of_data_date=end, force=True, run_name="run")
    diag = diagnose(result, start=start, end=end, baseline_compare_path=baseline_path)
    write_diagnostics(target, diag)
    click.echo(f"OK: {target}")


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


@main.command("update")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--full", is_flag=True, help="Reprocess the full history (ignore checkpoints).")
@click.option(
    "--data-until",
    default=None,
    help="Only use data up to this date (DD-MM-YYYY or YYYY-MM-DD). Default: latest.",
)
@click.option("--no-report", is_flag=True, help="Skip building report.html.")
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
@click.option(
    "--outputs-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUTS_ROOT,
    show_default=True,
    help="Outputs root; results go to <root>/<data date>/historic/<config>/.",
)
def cmd_update(
    config_path: Path,
    full: bool,
    data_until: str | None,
    no_report: bool,
    fred_key: str | None,
    outputs_root: Path,
) -> None:
    """Process only dates not yet in outputs/<date>/historic/<config>/ (or all with --full)."""
    # Lazy import: historic pulls the engine; keeps `roro --help` fast.
    from roro.historic import (  # noqa: PLC0415
        TypeRun,
        friendly_error,
        parse_user_date,
        run_update,
    )

    try:
        until = parse_user_date(data_until) if data_until else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--data-until") from None
    client = _build_fred_client(fred_key or os.environ.get("FRED_API_KEY", ""))
    try:
        run_update(
            config_path,
            type_run=TypeRun.ALL if full else TypeRun.NEW_DATA,
            data_until=until,
            build_report=not no_report,
            fred_client=client,
            outputs_root=outputs_root,
            echo=click.echo,
        )
    except Exception as exc:
        message = friendly_error(exc)
        if message is None:
            raise
        raise click.ClickException(message) from None
