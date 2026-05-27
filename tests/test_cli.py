"""CLI smoke tests for `roro run`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from roro import cli as cli_mod
from roro.cli import main
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def test_cli_run_dispatches(
    tiny_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force engine to use a mock fred client by patching _build_fred_client
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    monkeypatch.setattr(
        cli_mod,
        "_build_fred_client",
        lambda key: MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
        ),
    )

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        f"data_path: {tiny_xlsx}\n"
        f"output_dir: {tmp_path / 'out'}\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "tripwire_ewma_halflife_days: 5\n"
        "percentile_window_years: 1\n"
        "bucket_scheme: TERCILE\n"
        "min_n_per_cut: 2\n"
        "direction_lookback_days: 5\n"
        "external_corr_window_days: 60\n"
        "external_corr_alert_threshold: 0.3\n"
        "bootstrap_min_days: 10\n"
        "methodology_version: 1.0.0\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--config",
            str(cfg_yaml),
            "--date",
            "2024-12-31",
            "--as-of-data-date",
            "2024-12-31",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "2024-12-31").exists()
