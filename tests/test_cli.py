"""CLI smoke tests for `roro run`."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

import roro.historic as historic_mod
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


def test_cli_gate_diagnostics_dispatches(
    tiny_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        ["gate-diagnostics", "--config", str(cfg_yaml),
         "--start", "2024-01-01", "--end", "2024-12-31",
         "--out", str(tmp_path / "diag")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "diag" / "gate_diagnostics.json").exists()
    assert (tmp_path / "diag" / "g3_g5_frontier.csv").exists()


def test_cli_update_full_then_up_to_date(
    rw_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    monkeypatch.setattr(
        cli_mod,
        "_build_fred_client",
        lambda key: MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
        ),
    )
    cfg_yaml = tmp_path / "smoke.yaml"
    cfg_yaml.write_text(
        f"data_path: {rw_xlsx.as_posix()}\n"
        "output_dir: ignored\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "percentile_window_years: 1\n"
        "min_n_per_cut: 2\n"
        "bootstrap_min_days: 10\n",
        encoding="utf-8",
    )
    root = tmp_path / "hist"
    base = ["update", "--config", str(cfg_yaml), "--no-report", "--historic-root", str(root)]
    runner = CliRunner()

    first = runner.invoke(main, [*base, "--data-until", "15-06-2023"])
    assert first.exit_code == 0, first.output
    assert "[FULL]" in first.output
    assert (root / "smoke" / "results_2023-06-15" / "snapshot.json").exists()

    second = runner.invoke(main, base)
    assert second.exit_code == 0, second.output
    assert "[RESUME]" in second.output

    third = runner.invoke(main, base)
    assert third.exit_code == 0, third.output
    assert "[UP_TO_DATE]" in third.output

    bad = runner.invoke(main, [*base, "--data-until", "2023/06/15"])
    assert bad.exit_code != 0
    assert "DD-MM-YYYY" in bad.output


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        ("no_data", "no data in data.xlsx"),
        ("network", "Cannot reach FRED"),
    ],
)
def test_cli_update_friendly_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: str, expected: str
) -> None:
    errors: dict[str, Exception] = {
        "no_data": historic_mod.NoDataError("no data in data.xlsx"),
        "network": urllib.error.URLError("proxy"),
    }

    def failing(*_a: object, **_k: object) -> None:
        raise errors[exc]

    monkeypatch.setattr(historic_mod, "run_update", failing)
    monkeypatch.setattr(cli_mod, "_build_fred_client", lambda key: object())
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("data_path: x.xlsx\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["update", "--config", str(cfg_yaml)])
    assert result.exit_code == 1
    assert f"Error: {expected}" in result.output
    assert "Traceback" not in result.output


def test_cli_update_unknown_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_a: object, **_k: object) -> None:
        raise RuntimeError("engine bug")

    monkeypatch.setattr(historic_mod, "run_update", failing)
    monkeypatch.setattr(cli_mod, "_build_fred_client", lambda key: object())
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("data_path: x.xlsx\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["update", "--config", str(cfg_yaml)])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
