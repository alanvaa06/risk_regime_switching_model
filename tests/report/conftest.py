"""Shared report-test fixtures."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest


def _write_minimal_run_dir(tmp_path: Path, xlsx_path: Path) -> Path:
    """Build a minimal valid run dir: snapshot.json + beta_series.csv + regimes.csv."""
    run = tmp_path / "run"
    run.mkdir()

    snapshot = {
        "run_date": "2024-12-31",
        "as_of_data_date": "2024-12-30",
        "methodology_version": "1.0.0",
        "config_resolved": {"data_path": str(xlsx_path)},
        "data_fingerprint": {},
        "code_version": {},
        "warnings": [],
    }
    (run / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    beta_rows = []
    for d in dates:
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
            beta_rows.append(
                {
                    "date": d,
                    "segment": seg,
                    "scheme": "cap_wtd",
                    "beta": 1.0,
                    "r2": 0.5,
                    "n": 10,
                    "suppressed": False,
                    "singular": False,
                }
            )
    pd.DataFrame(beta_rows).to_csv(run / "beta_series.csv", index=False)

    reg_rows = []
    for d in dates:
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
            reg_rows.append(
                {
                    "date": d,
                    "segment": seg,
                    "percentile_5y": 0.5,
                    "tercile": "Transitional",
                    "quintile": "Q3",
                    "direction": 0.0,
                    "n": 10,
                    "thin_cut": False,
                    "bootstrap": False,
                }
            )
    pd.DataFrame(reg_rows).to_csv(run / "regimes.csv", index=False)
    return run


@pytest.fixture
def minimal_run_dir(tmp_path: Path, tiny_xlsx: Path) -> Path:
    """A minimal valid engine run dir backed by the tiny_xlsx fixture."""
    return _write_minimal_run_dir(tmp_path, tiny_xlsx)


@pytest.fixture
def make_run_dir(tmp_path: Path) -> Callable[[Path], Path]:
    """Factory: build a minimal run dir against any xlsx path."""
    def _factory(xlsx_path: Path) -> Path:
        return _write_minimal_run_dir(tmp_path, xlsx_path)
    return _factory
