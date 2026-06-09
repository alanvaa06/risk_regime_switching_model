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


def write_hmm_csv(run_dir: Path) -> Path:
    """Add a regimes_hmm.csv to an existing run dir (segments match the minimal fixture)."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rows = []
    labels = ("Risk-off", "Transitional", "Risk-on")
    for i, d in enumerate(dates):
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
            lab = labels[i % 3]
            p = {"Risk-off": (0.7, 0.2, 0.1), "Transitional": (0.2, 0.6, 0.2),
                 "Risk-on": (0.1, 0.2, 0.7)}[lab]
            rows.append({
                "date": d, "segment": seg, "state": labels.index(lab), "label": lab,
                "p_risk_off": p[0], "p_transitional": p[1], "p_risk_on": p[2],
                "confidence": max(p), "cold_start": False, "thin_cut": seg == "LatAm",
            })
    path = run_dir / "regimes_hmm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_jm_csv(run_dir: Path) -> Path:
    """Add a regimes_jm.csv to an existing run dir (segments match the minimal fixture)."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rows = []
    labels = ("Risk-off", "Transitional", "Risk-on")
    for i, d in enumerate(dates):
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
            lab = labels[i % 3]
            p = {"Risk-off": (0.7, 0.2, 0.1), "Transitional": (0.2, 0.6, 0.2),
                 "Risk-on": (0.1, 0.2, 0.7)}[lab]
            rows.append({
                "date": d, "segment": seg, "state": labels.index(lab), "label": lab,
                "p_risk_off": p[0], "p_transitional": p[1], "p_risk_on": p[2],
                "confidence": max(p), "cold_start": False, "thin_cut": False,
            })
    path = run_dir / "regimes_jm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(name="write_jm_csv")
def _write_jm_csv_fixture() -> Callable[[Path], Path]:
    """Fixture that returns the write_jm_csv helper callable."""
    return write_jm_csv


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
