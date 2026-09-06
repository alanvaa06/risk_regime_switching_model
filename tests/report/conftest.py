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


def write_attribution_csvs(run_dir: Path) -> None:
    """Add minimal attribution artifacts (segments match the minimal fixture)."""
    d = pd.Timestamp("2024-12-31")
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    level_rows = []
    delta_rows = []
    pc1_rows = []
    for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI"):
        for weighting in ("cap", "eq"):
            for i, (series, block, quad) in enumerate((
                ("United States__Eq", "DM_Eq", "HI/+"), ("Brazil__Eq", "EM_Eq", "HI/-"),
                ("Germany__FI", "DM_FI", "LO/-"), ("Mexico__FI", "EM_FI", "LO/+"),
            )):
                c = (0.3, -0.1, 0.05, -0.02)[i]
                level_rows.append({
                    "date": d, "cut": seg, "weighting": weighting, "series": series,
                    "block": block, "latam": series.startswith(("Brazil", "Mexico")),
                    "vol": 0.1 * (i + 1), "ret3m": c, "weight": 0.25, "leverage": 1.0,
                    "contribution": c, "share": c / 0.23, "quadrant": quad,
                    "xbar": 0.25, "ybar": 0.05,
                })
                delta_rows.append({
                    "date": d, "cut": seg, "weighting": weighting, "horizon": "fixed",
                    "anchor_date": dates[-64], "label_anchor": "Risk-off", "label_t": "Risk-on",
                    "beta_anchor": 0.1, "beta_t": 0.23, "series": series, "block": block,
                    "effect_return": c / 2, "effect_position": c / 4,
                    "effect_interaction": c / 8, "effect_universe": 0.0,
                    "delta_total": c * 7 / 8,
                })
            if weighting == "cap":
                for series in ("United States__Eq", "Brazil__Eq", "Germany__FI", "Mexico__FI"):
                    pc1_rows.append({
                        "date": d, "cut": seg, "series": series, "pc1_load_sq": 0.25,
                        "var_share": 0.25, "decoupling": 0.0, "row_mean_corr": 0.3,
                    })
    conc_rows = [
        {"date": dt, "cut": seg, "weighting": w, "n": 4, "beta": 0.23, "hhi": 0.4,
         "top1_series": "United States__Eq", "top1_share": 0.6, "top5_share": 1.0,
         "beta_ex_top1": 0.0, "pct_today": 0.8, "pct_ex_top1": 0.4, "fragile_flag": True}
        for dt in dates for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI") for w in ("cap", "eq")
    ]
    rollup_rows = [
        {"date": d, "cut": seg, "weighting": w, "group_kind": "block", "group": blk,
         "contribution_sum": val, "n": 1}
        for seg in ("global", "DM", "EM", "EM_Eq", "EM_FI") for w in ("cap", "eq")
        for blk, val in (("DM_Eq", 0.3), ("EM_Eq", -0.1), ("DM_FI", 0.05), ("EM_FI", -0.02))
    ]
    pd.DataFrame(level_rows).to_csv(run_dir / "attribution.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(run_dir / "attribution_delta.csv", index=False)
    pd.DataFrame(rollup_rows).to_csv(run_dir / "attribution_rollup.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(run_dir / "concentration.csv", index=False)
    pd.DataFrame(pc1_rows).to_csv(run_dir / "attribution_pc1.csv", index=False)


@pytest.fixture
def attribution_run_dir(minimal_run_dir: Path) -> Path:
    write_attribution_csvs(minimal_run_dir)
    return minimal_run_dir
