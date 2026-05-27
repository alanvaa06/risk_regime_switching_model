"""Excel + FRED ingest and run output writing."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from roro.config import to_dict as config_to_dict
from roro.fred_client import FRED_SERIES_IDS, FredClient
from roro.types import (
    AlertSet,
    BetaBySegment,
    CorrelationFrame,
    FredFrame,
    PriceFrame,
    RegimeFrame,
    RunResult,
    Universe,
    ValidationFrame,
)

COMPOSITE_NAMES: frozenset[str] = frozenset({"DM", "EM", "Europe", "Asia", "World", "LatAm"})


def load_panel(xlsx_path: Path | str) -> Universe:
    df = pd.read_excel(xlsx_path, sheet_name="Panel")
    missing = {"Country", "Segment", "Equity_Mkt_Cap_Val", "Fixed_Income_Mkt_Cap_Val"} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")

    is_composite = df["Country"].isin(COMPOSITE_NAMES)
    countries = df.loc[~is_composite].reset_index(drop=True).copy()
    composites = df.loc[is_composite].reset_index(drop=True).copy()
    return Universe(countries=countries, composites=composites)


def load_prices(xlsx_path: Path | str) -> PriceFrame:
    """Load Equity_LC and Fixed_Income_LC.

    Layout: row 0 = ticker IDs, row 1 = country names, row 2+ = daily prices.
    Skip row 0 (tickers) and use row 1 (country names) as the column header.
    """
    eq = _read_price_sheet(xlsx_path, "Equity_LC")
    fi = _read_price_sheet(xlsx_path, "Fixed_Income_LC")
    # Align FI columns to the equity universe (some FI countries may be absent in edge data)
    common = [c for c in eq.columns if c in fi.columns]
    return PriceFrame(equity_lc=eq[common], fi_lc=fi[common])


def load_fred(client: FredClient, start: date, end: date) -> FredFrame:
    series: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for sid in FRED_SERIES_IDS:
        s = client.fetch(sid, start, end)
        series[sid] = s
        hashes[sid] = _hash_series(s)
    return FredFrame(series=series, pulled_at=datetime.now(), series_hashes=hashes)


def _hash_series(s: pd.Series) -> str:
    payload = pd.util.hash_pandas_object(s, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _read_price_sheet(xlsx_path: Path | str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    countries = raw.iloc[1, 1:].tolist()
    data = raw.iloc[2:].copy()
    data.columns = [raw.iloc[1, 0]] + countries  # first col = date
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index()
    data.columns.name = None
    return data.astype(float)


def compute_data_fingerprint(xlsx_path: Path) -> dict[str, str]:
    data = xlsx_path.read_bytes()
    return {
        "data_xlsx_sha256": hashlib.sha256(data).hexdigest(),
        "data_xlsx_mtime": str(int(xlsx_path.stat().st_mtime)),
    }


def code_version() -> dict[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return {"git_sha": sha, "dirty": "true" if status.strip() else "false"}
    except Exception:  # noqa: BLE001
        return {"git_sha": "unknown", "dirty": "unknown"}


def write_run(
    result: RunResult,
    *,
    run_date: str,
    out_dir: Path,
    as_of_data_date: str,
    force: bool = False,
) -> Path:
    final = out_dir / run_date
    tmp = out_dir / f"{run_date}.tmp"
    if final.exists() and not force:
        raise FileExistsError(f"Run dir exists: {final}. Use force=True to overwrite.")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    _write_beta(result.beta, tmp / "beta_series.csv")
    _write_regime(result.regime, tmp / "regimes.csv")
    _write_correlation(result.correlation, tmp / "correlation.csv")
    _write_validation(result.validation, tmp / "external_validation.csv")
    _write_alerts(result.alerts, tmp / "alerts.csv")
    _write_tripwire(result.tripwire, tmp / "tripwire.csv")

    snapshot = _build_snapshot(result, run_date=run_date, as_of_data_date=as_of_data_date)
    (tmp / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def read_run(run_dir: Path) -> dict[str, Any]:
    return {
        "snapshot": json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8")),
        "beta_series": pd.read_csv(run_dir / "beta_series.csv", parse_dates=["date"]),
        "regimes": pd.read_csv(run_dir / "regimes.csv", parse_dates=["date"]),
        "correlation": pd.read_csv(run_dir / "correlation.csv", parse_dates=["date"]),
        "external_validation": pd.read_csv(
            run_dir / "external_validation.csv", parse_dates=["date"]
        ),
        "alerts": (
            pd.read_csv(run_dir / "alerts.csv", parse_dates=["date"])
            if (run_dir / "alerts.csv").stat().st_size > 0
            else pd.DataFrame()
        ),
        "tripwire": pd.read_csv(run_dir / "tripwire.csv", parse_dates=["date"]),
    }


def _stack_segment_frame(frames: dict[str, pd.DataFrame], scheme_label: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cut, df in frames.items():
        if df.empty:
            continue
        copy = df.copy()
        copy = copy.reset_index().rename(columns={"index": "date"})
        copy.insert(1, "segment", cut)
        copy.insert(2, "scheme", scheme_label)
        rows.append(copy)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_beta(bbs: BetaBySegment, path: Path) -> None:
    cap = _stack_segment_frame({k: v.cap_wtd for k, v in bbs.by_segment.items()}, "cap_wtd")
    eq = _stack_segment_frame({k: v.eq_wtd for k, v in bbs.by_segment.items()}, "eq_wtd")
    if cap.empty and eq.empty:
        path.write_text(
            "date,segment,scheme,beta,r2,n,suppressed,singular\n", encoding="utf-8"
        )
        return
    pd.concat([cap, eq], ignore_index=True).to_csv(path, index=False)


def _melt_with_date(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    reset = frame.reset_index()
    id_col = reset.columns[0]
    melted = reset.melt(id_vars=[id_col], var_name="segment", value_name=value_name)
    return melted.rename(columns={id_col: "date"})


def _write_regime(rf: RegimeFrame, path: Path) -> None:
    if rf.tercile.empty:
        path.write_text(
            "date,segment,percentile_5y,tercile,quintile,direction,n,thin_cut,bootstrap\n",
            encoding="utf-8",
        )
        return
    merged = _melt_with_date(rf.tercile, "tercile")
    pct = _melt_with_date(rf.percentile_5y, "percentile_5y")
    merged = merged.merge(pct, on=["date", "segment"], how="left")
    for label, frame in (
        ("quintile", rf.quintile),
        ("direction", rf.direction),
        ("n", rf.n_per_segment),
        ("thin_cut", rf.thin_cut_flag),
        ("bootstrap", rf.bootstrap_flag),
    ):
        m = _melt_with_date(frame, label)
        merged = merged.merge(m, on=["date", "segment"], how="left")
    merged.to_csv(path, index=False)


def _write_correlation(cf: CorrelationFrame, path: Path) -> None:
    if cf.avg_pairwise_3m.empty:
        path.write_text(
            "date,segment,avg_pairwise_3m,pc1_variance_share\n", encoding="utf-8"
        )
        return
    avg = _melt_with_date(cf.avg_pairwise_3m, "avg_pairwise_3m")
    pc1 = _melt_with_date(cf.pc1_variance_share, "pc1_variance_share")
    avg.merge(pc1, on=["date", "segment"], how="left").to_csv(path, index=False)


def _write_validation(vf: ValidationFrame, path: Path) -> None:
    if vf.rolling_corr_60d.empty:
        path.write_text(
            "date,segment,fred_series,rolling_corr_60d\n", encoding="utf-8"
        )
        return
    stacked = vf.rolling_corr_60d.stack(level=[0, 1], future_stack=True)
    series = pd.Series(stacked, name="rolling_corr_60d")
    df = series.reset_index()
    df.columns = pd.Index(["date", "segment", "fred_series", "rolling_corr_60d"])
    df.to_csv(path, index=False)


def _write_alerts(a: AlertSet, path: Path) -> None:
    rows: list[pd.DataFrame] = []
    if not a.bucket_transitions.empty:
        rows.append(a.bucket_transitions.assign(kind="bucket_transition"))
    if not a.disagreement_events.empty:
        rows.append(a.disagreement_events.assign(kind="disagreement"))
    if not a.validation_degradation.empty:
        rows.append(a.validation_degradation.assign(kind="validation_degradation"))
    if not rows:
        path.write_text("date,kind,segment\n", encoding="utf-8")
        return
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _write_tripwire(bbs: BetaBySegment, path: Path) -> None:
    _write_beta(bbs, path)


def _build_snapshot(
    result: RunResult, *, run_date: str, as_of_data_date: str
) -> dict[str, Any]:
    return {
        "run_date": run_date,
        "as_of_data_date": as_of_data_date,
        "methodology_version": result.config.methodology_version,
        "config_resolved": config_to_dict(result.config),
        "data_fingerprint": result.data_fingerprint,
        "code_version": result.code_version,
        "warnings": result.warnings,
    }
