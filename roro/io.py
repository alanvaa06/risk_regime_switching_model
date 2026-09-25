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
    AttributionFrame,
    BetaBySegment,
    CorrelationFrame,
    FredFrame,
    HmmRegimeFrame,
    JmRegimeFrame,
    PriceFrame,
    RegimeFrame,
    ResumeState,
    RunResult,
    SegmentPrior,
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


_BETA_COLUMNS: tuple[str, ...] = (
    "date", "segment", "scheme", "beta", "r2", "n", "suppressed", "singular",
)
_PROB_COLUMNS: list[str] = ["p_risk_off", "p_transitional", "p_risk_on"]


def cut_prices(prices: PriceFrame, until: pd.Timestamp) -> PriceFrame:
    """Prices on or before ``until`` (the data_until cut of an update run)."""
    return PriceFrame(equity_lc=prices.equity_lc.loc[:until], fi_lc=prices.fi_lc.loc[:until])


def beta_long(bbs: BetaBySegment) -> pd.DataFrame:
    """Long (date, segment, scheme, beta, r2, n, suppressed, singular) frame of beta_series.csv."""
    cap = _stack_segment_frame({k: v.cap_wtd for k, v in bbs.by_segment.items()}, "cap_wtd")
    eq = _stack_segment_frame({k: v.eq_wtd for k, v in bbs.by_segment.items()}, "eq_wtd")
    if cap.empty and eq.empty:
        return pd.DataFrame(columns=list(_BETA_COLUMNS))
    return pd.concat([cap, eq], ignore_index=True)


def read_resume_state(run_dir: Path, last_date: pd.Timestamp) -> ResumeState:
    """Everything a RESUME run needs from a checkpoint folder (exact float round trip)."""
    beta_path = run_dir / "beta_series.csv"
    beta = pd.read_csv(
        beta_path,
        parse_dates=["date"],
        float_precision="round_trip",
        dtype={"segment": str, "scheme": str},
    )
    return ResumeState(
        checkpoint_date=last_date,
        beta_series=beta,
        hmm=_read_overlay_prior(
            run_dir / "regimes_hmm.csv", run_dir / "hmm_refit_log.csv", last_date
        ),
        jm=_read_overlay_prior(
            run_dir / "regimes_jm.csv", run_dir / "jm_refit_log.csv", last_date
        ),
    )


def _read_cold_start(col: pd.Series, *, source: Path) -> pd.Series:
    """Map True/False/"True"/"False" -> bool, NaN -> True (cold by default); else raise."""
    mapped = col.map({True: True, False: False, "True": True, "False": False})
    mapped = mapped.mask(col.isna(), True)
    unmapped = mapped.isna()
    if bool(unmapped.any()):
        bad = col.loc[unmapped].iloc[0]
        raise ValueError(f"{source}: unexpected cold_start value {bad!r}")
    return mapped.astype(bool)


def _read_overlay_prior(
    rows_path: Path, log_path: Path, last_date: pd.Timestamp
) -> dict[str, SegmentPrior] | None:
    if not rows_path.exists():
        return None
    rows = pd.read_csv(
        rows_path,
        parse_dates=["date"],
        float_precision="round_trip",
        dtype={"segment": str},
    )
    log = (
        pd.read_csv(log_path, parse_dates=["refit_date"], dtype={"segment": str})
        if log_path.exists()
        else pd.DataFrame(columns=["segment", "refit_date"])
    )
    segment_str = rows["segment"].astype(str)
    out: dict[str, SegmentPrior] = {}
    for seg in sorted(segment_str.unique()):
        g = rows.loc[segment_str == seg].set_index("date").sort_index()
        refits = log.loc[log["segment"].astype(str) == seg, "refit_date"]
        cold_start = _read_cold_start(g["cold_start"], source=rows_path)
        out[seg] = SegmentPrior(
            probs=g[_PROB_COLUMNS],
            cold_start=cold_start,
            refit_dates=tuple(sorted(pd.Timestamp(d) for d in refits)),
            last_date=last_date,
        )
    return out


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

    if result.regime_hmm is not None:
        _write_regime_hmm(result.regime_hmm, tmp / "regimes_hmm.csv")
        _write_hmm_refit_log(result.regime_hmm, tmp / "hmm_refit_log.csv")

    if result.regime_jm is not None:
        _write_regime_jm(result.regime_jm, tmp / "regimes_jm.csv")
        _write_jm_refit_log(result.regime_jm, tmp / "jm_refit_log.csv")

    if result.attribution is not None:
        _write_attribution(result.attribution, tmp)

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
    frame = beta_long(bbs)
    if frame.empty:
        path.write_text(",".join(_BETA_COLUMNS) + "\n", encoding="utf-8")
        return
    frame.to_csv(path, index=False)


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
    if not a.concentration_alerts.empty:
        rows.append(a.concentration_alerts.assign(kind="concentration"))
    if not rows:
        path.write_text("date,kind,segment\n", encoding="utf-8")
        return
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _write_tripwire(bbs: BetaBySegment, path: Path) -> None:
    _write_beta(bbs, path)


def _write_regime_hmm(hf: HmmRegimeFrame, path: Path) -> None:
    cols = (
        "date,segment,state,label,p_risk_off,p_transitional,"
        "p_risk_on,confidence,cold_start,thin_cut\n"
    )
    if hf.label.empty:
        path.write_text(cols, encoding="utf-8")
        return
    merged = _melt_with_date(hf.label, "label")
    for name, frame in (
        ("state", hf.state),
        ("p_risk_off", hf.prob_risk_off),
        ("p_transitional", hf.prob_transitional),
        ("p_risk_on", hf.prob_risk_on),
        ("confidence", hf.confidence),
        ("cold_start", hf.cold_start_flag),
        ("thin_cut", hf.thin_cut_flag),
    ):
        merged = merged.merge(_melt_with_date(frame, name), on=["date", "segment"], how="left")
    merged = merged[
        ["date", "segment", "state", "label", "p_risk_off", "p_transitional",
         "p_risk_on", "confidence", "cold_start", "thin_cut"]
    ]
    merged.to_csv(path, index=False)


def _write_hmm_refit_log(hf: HmmRegimeFrame, path: Path) -> None:
    rows = [
        {"segment": seg, "refit_date": d}
        for seg, dates in hf.refit_dates.items()
        for d in dates
    ]
    df = pd.DataFrame(rows, columns=["segment", "refit_date"])
    df.to_csv(path, index=False)


def _write_regime_jm(jf: JmRegimeFrame, path: Path) -> None:
    cols = (
        "date,segment,state,label,p_risk_off,p_transitional,"
        "p_risk_on,confidence,cold_start,thin_cut\n"
    )
    if jf.label.empty:
        path.write_text(cols, encoding="utf-8")
        return
    merged = _melt_with_date(jf.label, "label")
    for name, frame in (
        ("state", jf.state),
        ("p_risk_off", jf.prob_risk_off.round(10)),
        ("p_transitional", jf.prob_transitional.round(10)),
        ("p_risk_on", jf.prob_risk_on.round(10)),
        ("confidence", jf.confidence.round(10)),
        ("cold_start", jf.cold_start_flag),
        ("thin_cut", jf.thin_cut_flag),
    ):
        merged = merged.merge(_melt_with_date(frame, name), on=["date", "segment"], how="left")
    merged = merged[
        ["date", "segment", "state", "label", "p_risk_off", "p_transitional",
         "p_risk_on", "confidence", "cold_start", "thin_cut"]
    ]
    merged.to_csv(path, index=False)


def _write_jm_refit_log(jf: JmRegimeFrame, path: Path) -> None:
    rows = [
        {"segment": seg, "refit_date": d}
        for seg, dates in jf.refit_dates.items()
        for d in dates
    ]
    df = pd.DataFrame(rows, columns=["segment", "refit_date"])
    df.to_csv(path, index=False)


_ATTRIBUTION_FILES: tuple[tuple[str, str], ...] = (
    ("level", "attribution.csv"),
    ("delta", "attribution_delta.csv"),
    ("rollup", "attribution_rollup.csv"),
    ("concentration", "concentration.csv"),
    ("pc1", "attribution_pc1.csv"),
)


def _write_attribution(af: AttributionFrame, run_dir: Path) -> None:
    """Five long-format CSVs (+ optional wide global history). Deterministic column order."""
    for attr, name in _ATTRIBUTION_FILES:
        frame: pd.DataFrame = getattr(af, attr)
        frame.to_csv(run_dir / name, index=False)
    if af.history_global is not None:
        af.history_global.to_csv(run_dir / "attribution_history_global.csv", index=True)


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _build_snapshot(
    result: RunResult, *, run_date: str, as_of_data_date: str
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "run_date": run_date,
        "as_of_data_date": as_of_data_date,
        "methodology_version": result.config.methodology_version,
        "config_resolved": config_to_dict(result.config),
        "data_fingerprint": result.data_fingerprint,
        "code_version": result.code_version,
        "warnings": result.warnings,
    }
    hf = result.regime_hmm
    if hf is not None and not hf.label.empty:
        last = hf.label.index[-1]
        snapshot["regime_hmm"] = {
            seg: {
                "label": hf.label.loc[last, seg],
                "p_risk_off": _safe_float(hf.prob_risk_off.loc[last, seg]),
                "p_transitional": _safe_float(hf.prob_transitional.loc[last, seg]),
                "p_risk_on": _safe_float(hf.prob_risk_on.loc[last, seg]),
                "confidence": _safe_float(hf.confidence.loc[last, seg]),
            }
            for seg in hf.label.columns
        }
    jf = result.regime_jm
    if jf is not None and not jf.label.empty:
        last_jm = jf.label.index[-1]
        snapshot["regime_jm"] = {
            seg: {
                "label": jf.label.loc[last_jm, seg],
                "p_risk_off": _safe_float(jf.prob_risk_off.loc[last_jm, seg]),
                "p_transitional": _safe_float(jf.prob_transitional.loc[last_jm, seg]),
                "p_risk_on": _safe_float(jf.prob_risk_on.loc[last_jm, seg]),
                "confidence": _safe_float(jf.confidence.loc[last_jm, seg]),
            }
            for seg in jf.label.columns
        }
    af = result.attribution
    if af is not None and not af.level.empty:
        snapshot["attribution"] = _attribution_snapshot(af)
    return snapshot


_SNAPSHOT_TOP: int = 3


def _attribution_snapshot(af: AttributionFrame) -> dict[str, Any]:
    """Per cut (cap-weighted, last date): beta, top-3 contributors, concentration, anchor."""
    out: dict[str, Any] = {}
    level = af.level[af.level["weighting"] == "cap"]
    conc = af.concentration[af.concentration["weighting"] == "cap"]
    for cut in sorted(level["cut"].unique()):
        rows = level[level["cut"] == cut]
        order = rows["contribution"].abs().sort_values(ascending=False, kind="stable").index
        top = rows.reindex(order).head(_SNAPSHOT_TOP)
        c_last = conc[conc["cut"] == cut]
        c_row = c_last.iloc[-1] if not c_last.empty else None
        anchor = af.anchors.get(cut)
        top3: list[dict[str, Any]] = []
        for _, r in top.iterrows():
            top3.append(
                {
                    "series": str(r["series"]),
                    "contribution": _safe_float(r["contribution"]),
                    "share": _safe_float(r["share"]),
                    "quadrant": str(r["quadrant"]),
                }
            )
        out[cut] = {
            "beta_cap": _safe_float(rows["contribution"].sum()),
            "top3": top3,
            "hhi": _safe_float(c_row["hhi"]) if c_row is not None else None,
            "top1_series": str(c_row["top1_series"]) if c_row is not None else None,
            "top1_share": _safe_float(c_row["top1_share"]) if c_row is not None else None,
            "fragile_flag": bool(c_row["fragile_flag"]) if c_row is not None else None,
            "anchor_date": anchor.strftime("%Y-%m-%d") if anchor is not None else None,
        }
    return out
