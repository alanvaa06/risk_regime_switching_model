"""Read engine run-dir CSVs + source xlsx prices into a typed DataBundle."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from roro.io import load_panel, load_prices
from roro.report.beta_vs_global import compute_beta_vs_global
from roro.report.bundle import DataBundle
from roro.report.errors import ReportInputError
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import ASSET_EQ, ASSET_FI

DEFAULT_WINDOW: int = 252
EWMA_HALFLIFE_DAYS: int = 30
BETA_WINDOW_DAYS: int = 63


def _read_snapshot(run_dir: Path) -> dict[str, object]:
    path = run_dir / "snapshot.json"
    if not path.exists():
        raise ReportInputError(f"Missing snapshot.json in run dir: {run_dir}")
    try:
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportInputError(f"Could not parse snapshot.json: {exc}") from exc
    return data


def _read_required_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise ReportInputError(f"Missing {name} in run dir: {run_dir}")
    return pd.read_csv(path, parse_dates=["date"])


def _build_series_panels(
    xlsx_path: Path,
    *,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-series vol, ret_3m, beta_vs_global, meta from xlsx."""
    universe = load_panel(xlsx_path)
    prices = load_prices(xlsx_path)

    # Build per-series price frame: column = "<country>_Eq" or "<country>_FI"
    eq = prices.equity_lc.copy()
    eq.columns = [f"{c}_{ASSET_EQ}" for c in eq.columns]
    fi = prices.fi_lc.copy()
    fi.columns = [f"{c}_{ASSET_FI}" for c in fi.columns]
    per_series_prices = pd.concat([eq, fi], axis=1).sort_index()

    daily = daily_log_returns(per_series_prices)
    vol = ewma_vol(daily, halflife=EWMA_HALFLIFE_DAYS)
    ret_3m = total_return_3m(per_series_prices)

    # Weights from Panel
    weight_rows: list[dict[str, object]] = []
    for _, row in universe.countries.iterrows():
        country = str(row["Country"])
        seg = str(row["Segment"])
        weight_rows.append(
            {
                "series_id": f"{country}_{ASSET_EQ}",
                "country": country,
                "asset": ASSET_EQ,
                "segment": seg,
                "weight": float(row["Equity_Mkt_Cap_Val"]),
            }
        )
        weight_rows.append(
            {
                "series_id": f"{country}_{ASSET_FI}",
                "country": country,
                "asset": ASSET_FI,
                "segment": seg,
                "weight": float(row["Fixed_Income_Mkt_Cap_Val"]),
            }
        )
    meta = pd.DataFrame(weight_rows).set_index("series_id")

    weights = meta["weight"]
    # Keep only columns present in both prices and meta
    common = [c for c in per_series_prices.columns if c in meta.index]
    daily = daily[common]
    vol = vol[common]
    ret_3m = ret_3m[common]
    beta = compute_beta_vs_global(daily, weights.loc[common], window=BETA_WINDOW_DAYS)

    # Slice to last `window` business days
    tail_dates = per_series_prices.index[-window:]
    return (
        vol.loc[tail_dates],
        ret_3m.loc[tail_dates],
        beta.loc[tail_dates],
        meta.loc[common],
    )


def _pivot_segment(
    df: pd.DataFrame, *, value_col: str, scheme_filter: str | None = None
) -> pd.DataFrame:
    """Pivot long-format engine CSV to wide (date x segment)."""
    work = df
    if scheme_filter is not None:
        work = work[work["scheme"] == scheme_filter]
    return work.pivot(index="date", columns="segment", values=value_col).sort_index()


def load_bundle(
    run_dir: Path,
    xlsx_path: Path,
    *,
    window: int = DEFAULT_WINDOW,
) -> DataBundle:
    """Load all data required to build the report.

    Args:
        run_dir: engine run directory (contains snapshot.json + CSVs).
        xlsx_path: source xlsx with Equity_LC + Fixed_Income_LC + Panel sheets.
        window: number of trailing business days to expose in the bundle.

    Returns:
        Frozen DataBundle.

    Raises:
        ReportInputError: snapshot.json / required CSVs missing or unparseable.
        FileNotFoundError: xlsx_path does not exist.
    """
    snapshot = _read_snapshot(run_dir)
    beta_series = _read_required_csv(run_dir, "beta_series.csv")
    regimes = _read_required_csv(run_dir, "regimes.csv")

    vol, ret_3m, beta_vs_global, meta = _build_series_panels(xlsx_path, window=window)
    dates = vol.index

    seg_beta = _pivot_segment(beta_series, value_col="beta", scheme_filter="cap_wtd")
    seg_tercile = _pivot_segment(regimes, value_col="tercile")

    # Align segment frames to `dates`
    seg_beta = seg_beta.reindex(dates)
    seg_tercile = seg_tercile.reindex(dates)

    return DataBundle(
        run_date=pd.Timestamp(str(snapshot["run_date"])),
        methodology_version=str(snapshot["methodology_version"]),
        dates=pd.DatetimeIndex(dates),
        vol=vol,
        ret_3m=ret_3m,
        beta_vs_global=beta_vs_global,
        meta=meta,
        seg_beta=seg_beta,
        seg_tercile=seg_tercile,
    )
