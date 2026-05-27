"""Excel + FRED ingest and run output writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.types import PriceFrame, Universe

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
