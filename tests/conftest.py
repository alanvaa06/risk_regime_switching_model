"""Shared pytest fixtures: tiny synthetic dataset that mirrors data.xlsx structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COUNTRIES: list[str] = ["United States", "Brazil", "Germany", "Mexico"]
_EQ_TICKERS: list[str] = ["SPX Index", "MXBR Index", "MXDE Index", "MXMX Index"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--regenerate-goldens", action="store_true", default=False)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": ["United States", "Brazil", "Germany", "Mexico", "DM", "LatAm"],
            "Segment": ["DM", "EM", "DM", "EM", "DM", "EM"],
            "Equity Index": ["SPX", "MXBR", "MXDE", "MXMX", "MXWO", "MXLA"],
            "Equity Index Curreny": ["USD"] * 6,
            "Bond Index": ["LBUSTRUU", "I00", "I05", "I05M", "I35", "H04"],
            "Bond Index Curreny": ["USD"] * 6,
            "Local Curreny": ["USD", "BRL", "EUR", "MXN", "USD", "USD"],
            "Curr": [1.0, 5.0, 1.16, 20.0, 1.0, 1.0],
            "Pair": ["USD", "USDBRL", "USDEUR", "USDMXN", "USD", "USD"],
            "Equity_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Equity_Mkt_Cap": [100, 10, 20, 5, 130, 15],
            "FI_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Fixed_Income_Mkt_Cap": [50, 5, 10, 3, 65, 8],
            "Equity_Mkt_Cap_Val": [100, 10, 20, 5, 130, 15],
            "Fixed_Income_Mkt_Cap_Val": [50, 5, 10, 3, 65, 8],
        }
    )


def build_xlsx(path: Path, eq_data: pd.DataFrame, fi_data: pd.DataFrame) -> Path:
    """Write a data.xlsx-shaped workbook (Panel + Equity_LC + Fixed_Income_LC)."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        _panel().to_excel(w, sheet_name="Panel", index=False)
        _write_two_header_sheet(w, "Equity_LC", _EQ_TICKERS, COUNTRIES, eq_data)
        _write_two_header_sheet(w, "Fixed_Income_LC", _EQ_TICKERS, COUNTRIES, fi_data)
    return path


def random_walk_prices(
    dates: pd.DatetimeIndex, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded geometric random walks with distinct per-country vols (noisy betas)."""
    rng = np.random.default_rng(seed)
    n, k = len(dates), len(COUNTRIES)
    eq_vol = np.array([0.008, 0.018, 0.011, 0.015])
    fi_vol = np.array([0.002, 0.005, 0.003, 0.004])
    eq_ret = rng.normal(0.0003, 1.0, (n, k)) * eq_vol
    fi_ret = rng.normal(0.0001, 1.0, (n, k)) * fi_vol
    eq = pd.DataFrame(100.0 * np.exp(np.cumsum(eq_ret, axis=0)), index=dates, columns=COUNTRIES)
    fi = pd.DataFrame(200.0 * np.exp(np.cumsum(fi_ret, axis=0)), index=dates, columns=COUNTRIES)
    return eq, fi


RW_DATES: pd.DatetimeIndex = pd.bdate_range("2020-01-01", "2024-12-31")
RW_SEED: int = 7


@pytest.fixture
def tiny_xlsx(tmp_path: Path) -> Path:
    """Build a 4-country + 2-composite tiny xlsx that mirrors data.xlsx layout."""
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    eq_data = pd.DataFrame({c: range(100, 100 + len(dates)) for c in COUNTRIES}, index=dates)
    fi_data = pd.DataFrame({c: range(200, 200 + len(dates)) for c in COUNTRIES}, index=dates)
    return build_xlsx(tmp_path / "tiny.xlsx", eq_data, fi_data)


@pytest.fixture
def rw_xlsx(tmp_path: Path) -> Path:
    """Same layout as tiny_xlsx, seeded random-walk prices (noisy, realistic betas)."""
    eq, fi = random_walk_prices(RW_DATES, RW_SEED)
    return build_xlsx(tmp_path / "rw.xlsx", eq, fi)


def _write_two_header_sheet(
    writer: pd.ExcelWriter,
    sheet: str,
    tickers: list[str],
    countries: list[str],
    data: pd.DataFrame,
) -> None:
    """Mimic data.xlsx layout: row 0 = tickers, row 1 = country names, row 2+ = data.

    Column 0 holds the date in row 2+. Row 0 and row 1 prepend an empty cell so
    all rows have aligned width (1 + N_countries).
    """
    header_row_0 = [""] + tickers
    header_row_1 = [""] + countries
    data_rows = data.reset_index().values.tolist()
    out = pd.DataFrame([header_row_0, header_row_1] + data_rows)
    out.to_excel(writer, sheet_name=sheet, index=False, header=False)
