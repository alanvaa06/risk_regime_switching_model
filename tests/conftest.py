"""Shared pytest fixtures: tiny synthetic dataset that mirrors data.xlsx structure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--regenerate-goldens", action="store_true", default=False)


@pytest.fixture
def tiny_xlsx(tmp_path: Path) -> Path:
    """Build a 4-country + 2-composite tiny xlsx that mirrors data.xlsx layout."""
    path = tmp_path / "tiny.xlsx"
    panel = pd.DataFrame(
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
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    eq_tickers = ["SPX Index", "MXBR Index", "MXDE Index", "MXMX Index"]
    eq_countries = ["United States", "Brazil", "Germany", "Mexico"]
    eq_data = pd.DataFrame(
        {c: range(100, 100 + len(dates)) for c in eq_countries},
        index=dates,
    )
    fi_data = pd.DataFrame(
        {c: range(200, 200 + len(dates)) for c in eq_countries},
        index=dates,
    )

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        panel.to_excel(w, sheet_name="Panel", index=False)
        _write_two_header_sheet(w, "Equity_LC", eq_tickers, eq_countries, eq_data)
        _write_two_header_sheet(w, "Fixed_Income_LC", eq_tickers, eq_countries, fi_data)
    return path


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
