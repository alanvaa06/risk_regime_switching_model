from pathlib import Path

import pandas as pd
import pytest

from roro.io import load_panel, load_prices


def test_panel_splits_countries_and_composites(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert set(u.countries["Country"]) == {"United States", "Brazil", "Germany", "Mexico"}
    assert set(u.composites["Country"]) == {"DM", "LatAm"}


def test_panel_carries_mcap_val_columns(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert "Equity_Mkt_Cap_Val" in u.countries.columns
    assert "Fixed_Income_Mkt_Cap_Val" in u.countries.columns
    assert (u.countries["Equity_Mkt_Cap_Val"] > 0).all()


def test_load_panel_missing_column_raises_sorted(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xlsx"
    # Missing Equity_Mkt_Cap_Val and Fixed_Income_Mkt_Cap_Val
    df = pd.DataFrame({"Country": ["US"], "Segment": ["DM"]})
    with pd.ExcelWriter(bad, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Panel", index=False)

    with pytest.raises(ValueError, match=r"Equity_Mkt_Cap_Val.*Fixed_Income_Mkt_Cap_Val"):
        load_panel(bad)


def test_prices_uses_country_row_as_header(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert "United States" in pf.equity_lc.columns
    assert "Brazil" in pf.equity_lc.columns
    # Header row must be country names, not tickers
    assert "SPX Index" not in pf.equity_lc.columns
    assert isinstance(pf.equity_lc.index, pd.DatetimeIndex)


def test_prices_aligned_columns(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert list(pf.equity_lc.columns) == list(pf.fi_lc.columns)
