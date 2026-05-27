from pathlib import Path

from roro.io import load_panel


def test_panel_splits_countries_and_composites(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert set(u.countries["Country"]) == {"United States", "Brazil", "Germany", "Mexico"}
    assert set(u.composites["Country"]) == {"DM", "LatAm"}


def test_panel_carries_mcap_val_columns(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert "Equity_Mkt_Cap_Val" in u.countries.columns
    assert "Fixed_Income_Mkt_Cap_Val" in u.countries.columns
    assert (u.countries["Equity_Mkt_Cap_Val"] > 0).all()
