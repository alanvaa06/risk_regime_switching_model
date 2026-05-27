import pandas as pd
import pytest

from roro.types import PriceFrame, Universe
from roro.validators import DataSourceError, validate_prices, validate_universe


def test_validate_universe_ok() -> None:
    u = Universe(
        countries=pd.DataFrame(
            {
                "Country": ["A", "B"],
                "Segment": ["DM", "EM"],
                "Equity_Mkt_Cap_Val": [1.0, 2.0],
                "Fixed_Income_Mkt_Cap_Val": [3.0, 4.0],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )
    warnings = validate_universe(u)
    assert warnings == []


def test_validate_universe_zero_mcap_raises() -> None:
    u = Universe(
        countries=pd.DataFrame(
            {
                "Country": ["A"],
                "Segment": ["DM"],
                "Equity_Mkt_Cap_Val": [0.0],
                "Fixed_Income_Mkt_Cap_Val": [1.0],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )
    with pytest.raises(DataSourceError):
        validate_universe(u)


def test_validate_prices_date_continuity_break_raises() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-31").tolist()
    # remove 4 contiguous business days mid-stream
    del dates[10:14]
    df = pd.DataFrame({"US": range(len(dates))}, index=pd.DatetimeIndex(dates))
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    with pytest.raises(DataSourceError, match="continuity"):
        validate_prices(pf)


def test_validate_prices_warns_on_per_series_nan() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    df = pd.DataFrame({"US": [1.0] * len(dates), "BR": [1.0] * len(dates)}, index=dates)
    df.loc[df.index[5], "BR"] = float("nan")
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    warnings = validate_prices(pf)
    assert any("BR" in w for w in warnings)


def test_validate_prices_weekend_spanning_gap_counted_as_one_run() -> None:
    """Fri + Mon missing must count as a 2-bday gap, not two separate 1-day gaps."""
    dates = pd.bdate_range("2024-01-01", "2024-01-31").tolist()
    # Find a Fri-Mon pair to drop. 2024-01-05 is Fri, 2024-01-08 is Mon.
    drop = {pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")}
    kept = [d for d in dates if d not in drop]
    df = pd.DataFrame({"US": range(len(kept))}, index=pd.DatetimeIndex(kept))
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    # 2-bday gap is within tolerance (MAX=3), so should pass without raising
    warnings = validate_prices(pf)
    assert warnings == [] or all("continuity" not in w for w in warnings)


def test_validate_prices_four_bday_gap_across_weekend_raises() -> None:
    """Thu+Fri+Mon+Tue missing = 4-bday continuous gap (must raise)."""
    dates = pd.bdate_range("2024-01-01", "2024-01-31").tolist()
    drop = {
        pd.Timestamp("2024-01-04"),  # Thu
        pd.Timestamp("2024-01-05"),  # Fri
        pd.Timestamp("2024-01-08"),  # Mon
        pd.Timestamp("2024-01-09"),  # Tue
    }
    kept = [d for d in dates if d not in drop]
    df = pd.DataFrame({"US": range(len(kept))}, index=pd.DatetimeIndex(kept))
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    with pytest.raises(DataSourceError, match="continuity"):
        validate_prices(pf)
