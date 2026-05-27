"""Hard schema + soft data-quality validation for engine inputs."""

from __future__ import annotations

import pandas as pd

from roro.types import PriceFrame, Universe

MAX_DATE_GAP_DAYS = 3


class DataSourceError(Exception):
    """Raised when input data is structurally unusable; engine should refuse to run."""


def validate_universe(u: Universe) -> list[str]:
    warnings: list[str] = []
    if (u.countries["Equity_Mkt_Cap_Val"] <= 0).any():
        raise DataSourceError("Equity_Mkt_Cap_Val must be > 0 for all countries")
    if (u.countries["Fixed_Income_Mkt_Cap_Val"] <= 0).any():
        raise DataSourceError("Fixed_Income_Mkt_Cap_Val must be > 0 for all countries")
    if not u.countries["Segment"].isin({"DM", "EM"}).all():
        raise DataSourceError("Segment must be DM or EM for every country")
    return warnings


def validate_prices(pf: PriceFrame) -> list[str]:
    warnings: list[str] = []
    for name, df in (("equity_lc", pf.equity_lc), ("fi_lc", pf.fi_lc)):
        _check_date_continuity(df, frame_name=name)
        warnings.extend(_collect_nan_warnings(df, frame_name=name))
    return warnings


def _check_date_continuity(df: pd.DataFrame, *, frame_name: str) -> None:
    if df.empty:
        raise DataSourceError(f"{frame_name}: price frame is empty")
    idx = pd.DatetimeIndex(df.index)
    bdays_expected = pd.bdate_range(idx.min(), idx.max())
    missing = bdays_expected.difference(idx)
    if len(missing) == 0:
        return
    # Find longest contiguous gap
    diffs = missing.to_series().diff().dt.days.fillna(1)
    longest = int((diffs == 1).astype(int).groupby((diffs != 1).cumsum()).cumsum().max())
    if longest > MAX_DATE_GAP_DAYS:
        raise DataSourceError(
            f"{frame_name}: date continuity broken — longest gap {longest} business days"
        )


def _collect_nan_warnings(df: pd.DataFrame, *, frame_name: str) -> list[str]:
    warnings: list[str] = []
    nan_counts = df.isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            warnings.append(f"{frame_name}: column {col!r} has {int(count)} NaN values")
    return warnings
