"""DataBundle frozen dataclass: typed input to all figure builders."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataBundle:
    """All data required to render the report.

    Index conventions:
    - Per-series panels (vol, ret_3m, beta_vs_global): index = business days,
      columns = series_id strings like "Brazil_Eq" or "Brazil_FI".
    - meta: index = series_id; columns = country, asset, segment, weight.
    - Segment-level panels: index = business days, columns = segment cuts.
    """

    run_date: pd.Timestamp
    methodology_version: str
    dates: pd.DatetimeIndex
    vol: pd.DataFrame
    ret_3m: pd.DataFrame
    beta_vs_global: pd.DataFrame
    meta: pd.DataFrame
    seg_beta: pd.DataFrame
    seg_tercile: pd.DataFrame
