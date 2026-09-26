"""Output layout: <outputs_root>/<data-date>/<kind>/<config-name>/.

The date is the last data date a run covers (not the day it ran), so every
artifact computed on the same data lands under one folder.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUTS_ROOT: Path = Path("outputs")


class OutputKind(StrEnum):
    HISTORIC = "historic"
    RUN = "run"
    BACKTEST = "backtest"
    GATE_DIAG = "gate_diag"


def output_dir(root: Path, data_date: str | pd.Timestamp, kind: OutputKind, name: str) -> Path:
    """``root/<YYYY-MM-DD>/<kind>/<name>``; ``data_date`` is normalised to ISO."""
    return root / f"{pd.Timestamp(data_date):%Y-%m-%d}" / kind.value / name
