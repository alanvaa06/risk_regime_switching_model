"""Date-first output layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.layout import OutputKind, output_dir


def test_output_dir_is_date_then_kind_then_name() -> None:
    got = output_dir(Path("outputs"), "2026-05-26", OutputKind.HISTORIC, "jm-eval")
    assert got == Path("outputs") / "2026-05-26" / "historic" / "jm-eval"


def test_output_dir_normalises_the_date() -> None:
    for date in ("2026-5-26", pd.Timestamp("2026-05-26 00:00")):
        assert output_dir(Path("o"), date, OutputKind.RUN, "c") == Path("o/2026-05-26/run/c")
