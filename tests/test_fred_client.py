from datetime import date, datetime

import pandas as pd

from roro.fred_client import MockFredClient
from roro.io import load_fred


def test_mock_fred_returns_requested_series() -> None:
    client = MockFredClient(
        seeded={"VIXCLS": pd.Series([10.0, 11.0], index=pd.bdate_range("2024-01-01", periods=2))}
    )
    s = client.fetch("VIXCLS", date(2024, 1, 1), date(2024, 1, 3))
    assert s.iloc[0] == 10.0


def test_load_fred_aggregates_all_series() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    seeded = {
        sid: pd.Series([1.0] * 5, index=idx)
        for sid in ("VIXCLS", "BAMLC0A4CBBB", "BAMLEMCBPIOAS", "BAMLH0A0HYM2", "T10Y2Y")
    }
    client = MockFredClient(seeded=seeded)
    ff = load_fred(client, start=date(2024, 1, 1), end=date(2024, 1, 5))
    assert set(ff.series) == set(seeded)
    assert isinstance(ff.pulled_at, datetime)
    assert all(len(h) == 64 for h in ff.series_hashes.values())  # sha256 hexdigest
