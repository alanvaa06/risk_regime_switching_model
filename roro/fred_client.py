"""FRED client Protocol + real fredapi client + in-memory mock."""

from __future__ import annotations

import time
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

FRED_SERIES_IDS: tuple[str, ...] = (
    "VIXCLS",
    "BAMLC0A4CBBB",
    "BAMLEMCBPIOAS",
    "BAMLH0A0HYM2",
    "T10Y2Y",
)


@runtime_checkable
class FredClient(Protocol):
    def fetch(self, series_id: str, start: date, end: date) -> pd.Series: ...


class FredApiClient:
    """Real client backed by fredapi.Fred. Retries 3x with 2s backoff."""

    def __init__(self, api_key: str, retries: int = 3, backoff_seconds: float = 2.0) -> None:
        from fredapi import Fred  # noqa: PLC0415  (lazy import to avoid fredapi side effects)

        self._fred = Fred(api_key=api_key)
        self._retries = retries
        self._backoff = backoff_seconds

    def fetch(self, series_id: str, start: date, end: date) -> pd.Series:
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                s = self._fred.get_series(
                    series_id, observation_start=start, observation_end=end
                )
                return pd.Series(s, name=series_id).astype(float)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._retries - 1:
                    time.sleep(self._backoff)
        assert last_exc is not None
        raise last_exc


class MockFredClient:
    def __init__(self, seeded: dict[str, pd.Series]) -> None:
        self._seeded = seeded

    def fetch(self, series_id: str, start: date, end: date) -> pd.Series:
        if series_id not in self._seeded:
            return pd.Series(dtype=float, name=series_id)
        s = self._seeded[series_id]
        mask = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
        return s.loc[mask].rename(series_id)
