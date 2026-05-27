"""Segment registry and universe partition into the 10 PRD cuts."""

from __future__ import annotations

from dataclasses import dataclass

from roro.types import Universe

ASSET_EQ: str = "Eq"
ASSET_FI: str = "FI"

SEGMENT_NAMES: tuple[str, ...] = (
    "global",
    "DM",
    "EM",
    "Equity",
    "FI",
    "DM_Eq",
    "EM_Eq",
    "DM_FI",
    "EM_FI",
    "LatAm",
)

LATAM_COUNTRIES: frozenset[str] = frozenset({"Brazil", "Mexico", "Chile", "Peru", "Colombia"})


@dataclass(frozen=True)
class SeriesId:
    country: str
    segment: str  # "DM" or "EM"
    asset_class: str  # ASSET_EQ or ASSET_FI
    mcap: float

    @property
    def column_key(self) -> str:
        """Key used to look up the price column in PriceFrame.equity_lc / fi_lc."""
        return self.country


def partition(u: Universe) -> dict[str, list[SeriesId]]:
    """Partition the universe into the 10 PRD cuts keyed by segment name."""
    all_series: list[SeriesId] = []
    for _, row in u.countries.iterrows():
        all_series.append(
            SeriesId(
                country=str(row["Country"]),
                segment=str(row["Segment"]),
                asset_class=ASSET_EQ,
                mcap=float(row["Equity_Mkt_Cap_Val"]),
            )
        )
        all_series.append(
            SeriesId(
                country=str(row["Country"]),
                segment=str(row["Segment"]),
                asset_class=ASSET_FI,
                mcap=float(row["Fixed_Income_Mkt_Cap_Val"]),
            )
        )

    return {
        "global": list(all_series),
        "DM": [s for s in all_series if s.segment == "DM"],
        "EM": [s for s in all_series if s.segment == "EM"],
        "Equity": [s for s in all_series if s.asset_class == ASSET_EQ],
        "FI": [s for s in all_series if s.asset_class == ASSET_FI],
        "DM_Eq": [s for s in all_series if s.segment == "DM" and s.asset_class == ASSET_EQ],
        "EM_Eq": [s for s in all_series if s.segment == "EM" and s.asset_class == ASSET_EQ],
        "DM_FI": [s for s in all_series if s.segment == "DM" and s.asset_class == ASSET_FI],
        "EM_FI": [s for s in all_series if s.segment == "EM" and s.asset_class == ASSET_FI],
        "LatAm": [s for s in all_series if s.country in LATAM_COUNTRIES],
    }
