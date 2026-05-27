import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SEGMENT_NAMES, SeriesId, partition
from roro.types import Universe


def _universe() -> Universe:
    return Universe(
        countries=pd.DataFrame(
            {
                "Country": ["US", "DE", "BR", "MX", "CL", "PE", "CO", "CN", "IN", "ZA"],
                "Segment": ["DM", "DM", "EM", "EM", "EM", "EM", "EM", "EM", "EM", "EM"],
                "Equity_Mkt_Cap_Val": [100, 20, 10, 5, 3, 2, 1, 40, 15, 7],
                "Fixed_Income_Mkt_Cap_Val": [50, 10, 5, 3, 2, 1, 1, 20, 8, 4],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM", "EM"]}),
    )


def test_partition_returns_all_segment_keys() -> None:
    cuts = partition(_universe())
    assert set(cuts) == set(SEGMENT_NAMES)


def test_global_cut_includes_every_country_each_class() -> None:
    cuts = partition(_universe())
    g = cuts["global"]
    assert len(g) == 20  # 10 countries × 2 classes


def test_latam_cut_membership_exact() -> None:
    cuts = partition(_universe())
    latam = cuts["LatAm"]
    assert {s.country for s in latam} == {"Brazil", "Mexico", "Chile", "Peru", "Colombia"} & set(
        _universe().countries["Country"]
    )


def test_dm_eq_cut_only_dm_equity() -> None:
    cuts = partition(_universe())
    for s in cuts["DM_Eq"]:
        assert s.segment == "DM"
        assert s.asset_class == ASSET_EQ


def test_series_id_carries_mcap() -> None:
    cuts = partition(_universe())
    em_eq = cuts["EM_Eq"]
    by_country = {s.country: s for s in em_eq}
    assert by_country["BR"].mcap > 0
    assert isinstance(by_country["BR"], SeriesId)
    assert by_country["BR"].asset_class != ASSET_FI
