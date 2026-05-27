import numpy as np
import pandas as pd

from roro.correlation import avg_pairwise_rolling, compute_correlation_panel, pc1_share_rolling
from roro.segments import ASSET_EQ, SeriesId


def test_avg_pairwise_in_minus_one_to_one() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 5)), index=pd.bdate_range("2024-01-01", periods=200))
    s = avg_pairwise_rolling(df, window=63)
    s = s.dropna()
    assert (s >= -1).all()
    assert (s <= 1).all()


def test_pc1_share_in_zero_to_one_and_min_n_inv() -> None:
    rng = np.random.default_rng(0)
    n = 5
    df = pd.DataFrame(rng.normal(size=(200, n)), index=pd.bdate_range("2024-01-01", periods=200))
    s = pc1_share_rolling(df, window=63)
    s = s.dropna()
    assert (s >= 1.0 / n - 1e-9).all()
    assert (s <= 1.0).all()


def test_correlation_panel_runs_per_segment() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    countries = [f"C{i}" for i in range(8)]
    eq = pd.DataFrame(rng.normal(size=(200, 8)), index=dates, columns=countries)
    fi = pd.DataFrame(rng.normal(size=(200, 8)), index=dates, columns=countries)
    series = [
        SeriesId(country=c, segment="DM" if i < 4 else "EM", asset_class=ASSET_EQ, mcap=1.0)
        for i, c in enumerate(countries)
    ]
    cuts = {"global": series, "DM_Eq": [s for s in series if s.segment == "DM"]}
    cf = compute_correlation_panel(
        daily_log_returns_eq=eq,
        daily_log_returns_fi=fi,
        cuts=cuts,
        window=63,
    )
    assert set(cf.avg_pairwise_3m.columns) == {"global", "DM_Eq"}
    assert set(cf.pc1_variance_share.columns) == {"global", "DM_Eq"}
