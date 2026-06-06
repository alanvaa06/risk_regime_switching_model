import numpy as np
import pandas as pd

from roro.report.vol_breadth import realized_vol, vol_percentile_matrix


def test_realized_vol_annualizes() -> None:
    idx = pd.bdate_range("2010-01-01", periods=10)
    r = pd.DataFrame({"A": [0.01] * 10}, index=idx)
    v = realized_vol(r, window=5)
    assert v["A"].iloc[:4].isna().all()
    assert abs(float(v["A"].iloc[5])) < 1e-9


def test_realized_vol_matches_manual() -> None:
    idx = pd.bdate_range("2010-01-01", periods=6)
    r = pd.DataFrame({"A": [0.0, 0.02, -0.02, 0.02, -0.02, 0.0]}, index=idx)
    v = realized_vol(r, window=3)
    expected = float(np.std([0.0, 0.02, -0.02], ddof=1) * np.sqrt(252))
    assert abs(float(v["A"].iloc[2]) - expected) < 1e-9


def test_vol_percentile_matrix_expanding_and_min_history() -> None:
    idx = pd.bdate_range("2010-01-01", periods=10)
    vol = pd.DataFrame({"A": np.arange(1.0, 11.0)}, index=idx)
    pct = vol_percentile_matrix(vol, min_history_days=4)
    assert pct["A"].iloc[:3].isna().all()
    assert (pct["A"].iloc[3:] == 1.0).all()
    arr = pct.dropna().to_numpy()
    assert (arr >= 0).all() and (arr <= 1).all()


def test_vol_percentile_matrix_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=300)
    vol = pd.DataFrame({"A": rng.normal(0.2, 0.05, 300)}, index=idx)
    fast = vol_percentile_matrix(vol, min_history_days=50)

    def brute(s: pd.Series) -> pd.Series:
        out = []
        for t in range(len(s)):
            hist = s.iloc[: t + 1].dropna()
            if len(hist) < 50:
                out.append(np.nan)
            else:
                out.append(float((hist <= s.iloc[t]).sum()) / len(hist))
        return pd.Series(out, index=s.index)

    expected = brute(vol["A"])
    pd.testing.assert_series_equal(fast["A"], expected, check_names=False)
