import numpy as np
import pandas as pd
from hypothesis import given, settings

from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from tests.strategies import finite_return_series, positive_price_series


def test_total_return_3m_log_additivity() -> None:
    prices = pd.DataFrame(
        {"A": [100, 110, 121, 133.1]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    r = total_return_3m(prices, window_days=2)
    # ln(121/100) ≈ ln(1.21)
    assert abs(r["A"].iloc[2] - np.log(1.21)) < 1e-12


def test_daily_log_returns_length() -> None:
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.bdate_range("2024-01-01", periods=3))
    r = daily_log_returns(prices)
    assert len(r) == 3
    assert pd.isna(r["A"].iloc[0])
    assert abs(r["A"].iloc[1] - np.log(110 / 100)) < 1e-12


def test_ewma_vol_non_negative_and_annualized() -> None:
    rng = np.random.default_rng(seed=0)
    r = pd.DataFrame(
        {"A": rng.normal(0, 0.01, size=300)},
        index=pd.bdate_range("2024-01-01", periods=300),
    )
    sigma = ewma_vol(r, halflife=30)
    assert (sigma.dropna() >= 0).all().all()
    # Annualized daily ~1% vol ≈ 0.16
    assert 0.10 < sigma["A"].iloc[-1] < 0.25


def test_ewma_vol_converges_with_constant_input() -> None:
    r = pd.DataFrame({"A": [0.01] * 500}, index=pd.bdate_range("2024-01-01", periods=500))
    sigma = ewma_vol(r, halflife=30)
    tail = sigma["A"].iloc[-50:]
    assert tail.std() < 1e-8


@given(prices=positive_price_series(length=100))
@settings(max_examples=25, deadline=None)
def test_total_return_3m_finite_when_prices_positive(prices: pd.Series) -> None:
    df = prices.to_frame(name="A")
    r = total_return_3m(df, window_days=21)
    assert r["A"].dropna().apply(np.isfinite).all()


@given(r=finite_return_series(length=200))
@settings(max_examples=25, deadline=None)
def test_ewma_vol_monotone_in_abs_returns(r: pd.Series) -> None:
    # Compare same series scaled by 2 → sigma should not decrease
    df1 = r.to_frame(name="A")
    df2 = (r * 2).to_frame(name="A")
    s1 = ewma_vol(df1, halflife=30)["A"].iloc[-1]
    s2 = ewma_vol(df2, halflife=30)["A"].iloc[-1]
    assert s2 >= s1 - 1e-12
