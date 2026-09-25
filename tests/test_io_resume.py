"""Checkpoint read-back (exact float round-trip) and the beta history check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.io import (
    _write_hmm_refit_log,
    _write_regime_hmm,
    beta_long,
    cut_prices,
    load_prices,
    read_resume_state,
)
from roro.resume import HistoryRevisedError, verify_beta_history
from roro.types import HmmRegimeFrame


def _hmm_frame() -> HmmRegimeFrame:
    idx = pd.bdate_range("2021-01-01", periods=30)
    rng = np.random.default_rng(5)
    raw = rng.random((30, 3))
    raw /= raw.sum(axis=1, keepdims=True)  # full-precision floats, not round numbers
    raw[:4] = np.nan

    def frame(col: int) -> pd.DataFrame:
        return pd.DataFrame({"global": raw[:, col], "EM": raw[::-1, col]}, index=idx)

    labels = pd.DataFrame("Risk-on", index=idx, columns=["global", "EM"])
    cold = pd.DataFrame(False, index=idx, columns=["global", "EM"])
    cold.iloc[:4] = True
    return HmmRegimeFrame(
        state=pd.DataFrame(2.0, index=idx, columns=["global", "EM"]),
        label=labels,
        prob_risk_off=frame(0),
        prob_transitional=frame(1),
        prob_risk_on=frame(2),
        confidence=frame(2),
        n_per_segment=pd.DataFrame(10, index=idx, columns=["global", "EM"]),
        thin_cut_flag=pd.DataFrame(False, index=idx, columns=["global", "EM"]),
        cold_start_flag=cold,
        refit_dates={"global": [idx[4], idx[20]], "EM": [idx[4]]},
    )


def test_overlay_prior_round_trips_exactly(tmp_path: Path) -> None:
    hf = _hmm_frame()
    _write_regime_hmm(hf, tmp_path / "regimes_hmm.csv")
    _write_hmm_refit_log(hf, tmp_path / "hmm_refit_log.csv")
    beta = pd.DataFrame(
        {"date": hf.label.index, "segment": "global", "scheme": "cap_wtd", "beta": 0.1}
    )
    beta.to_csv(tmp_path / "beta_series.csv", index=False)
    last = hf.label.index[-1]
    state = read_resume_state(tmp_path, last)
    assert state.jm is None
    assert state.hmm is not None
    assert sorted(state.hmm) == ["EM", "global"]
    prior = state.hmm["global"]
    np.testing.assert_array_equal(
        prior.probs["p_risk_on"].to_numpy(), hf.prob_risk_on["global"].to_numpy()
    )
    np.testing.assert_array_equal(
        prior.cold_start.to_numpy(), hf.cold_start_flag["global"].to_numpy()
    )
    assert prior.refit_dates == tuple(hf.refit_dates["global"])
    assert prior.last_date == last


def test_nan_beta_rows_kept_in_prior(tmp_path: Path) -> None:
    hf = _hmm_frame()
    _write_regime_hmm(hf, tmp_path / "regimes_hmm.csv")
    _write_hmm_refit_log(hf, tmp_path / "hmm_refit_log.csv")
    beta = pd.DataFrame(
        {"date": hf.label.index, "segment": "global", "scheme": "cap_wtd", "beta": 0.1}
    )
    beta.to_csv(tmp_path / "beta_series.csv", index=False)
    last = hf.label.index[-1]
    state = read_resume_state(tmp_path, last)
    assert state.hmm is not None
    prior = state.hmm["global"]
    pd.testing.assert_index_equal(prior.probs.index, hf.label.index, check_names=False)
    assert prior.cold_start.dtype == bool


def test_beta_long_round_trips_through_checkpoint(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    cfg = EngineConfig(
        data_path=tiny_xlsx, output_dir=tmp_path / "out", ewma_halflife_days=10,
        return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
        min_n_per_cut=2, bootstrap_min_days=10,
    )
    result = engine_run(
        cfg, fred_client=MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}),
        run_date="r", as_of_data_date="2024-12-31", force=True,
    )
    state = read_resume_state(tmp_path / "out" / "r", pd.Timestamp("2024-12-31"))
    # Exact equality with the in-memory betas proves the CSV round trip is lossless.
    verify_beta_history(beta_long(result.beta), state.beta_series,
                        through=pd.Timestamp("2024-12-31"))


def _long(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.DataFrame(
        {"date": dates, "segment": "global", "scheme": "cap_wtd", "beta": values}
    )


def test_verify_beta_history_accepts_equal_and_nan() -> None:
    old = _long([0.1, float("nan"), 0.3])
    new = _long([0.1, float("nan"), 0.3, 0.4])  # an extra (new) date is ignored
    verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_verify_beta_history_flags_changed_value() -> None:
    old = _long([0.1, 0.2, 0.3])
    new = _long([0.1, 0.2 + 1e-9, 0.3])
    with pytest.raises(HistoryRevisedError, match="2021-01-04 global cap_wtd"):
        verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_verify_beta_history_flags_missing_date() -> None:
    old = _long([0.1, 0.2, 0.3])
    new = old.drop(index=1)
    with pytest.raises(HistoryRevisedError, match="rows differ"):
        verify_beta_history(new, old, through=pd.Timestamp("2021-01-05"))


def test_cut_prices(tiny_xlsx: Path) -> None:
    cut = cut_prices(load_prices(tiny_xlsx), pd.Timestamp("2023-06-30"))
    assert cut.equity_lc.index.max() == pd.Timestamp("2023-06-30")
    assert cut.fi_lc.index.max() == pd.Timestamp("2023-06-30")
