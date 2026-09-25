"""Resume primitives: block alignment, checkpoint row seeding, refit-date filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roro.resume import (
    HistoryRevisedError,
    copied_through,
    prior_refits_before,
    resume_block_start,
    seed_prior_rows,
)
from roro.types import SegmentPrior


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.mark.parametrize(
    ("n_old", "expected"),
    [
        (100, None),  # checkpoint still in warmup -> nothing to reuse
        (252, None),  # first new row is the very first block start -> nothing to reuse
        (253, 252),  # one day into block 0
        (314, 252),  # last day of block 0 already processed, first new row still in block 0
        (315, 315),  # first new row opens block 1 exactly on its boundary
        (316, 315),
        (400, 378),
        (500, None),  # no new rows at all
    ],
)
def test_resume_block_start(n_old: int, expected: int | None) -> None:
    idx = _idx(500)
    got = resume_block_start(
        idx, idx[n_old - 1], min_history_days=252, refit_interval_days=63
    )
    assert got == expected


@pytest.mark.parametrize(
    ("n_old", "expected_pos"),
    [
        (100, None),  # warmup: nothing copied
        (252, None),  # first new row is the first block start: nothing copied
        (253, 251),  # block 0 open -> rows [0, 252) copied, last copied = index[251]
        (316, 314),  # block 1 open at 315
        (500, None),  # no new rows
    ],
)
def test_copied_through(n_old: int, expected_pos: int | None) -> None:
    idx = _idx(500)
    got = copied_through(idx, idx[n_old - 1], min_history_days=252, refit_interval_days=63)
    assert got == (None if expected_pos is None else idx[expected_pos])


def test_copied_through_zero_warmup_block_zero_is_none() -> None:
    idx = _idx(100)
    assert copied_through(idx, idx[9], min_history_days=0, refit_interval_days=63) is None


def _prior(n: int = 10) -> SegmentPrior:
    idx = _idx(n)
    probs = pd.DataFrame(
        {
            "p_risk_off": np.linspace(0.1, 0.2, n),
            "p_transitional": np.linspace(0.3, 0.4, n),
            "p_risk_on": np.linspace(0.6, 0.4, n),
        },
        index=idx,
    )
    cold = pd.Series([True] * 3 + [False] * (n - 3), index=idx)
    refits = (idx[3], idx[6])
    return SegmentPrior(probs=probs, cold_start=cold, refit_dates=refits, last_date=idx[-1])


def test_seed_prior_rows_returns_ordered_columns() -> None:
    prior = _prior()
    probs, cold = seed_prior_rows(prior, prior.probs.index[:5])
    np.testing.assert_array_equal(
        probs, prior.probs[["p_risk_off", "p_transitional", "p_risk_on"]].to_numpy()[:5]
    )
    np.testing.assert_array_equal(cold, [True, True, True, False, False])


def test_seed_prior_rows_missing_date_raises() -> None:
    prior = _prior()
    wanted = pd.DatetimeIndex([prior.probs.index[0], pd.Timestamp("2030-01-01")])
    with pytest.raises(HistoryRevisedError, match="2030-01-01"):
        seed_prior_rows(prior, wanted)


def test_prior_refits_before_includes_last_copied() -> None:
    prior = _prior()
    idx = prior.probs.index
    assert prior_refits_before(prior, idx[5]) == [idx[3]]
    assert prior_refits_before(prior, idx[6]) == [idx[3], idx[6]]
    assert prior_refits_before(prior, idx[2]) == []
    assert prior_refits_before(prior, None) == []
