"""Resume primitives for incremental runs: block alignment, checkpoint seeding.

The HMM/JM walk-forwards fit at the start of each refit block on data strictly
before it, then infer the block causally. Rows in closed blocks are therefore a
pure function of the data up to the block's end: they can be copied from a
checkpoint and the loop restarted at the open block, reproducing a full rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from roro.types import SegmentPrior

PROB_COLUMNS: tuple[str, str, str] = ("p_risk_off", "p_transitional", "p_risk_on")


class HistoryRevisedError(RuntimeError):
    """Recomputed history differs from the checkpoint (source data was revised)."""


def resume_block_start(
    index: pd.Index,
    last_date: pd.Timestamp,
    *,
    min_history_days: int,
    refit_interval_days: int,
) -> int | None:
    """Position of the refit block containing the first row after ``last_date``.

    Returns None when nothing can be reused: the checkpoint ends inside the warmup
    (or exactly at the first block start), or there are no new rows.
    """
    n_old = int(index.searchsorted(last_date, side="right"))
    if n_old <= min_history_days or n_old >= len(index):
        return None
    k = (n_old - min_history_days) // refit_interval_days
    return min_history_days + k * refit_interval_days


def seed_prior_rows(
    prior: SegmentPrior, dates: pd.Index
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Checkpoint probabilities (ordered columns) and cold-start flags for ``dates``."""
    missing = dates.difference(prior.probs.index)
    if len(missing) > 0:
        raise HistoryRevisedError(
            f"checkpoint has no overlay row for {pd.Timestamp(missing[0]).date()}"
        )
    probs = prior.probs.loc[dates, list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    cold = prior.cold_start.loc[dates].to_numpy(dtype=bool)
    return probs, cold


def prior_refits_before(prior: SegmentPrior, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    """Converged refit dates strictly before ``cutoff`` (the open block's first date)."""
    return [d for d in prior.refit_dates if d < cutoff]


BETA_TOLERANCE: float = 1e-12
_BETA_KEYS: list[str] = ["date", "segment", "scheme"]


def verify_beta_history(
    current: pd.DataFrame, checkpoint: pd.DataFrame, *, through: pd.Timestamp
) -> None:
    """Raise HistoryRevisedError unless current betas up to ``through`` equal the checkpoint's.

    Both frames are long beta_series layout. Same (date, segment, scheme) rows,
    |diff| <= BETA_TOLERANCE, NaN == NaN.
    """
    cur = (
        current.loc[current["date"] <= through]
        .set_index(_BETA_KEYS)["beta"]
        .astype(float)
        .sort_index()
    )
    old = checkpoint.set_index(_BETA_KEYS)["beta"].astype(float).sort_index()
    if not cur.index.equals(old.index):
        first = cur.index.symmetric_difference(old.index)[0]
        raise HistoryRevisedError(
            f"date/segment rows differ from checkpoint (first: "
            f"{pd.Timestamp(first[0]).date()} {first[1]} {first[2]})"
        )
    same = ((cur - old).abs() <= BETA_TOLERANCE) | (cur.isna() & old.isna())
    if not bool(same.all()):
        d, seg, scheme = same.index[~same.to_numpy()][0]
        raise HistoryRevisedError(f"beta changed on {pd.Timestamp(d).date()} {seg} {scheme}")
