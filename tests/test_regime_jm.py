"""Tests for the causal JM walk-forward + per-segment classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import roro.regime_jm as _jm_mod
from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient
from roro.jump_model import JumpFit
from roro.regime_jm import classify_jm, walk_forward
from roro.types import BetaBySegment, BetaFrame, JmRegimeFrame


def _two_regime_series(n: int = 600) -> pd.Series:
    rng = np.random.default_rng(11)
    lo = rng.normal(-2.0, 0.3, n // 2)
    hi = rng.normal(2.0, 0.3, n - n // 2)
    vals = np.concatenate([lo, hi])
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(vals, index=idx)


def test_walk_forward_returns_hmm_parity_keys() -> None:
    out = walk_forward(_two_regime_series(), jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=252, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert set(out) == {"state", "label", "prob_risk_off", "prob_transitional",
                        "prob_risk_on", "confidence", "cold_start", "refit_dates"}
    assert out["label"].iloc[:252].eq("Unknown").all()         # warmup
    assert bool(out["cold_start"].iloc[:252].all())
    assert out["label"].isin({"Risk-off", "Transitional", "Risk-on", "Unknown"}).all()


def test_walk_forward_deterministic() -> None:
    s = _two_regime_series()
    kw: dict[str, Any] = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["label"], b["label"])
    pd.testing.assert_series_equal(a["state"], b["state"])


def test_walk_forward_no_lookahead() -> None:
    # Label at date t is invariant to appending future rows (causal online inference).
    s = _two_regime_series(600)
    kw: dict[str, Any] = dict(jump_penalty=20.0, refit_interval_days=21, min_history_days=252,
              n_states=3, window="expanding", rolling_window_days=2000,
              continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0)
    full = walk_forward(s, **kw)["label"]
    prefix = walk_forward(s.iloc[:500], **kw)["label"]
    pd.testing.assert_series_equal(full.iloc[:500], prefix)


def test_walk_forward_drops_nan_emits_unknown() -> None:
    s = _two_regime_series(400)
    s.iloc[300] = np.nan
    out = walk_forward(s, jump_penalty=20.0, refit_interval_days=21,
                       min_history_days=120, n_states=3, window="expanding",
                       rolling_window_days=2000, continuous=False, n_init=5,
                       max_iter=30, tol=1e-8, seed=0)
    assert out["label"].iloc[300] == "Unknown"
    assert bool(out["cold_start"].iloc[300])


def test_walk_forward_rolling_no_lookahead_and_deterministic() -> None:
    # Rolling window: causal + deterministic, mirroring the expanding no-lookahead test.
    s = _two_regime_series(500)
    kw: dict[str, Any] = dict(
        jump_penalty=20.0, refit_interval_days=21, min_history_days=200,
        n_states=3, window="rolling", rolling_window_days=120,
        continuous=False, n_init=5, max_iter=30, tol=1e-8, seed=0,
    )
    full = walk_forward(s, **kw)["label"]
    prefix = walk_forward(s.iloc[:400], **kw)["label"]
    pd.testing.assert_series_equal(full.iloc[:400], prefix)  # no lookahead under rolling
    a = walk_forward(s, **kw)
    b = walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["state"], b["state"])    # deterministic


def test_walk_forward_last_good_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # When later refits do not converge, the block must reuse the last converged
    # (centroids, scaler) -- emitting real labels, not NaN/Unknown, and recording
    # only the converged refit date.
    real_fit = _jm_mod.fit_jump_model  # type: ignore[attr-defined]
    calls = {"n": 0}

    def fake_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        f = real_fit(*args, **kwargs)
        if calls["n"] == 1:
            return f  # first refit converges
        return JumpFit(centroids=f.centroids, labels=f.labels,
                       objective=f.objective, converged=False)  # later refits "fail"

    monkeypatch.setattr(_jm_mod, "fit_jump_model", fake_fit)
    out = walk_forward(_two_regime_series(400), jump_penalty=20.0,
                       refit_interval_days=40, min_history_days=120, n_states=3,
                       window="expanding", rolling_window_days=2000, continuous=False,
                       n_init=4, max_iter=30, tol=1e-8, seed=0)
    assert calls["n"] >= 2                                   # multiple refits occurred
    assert len(out["refit_dates"]) == 1                     # only the converged refit recorded
    active = out["label"].iloc[120:]
    assert (active != "Unknown").any()                      # last_good kept producing labels
    assert not active.isna().any()


def _bbs(n: int = 400) -> BetaBySegment:
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(2)
    beta = np.concatenate([rng.normal(-2, 0.3, n // 2), rng.normal(2, 0.3, n - n // 2)])
    cap = pd.DataFrame({"beta": beta, "n": np.full(n, 12)}, index=idx)
    bf = BetaFrame(cap_wtd=cap, eq_wtd=pd.DataFrame(), slope_spread=pd.Series(dtype=float))
    return BetaBySegment(by_segment={"global": bf, "LatAm": bf})


def test_classify_jm_returns_frame() -> None:
    cfg = EngineConfig(data_path=Path("d.xlsx"), output_dir=Path("o"),
                       jm_min_history_days=120, jm_refit_interval_days=40, jm_n_init=4)
    frame = classify_jm(_bbs(), cfg=cfg, thin_cuts=frozenset({"LatAm"}))
    assert isinstance(frame, JmRegimeFrame)
    assert "global" in frame.label.columns and "LatAm" in frame.label.columns
    assert bool(frame.thin_cut_flag["LatAm"].all())
    assert not bool(frame.thin_cut_flag["global"].any())
    assert frame.label["global"].isin({"Risk-off", "Transitional", "Risk-on", "Unknown"}).all()
    assert frame.jump_penalty_used["global"] == cfg.jm_jump_penalty


def test_engine_jm_byte_identical_and_artifacts(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}

    def _run(out: str) -> bytes:
        cfg = EngineConfig(
            data_path=tiny_xlsx, output_dir=tmp_path / out, ewma_halflife_days=10,
            return_window_days=21, tripwire_window_days=10, percentile_window_years=1,
            min_n_per_cut=2, bootstrap_min_days=10, jm_enabled=True,
            jm_min_history_days=120, jm_refit_interval_days=60, jm_n_init=4,
        )
        engine_run(cfg, fred_client=MockFredClient(seeded=seeded),
                   run_date="2024-12-31", as_of_data_date="2024-12-31", force=True)
        return (tmp_path / out / "2024-12-31" / "regimes_jm.csv").read_bytes()

    a = _run("a")
    b = _run("b")
    assert a == b  # AJ-2: byte-identical regimes_jm.csv across runs

    run_dir = tmp_path / "a" / "2024-12-31"
    assert (run_dir / "jm_refit_log.csv").exists()                  # refit log written
    snap = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert "regime_jm" in snap                                      # snapshot block present
    assert "global" in snap["regime_jm"]
    assert "label" in snap["regime_jm"]["global"]
    cfg_resolved = snap["config_resolved"]
    assert cfg_resolved["jm_enabled"] is True                       # jm_* captured in snapshot
    assert "jm_jump_penalty" in cfg_resolved


def test_walk_forward_cjm_deterministic_and_causal() -> None:
    s = _two_regime_series(500)
    kw: dict[str, Any] = dict(jump_penalty=20.0, refit_interval_days=21,
              min_history_days=200, n_states=3, window="expanding",
              rolling_window_days=2000, continuous=True, n_init=4, max_iter=30,
              tol=1e-8, seed=0)
    a, b = walk_forward(s, **kw), walk_forward(s, **kw)
    pd.testing.assert_series_equal(a["prob_risk_off"], b["prob_risk_off"])
    full = walk_forward(s, **kw)["prob_risk_on"]
    prefix = walk_forward(s.iloc[:420], **kw)["prob_risk_on"]
    pd.testing.assert_series_equal(full.iloc[:420], prefix)
