import pandas as pd

from roro.alerts import detect_alerts
from roro.types import (
    AttributionFrame,
    CorrelationFrame,
    HmmRegimeFrame,
    JmRegimeFrame,
    RegimeFrame,
    ValidationFrame,
)


def _regime(*, thin: frozenset[str] = frozenset()) -> RegimeFrame:
    """Regime fixture; ``thin`` names the cuts flagged as thin in ``thin_cut_flag``."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    tercile = pd.DataFrame(
        {
            "global": ["Transitional"] * 8 + ["Risk-off"] * 2,
            "DM_Eq": ["Risk-on"] * 5 + ["Risk-off"] * 5,
        },
        index=idx,
    )
    pct = pd.DataFrame(0.5, index=idx, columns=tercile.columns)
    other = pd.DataFrame(False, index=idx, columns=tercile.columns)
    thin_flag = other.copy()
    for cut in thin:
        thin_flag[cut] = True
    return RegimeFrame(
        percentile_5y=pct,
        tercile=tercile,
        quintile=tercile,
        direction=tercile,
        n_per_segment=pct,
        thin_cut_flag=thin_flag,
        bootstrap_flag=other,
    )


def test_bucket_transitions_picked_up() -> None:
    out = detect_alerts(
        regime=_regime(),
        correlation=CorrelationFrame(
            avg_pairwise_3m=pd.DataFrame(),
            pc1_variance_share=pd.DataFrame(),
        ),
        validation=ValidationFrame(
            rolling_corr_60d=pd.DataFrame(),
            internal_consistency=pd.DataFrame(),
            correlation_alerts=pd.DataFrame(),
        ),
    )
    bt = out.bucket_transitions
    assert {"global", "DM_Eq"}.issubset(set(bt["segment"]))


def test_detect_alerts_emits_hmm_transitions() -> None:
    idx = pd.bdate_range("2014-01-01", periods=3)
    empty = pd.DataFrame(index=idx)
    rf = RegimeFrame(percentile_5y=empty, tercile=empty, quintile=empty,
                     direction=empty, n_per_segment=empty, thin_cut_flag=empty,
                     bootstrap_flag=empty)
    labels = pd.DataFrame({"global": ["Risk-off", "Risk-off", "Risk-on"]}, index=idx)
    hmm = HmmRegimeFrame(state=labels, label=labels, prob_risk_off=empty,
                         prob_transitional=empty, prob_risk_on=empty, confidence=empty,
                         n_per_segment=empty, thin_cut_flag=empty, cold_start_flag=empty)
    cf = CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty)
    vf = ValidationFrame(rolling_corr_60d=empty, internal_consistency=empty,
                         correlation_alerts=empty)
    out = detect_alerts(regime=rf, correlation=cf, validation=vf, regime_hmm=hmm)
    assert (out.hmm_bucket_transitions["to_bucket"] == "Risk-on").any()


def test_detect_alerts_emits_jm_transitions() -> None:
    idx = pd.bdate_range("2014-01-01", periods=3)
    empty = pd.DataFrame(index=idx)
    rf = RegimeFrame(percentile_5y=empty, tercile=empty, quintile=empty,
                     direction=empty, n_per_segment=empty, thin_cut_flag=empty,
                     bootstrap_flag=empty)
    labels = pd.DataFrame({"global": ["Risk-off", "Risk-off", "Risk-on"]}, index=idx)
    jm = JmRegimeFrame(state=labels, label=labels, prob_risk_off=empty,
                       prob_transitional=empty, prob_risk_on=empty, confidence=empty,
                       n_per_segment=empty, thin_cut_flag=empty, cold_start_flag=empty)
    cf = CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty)
    vf = ValidationFrame(rolling_corr_60d=empty, internal_consistency=empty,
                         correlation_alerts=empty)
    out = detect_alerts(regime=rf, correlation=cf, validation=vf, regime_jm=jm)
    trans = out.jm_bucket_transitions
    assert not trans.empty
    assert {"date", "segment", "from_bucket", "to_bucket"}.issubset(trans.columns)
    assert (trans["to_bucket"] == "Risk-on").any()


def _empty_corr_val() -> tuple[CorrelationFrame, ValidationFrame]:
    return (
        CorrelationFrame(avg_pairwise_3m=pd.DataFrame(), pc1_variance_share=pd.DataFrame()),
        ValidationFrame(rolling_corr_60d=pd.DataFrame(), internal_consistency=pd.DataFrame(),
                        correlation_alerts=pd.DataFrame()),
    )


def test_concentration_alerts_on_transition_day_and_fragile() -> None:
    rf = _regime()  # global flips Transitional->Risk-off on idx[8]; DM_Eq on idx[5]
    idx = rf.tercile.index
    conc = pd.DataFrame(
        [
            # global, transition day, concentrated -> alert (transition_day)
            {"date": idx[8], "cut": "global", "weighting": "cap", "top1_series": "A__Eq",
             "top1_share": 0.7, "hhi": 0.5, "fragile_flag": False},
            # global, calm day, concentrated but not fragile -> no alert
            {"date": idx[3], "cut": "global", "weighting": "cap", "top1_series": "A__Eq",
             "top1_share": 0.7, "hhi": 0.5, "fragile_flag": False},
            # DM_Eq, calm day, fragile -> alert (fragile)
            {"date": idx[2], "cut": "DM_Eq", "weighting": "cap", "top1_series": "B__FI",
             "top1_share": 0.3, "hhi": 0.2, "fragile_flag": True},
            # eq weighting never alerts
            {"date": idx[8], "cut": "global", "weighting": "eq", "top1_series": "A__Eq",
             "top1_share": 0.9, "hhi": 0.8, "fragile_flag": True},
        ]
    )
    af = AttributionFrame(level=pd.DataFrame(), delta=pd.DataFrame(), rollup=pd.DataFrame(),
                          concentration=conc, pc1=pd.DataFrame())
    corr, val = _empty_corr_val()
    out = detect_alerts(regime=rf, correlation=corr, validation=val, attribution=af,
                        top1_alert=0.5)
    ca = out.concentration_alerts
    assert list(ca.columns) == ["date", "segment", "weighting", "top1_series", "top1_share",
                                "hhi", "fragile_flag", "trigger"]
    assert set(zip(ca["segment"], ca["trigger"], strict=True)) == {
        ("global", "transition_day"), ("DM_Eq", "fragile"),
    }


def test_concentration_alerts_suppress_fragile_on_thin_cuts() -> None:
    # DM_Eq is a thin cut: its fragility flag must not raise an alert, but a
    # transition-day concentration alert on the same cut still must (spec section 11).
    rf = _regime(thin=frozenset({"DM_Eq"}))  # DM_Eq flips Risk-on->Risk-off on idx[5]
    idx = rf.tercile.index
    conc = pd.DataFrame(
        [
            # DM_Eq, calm day, fragile, but thin cut -> suppressed
            {"date": idx[2], "cut": "DM_Eq", "weighting": "cap", "top1_series": "B__FI",
             "top1_share": 0.3, "hhi": 0.2, "fragile_flag": True},
            # DM_Eq, transition day, concentrated -> alert survives on a thin cut
            {"date": idx[5], "cut": "DM_Eq", "weighting": "cap", "top1_series": "B__FI",
             "top1_share": 0.7, "hhi": 0.5, "fragile_flag": False},
            # global is not thin: its fragile row still alerts
            {"date": idx[3], "cut": "global", "weighting": "cap", "top1_series": "A__Eq",
             "top1_share": 0.3, "hhi": 0.2, "fragile_flag": True},
        ]
    )
    af = AttributionFrame(level=pd.DataFrame(), delta=pd.DataFrame(), rollup=pd.DataFrame(),
                          concentration=conc, pc1=pd.DataFrame())
    corr, val = _empty_corr_val()
    out = detect_alerts(regime=rf, correlation=corr, validation=val, attribution=af,
                        top1_alert=0.5)
    ca = out.concentration_alerts
    assert set(zip(ca["segment"], ca["trigger"], strict=True)) == {
        ("DM_Eq", "transition_day"), ("global", "fragile"),
    }
    assert not ((ca["segment"] == "DM_Eq") & (ca["trigger"] == "fragile")).any()


def test_concentration_alerts_empty_without_attribution() -> None:
    corr, val = _empty_corr_val()
    out = detect_alerts(regime=_regime(), correlation=corr, validation=val)
    assert out.concentration_alerts.empty
