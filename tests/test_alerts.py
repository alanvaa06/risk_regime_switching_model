import pandas as pd

from roro.alerts import detect_alerts
from roro.types import CorrelationFrame, RegimeFrame, ValidationFrame


def _regime() -> RegimeFrame:
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
    return RegimeFrame(
        percentile_5y=pct,
        tercile=tercile,
        quintile=tercile,
        direction=tercile,
        n_per_segment=pct,
        thin_cut_flag=other,
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
