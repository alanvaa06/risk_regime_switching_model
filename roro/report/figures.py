"""Pure figure builders: scatter pair + segment β time-series."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from roro.report.bundle import DataBundle

SegmentFilter = Literal["Full", "DM", "EM", "EM_Eq", "EM_FI"]

SCATTER_SEGMENTS: tuple[SegmentFilter, ...] = ("Full", "DM", "EM", "EM_Eq", "EM_FI")

COLOR_DM: str = "#1f77b4"
COLOR_EM: str = "#ff7f0e"

_MIN_OBS_FOR_OLS: int = 2


def _series_for_segment(meta: pd.DataFrame, segment: SegmentFilter) -> list[str]:
    """Return the series_id list matching the segment filter."""
    if segment == "Full":
        return list(meta.index)
    if segment == "DM":
        return list(meta.index[meta["segment"] == "DM"])
    if segment == "EM":
        return list(meta.index[meta["segment"] == "EM"])
    if segment == "EM_Eq":
        return list(meta.index[(meta["segment"] == "EM") & (meta["asset"] == "Eq")])
    if segment == "EM_FI":
        return list(meta.index[(meta["segment"] == "EM") & (meta["asset"] == "FI")])
    raise ValueError(f"Unknown segment: {segment}")


def _ols_slope_intercept(
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> tuple[float, float]:
    """Plain OLS y = a*x + b; returns (slope, intercept). Empty input -> (nan, nan)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < _MIN_OBS_FOR_OLS:
        return float("nan"), float("nan")
    xv, yv = x[mask], y[mask]
    slope, intercept = np.polyfit(xv, yv, 1)
    return float(slope), float(intercept)


def _scatter_traces_for_date(
    bundle: DataBundle,
    x_panel: pd.DataFrame,
    y_panel: pd.DataFrame,
    date: pd.Timestamp,
    segment: SegmentFilter,
) -> list[go.Scatter]:
    """Build traces (markers + trend lines) for one (date, segment)."""
    series_ids = _series_for_segment(bundle.meta, segment)
    sub = bundle.meta.loc[series_ids]
    x_row = x_panel.loc[date]
    y_row = y_panel.loc[date]
    traces: list[go.Scatter] = []
    for color_label, color in (("DM", COLOR_DM), ("EM", COLOR_EM)):
        group_ids = list(sub.index[sub["segment"] == color_label])
        if not group_ids:
            continue
        x = x_row.reindex(group_ids).to_numpy(dtype=float)
        y = y_row.reindex(group_ids).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        countries = sub.loc[group_ids, "country"].to_numpy()
        assets = sub.loc[group_ids, "asset"].to_numpy()
        traces.append(
            go.Scatter(
                x=x[valid],
                y=y[valid],
                mode="markers",
                marker={"color": color, "size": 8},
                name=color_label,
                text=[f"{c}_{a}" for c, a in zip(countries[valid], assets[valid], strict=True)],
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )
        slope, intercept = _ols_slope_intercept(x[valid], y[valid])
        if np.isfinite(slope):
            x_line = np.array([np.nanmin(x[valid]), np.nanmax(x[valid])])
            traces.append(
                go.Scatter(
                    x=x_line,
                    y=slope * x_line + intercept,
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    name=f"{color_label} fit",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


def _build_scatter(
    bundle: DataBundle,
    x_panel: pd.DataFrame,
    y_panel: pd.DataFrame,
    *,
    x_title: str,
    y_title: str,
    title_prefix: str,
) -> go.Figure:
    dates = bundle.dates
    default_segment: SegmentFilter = "Full"

    initial_traces = _scatter_traces_for_date(bundle, x_panel, y_panel, dates[-1], default_segment)

    frames: list[go.Frame] = []
    for d in dates:
        frames.append(
            go.Frame(
                name=str(d.date()),
                data=_scatter_traces_for_date(bundle, x_panel, y_panel, d, default_segment),
            )
        )

    # Fixed axis ranges from global panel min/max with 5% padding
    x_all = x_panel.to_numpy(dtype=float)
    y_all = y_panel.to_numpy(dtype=float)
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    slider_steps = [
        {
            "method": "animate",
            "label": str(d.date()),
            "args": [[str(d.date())], {"mode": "immediate", "frame": {"duration": 0}}],
        }
        for d in dates
    ]

    segment_buttons = [
        {
            "method": "animate",
            "label": seg,
            "args": [
                None,
                {"frame": {"duration": 0}, "mode": "immediate", "_seg": seg},
            ],
        }
        for seg in SCATTER_SEGMENTS
    ]

    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=f"{title_prefix} — {dates[-1].date()} — Full",
            xaxis={"title": x_title, "range": [x_min - x_pad, x_max + x_pad]},
            yaxis={"title": y_title, "range": [y_min - y_pad, y_max + y_pad]},
            sliders=[
                {
                    "active": len(dates) - 1,
                    "currentvalue": {"prefix": "Date: "},
                    "steps": slider_steps,
                }
            ],
            updatemenus=[
                {
                    "type": "dropdown",
                    "showactive": True,
                    "buttons": segment_buttons,
                    "x": 1.02,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ],
            template="plotly_white",
        ),
        frames=frames,
    )
    return fig


def scatter_vol_return(bundle: DataBundle) -> go.Figure:
    """Risk-return scatter: x = EWMA vol, y = 3M log return."""
    return _build_scatter(
        bundle,
        x_panel=bundle.vol,
        y_panel=bundle.ret_3m,
        x_title="EWMA annualized volatility",
        y_title="3M total log return",
        title_prefix="Risk vs Return",
    )
