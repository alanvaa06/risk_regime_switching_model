"""Pure figure builders: scatter pair + segment β time-series."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from roro.report.bundle import DataBundle

SegmentFilter = Literal["Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI"]

SCATTER_SEGMENTS: tuple[SegmentFilter, ...] = (
    "Full", "DM", "EM", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI",
)

COLOR_DM: str = "#1f77b4"
COLOR_EM: str = "#2ca02c"

_MIN_OBS_FOR_OLS: int = 2
_TRACES_PER_SEGMENT: int = 4  # DM markers + DM fit + EM markers + EM fit


def _series_for_segment(meta: pd.DataFrame, segment: SegmentFilter) -> list[str]:  # noqa: PLR0911
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
    if segment == "DM_Eq":
        return list(meta.index[(meta["segment"] == "DM") & (meta["asset"] == "Eq")])
    if segment == "DM_FI":
        return list(meta.index[(meta["segment"] == "DM") & (meta["asset"] == "FI")])
    raise ValueError(f"Unknown segment: {segment}")


def _ols_slope_intercept(
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> tuple[float, float]:
    """Plain OLS y = a*x + b; returns (slope, intercept). Empty/degenerate input -> (nan, nan)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < _MIN_OBS_FOR_OLS:
        return float("nan"), float("nan")
    xv, yv = x[mask], y[mask]
    # Degenerate x (zero variance) yields a rank-deficient fit; skip silently.
    if np.ptp(xv) == 0.0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(xv, yv, 1)
    return float(slope), float(intercept)


def _scatter_traces_for_date(
    bundle: DataBundle,
    x_panel: pd.DataFrame,
    y_panel: pd.DataFrame,
    date: pd.Timestamp,
    segment: SegmentFilter,
) -> list[go.Scatter]:
    """Build exactly _TRACES_PER_SEGMENT traces for one (date, segment).

    Output order: [DM markers, DM fit, EM markers, EM fit]. Empty partitions
    yield empty-array Scatter placeholders so the trace count is invariant.
    """
    series_ids = _series_for_segment(bundle.meta, segment)
    sub = bundle.meta.loc[series_ids]
    x_row = x_panel.loc[date]
    y_row = y_panel.loc[date]
    traces: list[go.Scatter] = []
    for color_label, color in (("DM", COLOR_DM), ("EM", COLOR_EM)):
        group_ids = list(sub.index[sub["segment"] == color_label])
        if not group_ids:
            traces.append(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker={"color": color, "size": 8},
                    name=f"{segment}:{color_label}",
                    hoverinfo="skip",
                )
            )
            traces.append(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    name=f"{segment}:{color_label} fit",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
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
                name=f"{segment}:{color_label}",
                text=[f"{c}_{a}" for c, a in zip(countries[valid], assets[valid], strict=True)],
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )
        slope, intercept = _ols_slope_intercept(x[valid], y[valid])
        if np.isfinite(slope) and valid.any():
            x_line = np.array([np.nanmin(x[valid]), np.nanmax(x[valid])])
            traces.append(
                go.Scatter(
                    x=x_line,
                    y=slope * x_line + intercept,
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    name=f"{segment}:{color_label} fit",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    name=f"{segment}:{color_label} fit",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


def _all_segment_traces_for_date(
    bundle: DataBundle,
    x_panel: pd.DataFrame,
    y_panel: pd.DataFrame,
    date: pd.Timestamp,
) -> list[go.Scatter]:
    """Concatenate traces for every segment in SCATTER_SEGMENTS, in order."""
    traces: list[go.Scatter] = []
    for seg in SCATTER_SEGMENTS:
        traces.extend(_scatter_traces_for_date(bundle, x_panel, y_panel, date, seg))
    return traces


def _visibility_mask(active: SegmentFilter) -> list[bool]:
    """Return a visibility list of length len(SCATTER_SEGMENTS) * _TRACES_PER_SEGMENT.

    Active segment's traces are True; everything else False.
    """
    mask: list[bool] = []
    for seg in SCATTER_SEGMENTS:
        mask.extend([seg == active] * _TRACES_PER_SEGMENT)
    return mask


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

    initial_traces = _all_segment_traces_for_date(bundle, x_panel, y_panel, dates[-1])
    for visible, trace in zip(_visibility_mask(default_segment), initial_traces, strict=True):
        trace.visible = visible

    frames: list[go.Frame] = []
    for d in dates:
        frames.append(
            go.Frame(
                name=str(d.date()),
                data=_all_segment_traces_for_date(bundle, x_panel, y_panel, d),
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
            "method": "update",
            "label": seg,
            "args": [
                {"visible": _visibility_mask(seg)},
                {"title": f"{title_prefix} — {dates[-1].date()} — {seg}"},
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


def scatter_beta_return(bundle: DataBundle) -> go.Figure:
    """Beta-return scatter: x = β vs cap-wtd global, y = 3M log return."""
    return _build_scatter(
        bundle,
        x_panel=bundle.beta_vs_global,
        y_panel=bundle.ret_3m,
        x_title="β vs cap-weighted global",
        y_title="3M total log return",
        title_prefix="Beta vs Return",
    )


BETA_TS_SEGMENTS: tuple[str, ...] = (
    "global",
    "DM",
    "EM",
    "EM_Eq",
    "EM_FI",
    "DM_Eq",
    "DM_FI",
    "LatAm",
)

REGIME_COLORS: dict[str, str] = {
    "Risk-off": "rgba(220, 50, 47, 0.12)",
    "Transitional": "rgba(128, 128, 128, 0.06)",
    "Risk-on": "rgba(46, 160, 67, 0.12)",
}


def _regime_runs(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Compress a date-indexed label series into consecutive runs of identical labels."""
    s = labels.dropna()
    if s.empty:
        return []
    runs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current_label = s.iloc[0]
    run_start = s.index[0]
    prev_date = s.index[0]
    for d, label in s.iloc[1:].items():
        if label != current_label:
            runs.append((run_start, prev_date, str(current_label)))
            current_label = label
            run_start = d
        prev_date = d
    runs.append((run_start, prev_date, str(current_label)))
    return runs


def beta_timeseries(bundle: DataBundle) -> go.Figure:
    """Segment β line over `dates` with tercile bands shaded.

    Segment dropdown switches both the line trace and the background vrects.
    """
    default_segment = "global"
    available = [s for s in BETA_TS_SEGMENTS if s in bundle.seg_beta.columns]
    if default_segment not in available:
        default_segment = available[0]

    # Initial trace + shapes
    initial_line = go.Scatter(
        x=bundle.dates,
        y=bundle.seg_beta[default_segment].to_numpy(dtype=float),
        mode="lines",
        line={"color": "#222", "width": 2},
        name=default_segment,
        hovertemplate="%{x|%Y-%m-%d}<br>β=%{y:.3f}<extra></extra>",
    )

    initial_shapes = []
    if default_segment in bundle.seg_tercile.columns:
        for start, end, label in _regime_runs(bundle.seg_tercile[default_segment]):
            color = REGIME_COLORS.get(label)
            if color is None:
                continue
            initial_shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": start,
                    "x1": end,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": color,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )

    # Dropdown buttons: each button updates trace y, name, title, and shapes
    buttons = []
    for seg in available:
        y = bundle.seg_beta[seg].to_numpy(dtype=float)
        shapes: list[dict[str, object]] = []
        if seg in bundle.seg_tercile.columns:
            for start, end, label in _regime_runs(bundle.seg_tercile[seg]):
                color = REGIME_COLORS.get(label)
                if color is None:
                    continue
                shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": start,
                        "x1": end,
                        "y0": 0,
                        "y1": 1,
                        "fillcolor": color,
                        "line": {"width": 0},
                        "layer": "below",
                    }
                )
        buttons.append(
            {
                "method": "update",
                "label": seg,
                "args": [
                    {"y": [y], "name": [seg]},
                    {
                        "title": f"Segment β with regime bands — {seg}",
                        "shapes": shapes,
                    },
                ],
            }
        )

    fig = go.Figure(
        data=[initial_line],
        layout=go.Layout(
            title=f"Segment β with regime bands — {default_segment}",
            xaxis={"title": "Date"},
            yaxis={"title": "Cap-weighted β"},
            shapes=initial_shapes,
            updatemenus=[
                {
                    "type": "dropdown",
                    "showactive": True,
                    "buttons": buttons,
                    "x": 1.02,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ],
            template="plotly_white",
        ),
    )
    return fig
