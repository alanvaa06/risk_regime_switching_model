"""Pure figure builders for the regime-attribution artifacts (all deterministic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from roro.report.bundle import DataBundle
from roro.report.figures import _band_shapes

QUADRANT_COLORS: dict[str, str] = {
    "HI/+": "#2ca02c",  # classic risk-on
    "HI/-": "#d62728",  # classic risk-off
    "LO/+": "#ff7f0e",  # flight-to-quality
    "LO/-": "#1f77b4",  # havens sold
}
_EFFECTS: tuple[tuple[str, str, str], ...] = (
    ("effect_return", "return", "#1f77b4"),
    ("effect_position", "position", "#ff7f0e"),
    ("effect_interaction", "interaction", "#9467bd"),
    ("effect_universe", "universe", "#7f7f7f"),
)
_BLOCK_ORDER: tuple[str, ...] = ("DM_Eq", "EM_Eq", "DM_FI", "EM_FI")
_HEIGHT: int = 700
_FONT: dict[str, object] = {"family": "system-ui, -apple-system, sans-serif", "size": 13}
_MENU: dict[str, object] = {
    "type": "dropdown", "showactive": True, "x": 1.12, "y": 1.0,
    "xanchor": "left", "yanchor": "top",
}
_TOP_N_DEFAULT: int = 12
_MARKER_MIN: float = 6.0
_MARKER_MAX: float = 40.0
_CUT_ORDER: tuple[str, ...] = (
    "global", "DM", "EM", "Equity", "FI", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI", "LatAm",
)


def _layout(title: str, buttons: list[dict[str, object]], **axes: object) -> go.Layout:
    return go.Layout(
        title=title, height=_HEIGHT, template="simple_white", font=_FONT,
        margin={"l": 80, "r": 220, "t": 60, "b": 80},
        updatemenus=[{**_MENU, "buttons": buttons}],
        **axes,
    )


def _combo_key(combo: tuple[str, str]) -> tuple[int, int]:
    cut, weighting = combo
    rank = _CUT_ORDER.index(cut) if cut in _CUT_ORDER else len(_CUT_ORDER)
    return (rank, 0 if weighting == "cap" else 1)


def _combos(df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(set(zip(df["cut"], df["weighting"], strict=True)), key=_combo_key)


def _quadrant_colors(sub: pd.DataFrame) -> list[str]:
    return [QUADRANT_COLORS[q] for q in sub["quadrant"]]


def attribution_bars(bundle: DataBundle, *, top_n: int = _TOP_N_DEFAULT) -> go.Figure:
    """Horizontal bars of the top-N |c_i| for one (cut, weighting), colored by quadrant."""
    assert bundle.attribution_level is not None
    level = bundle.attribution_level
    per: dict[tuple[str, str], pd.DataFrame] = {}
    for combo in _combos(level):
        sub = level[(level["cut"] == combo[0]) & (level["weighting"] == combo[1])]
        order = sub["contribution"].abs().sort_values(ascending=True, kind="stable").index
        per[combo] = sub.reindex(order).tail(top_n)
    default = ("global", "cap") if ("global", "cap") in per else next(iter(per))

    def _title(combo: tuple[str, str]) -> str:
        full = level[(level["cut"] == combo[0]) & (level["weighting"] == combo[1])]
        beta = float(full["contribution"].sum())
        top1 = float(full["share"].abs().max()) if not full.empty else float("nan")
        return (f"Slope attribution — {combo[0]} · {combo[1]} · β̂ = {beta:.3f} · "
                f"max |share| = {top1:.0%}")

    def _custom(sub: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return np.column_stack([sub["quadrant"], sub["share"], sub["vol"], sub["ret3m"]])

    d0 = per[default]
    trace = go.Bar(
        x=d0["contribution"], y=d0["series"], orientation="h",
        marker={"color": _quadrant_colors(d0)},
        customdata=_custom(d0),
        hovertemplate=("%{y}<br>c=%{x:.4f}<br>share=%{customdata[1]:.1%}<br>"
                       "quadrant=%{customdata[0]}<br>vol=%{customdata[2]:.2%} "
                       "ret3m=%{customdata[3]:.2%}<extra></extra>"),
    )
    buttons: list[dict[str, object]] = [
        {
            "method": "update",
            "label": f"{c[0]} · {c[1]}",
            "args": [
                {
                    "x": [per[c]["contribution"]], "y": [per[c]["series"]],
                    "marker": [{"color": _quadrant_colors(per[c])}],
                    "customdata": [_custom(per[c])],
                },
                {"title": _title(c)},
            ],
        }
        for c in per
    ]
    return go.Figure(
        data=[trace],
        layout=_layout(
            _title(default), buttons,
            xaxis={"title": "contribution to β̂ (share of β̂ can exceed 100%)"},
            yaxis={"title": "asset"},
        ),
    )


def attribution_waterfall(bundle: DataBundle) -> go.Figure:
    """Δβ̂ from anchor to today, split into 4 effects per block; menu = cut·weighting·horizon."""
    assert bundle.attribution_delta is not None
    delta = bundle.attribution_delta
    keys = sorted(
        set(zip(delta["cut"], delta["weighting"], delta["horizon"], strict=True)),
        key=lambda k: (_combo_key((k[0], k[1])), 0 if k[2] == "anchor" else 1),
    )
    per: dict[tuple[str, str, str], pd.DataFrame] = {}
    meta: dict[tuple[str, str, str], pd.Series] = {}
    for k in keys:
        sub = delta[
            (delta["cut"] == k[0]) & (delta["weighting"] == k[1]) & (delta["horizon"] == k[2])
        ]
        g = sub.groupby("block")[[e[0] for e in _EFFECTS]].sum()
        per[k] = g.reindex([b for b in _BLOCK_ORDER if b in g.index])
        meta[k] = sub.iloc[0]
    default = ("global", "cap", "anchor") if ("global", "cap", "anchor") in per else keys[0]

    def _title(k: tuple[str, str, str]) -> str:
        m = meta[k]
        a = pd.Timestamp(m["anchor_date"]).strftime("%Y-%m-%d")
        return (f"Δβ̂ waterfall — {k[0]} · {k[1]} · {k[2]} anchor {a} "
                f"({m['label_anchor']} → {m['label_t']}) · "
                f"β̂ {float(m['beta_anchor']):.3f} → {float(m['beta_t']):.3f}")

    g0 = per[default]
    traces = [
        go.Bar(name=label, x=list(g0.index), y=g0[col], marker={"color": color})
        for col, label, color in _EFFECTS
    ]
    buttons: list[dict[str, object]] = [
        {
            "method": "update",
            "label": f"{k[0]} · {k[1]} · {k[2]}",
            "args": [
                {"x": [list(per[k].index)] * len(_EFFECTS),
                 "y": [per[k][col] for col, _, _ in _EFFECTS]},
                {"title": _title(k)},
            ],
        }
        for k in keys
    ]
    layout = _layout(
        _title(default), buttons,
        xaxis={"title": "block"}, yaxis={"title": "Δβ̂ contribution"},
    )
    layout.barmode = "relative"
    return go.Figure(data=traces, layout=layout)


def _xbar_shape(xbar: float) -> dict[str, object]:
    return {"type": "line", "x0": xbar, "x1": xbar, "y0": 0, "y1": 1, "xref": "x",
            "yref": "paper", "line": {"color": "#999", "dash": "dot", "width": 1}}


def attribution_scatter(bundle: DataBundle) -> go.Figure:
    """Vol vs 3M return, marker size ∝ |c_i|, color by quadrant, exact WLS line + x̄ marker."""
    assert bundle.attribution_level is not None
    level = bundle.attribution_level
    per = {
        c: level[(level["cut"] == c[0]) & (level["weighting"] == c[1])].sort_values("series")
        for c in _combos(level)
    }
    default = ("global", "cap") if ("global", "cap") in per else next(iter(per))

    def _size(sub: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        a = sub["contribution"].abs().to_numpy(dtype=float)
        top = float(a.max()) if a.size and a.max() > 0 else 1.0
        return _MARKER_MIN + (_MARKER_MAX - _MARKER_MIN) * a / top

    def _line(sub: pd.DataFrame) -> tuple[list[float], list[float]]:
        beta = float(sub["contribution"].sum())
        xbar, ybar = float(sub["xbar"].iloc[0]), float(sub["ybar"].iloc[0])
        xs = [float(sub["vol"].min()), float(sub["vol"].max())]
        return xs, [ybar + beta * (x - xbar) for x in xs]

    def _marker(sub: pd.DataFrame) -> dict[str, object]:
        return {"size": _size(sub), "color": _quadrant_colors(sub),
                "line": {"width": 0.5, "color": "#333"}}

    s0 = per[default]
    xs0, ys0 = _line(s0)
    traces = [
        go.Scatter(
            x=s0["vol"], y=s0["ret3m"], mode="markers", text=s0["series"], name="assets",
            marker=_marker(s0),
            hovertemplate="%{text}<br>vol=%{x:.2%} ret3m=%{y:.2%}<extra></extra>",
        ),
        go.Scatter(x=xs0, y=ys0, mode="lines", name="WLS line (regime slope)",
                   line={"color": "#333", "dash": "dash"}),
    ]
    buttons: list[dict[str, object]] = []
    for c, sub in per.items():
        xs, ys = _line(sub)
        buttons.append({
            "method": "update",
            "label": f"{c[0]} · {c[1]}",
            "args": [
                {"x": [sub["vol"], xs], "y": [sub["ret3m"], ys], "text": [sub["series"], None],
                 "marker": [_marker(sub), {}]},
                {"title": f"Vol vs return, sized by |contribution| — {c[0]} · {c[1]}",
                 "shapes": [_xbar_shape(float(sub["xbar"].iloc[0]))]},
            ],
        })
    layout = _layout(
        f"Vol vs return, sized by |contribution| — {default[0]} · {default[1]}", buttons,
        xaxis={"title": "EWMA vol (annualized)"}, yaxis={"title": "3M log return"},
    )
    layout.shapes = [_xbar_shape(float(s0["xbar"].iloc[0]))]
    return go.Figure(data=traces, layout=layout)


def concentration_timeseries(bundle: DataBundle) -> go.Figure:
    """top-1 share + HHI of |c_i| over time (cap-weighted) with hysteresis-smoothed bands."""
    assert bundle.concentration is not None
    conc = bundle.concentration[bundle.concentration["weighting"] == "cap"]
    present = set(conc["cut"])
    cuts = [c for c in bundle.seg_tercile.columns if c in present]
    per = {c: conc[conc["cut"] == c].sort_values("date", kind="stable") for c in cuts}
    default = "global" if "global" in per else cuts[0]
    s0 = per[default]
    traces = [
        go.Scatter(x=s0["date"], y=s0["top1_share"], mode="lines", name="top-1 share",
                   line={"color": "#d62728"}),
        go.Scatter(x=s0["date"], y=s0["hhi"], mode="lines", name="HHI", line={"color": "#1f77b4"}),
    ]
    buttons: list[dict[str, object]] = [
        {
            "method": "update",
            "label": c,
            "args": [
                {"x": [per[c]["date"]] * 2, "y": [per[c]["top1_share"], per[c]["hhi"]]},
                {"title": f"Concentration of the slope — {c}",
                 "shapes": _band_shapes(bundle.seg_tercile[c], smooth=True)},
            ],
        }
        for c in cuts
    ]
    layout = _layout(
        f"Concentration of the slope — {default}", buttons,
        xaxis={"title": "Date"}, yaxis={"title": "share of Σ|c|", "range": (0.0, 1.0)},
    )
    layout.shapes = _band_shapes(bundle.seg_tercile[default], smooth=True)
    return go.Figure(data=traces, layout=layout)


def pc1_loadings_bars(bundle: DataBundle) -> go.Figure:
    """Per-asset PC1 loading² vs variance share, sorted by decoupling (havens on the right)."""
    assert bundle.attribution_pc1 is not None
    pc1 = bundle.attribution_pc1
    cuts = sorted(set(pc1["cut"]), key=lambda c: _combo_key((c, "cap")))
    per = {c: pc1[pc1["cut"] == c].sort_values("decoupling", kind="stable") for c in cuts}
    default = "global" if "global" in per else cuts[0]
    s0 = per[default]
    traces = [
        go.Bar(name="PC1 loading²", x=s0["series"], y=s0["pc1_load_sq"],
               marker={"color": "#1f77b4"}),
        go.Bar(name="variance share", x=s0["series"], y=s0["var_share"],
               marker={"color": "#ff7f0e"}),
    ]
    buttons: list[dict[str, object]] = [
        {
            "method": "update",
            "label": c,
            "args": [
                {"x": [per[c]["series"]] * 2, "y": [per[c]["pc1_load_sq"], per[c]["var_share"]]},
                {"title": f"PC1 loadings vs variance share (sorted by decoupling) — {c}"},
            ],
        }
        for c in cuts
    ]
    layout = _layout(
        f"PC1 loadings vs variance share (sorted by decoupling) — {default}", buttons,
        xaxis={"title": "asset"}, yaxis={"title": "share"},
    )
    layout.barmode = "group"
    return go.Figure(data=traces, layout=layout)
