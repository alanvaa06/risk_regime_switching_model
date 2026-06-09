"""Assemble Plotly figures into a single self-contained HTML page."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

_CSS = """
:root { color-scheme: light; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 0;
  background: #fafafa;
  color: #222;
}
.container { max-width: 1280px; margin: 0 auto; padding: 24px; }
header { border-bottom: 1px solid #ddd; margin-bottom: 24px; padding-bottom: 12px; }
header h1 { margin: 0 0 4px; font-size: 24px; }
header .meta { color: #666; font-size: 13px; }
section { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
  padding: 16px; margin-bottom: 24px; }
section h2 { margin: 0 0 12px; font-size: 18px; }
footer { color: #888; font-size: 12px; text-align: center; padding: 16px; }
"""


@dataclass(frozen=True)
class FigureSpec:
    """Explicit per-figure specification: figure, div id, and section title."""

    figure: go.Figure
    div_id: str
    title: str


def _band_toggle_markup(
    beta_div_id: str,
    lookup: dict[str, dict[str, list[dict[str, object]]]],
) -> str:
    """HTML <select> + JS that swaps the beta chart's regime bands by method.

    Segment changes are detected via plotly_relayout (the segment dropdown sets a
    `title` containing the segment name); the method <select> sets the method.
    On either, the beta div is relayouted with lookup[segment][method] shapes.
    """
    payload = json.dumps(lookup)
    return f"""
<div style="max-width:1280px;margin:0 auto;padding:0 24px 12px;">
  <label for="hmm-band-method" style="font-size:13px;color:#444;">Regime bands:</label>
  <select id="hmm-band-method" style="font-size:13px;margin-left:6px;">
    <option value="percentile">Percentile</option>
    <option value="hmm">HMM</option>
    <option value="jm">Jump Model</option>
  </select>
</div>
<script>
(function() {{
  var lookup = {payload};
  var betaId = "{beta_div_id}";
  var seg = "global";
  var method = "percentile";
  function apply() {{
    var div = document.getElementById(betaId);
    if (!div || !window.Plotly) return;
    var byMethod = lookup[seg];
    if (!byMethod) return;
    window.Plotly.relayout(div, {{shapes: byMethod[method] || []}});
  }}
  function wire() {{
    var div = document.getElementById(betaId);
    var sel = document.getElementById("hmm-band-method");
    if (!div || !sel || !window.Plotly) {{ setTimeout(wire, 100); return; }}
    sel.addEventListener("change", function() {{ method = sel.value; apply(); }});
    div.on("plotly_relayout", function(e) {{
      var t = e && (e["title"] || (e["title.text"]));
      if (typeof t === "string" && t.indexOf("—") !== -1) {{
        seg = t.split("—").pop().trim();
        apply();
      }}
    }});
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", wire);
  }} else {{ wire(); }}
}})();
</script>
"""


def assemble(
    specs: list[FigureSpec],
    *,
    run_date: pd.Timestamp,
    methodology_version: str,
    beta_div_id: str | None = None,
    beta_band_lookup: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> str:
    """Assemble figure specs into one self-contained HTML report page.

    Args:
        specs: list of FigureSpec (figure + div_id + title), at least one required.
        run_date: report run date for header.
        methodology_version: engine methodology version string.
        beta_div_id: div_id of the beta timeseries figure (for band toggle wiring).
        beta_band_lookup: mapping of segment -> method -> list of shape dicts.
            When both beta_div_id and beta_band_lookup are provided, a <select> + JS
            toggle is injected directly above the beta-timeseries section.

    Returns:
        Full HTML document string (starts with <!DOCTYPE html>).

    Raises:
        ValueError: if ``specs`` is empty.
    """
    if not specs:
        raise ValueError("assemble requires at least one figure")

    # First fig pulls plotly.js from CDN; subsequent figs reuse it.
    fig_html_blocks: list[str] = []
    for idx, spec in enumerate(specs):
        include: str | bool = "cdn" if idx == 0 else False
        block = spec.figure.to_html(
            include_plotlyjs=include,
            full_html=False,
            div_id=spec.div_id,
            config={"displaylogo": False},
        )
        fig_html_blocks.append(f'<section><h2>{spec.title}</h2>{block}</section>')

    if beta_band_lookup is not None and beta_div_id is not None:
        toggle = _band_toggle_markup(beta_div_id, beta_band_lookup)
        beta_pos = next(
            (i for i, s in enumerate(specs) if s.div_id == beta_div_id), len(fig_html_blocks)
        )
        fig_html_blocks.insert(beta_pos, toggle)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RoRo Risk-Regime Report — {run_date.date()}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<header>
<h1>RoRo Risk-Regime Report</h1>
<div class="meta">Run date: {run_date.date()} &nbsp;|&nbsp;
Methodology: v{methodology_version}</div>
</header>
{''.join(fig_html_blocks)}
<footer>Generated by RoRo · methodology v{methodology_version}</footer>
</div>
</body>
</html>
"""
