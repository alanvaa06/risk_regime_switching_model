"""Assemble three or four Plotly figures into a single self-contained HTML page."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

DIV_IDS: tuple[str, ...] = ("fig_scatter_vol", "fig_scatter_beta", "fig_beta_ts", "fig_hmm_probs")
SECTION_TITLES: tuple[str, ...] = (
    "Risk vs Return",
    "Beta vs Return",
    "Segment β with regime bands",
    "HMM regime probabilities",
)
_MIN_FIGURES: int = 3
_MAX_FIGURES: int = 4

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
    figures: list[go.Figure],
    *,
    run_date: pd.Timestamp,
    methodology_version: str,
    beta_div_index: int | None = None,
    beta_band_lookup: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> str:
    """Assemble three or four figures into one self-contained HTML page.

    Args:
        figures: list of 3 or 4 Plotly figures
            (scatter_vol_return, scatter_beta_return, beta_timeseries[, hmm_probs]).
        run_date: report run date for header.
        methodology_version: engine methodology version string.
        beta_div_index: index into figures of the beta timeseries figure (for toggle wiring).
        beta_band_lookup: mapping of segment -> method -> list of shape dicts.
            When provided, a <select> + JS toggle is injected after the figures.

    Returns:
        Full HTML document string (starts with <!DOCTYPE html>).

    Raises:
        ValueError: if `figures` does not contain 3-4 entries.
    """
    if not _MIN_FIGURES <= len(figures) <= _MAX_FIGURES:
        raise ValueError(
            f"assemble requires {_MIN_FIGURES}-{_MAX_FIGURES} figures, got {len(figures)}"
        )

    # First fig pulls plotly.js from CDN; subsequent figs reuse it.
    fig_html_blocks: list[str] = []
    for idx, fig in enumerate(figures):
        div_id = DIV_IDS[idx]
        title = SECTION_TITLES[idx]
        include: str | bool = "cdn" if idx == 0 else False
        block = fig.to_html(
            include_plotlyjs=include,
            full_html=False,
            div_id=div_id,
            config={"displaylogo": False},
        )
        fig_html_blocks.append(f'<section><h2>{title}</h2>{block}</section>')

    toggle_html = ""
    if beta_band_lookup is not None and beta_div_index is not None:
        toggle_html = _band_toggle_markup(DIV_IDS[beta_div_index], beta_band_lookup)

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
{toggle_html}<footer>Generated by RoRo · methodology v{methodology_version}</footer>
</div>
</body>
</html>
"""
