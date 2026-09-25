"""Top-level build_report orchestrator: run_dir + xlsx -> report.html."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.report.attribution_figs import (
    attribution_bars,
    attribution_scatter,
    attribution_waterfall,
    concentration_timeseries,
    pc1_loadings_bars,
)
from roro.report.figures import (
    beta_band_lookup,
    beta_timeseries,
    regime_probability_area,
    regime_probability_area_jm,
    scatter_beta_return,
    scatter_vol_return,
    vol_breadth_heatmap,
    vol_pct_asset_heatmap,
)
from roro.report.html import FigureSpec, assemble
from roro.report.load import DEFAULT_WINDOW, load_bundle


def _has_rows(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty


def build_report(
    run_dir: Path,
    xlsx_path: Path,
    out_path: Path,
    *,
    window: int = DEFAULT_WINDOW,
    data_until: pd.Timestamp | None = None,
) -> Path:
    """Build a single self-contained HTML report from an engine run dir.

    Args:
        run_dir: engine run directory (snapshot.json + CSVs).
        xlsx_path: source xlsx with Equity_LC + Fixed_Income_LC + Panel.
        out_path: destination HTML file.
        window: trailing business-day window exposed in the bundle.
        data_until: ignore xlsx prices after this date (None = use all rows).

    Returns:
        Path that was written (== out_path).

    Raises:
        ReportInputError: required inputs missing or invalid.
        FileNotFoundError: xlsx_path does not exist.
    """
    bundle = load_bundle(run_dir, xlsx_path, window=window, data_until=data_until)
    specs = [
        FigureSpec(scatter_vol_return(bundle), "fig_scatter_vol", "Risk vs Return"),
        FigureSpec(scatter_beta_return(bundle), "fig_scatter_beta", "Beta vs Return"),
        FigureSpec(beta_timeseries(bundle), "fig_beta_ts", "Segment β with regime bands"),
    ]
    lookup = None
    if bundle.seg_hmm_label is not None:
        specs.append(FigureSpec(
            regime_probability_area(bundle), "fig_hmm_probs", "HMM regime probabilities"
        ))
    if bundle.seg_jm_label is not None:
        specs.append(FigureSpec(
            regime_probability_area_jm(bundle), "fig_jm_probs", "JM regime probabilities"
        ))
    if bundle.seg_hmm_label is not None or bundle.seg_jm_label is not None:
        lookup = beta_band_lookup(bundle)
    specs.append(FigureSpec(
        vol_breadth_heatmap(bundle), "fig_vol_breadth",
        "Volatility breadth (sorted percentile)",
    ))
    specs.append(FigureSpec(
        vol_pct_asset_heatmap(bundle), "fig_vol_assets",
        "Volatility percentile by asset",
    ))
    if _has_rows(bundle.attribution_level):
        specs.append(FigureSpec(attribution_bars(bundle), "fig_attrib_bars", "Slope attribution"))
    if _has_rows(bundle.attribution_delta):
        specs.append(FigureSpec(attribution_waterfall(bundle), "fig_attrib_waterfall",
                                "Δβ̂ waterfall"))
    if _has_rows(bundle.attribution_level):
        specs.append(FigureSpec(attribution_scatter(bundle), "fig_attrib_scatter",
                                "Vol vs return, sized by |contribution|"))
    if _has_rows(bundle.concentration):
        specs.append(FigureSpec(concentration_timeseries(bundle), "fig_concentration_ts",
                                "Concentration of the slope"))
    if _has_rows(bundle.attribution_pc1):
        specs.append(FigureSpec(pc1_loadings_bars(bundle), "fig_pc1_loadings",
                                "PC1 loadings vs variance share"))
    html = assemble(
        specs,
        run_date=bundle.run_date,
        methodology_version=bundle.methodology_version,
        beta_div_id="fig_beta_ts",
        beta_band_lookup=lookup,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
