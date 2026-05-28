"""Render the three report figures to static PNGs for the README.

Requires Plotly's static-image backend ``kaleido`` (install ad hoc:
``uv pip install kaleido``). Note: kaleido 0.2.1 is known to deadlock on some
Windows setups; this script renders reliably on Linux / CI.

Usage:
    uv pip install kaleido
    uv run python scripts/render_report_images.py --run-dir outputs/2026-05-27
"""

from __future__ import annotations

import argparse
from pathlib import Path

from roro.report.figures import (
    beta_timeseries,
    scatter_beta_return,
    scatter_vol_return,
)
from roro.report.load import load_bundle

ASSETS_DIR = Path("docs/assets")
IMAGE_WIDTH = 1100
IMAGE_HEIGHT = 700
IMAGE_SCALE = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, default=Path("data.xlsx"))
    parser.add_argument("--window", type=int, default=252)
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.run_dir, args.xlsx, window=args.window)

    figures = {
        "report_risk_return.png": scatter_vol_return(bundle),
        "report_beta_return.png": scatter_beta_return(bundle),
        "report_beta_timeseries.png": beta_timeseries(bundle),
    }
    for name, fig in figures.items():
        out_path = ASSETS_DIR / name
        fig.write_image(
            str(out_path),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            scale=IMAGE_SCALE,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
