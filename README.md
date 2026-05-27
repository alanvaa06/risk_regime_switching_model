# RoRo Risk-Regime Engine

Daily cross-asset risk-regime classifier across 32 countries × {equity, fixed income} plus 10 segmentation cuts. Diagnostic only (v1.0).

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Run

Either set the env var directly:

```bash
export FRED_API_KEY="..."          # Linux/Mac
$env:FRED_API_KEY = "..."          # Windows PowerShell (current session)
```

Or copy `.env.example` to `.env` and fill in your key (gitignored, persists across sessions):

```bash
cp .env.example .env
# edit .env
```

Then:

```bash
roro run --config configs/default.yaml --date 2026-05-27 --as-of-data-date 2026-05-26
```

Outputs land in `outputs/2026-05-27/`:

- `beta_series.csv` — per-segment β (cap-wtd + eq-wtd + slope spread)
- `regimes.csv` — percentile, tercile, quintile, direction, flags per segment
- `correlation.csv` — avg pairwise + PC1 variance share
- `external_validation.csv` — rolling 60d ρ vs each FRED series
- `tripwire.csv` — 1M fast-signal β mirror
- `alerts.csv` — bucket transitions, disagreement events, validation degradation
- `snapshot.json` — config + data fingerprint + warnings

## Backtest

```bash
roro backtest --config configs/default.yaml --start 2010-01-01 --end 2024-12-31 --assert-gates
```

Acceptance gates per PRD §10 are written to `backtest/acceptance_report.json`.

## Development

```bash
uv run pytest
uv run mypy roro/
uv run ruff check .
```

## Architecture

See `docs/superpowers/specs/2026-05-27-roro-engine-design.md`.
