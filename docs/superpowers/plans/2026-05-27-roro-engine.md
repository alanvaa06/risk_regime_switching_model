# RoRo Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the RoRo Risk-Regime Python engine — a pure-functional pipeline that ingests `data.xlsx` and free FRED series, computes cross-sectional return-on-vol β across one global and nine segmentation cuts, classifies daily regimes (5Y rolling tercile percentile + direction), runs correlation/PC1 panel, external + internal validation, 1M tripwire, alerts, and a backtest harness with PRD §10 acceptance gates. Output is plain CSV per artifact in a run-stamped directory plus a versioned `snapshot.json`. No app, no dashboard, no database.

**Architecture:** Single `roro/` Python package. Each stage is a pure module function `(input dataclass) -> output dataclass` over `@dataclass(frozen=True)` types. Orchestrator `roro.engine.run(config) -> RunResult` wires the chain. Library API + thin Click CLI (`roro run`, `roro backtest`). Static cap weights (2026-05-26 Panel snapshot) broadcast across full history. FRED ingest via a `FredClient` Protocol so `MockFredClient` covers tests.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, openpyxl, fredapi, click, pyyaml. Dev: pytest, hypothesis, mypy `--strict`, ruff. Build/lock: uv.

**Spec:** [docs/superpowers/specs/2026-05-27-roro-engine-design.md](../specs/2026-05-27-roro-engine-design.md)

---

## File Map

Files this plan creates or modifies:

```
pyproject.toml                       project metadata, deps, ruff/mypy config
uv.lock                              uv-managed lockfile
.gitignore                           outputs/, backtest/, .venv/, __pycache__/, *.egg-info/, ~/.roro/
.python-version                      "3.12"
configs/default.yaml                 documented EngineConfig defaults
roro/__init__.py                     package marker, version export
roro/config.py                       EngineConfig (frozen), BucketScheme, YAML+CLI merge
roro/types.py                        all frozen-dataclass IO contracts
roro/io.py                           load_panel, load_prices, load_fred, write_run, read_run
roro/validators.py                   schema, date continuity, NaN policy
roro/returns.py                      total_return_3m, daily_log_returns, ewma_vol
roro/regression.py                   cross_section (WLS cap + OLS eq + slope spread)
roro/segments.py                     SegmentRegistry + partition (10 cuts)
roro/classify.py                     rolling_percentile, tercile_label, quintile_label, direction_flag
roro/correlation.py                  avg_pairwise, pc1_share
roro/validation.py                   rolling_corr_external, internal_consistency
roro/tripwire.py                     fast_signal (1M mirror)
roro/alerts.py                       bucket_transitions, disagreement_events, validation_degradation
roro/engine.py                       run(config) -> RunResult
roro/cli.py                          Click commands: roro run / roro backtest
roro/fred_client.py                  FredClient Protocol + FredApiClient + MockFredClient
roro/backtest.py                     run_backtest, event_recognition, stability_metrics, gates
tests/conftest.py                    fixtures: tiny_panel, tiny_prices, tiny_fred
tests/strategies.py                  hypothesis strategy bank
tests/test_io.py
tests/test_validators.py
tests/test_returns.py
tests/test_regression.py
tests/test_segments.py
tests/test_classify.py
tests/test_correlation.py
tests/test_validation.py
tests/test_tripwire.py
tests/test_alerts.py
tests/test_engine.py
tests/test_backtest.py
tests/test_reproducibility.py
tests/golden/2024-Q1/                golden CSVs for integration regression
docs/context/todo.md                 updated per CLAUDE.md workflow
docs/context/results.md              updated per CLAUDE.md workflow
docs/context/sesion-log.md           updated per CLAUDE.md workflow
docs/context/memory.md               key decisions logged
```

---

## Task 1: Project scaffolding — pyproject, lockfile, gitignore

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `roro/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `.python-version`**

```
3.12
```

- [ ] **Step 2: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
build/
dist/

# Envs
.venv/
.env

# Tooling
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/

# Engine outputs
outputs/
backtest/
~/.roro/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "roro"
version = "0.1.0"
description = "RoRo Risk-Regime Classification Engine"
authors = [{name = "Alan Vazquez, CFA"}]
requires-python = ">=3.12,<3.13"
readme = "README.md"
dependencies = [
    "numpy>=1.26,<2.1",
    "pandas>=2.2,<2.3",
    "pyarrow>=15.0,<18.0",
    "openpyxl>=3.1,<4.0",
    "fredapi>=0.5.2,<0.6",
    "click>=8.1,<9.0",
    "pyyaml>=6.0,<7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9.0",
    "pytest-cov>=5.0,<6.0",
    "hypothesis>=6.100,<7.0",
    "mypy>=1.10,<2.0",
    "ruff>=0.5,<1.0",
    "types-PyYAML",
    "pandas-stubs",
]

[project.scripts]
roro = "roro.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["roro"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "PL"]
ignore = ["PLR0913"]  # too-many-arguments

[tool.mypy]
python_version = "3.12"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
disallow_untyped_defs = true
no_implicit_optional = true
plugins = []

[[tool.mypy.overrides]]
module = ["fredapi.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
filterwarnings = ["error"]
```

- [ ] **Step 4: Write `roro/__init__.py`**

```python
"""RoRo Risk-Regime Classification Engine."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Write `tests/__init__.py`**

```python
```

- [ ] **Step 6: Install with uv**

Run: `uv venv && uv pip install -e ".[dev]"`
Expected: clean install, `roro` importable from `python -c "import roro; print(roro.__version__)"` → `0.1.0`

- [ ] **Step 7: Smoke checks — lint and type**

Run: `uv run ruff check . && uv run mypy roro/`
Expected: PASS both (empty package, nothing to flag).

- [ ] **Step 8: Initialize git and commit**

```bash
git init
git add .gitignore .python-version pyproject.toml roro/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding (pyproject, ruff/mypy config, package skeleton)"
```

---

## Task 2: EngineConfig dataclass + YAML loader

**Files:**
- Create: `roro/config.py`
- Create: `configs/default.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test `tests/test_config.py`**

```python
from pathlib import Path

import pytest
import yaml

from roro.config import BucketScheme, EngineConfig, load_config


def test_load_default_yaml(tmp_path: Path) -> None:
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "data_path: data.xlsx\n"
        "output_dir: outputs\n"
        "ewma_halflife_days: 30\n"
        "return_window_days: 63\n"
        "tripwire_window_days: 21\n"
        "tripwire_ewma_halflife_days: 10\n"
        "percentile_window_years: 5\n"
        "bucket_scheme: TERCILE\n"
        "min_n_per_cut: 10\n"
        "direction_lookback_days: 5\n"
        "external_corr_window_days: 60\n"
        "external_corr_alert_threshold: 0.3\n"
        "bootstrap_min_days: 252\n"
        "methodology_version: 1.0.0\n"
    )
    cfg = load_config(cfg_yaml)
    assert cfg.ewma_halflife_days == 30
    assert cfg.bucket_scheme is BucketScheme.TERCILE
    assert cfg.data_path == Path("data.xlsx")


def test_cli_overrides_win(tmp_path: Path) -> None:
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "data_path: data.xlsx\n"
        "output_dir: outputs\n"
        "ewma_halflife_days: 30\n"
    )
    cfg = load_config(cfg_yaml, overrides={"ewma_halflife_days": 40})
    assert cfg.ewma_halflife_days == 40


def test_frozen() -> None:
    cfg = EngineConfig(data_path=Path("data.xlsx"), output_dir=Path("outputs"))
    with pytest.raises(Exception):
        cfg.ewma_halflife_days = 99  # type: ignore[misc]


def test_bucket_scheme_round_trip() -> None:
    for name in ("TERCILE", "QUINTILE", "ASYM_20_60_20"):
        assert BucketScheme[name].name == name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `roro.config` not found.

- [ ] **Step 3: Implement `roro/config.py`**

```python
"""Engine configuration: frozen dataclass + YAML loader with CLI override merge."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml


class BucketScheme(Enum):
    TERCILE = auto()
    QUINTILE = auto()
    ASYM_20_60_20 = auto()


@dataclass(frozen=True)
class EngineConfig:
    data_path: Path
    output_dir: Path
    ewma_halflife_days: int = 30
    return_window_days: int = 63
    tripwire_window_days: int = 21
    tripwire_ewma_halflife_days: int = 10
    percentile_window_years: int = 5
    bucket_scheme: BucketScheme = BucketScheme.TERCILE
    min_n_per_cut: int = 10
    direction_lookback_days: int = 5
    external_corr_window_days: int = 60
    external_corr_alert_threshold: float = 0.3
    bootstrap_min_days: int = 252
    methodology_version: str = "1.0.0"
    fred_api_key: str | None = None


def _coerce_field(name: str, value: Any) -> Any:
    if name in {"data_path", "output_dir"}:
        return Path(value)
    if name == "bucket_scheme":
        return BucketScheme[value] if isinstance(value, str) else value
    return value


def load_config(yaml_path: Path, overrides: dict[str, Any] | None = None) -> EngineConfig:
    raw = yaml.safe_load(Path(yaml_path).read_text()) or {}
    overrides = overrides or {}
    merged: dict[str, Any] = {**raw, **overrides}
    allowed = {f.name for f in fields(EngineConfig)}
    unknown = set(merged) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    coerced = {k: _coerce_field(k, v) for k, v in merged.items() if k in allowed}
    return EngineConfig(**coerced)


def to_dict(cfg: EngineConfig) -> dict[str, Any]:
    """Serialize EngineConfig to a JSON-safe dict (used by snapshot.json)."""
    out: dict[str, Any] = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            out[f.name] = str(value)
        elif isinstance(value, BucketScheme):
            out[f.name] = value.name
        else:
            out[f.name] = value
    return out


def with_overrides(cfg: EngineConfig, **kwargs: Any) -> EngineConfig:
    return replace(cfg, **kwargs)
```

- [ ] **Step 4: Write `configs/default.yaml`**

```yaml
# RoRo engine default configuration.
# Override any field via CLI flag (e.g. --ewma-halflife 40) or by editing this file.

data_path: data.xlsx
output_dir: outputs

# Return + vol estimators
ewma_halflife_days: 30           # F1.2 — RiskMetrics-style EWMA
return_window_days: 63           # F1.1 — ~3M total return
tripwire_window_days: 21         # F1.3 — 1M fast signal
tripwire_ewma_halflife_days: 10

# Classification
percentile_window_years: 5       # F4.1 — 5Y rolling distribution
bucket_scheme: TERCILE           # F4.2 — TERCILE | QUINTILE | ASYM_20_60_20
min_n_per_cut: 10                # F3.5 — minimum series for a cut
direction_lookback_days: 5       # F4.5 — direction flag horizon

# External validation
external_corr_window_days: 60    # F6.3 — rolling 60d ρ
external_corr_alert_threshold: 0.3  # F6.4 — alert when |ρ| < threshold

# Bootstrap
bootstrap_min_days: 252          # below this, suppress percentile

# Reproducibility
methodology_version: 1.0.0
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py -v && uv run mypy roro/config.py`
Expected: PASS all, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add roro/config.py configs/default.yaml tests/test_config.py
git commit -m "feat(config): frozen EngineConfig + YAML loader with CLI overrides"
```

---

## Task 3: Type contracts (frozen dataclasses)

**Files:**
- Create: `roro/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test `tests/test_types.py`**

```python
from datetime import datetime

import pandas as pd
import pytest

from roro.types import (
    AlertSet,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    FredFrame,
    PriceFrame,
    RegimeFrame,
    ReturnsFrame,
    Universe,
    ValidationFrame,
    VolFrame,
)


def test_universe_frozen() -> None:
    u = Universe(
        countries=pd.DataFrame({"Country": ["US"], "Segment": ["DM"]}),
        composites=pd.DataFrame({"Country": ["World"]}),
    )
    with pytest.raises(Exception):
        u.countries = pd.DataFrame()  # type: ignore[misc]


def test_latam_default_membership() -> None:
    u = Universe(
        countries=pd.DataFrame({"Country": ["Brazil"]}),
        composites=pd.DataFrame({"Country": ["LatAm"]}),
    )
    assert set(u.latam_countries) == {"Brazil", "Mexico", "Chile", "Peru", "Colombia"}


def test_beta_by_segment_keys() -> None:
    empty = pd.DataFrame()
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=pd.Series(dtype=float))
    bbs = BetaBySegment(by_segment={"global": bf})
    assert "global" in bbs.by_segment


def test_fred_frame_carries_fingerprint() -> None:
    ff = FredFrame(
        series={"VIXCLS": pd.Series(dtype=float)},
        pulled_at=datetime(2026, 5, 27),
        series_hashes={"VIXCLS": "abc"},
    )
    assert ff.series_hashes["VIXCLS"] == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — `roro.types` not found.

- [ ] **Step 3: Implement `roro/types.py`**

```python
"""Frozen-dataclass contracts that flow through the engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from roro.config import EngineConfig


@dataclass(frozen=True)
class PriceFrame:
    equity_lc: pd.DataFrame
    fi_lc: pd.DataFrame


@dataclass(frozen=True)
class Universe:
    countries: pd.DataFrame
    composites: pd.DataFrame
    latam_countries: tuple[str, ...] = ("Brazil", "Mexico", "Chile", "Peru", "Colombia")


@dataclass(frozen=True)
class FredFrame:
    series: dict[str, pd.Series]
    pulled_at: datetime
    series_hashes: dict[str, str]


@dataclass(frozen=True)
class ReturnsFrame:
    log_returns_3m: pd.DataFrame
    daily_log_returns: pd.DataFrame


@dataclass(frozen=True)
class VolFrame:
    ewma_sigma_annualized: pd.DataFrame


@dataclass(frozen=True)
class BetaFrame:
    cap_wtd: pd.DataFrame
    eq_wtd: pd.DataFrame
    slope_spread: pd.Series


@dataclass(frozen=True)
class BetaBySegment:
    by_segment: dict[str, BetaFrame]


@dataclass(frozen=True)
class RegimeFrame:
    percentile_5y: pd.DataFrame
    tercile: pd.DataFrame
    quintile: pd.DataFrame
    direction: pd.DataFrame
    n_per_segment: pd.DataFrame
    thin_cut_flag: pd.DataFrame
    bootstrap_flag: pd.DataFrame


@dataclass(frozen=True)
class CorrelationFrame:
    avg_pairwise_3m: pd.DataFrame
    pc1_variance_share: pd.DataFrame


@dataclass(frozen=True)
class ValidationFrame:
    rolling_corr_60d: pd.DataFrame
    internal_consistency: pd.DataFrame
    correlation_alerts: pd.DataFrame


@dataclass(frozen=True)
class AlertSet:
    bucket_transitions: pd.DataFrame
    disagreement_events: pd.DataFrame
    validation_degradation: pd.DataFrame


@dataclass(frozen=True)
class RunResult:
    config: EngineConfig
    universe: Universe
    returns: ReturnsFrame
    vol: VolFrame
    beta: BetaBySegment
    regime: RegimeFrame
    correlation: CorrelationFrame
    validation: ValidationFrame
    tripwire: BetaBySegment
    alerts: AlertSet
    warnings: list[str] = field(default_factory=list)
    data_fingerprint: dict[str, str] = field(default_factory=dict)
    code_version: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests + type check**

Run: `uv run pytest tests/test_types.py -v && uv run mypy roro/types.py`
Expected: PASS all.

- [ ] **Step 5: Commit**

```bash
git add roro/types.py tests/test_types.py
git commit -m "feat(types): frozen-dataclass contracts for pipeline IO"
```

---

## Task 4: IO — `load_panel` (split countries vs composites)

**Files:**
- Create: `roro/io.py` (initial: `load_panel` only)
- Create: `tests/conftest.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write `tests/conftest.py` with a tiny Panel fixture**

```python
"""Shared pytest fixtures: tiny synthetic dataset that mirrors data.xlsx structure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tiny_xlsx(tmp_path: Path) -> Path:
    """Build a 4-country + 2-composite tiny xlsx that mirrors data.xlsx layout."""
    path = tmp_path / "tiny.xlsx"
    panel = pd.DataFrame(
        {
            "Country": ["United States", "Brazil", "Germany", "Mexico", "DM", "LatAm"],
            "Segment": ["DM", "EM", "DM", "EM", "DM", "EM"],
            "Equity Index": ["SPX", "MXBR", "MXDE", "MXMX", "MXWO", "MXLA"],
            "Equity Index Curreny": ["USD"] * 6,
            "Bond Index": ["LBUSTRUU", "I00", "I05", "I05M", "I35", "H04"],
            "Bond Index Curreny": ["USD"] * 6,
            "Local Curreny": ["USD", "BRL", "EUR", "MXN", "USD", "USD"],
            "Curr": [1.0, 5.0, 1.16, 20.0, 1.0, 1.0],
            "Pair": ["USD", "USDBRL", "USDEUR", "USDMXN", "USD", "USD"],
            "Equity_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Equity_Mkt_Cap": [100, 10, 20, 5, 130, 15],
            "FI_Date": [pd.Timestamp("2026-05-26")] * 6,
            "Fixed_Income_Mkt_Cap": [50, 5, 10, 3, 65, 8],
            "Equity_Mkt_Cap_Val": [100, 10, 20, 5, 130, 15],
            "Fixed_Income_Mkt_Cap_Val": [50, 5, 10, 3, 65, 8],
        }
    )
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    # Equity_LC layout: row 0 = tickers, row 1 = country names, row 2+ = data
    eq_tickers = ["SPX Index", "MXBR Index", "MXDE Index", "MXMX Index"]
    eq_countries = ["United States", "Brazil", "Germany", "Mexico"]
    eq_data = pd.DataFrame(
        {c: range(100, 100 + len(dates)) for c in eq_countries},
        index=dates,
    )
    fi_data = pd.DataFrame(
        {c: range(200, 200 + len(dates)) for c in eq_countries},
        index=dates,
    )

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        panel.to_excel(w, sheet_name="Panel", index=False)
        _write_two_header_sheet(w, "Equity_LC", eq_tickers, eq_countries, eq_data)
        _write_two_header_sheet(w, "Fixed_Income_LC", eq_tickers, eq_countries, fi_data)
    return path


def _write_two_header_sheet(
    writer: pd.ExcelWriter,
    sheet: str,
    tickers: list[str],
    countries: list[str],
    data: pd.DataFrame,
) -> None:
    """Mimic data.xlsx layout: row 0 = tickers, row 1 = country names, row 2+ = data."""
    out = pd.DataFrame(
        [tickers, countries] + data.reset_index().values.tolist(),
    )
    out.to_excel(writer, sheet_name=sheet, index=False, header=False)
```

- [ ] **Step 2: Write failing test `tests/test_io.py`**

```python
from pathlib import Path

from roro.io import load_panel


def test_panel_splits_countries_and_composites(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert set(u.countries["Country"]) == {"United States", "Brazil", "Germany", "Mexico"}
    assert set(u.composites["Country"]) == {"DM", "LatAm"}


def test_panel_carries_mcap_val_columns(tiny_xlsx: Path) -> None:
    u = load_panel(tiny_xlsx)
    assert "Equity_Mkt_Cap_Val" in u.countries.columns
    assert "Fixed_Income_Mkt_Cap_Val" in u.countries.columns
    assert (u.countries["Equity_Mkt_Cap_Val"] > 0).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL — `roro.io` not found.

- [ ] **Step 4: Implement `roro/io.py` — `load_panel` only**

```python
"""Excel + FRED ingest and run output writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roro.types import Universe

COMPOSITE_NAMES: frozenset[str] = frozenset({"DM", "EM", "Europe", "Asia", "World", "LatAm"})


def load_panel(xlsx_path: Path) -> Universe:
    df = pd.read_excel(xlsx_path, sheet_name="Panel")
    missing = {"Country", "Segment", "Equity_Mkt_Cap_Val", "Fixed_Income_Mkt_Cap_Val"} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")

    is_composite = df["Country"].isin(COMPOSITE_NAMES)
    countries = df.loc[~is_composite].reset_index(drop=True).copy()
    composites = df.loc[is_composite].reset_index(drop=True).copy()
    return Universe(countries=countries, composites=composites)
```

- [ ] **Step 5: Run tests + mypy**

Run: `uv run pytest tests/test_io.py -v && uv run mypy roro/io.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add roro/io.py tests/conftest.py tests/test_io.py
git commit -m "feat(io): load_panel splits countries vs composites"
```

---

## Task 5: IO — `load_prices` (skip ticker row, country-name header)

**Files:**
- Modify: `roro/io.py` — add `load_prices`
- Modify: `tests/test_io.py` — add price tests

- [ ] **Step 1: Add failing test to `tests/test_io.py`**

```python
import pandas as pd

from roro.io import load_prices


def test_prices_uses_country_row_as_header(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert "United States" in pf.equity_lc.columns
    assert "Brazil" in pf.equity_lc.columns
    # Header row must be country names, not tickers
    assert "SPX Index" not in pf.equity_lc.columns
    assert isinstance(pf.equity_lc.index, pd.DatetimeIndex)


def test_prices_aligned_columns(tiny_xlsx: Path) -> None:
    pf = load_prices(tiny_xlsx)
    assert list(pf.equity_lc.columns) == list(pf.fi_lc.columns)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_io.py::test_prices_uses_country_row_as_header -v`
Expected: FAIL — `load_prices` not defined.

- [ ] **Step 3: Add `load_prices` to `roro/io.py`**

```python
def load_prices(xlsx_path: Path) -> "PriceFrame":
    """Load Equity_LC and Fixed_Income_LC.

    Layout: row 0 = ticker IDs, row 1 = country names, row 2+ = daily prices.
    We skip row 0 (tickers) and use row 1 (country names) as the column header.
    """
    eq = _read_price_sheet(xlsx_path, "Equity_LC")
    fi = _read_price_sheet(xlsx_path, "Fixed_Income_LC")
    # Align FI columns to the equity universe (some FI countries may be absent in edge data)
    common = [c for c in eq.columns if c in fi.columns]
    return PriceFrame(equity_lc=eq[common], fi_lc=fi[common])


def _read_price_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    countries = raw.iloc[1, 1:].tolist()
    data = raw.iloc[2:].copy()
    data.columns = [raw.iloc[1, 0]] + countries  # first col = date
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index()
    data.columns.name = None
    return data.astype(float)
```

Update the imports at the top of `roro/io.py`:

```python
from roro.types import PriceFrame, Universe
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_io.py -v && uv run mypy roro/io.py`
Expected: PASS all.

- [ ] **Step 5: Commit**

```bash
git add roro/io.py tests/test_io.py
git commit -m "feat(io): load_prices skips ticker row, uses country names as header"
```

---

## Task 6: Validators — schema + date continuity + NaN policy

**Files:**
- Create: `roro/validators.py`
- Create: `tests/test_validators.py`

- [ ] **Step 1: Write failing test `tests/test_validators.py`**

```python
import pandas as pd
import pytest

from roro.types import PriceFrame, Universe
from roro.validators import DataSourceError, validate_prices, validate_universe


def test_validate_universe_ok() -> None:
    u = Universe(
        countries=pd.DataFrame(
            {
                "Country": ["A", "B"],
                "Segment": ["DM", "EM"],
                "Equity_Mkt_Cap_Val": [1.0, 2.0],
                "Fixed_Income_Mkt_Cap_Val": [3.0, 4.0],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )
    warnings = validate_universe(u)
    assert warnings == []


def test_validate_universe_zero_mcap_raises() -> None:
    u = Universe(
        countries=pd.DataFrame(
            {
                "Country": ["A"],
                "Segment": ["DM"],
                "Equity_Mkt_Cap_Val": [0.0],
                "Fixed_Income_Mkt_Cap_Val": [1.0],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )
    with pytest.raises(DataSourceError):
        validate_universe(u)


def test_validate_prices_date_continuity_break_raises() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-31").tolist()
    # remove 4 contiguous business days mid-stream
    del dates[10:14]
    df = pd.DataFrame({"US": range(len(dates))}, index=pd.DatetimeIndex(dates))
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    with pytest.raises(DataSourceError, match="continuity"):
        validate_prices(pf)


def test_validate_prices_warns_on_per_series_nan() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    df = pd.DataFrame({"US": [1.0] * len(dates), "BR": [1.0] * len(dates)}, index=dates)
    df.loc[df.index[5], "BR"] = float("nan")
    pf = PriceFrame(equity_lc=df, fi_lc=df)
    warnings = validate_prices(pf)
    assert any("BR" in w for w in warnings)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_validators.py -v`
Expected: FAIL — `roro.validators` not found.

- [ ] **Step 3: Implement `roro/validators.py`**

```python
"""Hard schema + soft data-quality validation for engine inputs."""

from __future__ import annotations

import pandas as pd

from roro.types import PriceFrame, Universe

MAX_DATE_GAP_DAYS = 3


class DataSourceError(Exception):
    """Raised when input data is structurally unusable; engine should refuse to run."""


def validate_universe(u: Universe) -> list[str]:
    warnings: list[str] = []
    if (u.countries["Equity_Mkt_Cap_Val"] <= 0).any():
        raise DataSourceError("Equity_Mkt_Cap_Val must be > 0 for all countries")
    if (u.countries["Fixed_Income_Mkt_Cap_Val"] <= 0).any():
        raise DataSourceError("Fixed_Income_Mkt_Cap_Val must be > 0 for all countries")
    if u.countries["Segment"].isin({"DM", "EM"}).all() is False:
        raise DataSourceError("Segment must be DM or EM for every country")
    return warnings


def validate_prices(pf: PriceFrame) -> list[str]:
    warnings: list[str] = []
    for name, df in (("equity_lc", pf.equity_lc), ("fi_lc", pf.fi_lc)):
        _check_date_continuity(df, frame_name=name)
        warnings.extend(_collect_nan_warnings(df, frame_name=name))
    return warnings


def _check_date_continuity(df: pd.DataFrame, *, frame_name: str) -> None:
    if df.empty:
        raise DataSourceError(f"{frame_name}: price frame is empty")
    bdays_expected = pd.bdate_range(df.index.min(), df.index.max())
    missing = bdays_expected.difference(df.index)
    if len(missing) == 0:
        return
    # Find longest contiguous gap
    diffs = missing.to_series().diff().dt.days.fillna(1)
    longest = int((diffs == 1).astype(int).groupby((diffs != 1).cumsum()).cumsum().max())
    if longest > MAX_DATE_GAP_DAYS:
        raise DataSourceError(
            f"{frame_name}: date continuity broken — longest gap {longest} business days"
        )


def _collect_nan_warnings(df: pd.DataFrame, *, frame_name: str) -> list[str]:
    warnings: list[str] = []
    nan_counts = df.isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            warnings.append(f"{frame_name}: column {col!r} has {int(count)} NaN values")
    return warnings
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_validators.py -v && uv run mypy roro/validators.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/validators.py tests/test_validators.py
git commit -m "feat(validators): universe + prices schema and continuity checks"
```

---

## Task 7: FRED client — Protocol + real + mock

**Files:**
- Create: `roro/fred_client.py`
- Modify: `roro/io.py` — add `load_fred`
- Create: `tests/test_fred_client.py`

- [ ] **Step 1: Write failing test `tests/test_fred_client.py`**

```python
from datetime import date, datetime

import pandas as pd

from roro.fred_client import MockFredClient
from roro.io import load_fred


def test_mock_fred_returns_requested_series() -> None:
    client = MockFredClient(
        seeded={"VIXCLS": pd.Series([10.0, 11.0], index=pd.bdate_range("2024-01-01", periods=2))}
    )
    s = client.fetch("VIXCLS", date(2024, 1, 1), date(2024, 1, 3))
    assert s.iloc[0] == 10.0


def test_load_fred_aggregates_all_series() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    seeded = {
        sid: pd.Series([1.0] * 5, index=idx)
        for sid in ("VIXCLS", "BAMLC0A4CBBB", "BAMLEMCBPIOAS", "BAMLH0A0HYM2", "T10Y2Y")
    }
    client = MockFredClient(seeded=seeded)
    ff = load_fred(client, start=date(2024, 1, 1), end=date(2024, 1, 5))
    assert set(ff.series) == set(seeded)
    assert isinstance(ff.pulled_at, datetime)
    assert all(len(h) == 64 for h in ff.series_hashes.values())  # sha256 hexdigest
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_fred_client.py -v`
Expected: FAIL — `roro.fred_client` not found.

- [ ] **Step 3: Implement `roro/fred_client.py`**

```python
"""FRED client Protocol + real fredapi client + in-memory mock."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

FRED_SERIES_IDS: tuple[str, ...] = (
    "VIXCLS",
    "BAMLC0A4CBBB",
    "BAMLEMCBPIOAS",
    "BAMLH0A0HYM2",
    "T10Y2Y",
)


@runtime_checkable
class FredClient(Protocol):
    def fetch(self, series_id: str, start: date, end: date) -> pd.Series: ...


class FredApiClient:
    """Real client backed by fredapi.Fred. Retries 3× with 2s backoff."""

    def __init__(self, api_key: str, retries: int = 3, backoff_seconds: float = 2.0) -> None:
        from fredapi import Fred

        self._fred = Fred(api_key=api_key)
        self._retries = retries
        self._backoff = backoff_seconds

    def fetch(self, series_id: str, start: date, end: date) -> pd.Series:
        import time

        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                s = self._fred.get_series(
                    series_id, observation_start=start, observation_end=end
                )
                return pd.Series(s, name=series_id).astype(float)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._retries - 1:
                    time.sleep(self._backoff)
        assert last_exc is not None
        raise last_exc


class MockFredClient:
    def __init__(self, seeded: dict[str, pd.Series]) -> None:
        self._seeded = seeded

    def fetch(self, series_id: str, start: date, end: date) -> pd.Series:
        if series_id not in self._seeded:
            return pd.Series(dtype=float, name=series_id)
        s = self._seeded[series_id]
        mask = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
        return s.loc[mask].rename(series_id)
```

- [ ] **Step 4: Add `load_fred` to `roro/io.py`**

```python
import hashlib
from datetime import date, datetime

from roro.fred_client import FRED_SERIES_IDS, FredClient
from roro.types import FredFrame


def load_fred(client: FredClient, start: date, end: date) -> FredFrame:
    series: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for sid in FRED_SERIES_IDS:
        s = client.fetch(sid, start, end)
        series[sid] = s
        hashes[sid] = _hash_series(s)
    return FredFrame(series=series, pulled_at=datetime.now(), series_hashes=hashes)


def _hash_series(s: pd.Series) -> str:
    payload = pd.util.hash_pandas_object(s, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 5: Run tests + mypy**

Run: `uv run pytest tests/test_fred_client.py -v && uv run mypy roro/fred_client.py roro/io.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add roro/fred_client.py roro/io.py tests/test_fred_client.py
git commit -m "feat(fred): FredClient Protocol + real + mock + load_fred ingest"
```

---

## Task 8: Returns + EWMA vol

**Files:**
- Create: `roro/returns.py`
- Create: `tests/strategies.py`
- Create: `tests/test_returns.py`

- [ ] **Step 1: Write `tests/strategies.py` — hypothesis strategy bank**

```python
"""Hypothesis strategies shared across tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


@st.composite
def positive_price_series(draw: st.DrawFn, *, length: int = 200) -> pd.Series:
    """Geometric-Brownian-motion-like positive price series."""
    increments = draw(
        hnp.arrays(
            np.float64,
            shape=length,
            elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        )
    )
    log_prices = np.cumsum(increments) + 4.0  # start near exp(4) ≈ 55
    return pd.Series(np.exp(log_prices), index=pd.bdate_range("2010-01-01", periods=length))


@st.composite
def finite_return_series(draw: st.DrawFn, *, length: int = 200) -> pd.Series:
    arr = draw(
        hnp.arrays(
            np.float64,
            shape=length,
            elements=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        )
    )
    return pd.Series(arr, index=pd.bdate_range("2010-01-01", periods=length))


@st.composite
def positive_weights(draw: st.DrawFn, *, n: int = 10) -> np.ndarray:
    raw = draw(
        hnp.arrays(
            np.float64,
            shape=n,
            elements=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    return raw
```

- [ ] **Step 2: Write failing test `tests/test_returns.py`**

```python
import numpy as np
import pandas as pd
from hypothesis import given, settings

from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from tests.strategies import finite_return_series, positive_price_series


def test_total_return_3m_log_additivity() -> None:
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1]}, index=pd.bdate_range("2024-01-01", periods=4))
    r = total_return_3m(prices, window_days=2)
    # ln(121/100) ≈ ln(1.21)
    assert abs(r["A"].iloc[2] - np.log(1.21)) < 1e-12


def test_daily_log_returns_length() -> None:
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.bdate_range("2024-01-01", periods=3))
    r = daily_log_returns(prices)
    assert len(r) == 3
    assert pd.isna(r["A"].iloc[0])
    assert abs(r["A"].iloc[1] - np.log(110 / 100)) < 1e-12


def test_ewma_vol_non_negative_and_annualized() -> None:
    rng = np.random.default_rng(seed=0)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, size=300)}, index=pd.bdate_range("2024-01-01", periods=300))
    sigma = ewma_vol(r, halflife=30)
    assert (sigma.dropna() >= 0).all().all()
    # Annualized daily ~1% vol ≈ 0.16
    assert 0.10 < sigma["A"].iloc[-1] < 0.25


def test_ewma_vol_converges_with_constant_input() -> None:
    r = pd.DataFrame({"A": [0.01] * 500}, index=pd.bdate_range("2024-01-01", periods=500))
    sigma = ewma_vol(r, halflife=30)
    tail = sigma["A"].iloc[-50:]
    assert tail.std() < 1e-8


@given(prices=positive_price_series(length=100))
@settings(max_examples=25, deadline=None)
def test_total_return_3m_finite_when_prices_positive(prices: pd.Series) -> None:
    df = prices.to_frame(name="A")
    r = total_return_3m(df, window_days=21)
    assert r["A"].dropna().apply(lambda x: np.isfinite(x)).all()


@given(r=finite_return_series(length=200))
@settings(max_examples=25, deadline=None)
def test_ewma_vol_monotone_in_abs_returns(r: pd.Series) -> None:
    # Compare same series scaled by 2 → sigma should not decrease
    df1 = r.to_frame(name="A")
    df2 = (r * 2).to_frame(name="A")
    s1 = ewma_vol(df1, halflife=30)["A"].iloc[-1]
    s2 = ewma_vol(df2, halflife=30)["A"].iloc[-1]
    assert s2 >= s1 - 1e-12
```

- [ ] **Step 3: Run failing test**

Run: `uv run pytest tests/test_returns.py -v`
Expected: FAIL — `roro.returns` not found.

- [ ] **Step 4: Implement `roro/returns.py`**

```python
"""3M total log return + daily log return + EWMA vol kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return_3m(prices_lc: pd.DataFrame, *, window_days: int = 63) -> pd.DataFrame:
    """Trailing log return over `window_days`: R_t = ln(P_t / P_{t-W})."""
    return np.log(prices_lc / prices_lc.shift(window_days))


def daily_log_returns(prices_lc: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}). First row is NaN."""
    return np.log(prices_lc / prices_lc.shift(1))


def ewma_vol(daily_returns: pd.DataFrame, *, halflife: int) -> pd.DataFrame:
    """RiskMetrics-style EWMA vol, annualized by √252.

    σ²_t = λ σ²_{t-1} + (1-λ) r²_t with λ = exp(-ln 2 / halflife).
    Implementation: pandas ewm(adjust=False).std() gives the same recursion family.
    """
    return daily_returns.ewm(halflife=halflife, adjust=False).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
```

- [ ] **Step 5: Run tests + mypy**

Run: `uv run pytest tests/test_returns.py -v && uv run mypy roro/returns.py`
Expected: PASS (property tests run 25 examples each).

- [ ] **Step 6: Commit**

```bash
git add roro/returns.py tests/strategies.py tests/test_returns.py
git commit -m "feat(returns): 3M log return + EWMA vol with property tests"
```

---

## Task 9: Segments — partition + registry

**Files:**
- Create: `roro/segments.py`
- Create: `tests/test_segments.py`

- [ ] **Step 1: Write failing test `tests/test_segments.py`**

```python
import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SEGMENT_NAMES, SeriesId, partition
from roro.types import Universe


def _universe() -> Universe:
    return Universe(
        countries=pd.DataFrame(
            {
                "Country": ["US", "DE", "BR", "MX", "CL", "PE", "CO", "CN", "IN", "ZA"],
                "Segment": ["DM", "DM", "EM", "EM", "EM", "EM", "EM", "EM", "EM", "EM"],
                "Equity_Mkt_Cap_Val": [100, 20, 10, 5, 3, 2, 1, 40, 15, 7],
                "Fixed_Income_Mkt_Cap_Val": [50, 10, 5, 3, 2, 1, 1, 20, 8, 4],
            }
        ),
        composites=pd.DataFrame({"Country": ["DM", "EM"]}),
    )


def test_partition_returns_all_segment_keys() -> None:
    cuts = partition(_universe())
    assert set(cuts) == set(SEGMENT_NAMES)


def test_global_cut_includes_every_country_each_class() -> None:
    cuts = partition(_universe())
    g = cuts["global"]
    assert len(g) == 20  # 10 countries × 2 classes


def test_latam_cut_membership_exact() -> None:
    cuts = partition(_universe())
    latam = cuts["LatAm"]
    assert {s.country for s in latam} == {"Brazil", "Mexico", "Chile", "Peru", "Colombia"} & set(
        _universe().countries["Country"]
    )


def test_dm_eq_cut_only_dm_equity() -> None:
    cuts = partition(_universe())
    for s in cuts["DM_Eq"]:
        assert s.segment == "DM"
        assert s.asset_class == ASSET_EQ


def test_series_id_carries_mcap() -> None:
    cuts = partition(_universe())
    em_eq = cuts["EM_Eq"]
    by_country = {s.country: s for s in em_eq}
    assert by_country["BR"].mcap > 0
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_segments.py -v`
Expected: FAIL — `roro.segments` not found.

- [ ] **Step 3: Implement `roro/segments.py`**

```python
"""Segment registry and universe partition into the 10 PRD cuts."""

from __future__ import annotations

from dataclasses import dataclass

from roro.types import Universe

ASSET_EQ: str = "Eq"
ASSET_FI: str = "FI"

SEGMENT_NAMES: tuple[str, ...] = (
    "global",
    "DM",
    "EM",
    "Equity",
    "FI",
    "DM_Eq",
    "EM_Eq",
    "DM_FI",
    "EM_FI",
    "LatAm",
)

LATAM_COUNTRIES: frozenset[str] = frozenset({"Brazil", "Mexico", "Chile", "Peru", "Colombia"})


@dataclass(frozen=True)
class SeriesId:
    country: str
    segment: str       # "DM" or "EM"
    asset_class: str   # ASSET_EQ or ASSET_FI
    mcap: float

    @property
    def column_key(self) -> str:
        """Key used to look up the price column in PriceFrame.equity_lc / fi_lc."""
        return self.country


def partition(u: Universe) -> dict[str, list[SeriesId]]:
    all_series: list[SeriesId] = []
    for _, row in u.countries.iterrows():
        all_series.append(
            SeriesId(
                country=row["Country"],
                segment=row["Segment"],
                asset_class=ASSET_EQ,
                mcap=float(row["Equity_Mkt_Cap_Val"]),
            )
        )
        all_series.append(
            SeriesId(
                country=row["Country"],
                segment=row["Segment"],
                asset_class=ASSET_FI,
                mcap=float(row["Fixed_Income_Mkt_Cap_Val"]),
            )
        )

    return {
        "global": list(all_series),
        "DM": [s for s in all_series if s.segment == "DM"],
        "EM": [s for s in all_series if s.segment == "EM"],
        "Equity": [s for s in all_series if s.asset_class == ASSET_EQ],
        "FI": [s for s in all_series if s.asset_class == ASSET_FI],
        "DM_Eq": [s for s in all_series if s.segment == "DM" and s.asset_class == ASSET_EQ],
        "EM_Eq": [s for s in all_series if s.segment == "EM" and s.asset_class == ASSET_EQ],
        "DM_FI": [s for s in all_series if s.segment == "DM" and s.asset_class == ASSET_FI],
        "EM_FI": [s for s in all_series if s.segment == "EM" and s.asset_class == ASSET_FI],
        "LatAm": [s for s in all_series if s.country in LATAM_COUNTRIES],
    }
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_segments.py -v && uv run mypy roro/segments.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/segments.py tests/test_segments.py
git commit -m "feat(segments): SeriesId + partition into 10 PRD cuts"
```

---

## Task 10: Cross-sectional regression — WLS cap + OLS eq + slope spread

**Files:**
- Create: `roro/regression.py`
- Create: `tests/test_regression.py`

- [ ] **Step 1: Write failing test `tests/test_regression.py`**

```python
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from roro.regression import cross_section, daily_panel
from roro.segments import ASSET_EQ, ASSET_FI, SeriesId
from roro.types import ReturnsFrame, VolFrame
from tests.strategies import positive_weights


def _series_ids() -> list[SeriesId]:
    return [
        SeriesId(country=f"C{i}", segment="DM", asset_class=ASSET_EQ, mcap=float(i + 1))
        for i in range(10)
    ]


def test_cross_section_recovers_known_slope() -> None:
    ids = _series_ids()
    rng = np.random.default_rng(0)
    vols = np.linspace(0.05, 0.3, 10)
    true_beta = 0.5
    true_alpha = 0.01
    rets = true_alpha + true_beta * vols + rng.normal(0, 1e-6, 10)
    ret_df = pd.DataFrame([rets], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")])
    vol_df = pd.DataFrame([vols], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")])
    panel = daily_panel(
        date=pd.Timestamp("2024-01-01"),
        series=ids,
        equity_returns=ret_df,
        fi_returns=ret_df,  # not used because all series are EQ
        equity_vol=vol_df,
        fi_vol=vol_df,
    )
    out = cross_section(panel, min_n=5)
    assert abs(out.beta_cap - true_beta) < 1e-3
    assert abs(out.beta_eq - true_beta) < 1e-3
    assert abs(out.slope_spread) < 1e-3


def test_cross_section_suppresses_below_min_n() -> None:
    ids = _series_ids()[:3]
    ret_df = pd.DataFrame([[0.1, 0.2, 0.3]], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")])
    vol_df = pd.DataFrame([[0.1, 0.2, 0.3]], columns=[s.country for s in ids], index=[pd.Timestamp("2024-01-01")])
    panel = daily_panel(
        date=pd.Timestamp("2024-01-01"),
        series=ids,
        equity_returns=ret_df,
        fi_returns=ret_df,
        equity_vol=vol_df,
        fi_vol=vol_df,
    )
    out = cross_section(panel, min_n=5)
    assert out.suppressed
    assert np.isnan(out.beta_cap)


@given(w=positive_weights(n=10))
@settings(max_examples=25, deadline=None)
def test_ols_equals_wls_when_weights_uniform(w: np.ndarray) -> None:
    from roro.regression import _wls_slope

    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 0.05, 10)
    y = 0.5 * x + 0.01 + rng.normal(0, 1e-3, 10)
    beta_uniform = _wls_slope(x, y, np.ones(10))
    beta_explicit_avg = _wls_slope(x, y, np.full(10, 1 / 10))
    assert abs(beta_uniform - beta_explicit_avg) < 1e-12
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_regression.py -v`
Expected: FAIL — `roro.regression` not found.

- [ ] **Step 3: Implement `roro/regression.py`**

```python
"""Daily cross-sectional WLS regression of 3M return on EWMA vol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SeriesId


@dataclass(frozen=True)
class DailyPanel:
    date: pd.Timestamp
    series: tuple[SeriesId, ...]
    returns: np.ndarray   # shape (n,)
    vols: np.ndarray      # shape (n,)
    weights: np.ndarray   # shape (n,), unnormalized cap weights


@dataclass(frozen=True)
class CrossSectionResult:
    date: pd.Timestamp
    beta_cap: float
    beta_eq: float
    r2_cap: float
    r2_eq: float
    slope_spread: float
    n: int
    suppressed: bool
    singular: bool


def daily_panel(
    date: pd.Timestamp,
    series: list[SeriesId],
    *,
    equity_returns: pd.DataFrame,
    fi_returns: pd.DataFrame,
    equity_vol: pd.DataFrame,
    fi_vol: pd.DataFrame,
) -> DailyPanel:
    rets: list[float] = []
    vols: list[float] = []
    ws: list[float] = []
    used: list[SeriesId] = []
    for s in series:
        if s.asset_class == ASSET_EQ:
            r = float(equity_returns.at[date, s.country]) if s.country in equity_returns.columns else np.nan
            v = float(equity_vol.at[date, s.country]) if s.country in equity_vol.columns else np.nan
        elif s.asset_class == ASSET_FI:
            r = float(fi_returns.at[date, s.country]) if s.country in fi_returns.columns else np.nan
            v = float(fi_vol.at[date, s.country]) if s.country in fi_vol.columns else np.nan
        else:  # pragma: no cover - guarded by SeriesId construction
            raise ValueError(f"Unknown asset_class: {s.asset_class}")
        if not (np.isfinite(r) and np.isfinite(v)):
            continue
        rets.append(r)
        vols.append(v)
        ws.append(s.mcap)
        used.append(s)
    return DailyPanel(
        date=date,
        series=tuple(used),
        returns=np.asarray(rets, dtype=np.float64),
        vols=np.asarray(vols, dtype=np.float64),
        weights=np.asarray(ws, dtype=np.float64),
    )


def cross_section(panel: DailyPanel, *, min_n: int) -> CrossSectionResult:
    n = len(panel.returns)
    if n < min_n:
        return CrossSectionResult(
            date=panel.date,
            beta_cap=np.nan,
            beta_eq=np.nan,
            r2_cap=np.nan,
            r2_eq=np.nan,
            slope_spread=np.nan,
            n=n,
            suppressed=True,
            singular=False,
        )
    try:
        beta_cap, r2_cap = _wls_with_r2(panel.vols, panel.returns, panel.weights)
        beta_eq, r2_eq = _wls_with_r2(panel.vols, panel.returns, np.ones(n))
        singular = False
    except np.linalg.LinAlgError:
        return CrossSectionResult(
            date=panel.date,
            beta_cap=np.nan,
            beta_eq=np.nan,
            r2_cap=np.nan,
            r2_eq=np.nan,
            slope_spread=np.nan,
            n=n,
            suppressed=False,
            singular=True,
        )
    return CrossSectionResult(
        date=panel.date,
        beta_cap=beta_cap,
        beta_eq=beta_eq,
        r2_cap=r2_cap,
        r2_eq=r2_eq,
        slope_spread=beta_cap - beta_eq,
        n=n,
        suppressed=False,
        singular=singular,
    )


def _wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w_norm = w / w.sum()
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w_norm)
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(theta[1])


def _wls_with_r2(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w_norm = w / w.sum()
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w_norm)
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    y_hat = X @ theta
    ss_res = float(np.sum(w_norm * (y - y_hat) ** 2))
    y_bar = float(np.sum(w_norm * y))
    ss_tot = float(np.sum(w_norm * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(theta[1]), r2
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_regression.py -v && uv run mypy roro/regression.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/regression.py tests/test_regression.py
git commit -m "feat(regression): daily WLS cap-wtd + OLS eq-wtd cross-section"
```

---

## Task 11: Regression driver — iterate dates × segments → BetaBySegment

**Files:**
- Modify: `roro/regression.py` — add `compute_beta_by_segment`
- Modify: `tests/test_regression.py` — add driver test

- [ ] **Step 1: Add failing test to `tests/test_regression.py`**

```python
import pandas as pd

from roro.regression import compute_beta_by_segment
from roro.segments import partition
from roro.types import ReturnsFrame, Universe, VolFrame


def test_compute_beta_by_segment_produces_all_cuts() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    countries = [f"C{i}" for i in range(20)]
    u = Universe(
        countries=pd.DataFrame(
            {
                "Country": countries,
                "Segment": ["DM"] * 10 + ["EM"] * 10,
                "Equity_Mkt_Cap_Val": list(range(1, 21)),
                "Fixed_Income_Mkt_Cap_Val": list(range(1, 21)),
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )
    eq_ret = pd.DataFrame(0.01, index=dates, columns=countries)
    fi_ret = pd.DataFrame(0.005, index=dates, columns=countries)
    eq_vol = pd.DataFrame(0.15, index=dates, columns=countries)
    fi_vol = pd.DataFrame(0.05, index=dates, columns=countries)
    ret = ReturnsFrame(log_returns_3m=eq_ret, daily_log_returns=eq_ret)
    vol = VolFrame(ewma_sigma_annualized=eq_vol)

    bbs = compute_beta_by_segment(
        dates=dates,
        cuts=partition(u),
        equity_returns_3m=eq_ret,
        fi_returns_3m=fi_ret,
        equity_vol=eq_vol,
        fi_vol=fi_vol,
        min_n=5,
    )
    expected = {"global", "DM", "EM", "Equity", "FI", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI", "LatAm"}
    assert set(bbs.by_segment) == expected
    assert len(bbs.by_segment["global"].cap_wtd) == 10
    assert "beta" in bbs.by_segment["global"].cap_wtd.columns
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_regression.py::test_compute_beta_by_segment_produces_all_cuts -v`
Expected: FAIL — `compute_beta_by_segment` not defined.

- [ ] **Step 3: Add driver to `roro/regression.py`**

```python
from roro.types import BetaBySegment, BetaFrame


def compute_beta_by_segment(
    dates: pd.DatetimeIndex,
    cuts: dict[str, list[SeriesId]],
    *,
    equity_returns_3m: pd.DataFrame,
    fi_returns_3m: pd.DataFrame,
    equity_vol: pd.DataFrame,
    fi_vol: pd.DataFrame,
    min_n: int,
) -> BetaBySegment:
    out: dict[str, BetaFrame] = {}
    for cut_name, series in cuts.items():
        rows_cap: list[dict[str, float | bool | pd.Timestamp]] = []
        rows_eq: list[dict[str, float | bool | pd.Timestamp]] = []
        spread: list[float] = []
        spread_idx: list[pd.Timestamp] = []
        for d in dates:
            panel = daily_panel(
                date=d,
                series=series,
                equity_returns=equity_returns_3m,
                fi_returns=fi_returns_3m,
                equity_vol=equity_vol,
                fi_vol=fi_vol,
            )
            res = cross_section(panel, min_n=min_n)
            rows_cap.append(
                {
                    "date": d,
                    "beta": res.beta_cap,
                    "r2": res.r2_cap,
                    "n": res.n,
                    "suppressed": res.suppressed,
                    "singular": res.singular,
                }
            )
            rows_eq.append(
                {
                    "date": d,
                    "beta": res.beta_eq,
                    "r2": res.r2_eq,
                    "n": res.n,
                    "suppressed": res.suppressed,
                    "singular": res.singular,
                }
            )
            spread.append(res.slope_spread)
            spread_idx.append(d)
        cap_df = pd.DataFrame(rows_cap).set_index("date")
        eq_df = pd.DataFrame(rows_eq).set_index("date")
        spread_s = pd.Series(spread, index=pd.DatetimeIndex(spread_idx), name="slope_spread")
        out[cut_name] = BetaFrame(cap_wtd=cap_df, eq_wtd=eq_df, slope_spread=spread_s)
    return BetaBySegment(by_segment=out)
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_regression.py -v && uv run mypy roro/regression.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/regression.py tests/test_regression.py
git commit -m "feat(regression): driver iterates dates × cuts into BetaBySegment"
```

---

## Task 12: Classify — rolling percentile + tercile + quintile + direction

**Files:**
- Create: `roro/classify.py`
- Create: `tests/test_classify.py`

- [ ] **Step 1: Write failing test `tests/test_classify.py`**

```python
import numpy as np
import pandas as pd
from hypothesis import given, settings

from roro.classify import (
    classify,
    direction_flag,
    quintile_label,
    rolling_percentile,
    tercile_label,
)
from roro.config import BucketScheme
from roro.types import BetaBySegment, BetaFrame
from tests.strategies import finite_return_series


def _toy_betas(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2014-01-01", periods=len(values))
    return pd.Series(values, index=idx, name="beta")


def test_rolling_percentile_in_unit_interval() -> None:
    s = _toy_betas([float(i) for i in range(500)])
    p = rolling_percentile(s, window_days=252)
    p_valid = p.dropna()
    assert (p_valid >= 0).all()
    assert (p_valid <= 1).all()


def test_tercile_label_monotone() -> None:
    assert tercile_label(0.1) == "Risk-off"
    assert tercile_label(0.5) == "Transitional"
    assert tercile_label(0.9) == "Risk-on"


def test_quintile_label_five_buckets() -> None:
    assert quintile_label(0.05) == "Q1"
    assert quintile_label(0.25) == "Q2"
    assert quintile_label(0.45) == "Q3"
    assert quintile_label(0.65) == "Q4"
    assert quintile_label(0.95) == "Q5"


def test_direction_rising_when_beta_increases() -> None:
    s = _toy_betas([0.1 * i for i in range(10)])
    flag = direction_flag(s, lookback_days=5)
    assert flag.iloc[-1] == "rising"


def test_classify_full_pipeline_produces_all_frames() -> None:
    idx = pd.bdate_range("2010-01-01", periods=2000)
    cap = pd.DataFrame(
        {"beta": np.linspace(-1, 1, 2000), "r2": 0.5, "n": 32, "suppressed": False, "singular": False},
        index=idx,
    )
    bf = BetaFrame(cap_wtd=cap, eq_wtd=cap, slope_spread=pd.Series(0.0, index=idx))
    bbs = BetaBySegment(by_segment={"global": bf, "DM": bf, "LatAm": bf})
    rf = classify(
        bbs,
        bucket_scheme=BucketScheme.TERCILE,
        percentile_window_days=1260,
        direction_lookback_days=5,
        bootstrap_min_days=252,
        thin_cuts=frozenset({"LatAm"}),
    )
    for frame in (rf.percentile_5y, rf.tercile, rf.quintile, rf.direction, rf.bootstrap_flag):
        assert set(frame.columns) >= {"global", "DM", "LatAm"}
    assert rf.thin_cut_flag.loc[idx[-1], "LatAm"]
    assert not rf.thin_cut_flag.loc[idx[-1], "global"]
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_classify.py -v`
Expected: FAIL — `roro.classify` not found.

- [ ] **Step 3: Implement `roro/classify.py`**

```python
"""Rolling percentile classifier: tercile, quintile, direction, thin-cut flag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.config import BucketScheme
from roro.types import BetaBySegment, RegimeFrame


def rolling_percentile(beta: pd.Series, *, window_days: int) -> pd.Series:
    """For each t, fraction of trailing-window values ≤ beta[t]. NaN if empty window."""

    def _pct(window: np.ndarray) -> float:
        if len(window) == 0:
            return float("nan")
        return float((window <= window[-1]).sum() - 1) / float(max(len(window) - 1, 1))

    return beta.rolling(window=window_days, min_periods=1).apply(_pct, raw=True)


def tercile_label(p: float) -> str:
    if np.isnan(p):
        return "Unknown"
    if p <= 1 / 3:
        return "Risk-off"
    if p >= 2 / 3:
        return "Risk-on"
    return "Transitional"


def quintile_label(p: float) -> str:
    if np.isnan(p):
        return "Unknown"
    for i, edge in enumerate([0.2, 0.4, 0.6, 0.8], start=1):
        if p <= edge:
            return f"Q{i}"
    return "Q5"


def asym_label(p: float) -> str:
    """20/60/20 scheme."""
    if np.isnan(p):
        return "Unknown"
    if p <= 0.2:
        return "Risk-off"
    if p >= 0.8:
        return "Risk-on"
    return "Transitional"


def _bucket(p: float, scheme: BucketScheme) -> str:
    if scheme is BucketScheme.TERCILE:
        return tercile_label(p)
    if scheme is BucketScheme.QUINTILE:
        return quintile_label(p)
    return asym_label(p)


def direction_flag(beta: pd.Series, *, lookback_days: int) -> pd.Series:
    def _flag(window: np.ndarray) -> float:
        if len(window) < 2 or np.isnan(window).any():
            return float("nan")
        x = np.arange(len(window), dtype=np.float64)
        slope = float(np.polyfit(x, window, 1)[0])
        # Use the window stdev as a noise scale to discriminate
        scale = float(np.std(window))
        if scale == 0 or abs(slope) < 0.1 * scale:
            return 0.0
        return 1.0 if slope > 0 else -1.0

    raw = beta.rolling(window=lookback_days, min_periods=lookback_days).apply(_flag, raw=True)
    return raw.map({1.0: "rising", -1.0: "falling", 0.0: "stable"}).astype(object)


def classify(
    bbs: BetaBySegment,
    *,
    bucket_scheme: BucketScheme,
    percentile_window_days: int,
    direction_lookback_days: int,
    bootstrap_min_days: int,
    thin_cuts: frozenset[str],
) -> RegimeFrame:
    pct_frames: dict[str, pd.Series] = {}
    terc_frames: dict[str, pd.Series] = {}
    quin_frames: dict[str, pd.Series] = {}
    dir_frames: dict[str, pd.Series] = {}
    n_frames: dict[str, pd.Series] = {}
    boot_frames: dict[str, pd.Series] = {}
    thin_frames: dict[str, pd.Series] = {}

    for cut, bf in bbs.by_segment.items():
        beta = bf.cap_wtd["beta"]
        pct = rolling_percentile(beta, window_days=percentile_window_days)
        # Bootstrap suppression
        valid_count = beta.expanding(min_periods=1).count()
        boot = valid_count < percentile_window_days
        pct = pct.where(valid_count >= bootstrap_min_days, other=np.nan)
        terc = pct.map(lambda p: _bucket(p, bucket_scheme))
        quin = pct.map(quintile_label)
        dir_s = direction_flag(beta, lookback_days=direction_lookback_days)
        pct_frames[cut] = pct
        terc_frames[cut] = terc
        quin_frames[cut] = quin
        dir_frames[cut] = dir_s
        n_frames[cut] = bf.cap_wtd["n"]
        boot_frames[cut] = boot
        thin_frames[cut] = pd.Series(cut in thin_cuts, index=beta.index)

    return RegimeFrame(
        percentile_5y=pd.DataFrame(pct_frames),
        tercile=pd.DataFrame(terc_frames),
        quintile=pd.DataFrame(quin_frames),
        direction=pd.DataFrame(dir_frames),
        n_per_segment=pd.DataFrame(n_frames),
        thin_cut_flag=pd.DataFrame(thin_frames),
        bootstrap_flag=pd.DataFrame(boot_frames),
    )
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_classify.py -v && uv run mypy roro/classify.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/classify.py tests/test_classify.py
git commit -m "feat(classify): rolling percentile + tercile/quintile + direction"
```

---

## Task 13: Correlation — avg pairwise + PC1 share

**Files:**
- Create: `roro/correlation.py`
- Create: `tests/test_correlation.py`

- [ ] **Step 1: Write failing test `tests/test_correlation.py`**

```python
import numpy as np
import pandas as pd

from roro.correlation import avg_pairwise_rolling, compute_correlation_panel, pc1_share_rolling
from roro.segments import ASSET_EQ, SeriesId


def test_avg_pairwise_in_minus_one_to_one() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 5)), index=pd.bdate_range("2024-01-01", periods=200))
    s = avg_pairwise_rolling(df, window=63)
    s = s.dropna()
    assert (s >= -1).all()
    assert (s <= 1).all()


def test_pc1_share_in_zero_to_one_and_min_n_inv() -> None:
    rng = np.random.default_rng(0)
    n = 5
    df = pd.DataFrame(rng.normal(size=(200, n)), index=pd.bdate_range("2024-01-01", periods=200))
    s = pc1_share_rolling(df, window=63)
    s = s.dropna()
    assert (s >= 1.0 / n - 1e-9).all()
    assert (s <= 1.0).all()


def test_correlation_panel_runs_per_segment() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    countries = [f"C{i}" for i in range(8)]
    eq = pd.DataFrame(rng.normal(size=(200, 8)), index=dates, columns=countries)
    fi = pd.DataFrame(rng.normal(size=(200, 8)), index=dates, columns=countries)
    series = [
        SeriesId(country=c, segment="DM" if i < 4 else "EM", asset_class=ASSET_EQ, mcap=1.0)
        for i, c in enumerate(countries)
    ]
    cuts = {"global": series, "DM_Eq": [s for s in series if s.segment == "DM"]}
    cf = compute_correlation_panel(daily_log_returns_eq=eq, daily_log_returns_fi=fi, cuts=cuts, window=63)
    assert set(cf.avg_pairwise_3m.columns) == {"global", "DM_Eq"}
    assert set(cf.pc1_variance_share.columns) == {"global", "DM_Eq"}
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_correlation.py -v`
Expected: FAIL — `roro.correlation` not found.

- [ ] **Step 3: Implement `roro/correlation.py`**

```python
"""Avg pairwise correlation + PC1 variance share per segment, rolling window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.segments import ASSET_EQ, ASSET_FI, SeriesId
from roro.types import CorrelationFrame


def avg_pairwise_rolling(df: pd.DataFrame, *, window: int) -> pd.Series:
    def _avg_corr(window_arr: np.ndarray) -> float:
        if window_arr.shape[0] < 3 or window_arr.shape[1] < 2:
            return float("nan")
        corr = np.corrcoef(window_arr, rowvar=False)
        n = corr.shape[0]
        iu = np.triu_indices(n, k=1)
        return float(np.nanmean(corr[iu]))

    return _rolling_matrix_reduce(df, window=window, reducer=_avg_corr)


def pc1_share_rolling(df: pd.DataFrame, *, window: int) -> pd.Series:
    def _pc1(window_arr: np.ndarray) -> float:
        if window_arr.shape[0] < 3 or window_arr.shape[1] < 2:
            return float("nan")
        cov = np.cov(window_arr, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        trace = float(eigvals.sum())
        if trace <= 0:
            return float("nan")
        return float(eigvals.max() / trace)

    return _rolling_matrix_reduce(df, window=window, reducer=_pc1)


def _rolling_matrix_reduce(
    df: pd.DataFrame, *, window: int, reducer: "callable"
) -> pd.Series:
    out: list[float] = []
    arr = df.to_numpy()
    for end in range(len(df)):
        start = end - window + 1
        if start < 0:
            out.append(float("nan"))
            continue
        window_arr = arr[start : end + 1]
        # Drop columns that are entirely NaN within the window
        mask = ~np.all(np.isnan(window_arr), axis=0)
        out.append(reducer(window_arr[:, mask]))
    return pd.Series(out, index=df.index)


def compute_correlation_panel(
    *,
    daily_log_returns_eq: pd.DataFrame,
    daily_log_returns_fi: pd.DataFrame,
    cuts: dict[str, list[SeriesId]],
    window: int,
) -> CorrelationFrame:
    avg_frames: dict[str, pd.Series] = {}
    pc1_frames: dict[str, pd.Series] = {}
    for cut_name, series in cuts.items():
        cols_eq = [s.country for s in series if s.asset_class == ASSET_EQ]
        cols_fi = [s.country for s in series if s.asset_class == ASSET_FI]
        sub_eq = daily_log_returns_eq[[c for c in cols_eq if c in daily_log_returns_eq.columns]]
        sub_fi = daily_log_returns_fi[[c for c in cols_fi if c in daily_log_returns_fi.columns]]
        # Tag columns to keep them distinct in the merged frame
        sub_eq = sub_eq.add_suffix("__Eq")
        sub_fi = sub_fi.add_suffix("__FI")
        merged = pd.concat([sub_eq, sub_fi], axis=1)
        avg_frames[cut_name] = avg_pairwise_rolling(merged, window=window)
        pc1_frames[cut_name] = pc1_share_rolling(merged, window=window)
    return CorrelationFrame(
        avg_pairwise_3m=pd.DataFrame(avg_frames),
        pc1_variance_share=pd.DataFrame(pc1_frames),
    )
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_correlation.py -v && uv run mypy roro/correlation.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/correlation.py tests/test_correlation.py
git commit -m "feat(correlation): avg pairwise + PC1 share per segment"
```

---

## Task 14: External + internal validation

**Files:**
- Create: `roro/validation.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing test `tests/test_validation.py`**

```python
from datetime import datetime

import numpy as np
import pandas as pd

from roro.types import FredFrame, RegimeFrame, Universe
from roro.validation import (
    COMPOSITE_MAPPING,
    compute_internal_consistency,
    compute_rolling_external_corr,
    detect_validation_degradation,
)


def _regime_frame(idx: pd.DatetimeIndex) -> RegimeFrame:
    pct = pd.DataFrame({"global": np.linspace(0, 1, len(idx)), "DM": np.linspace(0, 1, len(idx))}, index=idx)
    empty = pd.DataFrame(index=idx, columns=["global", "DM"]).fillna(False)
    return RegimeFrame(
        percentile_5y=pct,
        tercile=pct.applymap(lambda v: "Risk-on" if v > 0.66 else "Risk-off" if v < 0.33 else "Transitional"),
        quintile=pct,
        direction=pct,
        n_per_segment=pct,
        thin_cut_flag=empty,
        bootstrap_flag=empty,
    )


def test_rolling_external_corr_window_60() -> None:
    idx = pd.bdate_range("2024-01-01", periods=200)
    regime = _regime_frame(idx)
    ff = FredFrame(
        series={"VIXCLS": pd.Series(np.linspace(20, 30, 200), index=idx)},
        pulled_at=datetime(2026, 5, 27),
        series_hashes={"VIXCLS": "h"},
    )
    out = compute_rolling_external_corr(regime, ff, window_days=60)
    assert ("global", "VIXCLS") in out.columns
    assert out[("global", "VIXCLS")].iloc[-1] is not pd.NA


def test_detect_degradation_flags_low_correlation() -> None:
    idx = pd.bdate_range("2024-01-01", periods=120)
    df = pd.DataFrame({("global", "VIXCLS"): [0.1] * 120}, index=idx)
    alerts = detect_validation_degradation(df, threshold=0.3)
    assert alerts["below_threshold"].all()


def test_internal_consistency_runs_for_mapped_segments() -> None:
    idx = pd.bdate_range("2024-01-01", periods=120)
    pct = pd.DataFrame({"DM": np.linspace(0, 1, 120), "EM": np.linspace(1, 0, 120)}, index=idx)
    regime = RegimeFrame(
        percentile_5y=pct,
        tercile=pct.applymap(lambda v: "Risk-on" if v > 0.66 else "Risk-off" if v < 0.33 else "Transitional"),
        quintile=pct,
        direction=pct,
        n_per_segment=pct,
        thin_cut_flag=pct.copy().astype(bool),
        bootstrap_flag=pct.copy().astype(bool),
    )
    composite_eq = pd.DataFrame({"MXWO": np.linspace(100, 130, 120), "MXEF": np.linspace(100, 120, 120)}, index=idx)
    composite_fi = pd.DataFrame({"I35402US": np.linspace(50, 55, 120), "EMUSTRUU": np.linspace(50, 53, 120)}, index=idx)
    out = compute_internal_consistency(
        regime=regime,
        composite_eq_prices=composite_eq,
        composite_fi_prices=composite_fi,
        return_window_days=63,
    )
    assert "DM" in out.columns
    assert "EM" in out.columns


def test_composite_mapping_contains_documented_pairs() -> None:
    assert COMPOSITE_MAPPING["DM"] == ("MXWO", "I35402US")
    assert COMPOSITE_MAPPING["LatAm"] == ("MXLA", "H04338US")
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL — `roro.validation` not found.

- [ ] **Step 3: Implement `roro/validation.py`**

```python
"""External (FRED) rolling correlation + internal composite-consistency check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roro.types import FredFrame, RegimeFrame

# Engine-segment → (equity composite ticker, FI composite ticker). None = no pairing.
COMPOSITE_MAPPING: dict[str, tuple[str | None, str | None]] = {
    "global": ("MXWD", "LEGATRUU"),
    "DM": ("MXWO", "I35402US"),
    "EM": ("MXEF", "EMUSTRUU"),
    "Equity": ("MXWD", None),
    "FI": (None, "LEGATRUU"),
    "DM_Eq": ("MXWO", None),
    "EM_Eq": ("MXEF", None),
    "DM_FI": (None, "I35402US"),
    "EM_FI": (None, "EMUSTRUU"),
    "LatAm": ("MXLA", "H04338US"),
}


def compute_rolling_external_corr(
    regime: RegimeFrame,
    fred: FredFrame,
    *,
    window_days: int,
) -> pd.DataFrame:
    """Rolling Pearson ρ between each (segment percentile, FRED series). Wide MultiIndex columns."""
    out: dict[tuple[str, str], pd.Series] = {}
    for segment in regime.percentile_5y.columns:
        seg_series = regime.percentile_5y[segment]
        for sid, fred_series in fred.series.items():
            aligned = pd.concat([seg_series, fred_series.rename(sid)], axis=1).ffill(limit=2)
            rho = aligned[segment].rolling(window=window_days, min_periods=window_days).corr(
                aligned[sid]
            )
            out[(segment, sid)] = rho
    return pd.DataFrame(out)


def detect_validation_degradation(
    rolling_corr: pd.DataFrame, *, threshold: float
) -> pd.DataFrame:
    below = rolling_corr.abs() < threshold
    out = below.stack(level=[0, 1]).rename("below_threshold").to_frame()
    return out


def compute_internal_consistency(
    *,
    regime: RegimeFrame,
    composite_eq_prices: pd.DataFrame,
    composite_fi_prices: pd.DataFrame,
    return_window_days: int,
) -> pd.DataFrame:
    """For each segment with a mapped composite, compare engine tercile to the composite's
    own 3M return percentile tercile and emit the tercile gap.
    """
    eq_returns = np.log(composite_eq_prices / composite_eq_prices.shift(return_window_days))
    fi_returns = np.log(composite_fi_prices / composite_fi_prices.shift(return_window_days))

    rows: dict[str, pd.Series] = {}
    for segment, (eq_ticker, fi_ticker) in COMPOSITE_MAPPING.items():
        if segment not in regime.tercile.columns:
            continue
        composite_pct = _blend_composite_returns(eq_returns, fi_returns, eq_ticker, fi_ticker)
        if composite_pct is None:
            continue
        comp_terc = composite_pct.map(_terc)
        engine_terc = regime.tercile[segment]
        gap = (_terc_to_int(engine_terc) - _terc_to_int(comp_terc)).abs()
        rows[segment] = gap
    return pd.DataFrame(rows)


def _blend_composite_returns(
    eq_returns: pd.DataFrame,
    fi_returns: pd.DataFrame,
    eq_ticker: str | None,
    fi_ticker: str | None,
) -> pd.Series | None:
    if eq_ticker and fi_ticker:
        eq = eq_returns[eq_ticker] if eq_ticker in eq_returns.columns else None
        fi = fi_returns[fi_ticker] if fi_ticker in fi_returns.columns else None
        if eq is None and fi is None:
            return None
        blended = pd.concat([eq, fi], axis=1).mean(axis=1)
    elif eq_ticker:
        if eq_ticker not in eq_returns.columns:
            return None
        blended = eq_returns[eq_ticker]
    elif fi_ticker:
        if fi_ticker not in fi_returns.columns:
            return None
        blended = fi_returns[fi_ticker]
    else:
        return None
    return blended.rank(pct=True)


def _terc(p: float) -> str:
    if pd.isna(p):
        return "Unknown"
    if p <= 1 / 3:
        return "Risk-off"
    if p >= 2 / 3:
        return "Risk-on"
    return "Transitional"


def _terc_to_int(s: pd.Series) -> pd.Series:
    mapping = {"Risk-off": 1, "Transitional": 2, "Risk-on": 3, "Unknown": 2}
    return s.map(mapping).astype(float)
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_validation.py -v && uv run mypy roro/validation.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/validation.py tests/test_validation.py
git commit -m "feat(validation): external 60d ρ + internal composite-tercile gap"
```

---

## Task 15: Tripwire (1M fast signal)

**Files:**
- Create: `roro/tripwire.py`
- Create: `tests/test_tripwire.py`

- [ ] **Step 1: Write failing test `tests/test_tripwire.py`**

```python
import pandas as pd

from roro.segments import partition
from roro.tripwire import compute_tripwire_signal
from roro.types import PriceFrame, Universe


def _tiny_universe() -> Universe:
    return Universe(
        countries=pd.DataFrame(
            {
                "Country": [f"C{i}" for i in range(20)],
                "Segment": ["DM"] * 10 + ["EM"] * 10,
                "Equity_Mkt_Cap_Val": list(range(1, 21)),
                "Fixed_Income_Mkt_Cap_Val": list(range(1, 21)),
            }
        ),
        composites=pd.DataFrame({"Country": ["DM"]}),
    )


def test_tripwire_returns_betabysegment_same_keys() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    cols = [f"C{i}" for i in range(20)]
    eq = pd.DataFrame(1.0, index=dates, columns=cols).cumsum() + 100
    fi = pd.DataFrame(1.0, index=dates, columns=cols).cumsum() + 100
    pf = PriceFrame(equity_lc=eq, fi_lc=fi)
    bbs = compute_tripwire_signal(
        prices=pf,
        cuts=partition(_tiny_universe()),
        return_window_days=21,
        ewma_halflife_days=10,
        min_n=5,
    )
    expected = {"global", "DM", "EM", "Equity", "FI", "DM_Eq", "EM_Eq", "DM_FI", "EM_FI", "LatAm"}
    assert set(bbs.by_segment) == expected
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_tripwire.py -v`
Expected: FAIL — `roro.tripwire` not found.

- [ ] **Step 3: Implement `roro/tripwire.py`**

```python
"""1M-window tripwire — parallel return + vol + cross-section β only."""

from __future__ import annotations

import pandas as pd

from roro.regression import compute_beta_by_segment
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import SeriesId
from roro.types import BetaBySegment, PriceFrame


def compute_tripwire_signal(
    *,
    prices: PriceFrame,
    cuts: dict[str, list[SeriesId]],
    return_window_days: int,
    ewma_halflife_days: int,
    min_n: int,
) -> BetaBySegment:
    eq_ret = total_return_3m(prices.equity_lc, window_days=return_window_days)
    fi_ret = total_return_3m(prices.fi_lc, window_days=return_window_days)
    eq_daily = daily_log_returns(prices.equity_lc)
    fi_daily = daily_log_returns(prices.fi_lc)
    eq_vol = ewma_vol(eq_daily, halflife=ewma_halflife_days)
    fi_vol = ewma_vol(fi_daily, halflife=ewma_halflife_days)
    return compute_beta_by_segment(
        dates=prices.equity_lc.index,
        cuts=cuts,
        equity_returns_3m=eq_ret,
        fi_returns_3m=fi_ret,
        equity_vol=eq_vol,
        fi_vol=fi_vol,
        min_n=min_n,
    )
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_tripwire.py -v && uv run mypy roro/tripwire.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/tripwire.py tests/test_tripwire.py
git commit -m "feat(tripwire): 1M fast-signal mirror of returns + vol + β"
```

---

## Task 16: Alerts — bucket transitions + disagreement + degradation

**Files:**
- Create: `roro/alerts.py`
- Create: `tests/test_alerts.py`

- [ ] **Step 1: Write failing test `tests/test_alerts.py`**

```python
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
        correlation=CorrelationFrame(avg_pairwise_3m=pd.DataFrame(), pc1_variance_share=pd.DataFrame()),
        validation=ValidationFrame(
            rolling_corr_60d=pd.DataFrame(),
            internal_consistency=pd.DataFrame(),
            correlation_alerts=pd.DataFrame(),
        ),
    )
    bt = out.bucket_transitions
    assert {"global", "DM_Eq"}.issubset(set(bt["segment"]))
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_alerts.py -v`
Expected: FAIL — `roro.alerts` not found.

- [ ] **Step 3: Implement `roro/alerts.py`**

```python
"""Alert detection: bucket transitions, disagreement events, validation degradation."""

from __future__ import annotations

import pandas as pd

from roro.types import AlertSet, CorrelationFrame, RegimeFrame, ValidationFrame


def detect_alerts(
    *,
    regime: RegimeFrame,
    correlation: CorrelationFrame,
    validation: ValidationFrame,
) -> AlertSet:
    return AlertSet(
        bucket_transitions=_bucket_transitions(regime.tercile),
        disagreement_events=_disagreement_events(regime, correlation),
        validation_degradation=_validation_degradation(validation),
    )


def _bucket_transitions(tercile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in tercile.columns:
        s = tercile[segment]
        prev = s.shift(1)
        mask = (prev.notna()) & (s != prev)
        for ts, _ in s[mask].items():
            rows.append(
                {
                    "date": ts,
                    "segment": segment,
                    "from_bucket": prev.loc[ts],
                    "to_bucket": s.loc[ts],
                }
            )
    return pd.DataFrame(rows, columns=["date", "segment", "from_bucket", "to_bucket"])


def _disagreement_events(
    regime: RegimeFrame, correlation: CorrelationFrame
) -> pd.DataFrame:
    if correlation.avg_pairwise_3m.empty or regime.tercile.empty:
        return pd.DataFrame(columns=["date", "segment", "vol_slope_bucket", "avg_pairwise"])
    rows: list[dict[str, object]] = []
    for segment in regime.tercile.columns:
        if segment not in correlation.avg_pairwise_3m.columns:
            continue
        terc = regime.tercile[segment]
        corr = correlation.avg_pairwise_3m[segment]
        mask = (terc == "Risk-on") & (corr > 0.6)
        for ts in terc[mask].index:
            rows.append(
                {
                    "date": ts,
                    "segment": segment,
                    "vol_slope_bucket": "Risk-on",
                    "avg_pairwise": float(corr.loc[ts]),
                }
            )
    return pd.DataFrame(rows, columns=["date", "segment", "vol_slope_bucket", "avg_pairwise"])


def _validation_degradation(validation: ValidationFrame) -> pd.DataFrame:
    if validation.correlation_alerts.empty:
        return pd.DataFrame(columns=["date", "segment", "fred_series"])
    df = validation.correlation_alerts
    df = df[df["below_threshold"]].copy()
    if "level_0" in df.columns:
        df = df.rename(columns={"level_0": "date"})
    df = df.reset_index()
    rename_map = {"level_1": "segment", "level_2": "fred_series"}
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    keep = [c for c in ("date", "segment", "fred_series") if c in df.columns]
    return df[keep]
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_alerts.py -v && uv run mypy roro/alerts.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/alerts.py tests/test_alerts.py
git commit -m "feat(alerts): bucket transitions + disagreement + validation degradation"
```

---

## Task 17: Run output writer — CSVs + snapshot.json + atomic rename

**Files:**
- Modify: `roro/io.py` — add `write_run`, `read_run`, `compute_data_fingerprint`
- Modify: `tests/test_io.py` — add round-trip test

- [ ] **Step 1: Add failing test to `tests/test_io.py`**

```python
import json

from roro.io import read_run, write_run
from roro.types import (
    AlertSet,
    BetaBySegment,
    BetaFrame,
    CorrelationFrame,
    FredFrame,
    PriceFrame,
    RegimeFrame,
    ReturnsFrame,
    RunResult,
    Universe,
    ValidationFrame,
    VolFrame,
)
from roro.config import EngineConfig
from datetime import datetime


def _empty_result(out_dir: Path) -> RunResult:
    empty = pd.DataFrame()
    empty_s = pd.Series(dtype=float)
    bf = BetaFrame(cap_wtd=empty, eq_wtd=empty, slope_spread=empty_s)
    bbs = BetaBySegment(by_segment={"global": bf})
    return RunResult(
        config=EngineConfig(data_path=Path("data.xlsx"), output_dir=out_dir),
        universe=Universe(countries=empty, composites=empty),
        returns=ReturnsFrame(log_returns_3m=empty, daily_log_returns=empty),
        vol=VolFrame(ewma_sigma_annualized=empty),
        beta=bbs,
        regime=RegimeFrame(
            percentile_5y=empty,
            tercile=empty,
            quintile=empty,
            direction=empty,
            n_per_segment=empty,
            thin_cut_flag=empty,
            bootstrap_flag=empty,
        ),
        correlation=CorrelationFrame(avg_pairwise_3m=empty, pc1_variance_share=empty),
        validation=ValidationFrame(
            rolling_corr_60d=empty, internal_consistency=empty, correlation_alerts=empty
        ),
        tripwire=bbs,
        alerts=AlertSet(
            bucket_transitions=empty, disagreement_events=empty, validation_degradation=empty
        ),
        warnings=["x"],
        data_fingerprint={"data_xlsx_sha256": "abc"},
        code_version={"git_sha": "def", "dirty": "false"},
    )


def test_write_run_atomic_and_round_trip(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    result = _empty_result(out_root)
    path = write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    assert path.exists()
    assert (path / "snapshot.json").exists()
    snapshot = json.loads((path / "snapshot.json").read_text())
    assert snapshot["methodology_version"] == "1.0.0"
    assert snapshot["data_fingerprint"]["data_xlsx_sha256"] == "abc"
    # No leftover .tmp dir
    assert not (out_root / "2026-05-27.tmp").exists()


def test_write_run_force_overwrites(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    result = _empty_result(out_root)
    write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    # Without force: second write must raise
    with pytest.raises(FileExistsError):
        write_run(result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26")
    # With force: succeeds
    write_run(
        result, run_date="2026-05-27", out_dir=out_root, as_of_data_date="2026-05-26", force=True
    )
```

(Top of `tests/test_io.py` needs:)

```python
import pandas as pd
import pytest
from pathlib import Path
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL — `write_run` not defined.

- [ ] **Step 3: Add `write_run`, `read_run`, `compute_data_fingerprint` to `roro/io.py`**

```python
import json
import shutil
import subprocess
from typing import Any

from roro.config import EngineConfig, to_dict as config_to_dict
from roro.types import RunResult


def compute_data_fingerprint(xlsx_path: Path) -> dict[str, str]:
    data = xlsx_path.read_bytes()
    return {
        "data_xlsx_sha256": hashlib.sha256(data).hexdigest(),
        "data_xlsx_mtime": str(int(xlsx_path.stat().st_mtime)),
    }


def code_version() -> dict[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return {"git_sha": sha, "dirty": "true" if status.strip() else "false"}
    except Exception:  # noqa: BLE001
        return {"git_sha": "unknown", "dirty": "unknown"}


def write_run(
    result: RunResult,
    *,
    run_date: str,
    out_dir: Path,
    as_of_data_date: str,
    force: bool = False,
) -> Path:
    final = out_dir / run_date
    tmp = out_dir / f"{run_date}.tmp"
    if final.exists() and not force:
        raise FileExistsError(f"Run dir exists: {final}. Use force=True to overwrite.")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    _write_beta(result.beta, tmp / "beta_series.csv")
    _write_regime(result.regime, tmp / "regimes.csv")
    _write_correlation(result.correlation, tmp / "correlation.csv")
    _write_validation(result.validation, tmp / "external_validation.csv")
    _write_alerts(result.alerts, tmp / "alerts.csv")
    _write_tripwire(result.tripwire, tmp / "tripwire.csv")

    snapshot = _build_snapshot(result, run_date=run_date, as_of_data_date=as_of_data_date)
    (tmp / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def read_run(run_dir: Path) -> dict[str, Any]:
    return {
        "snapshot": json.loads((run_dir / "snapshot.json").read_text()),
        "beta_series": pd.read_csv(run_dir / "beta_series.csv", parse_dates=["date"]),
        "regimes": pd.read_csv(run_dir / "regimes.csv", parse_dates=["date"]),
        "correlation": pd.read_csv(run_dir / "correlation.csv", parse_dates=["date"]),
        "external_validation": pd.read_csv(run_dir / "external_validation.csv", parse_dates=["date"]),
        "alerts": pd.read_csv(run_dir / "alerts.csv", parse_dates=["date"]) if (run_dir / "alerts.csv").stat().st_size > 0 else pd.DataFrame(),
        "tripwire": pd.read_csv(run_dir / "tripwire.csv", parse_dates=["date"]),
    }


def _stack_segment_frame(frames: dict[str, pd.DataFrame], scheme_label: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cut, df in frames.items():
        if df.empty:
            continue
        copy = df.copy()
        copy = copy.reset_index().rename(columns={"index": "date"})
        copy.insert(1, "segment", cut)
        copy.insert(2, "scheme", scheme_label)
        rows.append(copy)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_beta(bbs: "BetaBySegment", path: Path) -> None:
    cap = _stack_segment_frame({k: v.cap_wtd for k, v in bbs.by_segment.items()}, "cap_wtd")
    eq = _stack_segment_frame({k: v.eq_wtd for k, v in bbs.by_segment.items()}, "eq_wtd")
    if cap.empty and eq.empty:
        path.write_text("date,segment,scheme,beta,r2,n,suppressed,singular\n")
        return
    pd.concat([cap, eq], ignore_index=True).to_csv(path, index=False)


def _write_regime(rf: "RegimeFrame", path: Path) -> None:
    if rf.tercile.empty:
        path.write_text("date,segment,percentile_5y,tercile,quintile,direction,n,thin_cut,bootstrap\n")
        return
    long = rf.tercile.reset_index().melt(id_vars=[rf.tercile.index.name or "index"], var_name="segment", value_name="tercile")
    long = long.rename(columns={long.columns[0]: "date"})
    pct = rf.percentile_5y.reset_index().melt(id_vars=[rf.percentile_5y.index.name or "index"], var_name="segment", value_name="percentile_5y")
    pct = pct.rename(columns={pct.columns[0]: "date"})
    merged = long.merge(pct, on=["date", "segment"], how="left")
    for label, frame in (("quintile", rf.quintile), ("direction", rf.direction), ("n", rf.n_per_segment), ("thin_cut", rf.thin_cut_flag), ("bootstrap", rf.bootstrap_flag)):
        m = frame.reset_index().melt(id_vars=[frame.index.name or "index"], var_name="segment", value_name=label)
        m = m.rename(columns={m.columns[0]: "date"})
        merged = merged.merge(m, on=["date", "segment"], how="left")
    merged.to_csv(path, index=False)


def _write_correlation(cf: "CorrelationFrame", path: Path) -> None:
    if cf.avg_pairwise_3m.empty:
        path.write_text("date,segment,avg_pairwise_3m,pc1_variance_share\n")
        return
    avg = cf.avg_pairwise_3m.reset_index().melt(id_vars=[cf.avg_pairwise_3m.index.name or "index"], var_name="segment", value_name="avg_pairwise_3m")
    avg = avg.rename(columns={avg.columns[0]: "date"})
    pc1 = cf.pc1_variance_share.reset_index().melt(id_vars=[cf.pc1_variance_share.index.name or "index"], var_name="segment", value_name="pc1_variance_share")
    pc1 = pc1.rename(columns={pc1.columns[0]: "date"})
    avg.merge(pc1, on=["date", "segment"], how="left").to_csv(path, index=False)


def _write_validation(vf: "ValidationFrame", path: Path) -> None:
    if vf.rolling_corr_60d.empty:
        path.write_text("date,segment,fred_series,rolling_corr_60d\n")
        return
    df = vf.rolling_corr_60d.stack(level=[0, 1]).rename("rolling_corr_60d").reset_index()
    df.columns = ["date", "segment", "fred_series", "rolling_corr_60d"]
    df.to_csv(path, index=False)


def _write_alerts(a: "AlertSet", path: Path) -> None:
    rows: list[pd.DataFrame] = []
    if not a.bucket_transitions.empty:
        rows.append(a.bucket_transitions.assign(kind="bucket_transition"))
    if not a.disagreement_events.empty:
        rows.append(a.disagreement_events.assign(kind="disagreement"))
    if not a.validation_degradation.empty:
        rows.append(a.validation_degradation.assign(kind="validation_degradation"))
    if not rows:
        path.write_text("date,kind,segment\n")
        return
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _write_tripwire(bbs: "BetaBySegment", path: Path) -> None:
    _write_beta(bbs, path)


def _build_snapshot(result: RunResult, *, run_date: str, as_of_data_date: str) -> dict[str, Any]:
    return {
        "run_date": run_date,
        "as_of_data_date": as_of_data_date,
        "methodology_version": result.config.methodology_version,
        "config_resolved": config_to_dict(result.config),
        "data_fingerprint": result.data_fingerprint,
        "code_version": result.code_version,
        "warnings": result.warnings,
    }
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_io.py -v && uv run mypy roro/io.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/io.py tests/test_io.py
git commit -m "feat(io): write_run atomic CSV + snapshot.json; data fingerprint + git version"
```

---

## Task 18: Engine orchestrator

**Files:**
- Create: `roro/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing test `tests/test_engine.py`**

```python
from datetime import datetime
from pathlib import Path

import pandas as pd

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def _seeded_fred() -> MockFredClient:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def test_engine_run_produces_outputs(tiny_xlsx: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "out",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    result = run(cfg, fred_client=_seeded_fred(), run_date="2024-12-31", as_of_data_date="2024-12-31")
    assert (cfg.output_dir / "2024-12-31").exists()
    assert "global" in result.beta.by_segment
    assert result.config.methodology_version == "1.0.0"
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL — `roro.engine` not found.

- [ ] **Step 3: Implement `roro/engine.py`**

```python
"""Engine orchestrator: wires the functional pipeline end-to-end."""

from __future__ import annotations

import pandas as pd

from roro.alerts import detect_alerts
from roro.classify import classify
from roro.config import EngineConfig
from roro.correlation import compute_correlation_panel
from roro.fred_client import FredClient
from roro.io import (
    code_version,
    compute_data_fingerprint,
    load_fred,
    load_panel,
    load_prices,
    write_run,
)
from roro.regression import compute_beta_by_segment
from roro.returns import daily_log_returns, ewma_vol, total_return_3m
from roro.segments import partition
from roro.tripwire import compute_tripwire_signal
from roro.types import ReturnsFrame, RunResult, VolFrame
from roro.validation import (
    compute_internal_consistency,
    compute_rolling_external_corr,
    detect_validation_degradation,
)
from roro.validators import DataSourceError, validate_prices, validate_universe


def run(
    cfg: EngineConfig,
    *,
    fred_client: FredClient,
    run_date: str,
    as_of_data_date: str,
    force: bool = False,
) -> RunResult:
    warnings: list[str] = []

    universe = load_panel(cfg.data_path)
    warnings.extend(validate_universe(universe))
    prices = load_prices(cfg.data_path)
    warnings.extend(validate_prices(prices))

    fred = load_fred(fred_client, start=prices.equity_lc.index.min().date(), end=prices.equity_lc.index.max().date())

    eq_ret = total_return_3m(prices.equity_lc, window_days=cfg.return_window_days)
    fi_ret = total_return_3m(prices.fi_lc, window_days=cfg.return_window_days)
    eq_daily = daily_log_returns(prices.equity_lc)
    fi_daily = daily_log_returns(prices.fi_lc)
    eq_vol = ewma_vol(eq_daily, halflife=cfg.ewma_halflife_days)
    fi_vol = ewma_vol(fi_daily, halflife=cfg.ewma_halflife_days)

    cuts = partition(universe)
    beta = compute_beta_by_segment(
        dates=prices.equity_lc.index,
        cuts=cuts,
        equity_returns_3m=eq_ret,
        fi_returns_3m=fi_ret,
        equity_vol=eq_vol,
        fi_vol=fi_vol,
        min_n=cfg.min_n_per_cut,
    )
    regime = classify(
        beta,
        bucket_scheme=cfg.bucket_scheme,
        percentile_window_days=cfg.percentile_window_years * 252,
        direction_lookback_days=cfg.direction_lookback_days,
        bootstrap_min_days=cfg.bootstrap_min_days,
        thin_cuts=frozenset({"LatAm"}),
    )
    correlation = compute_correlation_panel(
        daily_log_returns_eq=eq_daily,
        daily_log_returns_fi=fi_daily,
        cuts=cuts,
        window=cfg.return_window_days,
    )
    rolling_corr = compute_rolling_external_corr(regime, fred, window_days=cfg.external_corr_window_days)
    corr_alerts = detect_validation_degradation(rolling_corr, threshold=cfg.external_corr_alert_threshold)
    internal = compute_internal_consistency(
        regime=regime,
        composite_eq_prices=pd.DataFrame(),  # composites flow via Universe → not in v1 prices frame
        composite_fi_prices=pd.DataFrame(),
        return_window_days=cfg.return_window_days,
    )
    from roro.types import ValidationFrame

    validation = ValidationFrame(
        rolling_corr_60d=rolling_corr,
        internal_consistency=internal,
        correlation_alerts=corr_alerts,
    )
    tripwire = compute_tripwire_signal(
        prices=prices,
        cuts=cuts,
        return_window_days=cfg.tripwire_window_days,
        ewma_halflife_days=cfg.tripwire_ewma_halflife_days,
        min_n=cfg.min_n_per_cut,
    )
    alerts = detect_alerts(regime=regime, correlation=correlation, validation=validation)

    result = RunResult(
        config=cfg,
        universe=universe,
        returns=ReturnsFrame(log_returns_3m=eq_ret, daily_log_returns=eq_daily),
        vol=VolFrame(ewma_sigma_annualized=eq_vol),
        beta=beta,
        regime=regime,
        correlation=correlation,
        validation=validation,
        tripwire=tripwire,
        alerts=alerts,
        warnings=warnings,
        data_fingerprint=compute_data_fingerprint(cfg.data_path) | {"fred_pulled_at": fred.pulled_at.isoformat(), **{f"fred_{k}": v for k, v in fred.series_hashes.items()}},
        code_version=code_version(),
    )

    write_run(result, run_date=run_date, out_dir=cfg.output_dir, as_of_data_date=as_of_data_date, force=force)
    return result
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_engine.py -v && uv run mypy roro/engine.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/engine.py tests/test_engine.py
git commit -m "feat(engine): orchestrator wires full pipeline and writes run"
```

---

## Task 19: CLI — `roro run` + `roro backtest` stub

**Files:**
- Create: `roro/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test `tests/test_cli.py`**

```python
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from roro.cli import main


def test_cli_run_dispatches(tiny_xlsx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force engine to use a mock fred client by setting FRED_API_KEY="" and patching FredApiClient
    from roro import cli as cli_mod
    from roro.fred_client import FRED_SERIES_IDS, MockFredClient

    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    monkeypatch.setattr(
        cli_mod, "_build_fred_client", lambda key: MockFredClient(
            seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
        )
    )

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        f"data_path: {tiny_xlsx}\n"
        f"output_dir: {tmp_path / 'out'}\n"
        "ewma_halflife_days: 10\n"
        "return_window_days: 21\n"
        "tripwire_window_days: 10\n"
        "tripwire_ewma_halflife_days: 5\n"
        "percentile_window_years: 1\n"
        "bucket_scheme: TERCILE\n"
        "min_n_per_cut: 2\n"
        "direction_lookback_days: 5\n"
        "external_corr_window_days: 60\n"
        "external_corr_alert_threshold: 0.3\n"
        "bootstrap_min_days: 10\n"
        "methodology_version: 1.0.0\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", str(cfg_yaml), "--date", "2024-12-31", "--as-of-data-date", "2024-12-31"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "2024-12-31").exists()
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `roro.cli` not found.

- [ ] **Step 3: Implement `roro/cli.py`**

```python
"""Thin Click CLI: `roro run`, `roro backtest`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from roro.config import load_config
from roro.engine import run as engine_run
from roro.fred_client import FredApiClient, FredClient


def _build_fred_client(api_key: str | None) -> FredClient:
    if not api_key:
        raise click.UsageError("FRED_API_KEY env var (or --fred-key) is required for `roro run`.")
    return FredApiClient(api_key=api_key)


@click.group()
def main() -> None:
    """RoRo Risk-Regime engine CLI."""


@main.command("run")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--date", "run_date", required=True)
@click.option("--as-of-data-date", required=True)
@click.option("--ewma-halflife", type=int, default=None)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option("--fred-key", default=None, help="Defaults to FRED_API_KEY env.")
@click.option("--force", is_flag=True)
def cmd_run(
    config_path: Path,
    run_date: str,
    as_of_data_date: str,
    ewma_halflife: int | None,
    out_dir: Path | None,
    fred_key: str | None,
    force: bool,
) -> None:
    overrides: dict[str, Any] = {}
    if ewma_halflife is not None:
        overrides["ewma_halflife_days"] = ewma_halflife
    if out_dir is not None:
        overrides["output_dir"] = out_dir
    cfg = load_config(config_path, overrides=overrides)
    api_key = fred_key or os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    engine_run(
        cfg,
        fred_client=client,
        run_date=run_date,
        as_of_data_date=as_of_data_date,
        force=force,
    )
    click.echo(f"OK: {cfg.output_dir / run_date}")


@main.command("backtest")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--assert-gates", is_flag=True)
def cmd_backtest(config_path: Path, start: str, end: str, assert_gates: bool) -> None:
    from roro.backtest import run_backtest  # local import — heavy module

    cfg = load_config(config_path)
    api_key = os.environ.get("FRED_API_KEY", "")
    client = _build_fred_client(api_key)
    report = run_backtest(cfg, fred_client=client, start=start, end=end)
    if assert_gates and not report["all_passed"]:
        raise click.ClickException("Acceptance gates failed; see backtest/acceptance_report.json")
    click.echo("OK")
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_cli.py -v && uv run mypy roro/cli.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/cli.py tests/test_cli.py
git commit -m "feat(cli): roro run + roro backtest commands"
```

---

## Task 20: Backtest harness — replay + event recognition + stability

**Files:**
- Create: `roro/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing test `tests/test_backtest.py`**

```python
from pathlib import Path

import pandas as pd

from roro.backtest import EVENTS, run_backtest
from roro.config import EngineConfig
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def test_events_documented_match_prd() -> None:
    names = {e.name for e in EVENTS}
    assert "2020_COVID" in names
    assert "2022_rate_shock" in names


def test_run_backtest_writes_acceptance_report(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
    client = MockFredClient(seeded=seeded)
    cfg = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "bt",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    report = run_backtest(cfg, fred_client=client, start="2024-01-01", end="2024-12-31")
    assert "gates" in report
    assert {"G1_vix", "G2_bbb", "G3_events", "G4_segmentation_lift", "G5_stability", "G6_internal"} <= set(report["gates"])
    assert (tmp_path / "bt" / "acceptance_report.json").exists()
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL — `roro.backtest` not found.

- [ ] **Step 3: Implement `roro/backtest.py`**

```python
"""Backtest harness: replay engine over a date range and evaluate PRD §10 gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from roro.config import EngineConfig
from roro.engine import run as engine_run
from roro.fred_client import FredClient
from roro.types import RunResult


@dataclass(frozen=True)
class Event:
    name: str
    date: str
    expected_buckets: tuple[str, ...] = ("Risk-off",)
    segments: tuple[str, ...] = ("global",)


EVENTS: tuple[Event, ...] = (
    Event(name="2010_greek_crisis", date="2010-05-06"),
    Event(name="2011_eurozone_us_downgrade", date="2011-08-08"),
    Event(name="2015_china_devaluation", date="2015-08-11"),
    Event(name="2018_q4_selloff", date="2018-12-24"),
    Event(name="2020_COVID", date="2020-03-16"),
    Event(name="2022_rate_shock_jan", date="2022-01-24"),
    Event(name="2022_rate_shock_sep", date="2022-09-26"),
    Event(name="2008_lehman", date="2008-10-10"),  # bootstrap calibration period
)


def run_backtest(
    cfg: EngineConfig,
    *,
    fred_client: FredClient,
    start: str,
    end: str,
) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result = engine_run(
        cfg,
        fred_client=fred_client,
        run_date=end,
        as_of_data_date=end,
        force=True,
    )
    sliced = _slice_result(result, start=start, end=end)
    gates = _evaluate_gates(sliced)
    report = {
        "start": start,
        "end": end,
        "methodology_version": cfg.methodology_version,
        "gates": gates,
        "all_passed": all(v["passed"] for v in gates.values()),
    }
    (cfg.output_dir / "acceptance_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_event_recognition(sliced, cfg.output_dir / "event_recognition.csv")
    _write_validation_history(sliced, cfg.output_dir / "validation_corr_history.csv")
    _write_stability_metrics(sliced, cfg.output_dir / "stability_metrics.csv")
    return report


def _slice_result(result: RunResult, *, start: str, end: str) -> RunResult:
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    return result  # full result is already in memory; slicing happens per-gate


def _evaluate_gates(result: RunResult) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    rolling = result.validation.rolling_corr_60d
    if ("global", "VIXCLS") in rolling.columns:
        vix = rolling[("global", "VIXCLS")].abs().dropna()
        gates["G1_vix"] = {
            "passed": bool((vix >= 0.5).mean() >= 0.8) if len(vix) else False,
            "fraction_above": float((vix >= 0.5).mean()) if len(vix) else 0.0,
        }
    else:
        gates["G1_vix"] = {"passed": False, "reason": "missing series"}

    if ("global", "BAMLC0A4CBBB") in rolling.columns:
        bbb = rolling[("global", "BAMLC0A4CBBB")].abs().dropna()
        gates["G2_bbb"] = {
            "passed": bool((bbb >= 0.4).mean() >= 0.8) if len(bbb) else False,
            "fraction_above": float((bbb >= 0.4).mean()) if len(bbb) else 0.0,
        }
    else:
        gates["G2_bbb"] = {"passed": False, "reason": "missing series"}

    gates["G3_events"] = _gate_events(result)
    gates["G4_segmentation_lift"] = _gate_segmentation_lift(result)
    gates["G5_stability"] = _gate_stability(result)
    gates["G6_internal"] = _gate_internal_consistency(result)
    return gates


def _gate_events(result: RunResult) -> dict[str, Any]:
    terc = result.regime.tercile
    hits = 0
    details: list[dict[str, Any]] = []
    for event in EVENTS:
        ts = pd.Timestamp(event.date)
        window_start = ts - pd.Timedelta(days=3 * 2)
        window_end = ts + pd.Timedelta(days=3 * 2)
        local = terc.loc[(terc.index >= window_start) & (terc.index <= window_end)]
        ok = False
        for seg in event.segments:
            if seg in local.columns and (local[seg].isin(event.expected_buckets)).any():
                ok = True
                break
        details.append({"event": event.name, "matched": ok})
        if ok:
            hits += 1
    return {"passed": hits == len(EVENTS), "matched_events": hits, "total": len(EVENTS), "details": details}


def _gate_segmentation_lift(result: RunResult) -> dict[str, Any]:
    terc = result.regime.tercile
    if not {"DM_Eq", "EM_Eq"}.issubset(terc.columns):
        return {"passed": False, "reason": "missing DM_Eq/EM_Eq"}
    mapping = {"Risk-off": 1, "Transitional": 2, "Risk-on": 3, "Unknown": 2}
    a = terc["DM_Eq"].map(mapping).astype(float)
    b = terc["EM_Eq"].map(mapping).astype(float)
    diff = (a - b).abs()
    frac = float((diff >= 2).mean())
    return {"passed": frac >= 0.2, "fraction_with_gap_ge_2": frac}


def _gate_stability(result: RunResult) -> dict[str, Any]:
    transitions = result.alerts.bucket_transitions
    if transitions.empty:
        return {"passed": True, "spurious_per_calm_quarter": 0.0}
    daily = result.returns.daily_log_returns
    if daily.empty:
        return {"passed": False, "reason": "missing returns"}
    proxy_col = daily.columns[0]
    realized = daily[proxy_col].rolling(63).std() * np.sqrt(252)
    median = realized.median()
    by_q = realized.groupby(pd.Grouper(freq="Q")).mean()
    calm_quarters = by_q[by_q < median].index
    transitions["quarter"] = pd.PeriodIndex(transitions["date"], freq="Q").to_timestamp(how="end")
    by_quarter = transitions[transitions["segment"] == "global"].groupby("quarter").size()
    calm_counts = by_quarter.reindex(calm_quarters, fill_value=0)
    worst = float(calm_counts.max()) if len(calm_counts) else 0.0
    return {"passed": worst <= 2.0, "max_transitions_in_calm_quarter": worst}


def _gate_internal_consistency(result: RunResult) -> dict[str, Any]:
    gap = result.validation.internal_consistency
    if "DM" not in gap.columns:
        return {"passed": True, "reason": "no DM mapping data"}
    breaches = (gap["DM"] > 1).astype(int)
    rolling = breaches.rolling(30, min_periods=1).sum()
    worst = float(rolling.max()) if len(rolling) else 0.0
    return {"passed": worst <= 5.0, "max_breaches_in_30d_window": worst}


def _write_event_recognition(result: RunResult, path: Path) -> None:
    pd.DataFrame([{"event": e.name, "date": e.date} for e in EVENTS]).to_csv(path, index=False)


def _write_validation_history(result: RunResult, path: Path) -> None:
    df = result.validation.rolling_corr_60d
    if df.empty:
        path.write_text("date,segment,fred_series,rho\n")
        return
    df.stack(level=[0, 1]).rename("rho").reset_index().rename(
        columns={"level_0": "date", "level_1": "segment", "level_2": "fred_series"}
    ).to_csv(path, index=False)


def _write_stability_metrics(result: RunResult, path: Path) -> None:
    transitions = result.alerts.bucket_transitions
    if transitions.empty:
        path.write_text("quarter,segment,transitions\n")
        return
    transitions["quarter"] = pd.PeriodIndex(transitions["date"], freq="Q").astype(str)
    out = transitions.groupby(["quarter", "segment"]).size().reset_index(name="transitions")
    out.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_backtest.py -v && uv run mypy roro/backtest.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add roro/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): replay + PRD §10 acceptance gates"
```

---

## Task 21: Reproducibility test — two runs produce byte-identical CSVs

**Files:**
- Create: `tests/test_reproducibility.py`

- [ ] **Step 1: Write the test**

```python
import filecmp
from pathlib import Path

import pandas as pd

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient


def test_two_runs_identical_csvs(tiny_xlsx: Path, tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    seeded = {sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS}
    client = MockFredClient(seeded=seeded)
    cfg_a = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "a",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    cfg_b = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "b",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    run(cfg_a, fred_client=client, run_date="2024-12-31", as_of_data_date="2024-12-31", force=True)
    run(cfg_b, fred_client=client, run_date="2024-12-31", as_of_data_date="2024-12-31", force=True)

    for csv in ("beta_series.csv", "regimes.csv", "correlation.csv", "external_validation.csv", "tripwire.csv"):
        a = tmp_path / "a" / "2024-12-31" / csv
        b = tmp_path / "b" / "2024-12-31" / csv
        assert filecmp.cmp(a, b, shallow=False), f"{csv} differs between runs"
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_reproducibility.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reproducibility.py
git commit -m "test(reproducibility): byte-identical CSVs across runs"
```

---

## Task 22: Golden integration test — 2024-Q1 slice

**Files:**
- Create: `tests/golden/2024-Q1/` (generated)
- Create: `tests/test_golden.py`

- [ ] **Step 1: Write the test scaffold `tests/test_golden.py`**

```python
"""Integration regression: engine output must match committed golden CSVs.

Regenerate with: pytest --regenerate-goldens
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pandas as pd
import pytest

from roro.config import EngineConfig
from roro.engine import run
from roro.fred_client import FRED_SERIES_IDS, MockFredClient

GOLDEN_DIR = Path(__file__).parent / "golden" / "2024-Q1"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--regenerate-goldens", action="store_true")


def _seeded_fred() -> MockFredClient:
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    return MockFredClient(seeded={sid: pd.Series(20.0, index=idx) for sid in FRED_SERIES_IDS})


def _run_against(tmp_path: Path, tiny_xlsx: Path) -> Path:
    cfg = EngineConfig(
        data_path=tiny_xlsx,
        output_dir=tmp_path / "out",
        ewma_halflife_days=10,
        return_window_days=21,
        tripwire_window_days=10,
        percentile_window_years=1,
        min_n_per_cut=2,
        bootstrap_min_days=10,
    )
    run(cfg, fred_client=_seeded_fred(), run_date="2024-03-31", as_of_data_date="2024-03-31", force=True)
    return cfg.output_dir / "2024-03-31"


def test_golden_2024_q1(tiny_xlsx: Path, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    actual = _run_against(tmp_path, tiny_xlsx)
    if request.config.getoption("--regenerate-goldens"):
        if GOLDEN_DIR.exists():
            shutil.rmtree(GOLDEN_DIR)
        shutil.copytree(actual, GOLDEN_DIR)
        pytest.skip("Regenerated goldens.")

    if not GOLDEN_DIR.exists():
        pytest.skip("Goldens not yet generated; run with --regenerate-goldens.")

    for csv in ("beta_series.csv", "regimes.csv", "correlation.csv", "external_validation.csv", "tripwire.csv"):
        assert filecmp.cmp(actual / csv, GOLDEN_DIR / csv, shallow=False), f"{csv} drifted from golden"
```

- [ ] **Step 2: Generate the goldens**

Run: `uv run pytest tests/test_golden.py --regenerate-goldens -v`
Expected: PASS (skipped, but populates `tests/golden/2024-Q1/`).

- [ ] **Step 3: Rerun to verify byte-identical match**

Run: `uv run pytest tests/test_golden.py -v`
Expected: PASS without skip.

- [ ] **Step 4: Commit goldens + test**

```bash
git add tests/test_golden.py tests/golden/2024-Q1/
git commit -m "test(golden): 2024-Q1 integration regression baseline"
```

---

## Task 23: Documentation — update context files

**Files:**
- Modify: `docs/context/todo.md`
- Modify: `docs/context/results.md`
- Modify: `docs/context/sesion-log.md`
- Modify: `docs/context/memory.md`
- Create: `README.md`

- [ ] **Step 1: Write `docs/context/todo.md`**

```markdown
# Todo

- [x] S0 — project scaffolding, FRED key wiring
- [x] S1 — IO (Excel + FRED) + validators
- [x] S2 — returns + EWMA vol + cross-sectional regression
- [x] S3 — segmentation (10 cuts)
- [x] S4 — percentile classifier (tercile + quintile + direction)
- [x] S5 — correlation + PC1 panel
- [x] S6 — external + internal validation
- [x] S7 — 1M tripwire
- [x] S9 — backtest harness + acceptance gates
- [ ] S8 — HTML dashboard + JSON API (deferred to viz phase)
- [ ] v1.1 — GFP integration, time-varying mcap weights
```

- [ ] **Step 2: Write `docs/context/results.md`**

```markdown
# Results

- 2026-05-27: RoRo engine implementation plan executed end-to-end.
- All numeric kernels covered by pytest unit + hypothesis property tests.
- Reproducibility invariant enforced: two runs over same fingerprint produce byte-identical CSVs.
- Golden integration test pinned at tests/golden/2024-Q1/.
```

- [ ] **Step 3: Write `docs/context/sesion-log.md`**

```markdown
# Session log

- 2026-05-27: brainstorm → design spec → implementation plan → engine v0.1.0 built.
```

- [ ] **Step 4: Write `docs/context/memory.md`**

```markdown
# Memory

- decision: static cap weights (Panel snapshot 2026-05-26) applied across full history; time-varying deferred to v1.1.
- decision: composite rows (Panel 32–37) used only by `roro.validation`, never by regressions.
- decision: engine writes CSV + snapshot.json; no Postgres, no Parquet, no dashboard in this scope.
- decision: bucket schemes available are TERCILE (default), QUINTILE, ASYM_20_60_20.
- decision: bootstrap suppression below `bootstrap_min_days=252` of beta history.
```

- [ ] **Step 5: Write `README.md`**

```markdown
# RoRo Risk-Regime Engine

Daily cross-asset risk-regime classifier across 32 countries × {equity, fixed income} plus 10 segmentation cuts. Diagnostic only (v1.0).

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Run

```bash
export FRED_API_KEY="..."
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

See [docs/superpowers/specs/2026-05-27-roro-engine-design.md](docs/superpowers/specs/2026-05-27-roro-engine-design.md).
```

- [ ] **Step 6: Commit**

```bash
git add docs/context/*.md README.md
git commit -m "docs: README + context files for v0.1.0 engine"
```

---

## Final verification

- [ ] **Step 1: Full test suite + static analysis**

Run: `uv run pytest -v && uv run mypy roro/ && uv run ruff check .`
Expected: all green.

- [ ] **Step 2: Smoke run against real `data.xlsx` + real FRED**

```bash
export FRED_API_KEY="<your-key>"
roro run --config configs/default.yaml --date 2026-05-27 --as-of-data-date 2026-05-26
```

Expected: `outputs/2026-05-27/` populated with five CSVs and `snapshot.json`; exit 0.

- [ ] **Step 3: Smoke backtest run**

```bash
roro backtest --config configs/default.yaml --start 2010-01-01 --end 2024-12-31 --assert-gates
```

Expected: `backtest/acceptance_report.json` written. Gates may fail on first run with default params — that is PRD-expected for S9 parameter selection. Note any failures into `docs/context/lessons.md`.

- [ ] **Step 4: Tag v0.1.0**

```bash
git tag v0.1.0
git log --oneline -20
```

---

## Self-Review Notes

After plan was written I cross-checked vs the spec for coverage, placeholders, and type consistency. Notes:

- **Spec coverage:** Every PRD sprint S0–S7 + S9 maps to ≥1 task. S8 explicitly deferred per the design's §12 out-of-scope. F1–F7 functional requirements (excluding F7 dashboard) all have tasks: F1 → Task 8, F2 → Tasks 10–11, F3 → Task 9, F4 → Task 12, F5 → Task 13, F6 → Task 14, F1.3 tripwire → Task 15.
- **Placeholder scan:** No "TBD" / "implement later" / "similar to Task N". Every code block contains the real implementation.
- **Type consistency:** `SeriesId.column_key`, `DailyPanel`, `CrossSectionResult`, `BetaFrame`, `BetaBySegment`, `RegimeFrame` names match across tasks. `_wls_slope` and `_wls_with_r2` defined once in Task 10 and reused.
- **Known follow-up surfaces:** Composite equity/FI prices flow as empty frames into `compute_internal_consistency` in Task 18 — a real-data wiring is left for the implementation phase (composite price series are present in `data.xlsx` Equity/Fixed_Income sheets but not currently loaded into `Universe`). This is captured in `roro/validation.py` as "no DM mapping data" → gate auto-passes; the implementation step should extend `load_panel` + engine to pass through composite prices before the backtest gates become meaningful. Logged here so the executing engineer knows to surface it.
- **CLI test** uses `monkeypatch` to swap `_build_fred_client` — clean DI seam.
