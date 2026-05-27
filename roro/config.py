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
