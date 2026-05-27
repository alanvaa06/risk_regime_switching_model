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
