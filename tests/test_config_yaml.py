"""Tests for the strict YAML configuration loader."""

from __future__ import annotations

import pytest

from portfolio_optimization_lab.config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_config


def test_shipped_default_yaml_loads_to_documented_defaults():
    """configs/default.yaml must map to the documented OptConfig defaults."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.tickers == ("AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "AMZN")
    assert cfg.benchmark == "SPY"
    assert cfg.rf_annual == 0.0
    assert cfg.periods_per_year == 252
    assert cfg.estimation_window == 252
    assert cfg.holding_period == 63
    assert cfg.max_weight == 0.40
    assert cfg.cost_bps == 10.0
    assert cfg.frontier_points == 30
    assert cfg.seed == 42
    assert cfg == DEFAULT_CONFIG  # identical to the programmatic default


def test_yaml_unknown_key_raises(tmp_path):
    """A typo'd key must raise ValueError, never fall back to a default."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "holdng_period" not in text  # the injected key is genuinely unknown
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        text.replace("cost_bps: 10", "cost_bps: 10\nholdng_period: 63"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="holdng_period"):
        load_config(bad)


def test_yaml_values_are_honored(tmp_path):
    """Non-default YAML values flow into the config (with type coercion)."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        text.replace("max_weight: 0.40", "max_weight: 0.25").replace(
            "rf_annual: 0.0", "rf_annual: 0.02"
        ),
        encoding="utf-8",
    )
    cfg = load_config(custom)
    assert cfg.max_weight == 0.25
    assert cfg.rf_annual == pytest.approx(0.02)
    assert cfg.cost_bps == 10.0  # int in YAML, coerced to float field


def test_yaml_invalid_values_raise(tmp_path):
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    out_of_range = tmp_path / "range.yaml"
    out_of_range.write_text(text.replace("max_weight: 0.40", "max_weight: 1.5"), encoding="utf-8")
    with pytest.raises(ValueError, match="max_weight"):
        load_config(out_of_range)
    wrong_type = tmp_path / "type.yaml"
    wrong_type.write_text(text.replace("benchmark: SPY", "benchmark: [SPY]"), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark"):
        load_config(wrong_type)
