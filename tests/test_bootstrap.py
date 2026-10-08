"""Tests for the circular block bootstrap of Sharpe differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization_lab.bootstrap import (
    bootstrap_sharpe_differences,
    circular_block_indices,
)


def test_circular_block_indices_wrap_and_cover():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(100, 10, rng)
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100
    # circular wrap-around: a block starting near the end wraps to small indices
    idx_wrap = circular_block_indices(10, 4, np.random.default_rng(1))
    assert len(idx_wrap) == 10


def test_circular_block_indices_deterministic_and_validated():
    a = circular_block_indices(50, 7, np.random.default_rng(42))
    b = circular_block_indices(50, 7, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        circular_block_indices(10, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        circular_block_indices(10, 11, np.random.default_rng(0))
    with pytest.raises(ValueError):
        circular_block_indices(0, 3, np.random.default_rng(0))


def _synthetic_daily(n: int = 1890, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # AR(1)-ish dependence so an IID bootstrap would be visibly wrong
    a = np.empty(n)
    b = np.empty(n)
    a[0] = b[0] = 0.0
    shocks_a = rng.normal(5e-4, 0.01, size=n)
    shocks_b = rng.normal(2e-4, 0.01, size=n)
    for t in range(1, n):
        a[t] = 0.3 * a[t - 1] + shocks_a[t]
        b[t] = 0.3 * b[t - 1] + shocks_b[t]
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"strat_a": a, "strat_b": b}, index=dates)


def test_bootstrap_sharpe_differences_shape_and_significance():
    daily = _synthetic_daily()
    out = bootstrap_sharpe_differences(daily, B=500, block=63, seed=42)
    assert out["B"] == 500 and out["block"] == 63 and out["seed"] == 42
    assert len(out["pairs"]) == 1
    pair = out["pairs"][0]
    assert set(pair) >= {"a", "b", "diff", "ci_low", "ci_high", "significant_95", "sensitivity"}
    assert pair["ci_low"] <= pair["diff"] <= pair["ci_high"]
    # strat_a (mean 5e-4) beats strat_b (mean 2e-4) at equal vol by construction:
    # the CI must exclude 0 on the positive side with enough resamples
    assert pair["diff"] > 0
    assert pair["significant_95"] is True
    assert pair["ci_low"] > 0
    # sensitivity blocks recorded
    assert set(pair["sensitivity"]) == {"21", "126"}


def test_bootstrap_reproducible_with_fixed_seed():
    daily = _synthetic_daily(n=400)
    out1 = bootstrap_sharpe_differences(daily, B=200, block=21, seed=7)
    out2 = bootstrap_sharpe_differences(daily, B=200, block=21, seed=7)
    assert out1["pairs"][0]["ci_low"] == out2["pairs"][0]["ci_low"]
    assert out1["pairs"][0]["ci_high"] == out2["pairs"][0]["ci_high"]


def test_bootstrap_noisy_difference_gives_wide_interval():
    """Two statistically identical series -> CI must include 0."""
    rng = np.random.default_rng(11)
    n = 800
    daily = pd.DataFrame(
        {
            "x": rng.normal(3e-4, 0.01, size=n),
            "y": rng.normal(3e-4, 0.01, size=n),
        }
    )
    out = bootstrap_sharpe_differences(daily, B=500, block=63, seed=42)
    pair = out["pairs"][0]
    assert pair["ci_low"] <= 0 <= pair["ci_high"]
    assert pair["significant_95"] is False


def test_bootstrap_rejects_invalid_arguments():
    daily = _synthetic_daily(n=100)
    with pytest.raises(ValueError):
        bootstrap_sharpe_differences(daily, B=0)
    with pytest.raises(ValueError):
        bootstrap_sharpe_differences(daily, B=10, block=1, confidence_level=1.5)
