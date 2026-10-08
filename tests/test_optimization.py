"""Tests for the optimization layer (offline, synthetic inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization_lab.optimization import (
    efficient_frontier,
    equal_weights,
    max_sharpe,
    min_variance,
    risk_contributions,
    risk_parity,
)


def _cov_from_corr(corr: np.ndarray, vols: np.ndarray) -> pd.DataFrame:
    cov = corr * np.outer(vols, vols)
    return pd.DataFrame(cov)


def test_min_variance_hedges_perfect_negatives():
    """Two equal-vol assets with correlation -1 admit a zero-variance mix."""
    cov = _cov_from_corr(np.array([[1.0, -1.0], [-1.0, 1.0]]), np.array([0.2, 0.2]))
    mu = pd.Series([0.05, 0.05], index=["A", "B"])
    w = min_variance(mu, cov)
    assert abs(float(w @ cov.to_numpy() @ w)) < 1e-10
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-6)


def test_budget_and_long_only_constraints_hold():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 4))
    cov = pd.DataFrame(np.cov(a.T) * 252)
    mu = pd.Series([0.1, 0.08, 0.12, 0.05])
    for w in (
        min_variance(mu, cov, max_weight=0.4),
        max_sharpe(mu, cov, max_weight=0.4),
        risk_parity(cov, max_weight=0.4),
    ):
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= -1e-12).all()
        assert (w <= 0.4 + 1e-6).all()


def test_risk_parity_equalizes_contributions():
    """For uncorrelated assets, risk parity = inverse-vol weights."""
    vols = np.array([0.10, 0.20, 0.30])
    cov = _cov_from_corr(np.eye(3), vols)
    w = risk_parity(cov)
    expected = (1.0 / vols) / (1.0 / vols).sum()
    np.testing.assert_allclose(w, expected, atol=1e-6)
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc, np.full(3, 1 / 3), atol=1e-8)


def test_frontier_volatility_is_monotone_on_upper_branch():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(800, 3))
    cov = pd.DataFrame(np.cov(a.T) * 252)
    mu = pd.Series([0.04, 0.09, 0.15])
    frontier = efficient_frontier(mu, cov, n_points=15).dropna()
    # the efficient (upper) branch starts at the global minimum-variance
    # return; beyond it, volatility must be non-decreasing in target return
    gmvr = frontier.loc[frontier["volatility"].idxmin(), "target_return"]
    upper = frontier[frontier["target_return"] >= gmvr]
    vols = upper["volatility"].to_numpy()
    assert (np.diff(vols) > -1e-8).all(), "upper frontier must be non-decreasing in vol"
    assert (frontier["sharpe"] > 0).all()


def test_equal_weights():
    np.testing.assert_allclose(equal_weights(5), np.full(5, 0.2))
