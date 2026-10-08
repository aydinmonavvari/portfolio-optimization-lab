"""Tests for estimation and the walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization_lab.backtest import in_sample_stats, walk_forward_backtest
from portfolio_optimization_lab.estimates import estimate, portfolio_stats


def _synthetic_returns(n=600, k=4, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    base = rng.normal(0.0003, 0.012, size=(n, k))
    frame = pd.DataFrame(base, index=dates, columns=[f"A{i}" for i in range(k)])
    return frame


def test_shrinkage_matrix_is_psd_and_between():
    r = _synthetic_returns()
    est = estimate(r, periods_per_year=252)
    sample = est.cov_sample.to_numpy()
    shrunk = est.cov_shrunk.to_numpy()
    eig = np.linalg.eigvalsh(shrunk)
    assert (eig > 0).all(), "shrunk covariance must be positive definite"
    off_s = np.abs(sample - np.diag(np.diag(sample))).sum()
    off_sh = np.abs(shrunk - np.diag(np.diag(shrunk))).sum()
    assert off_sh <= off_s + 1e-12, "shrinkage must pull off-diagonals toward zero"
    assert 0.0 <= est.shrinkage <= 1.0


def test_portfolio_stats_math():
    mu = pd.Series([0.10, 0.06])
    cov = pd.DataFrame(np.diag([0.04, 0.01]))
    stats = portfolio_stats(np.array([0.5, 0.5]), mu, cov, rf_annual=0.0)
    assert stats["return"] == pytest.approx(0.08)
    # var = 0.25*0.04 + 0.25*0.01 = 0.0125 -> vol = sqrt(0.0125)
    assert stats["volatility"] == pytest.approx(np.sqrt(0.0125))
    assert stats["sharpe"] == pytest.approx(0.08 / np.sqrt(0.0125))


def test_walkforward_estimation_and_oos_windows_are_disjoint():
    """THE leakage test: estimators must never see the holding period."""
    n = 252 * 2 + 63 * 3
    r = _synthetic_returns(n=n)
    estimation_window, holding_period = 252, 63
    # instrument estimate() to record the slice it is given
    seen_windows: list[tuple[int, int]] = []
    original_estimate = __import__(
        "portfolio_optimization_lab.backtest", fromlist=["estimate"]
    ).estimate

    def spy(frame, *args, **kwargs):
        seen_windows.append((frame.index.min(), frame.index.max()))
        return original_estimate(frame, *args, **kwargs)

    import portfolio_optimization_lab.backtest as bt

    bt.estimate = spy
    try:
        daily, performance, weights = walk_forward_backtest(
            r, estimation_window=estimation_window, holding_period=holding_period
        )
    finally:
        bt.estimate = original_estimate

    # each estimation window must end EXACTLY one day before the holding
    # period it feeds -> estimator and evaluation windows are disjoint
    assert len(seen_windows) >= 3
    for k, (_wmin, wmax) in enumerate(seen_windows):
        hold_start = r.index[estimation_window + k * holding_period]
        assert wmax < hold_start, f"rebalance {k}: estimator saw its own OOS data"
        assert wmax == r.index[estimation_window + k * holding_period - 1]
    # OOS daily returns exist only after the first estimation window
    assert daily.index.min() == r.index[estimation_window]
    # performance table covers all strategies with finite Sharpe
    assert (performance["sharpe"].abs() < np.inf).all()


def test_in_sample_sharpe_uses_same_inputs():
    r = _synthetic_returns()
    stats = in_sample_stats(r)
    assert (stats["sharpe"] > -np.inf).all()
    assert (stats["volatility"] > 0).all()
