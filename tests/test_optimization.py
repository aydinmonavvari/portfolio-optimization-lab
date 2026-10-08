"""Tests for the optimization layer (offline, synthetic inputs)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization_lab.optimization import (
    efficient_frontier,
    equal_weights,
    max_sharpe,
    min_variance,
    reset_solver_diagnostics,
    risk_contributions,
    risk_parity,
    solver_diagnostics,
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


# ---------------------------------------------------------------------------
# Optimality regression: max_sharpe must equal the EXACT long-only + capped
# tangency portfolio.
#
# Exact reference: every candidate optimum of
#     max (mu-rf)'w / sqrt(w'Sigma w)  s.t.  1'w = 1, 0 <= w <= cap
# has an activity pattern in {excluded, free, pinned at cap} per asset. For
# each pattern the Charnes-Cooper transform y = w / ((mu-rf)'w) turns the
# restricted problem into an equality-constrained convex QP (one linear
# solve), whose solution is the EXACT restricted optimum; the best feasible
# pattern solution is the global optimum. Plain support enumeration (the
# C = {} sub-case) is subsumed: once the cap binds, the assets pinned at the
# cap are part of the optimal support and must be enumerated too.
# ---------------------------------------------------------------------------


def _instance_cov(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Well-conditioned random covariance (60-sample correlation + jitter)."""
    a = rng.normal(size=(60, n))
    corr = np.corrcoef(a.T) + 1e-6 * np.eye(n)
    vols = rng.uniform(0.10, 0.35, size=n)
    return pd.DataFrame(corr * np.outer(vols, vols))


def _random_instance(seed: int) -> tuple[pd.Series, pd.DataFrame]:
    n = int(np.random.default_rng(seed).integers(3, 7))
    rng = np.random.default_rng(10_000 + seed)
    mu = pd.Series(rng.uniform(0.0, 0.25, size=n))
    return mu, _instance_cov(n, rng)


def _exact_capped_tangency(
    mu: np.ndarray, sigma: np.ndarray, cap: float
) -> tuple[float, np.ndarray | None]:
    """Exact optimum and its Sharpe by exhaustive activity-pattern enumeration."""
    n = len(mu)
    best_sharpe, best_w = -np.inf, None
    for pattern in itertools.product((0, 1, 2), repeat=n):  # 0 excl, 1 free, 2 capped
        free = [i for i, p in enumerate(pattern) if p == 1]
        capped = [i for i, p in enumerate(pattern) if p == 2]
        budget = 1.0 - len(capped) * cap
        if budget < -1e-12:
            continue
        if not free:
            if abs(budget) > 1e-12 or not capped:
                continue
            w = np.zeros(n)
            w[capped] = cap
            sharpe = _sharpe(w, mu, sigma)
            if sharpe > best_sharpe:
                best_sharpe, best_w = sharpe, w
            continue
        f = len(free)
        # QP in z = (y_free, t): min z'Qz  s.t.  1'y_f = budget*t, e'y_f + cap*e_C'1*t = 1
        q = np.zeros((f + 1, f + 1))
        q[:f, :f] = sigma[np.ix_(free, free)]
        q[:f, f] = cap * sigma[np.ix_(free, capped)].sum(axis=1)
        q[f, :f] = cap * sigma[np.ix_(capped, free)].sum(axis=0)
        q[f, f] = cap**2 * sigma[np.ix_(capped, capped)].sum()
        a = np.zeros((2, f + 1))
        a[0, :f] = 1.0
        a[0, f] = -budget
        a[1, :f] = mu[free]
        a[1, f] = cap * mu[capped].sum()
        kkt = np.block([[q, a.T], [a, np.zeros((2, 2))]])
        rhs = np.concatenate([np.zeros(f + 1), [0.0, 1.0]])
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            continue
        y = np.zeros(n)
        y[free] = sol[:f]
        y[capped] = cap * sol[f]
        t = sol[f]
        if t <= 1e-12 or y.min() < -1e-9 * t or (y - cap * t).max() > 1e-9 * t:
            continue
        sharpe = 1.0 / np.sqrt(max(float(y @ sigma @ y), 1e-18))
        if sharpe > best_sharpe:
            w = y / t
            assert abs(w.sum() - 1.0) < 1e-9 and w.min() > -1e-9 and w.max() < cap + 1e-9
            best_sharpe, best_w = sharpe, w
    return best_sharpe, best_w


def _sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    excess = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    return excess / vol if vol > 0 else -np.inf


# (seed, has_negative_closed_form, cap_binds_on_closed_form): fixed, seeded
# instances; seeds 2 and 6 are the required negative-weight + binding-cap
# cases; seeds 7/80/156 additionally fire the old clip+renormalize shortcut
# (negative closed-form whose clipped version is cap-feasible), so the suite
# fails on the pre-fix implementation.
_OPTIMALITY_CASES = [
    (1, False, False),
    (2, True, True),
    (3, False, False),
    (6, True, True),
    (7, True, False),
    (80, True, False),
    (156, True, True),
]


@pytest.mark.parametrize("seed,has_neg,cap_binds", _OPTIMALITY_CASES)
def test_max_sharpe_matches_exact_support_enumeration(seed: int, has_neg: bool, cap_binds: bool):
    """max_sharpe attains the exact long-only+capped optimum (rel. tol 1e-8)."""
    cap = 0.40
    mu, cov = _random_instance(seed)
    sigma = cov.to_numpy()

    # guard the test's own coverage: closed-form tangency properties
    w_cf = np.linalg.solve(sigma, mu.to_numpy())
    w_cf = w_cf / w_cf.sum()
    assert (w_cf.min() < 0.0) == has_neg
    assert (w_cf.max() > cap) == cap_binds

    w = max_sharpe(mu, cov, rf_annual=0.0, max_weight=cap)
    # the returned portfolio must be feasible
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert w.min() >= -1e-12
    assert w.max() <= cap + 1e-9

    exact_sharpe, exact_w = _exact_capped_tangency(mu.to_numpy(), sigma, cap)
    assert exact_w is not None
    impl_sharpe = _sharpe(w, mu.to_numpy(), sigma)
    rel_err = abs(impl_sharpe - exact_sharpe) / abs(exact_sharpe)
    assert rel_err < 1e-8, (
        f"seed {seed}: max_sharpe {impl_sharpe:.12f} != exact {exact_sharpe:.12f}"
    )
    # and the exact reference must never be beaten by a lower bound mismatch:
    assert impl_sharpe >= exact_sharpe - 1e-10


def test_solver_diagnostics_expose_safety_net_counters():
    """Solver fallback counters are exposed, shaped, and move on real calls."""
    reset_solver_diagnostics()
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 4))
    cov = pd.DataFrame(np.cov(a.T) * 252)
    mu = pd.Series([0.1, 0.08, 0.12, 0.05])
    min_variance(mu, cov, max_weight=0.4)
    diags = solver_diagnostics()
    assert diags["min_variance_clip_renorm"] >= 0  # counter exposed and finite
    assert set(diags) == {
        "min_variance_clip_renorm",
        "max_sharpe_closed_form",
        "max_sharpe_frontier_scan",
    }
    # the max_sharpe branch counters must move on a clean call
    reset_solver_diagnostics()
    max_sharpe(mu, cov, 0.0, 0.4)
    diags = solver_diagnostics()
    assert diags["max_sharpe_closed_form"] + diags["max_sharpe_frontier_scan"] == 1
