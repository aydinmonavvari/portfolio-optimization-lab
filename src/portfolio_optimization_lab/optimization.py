"""Mean–variance and risk-parity portfolio construction.

All optimizers solve constrained problems with ``scipy.optimize.minimize``
(SLSQP) under: long-only weights, full investment (weights sum to 1) and an
optional per-asset upper bound. Objectives are rescaled internally (annualized
variance entries are O(1e-2), which stalls SLSQP near the initial point);
risk parity uses the dedicated cyclical coordinate descent algorithm of
Griveau-Billion, Richard & Roncalli (2013), which is the standard robust
method for the equal-risk-contribution problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_OBJECTIVE_SCALE = 1.0e3  # bring annualized-variance objectives to O(1)


class TangencyInfeasibleError(RuntimeError):
    """Raised when no cap-feasible portfolio has positive expected excess return.

    In such windows every risky portfolio is dominated by the risk-free asset
    and the tangency portfolio is mathematically undefined. Callers must
    handle this case explicitly (the walk-forward backtest falls back to the
    minimum-variance portfolio and logs the fallback).
    """



def _bounds(n: int, max_weight: float | None) -> list[tuple[float, float]]:
    hi = max_weight if max_weight is not None else 1.0
    return [(0.0, hi)] * n


def _budget_constraint() -> dict:
    return {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}


def equal_weights(n: int) -> np.ndarray:
    """The 1/n portfolio."""
    return np.full(n, 1.0 / n)


def min_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float | None = None,
    target_return: float | None = None,
) -> np.ndarray:
    """Minimum-variance portfolio, optionally at a target expected return."""
    sigma = cov.to_numpy() * _OBJECTIVE_SCALE
    m = mu.to_numpy()
    n = len(mu)

    def obj(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    cons = [_budget_constraint()]
    if max_weight is not None:
        cons.append({"type": "ineq", "fun": lambda w: max_weight - w})
    if target_return is not None:
        cons.append(
            {"type": "eq", "fun": lambda w: float(w @ m) - target_return}
        )
    res = minimize(
        obj,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"min-variance optimizer failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf_annual: float = 0.0,
    max_weight: float | None = None,
) -> np.ndarray:
    """Maximum Sharpe ratio (tangency) portfolio under the constraint set.

    Strategy: the unconstrained tangency portfolio has the closed form
    ``w ∝ Sigma^-1 (mu - rf)``. If that solution already satisfies the
    long-only and cap constraints it IS the constrained optimum and is
    returned directly. Otherwise the problem is solved numerically via the
    convex reformulation of Cornuejols & Tütüncü (2006): minimize
    ``y' Sigma y`` s.t. ``(mu - rf)' y = 1``, ``y >= 0``, ``y_i <= cap*sum(y)``,
    then ``w = y / sum(y)``.
    """
    sigma = cov.to_numpy()
    excess = mu.to_numpy() - rf_annual
    n = len(mu)
    if (excess <= 0).all():
        raise TangencyInfeasibleError(
            "no asset has positive excess return; the tangency portfolio is undefined"
        )

    # 1) closed-form candidate (exact when constraints are inactive)
    try:
        w_cf = np.linalg.solve(sigma, excess)
        w_cf = np.clip(w_cf, 0.0, None)
        if w_cf.sum() > 0:
            w_cf /= w_cf.sum()
            if max_weight is None or (w_cf <= max_weight + 1e-9).all():
                return w_cf
    except np.linalg.LinAlgError:
        w_cf = None

    # 2) frontier scan: the tangency portfolio is the max-Sharpe point on the
    # constrained efficient frontier, which we can trace robustly with the
    # (reliable) min-variance solver, then refine by bisection on the target
    cap = max_weight if max_weight is not None else 1.0
    order = np.argsort(-excess)
    w_greedy = np.zeros(n)
    remaining = 1.0
    for i in order:
        take = min(remaining, cap)
        w_greedy[i] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    max_excess = float(excess @ w_greedy)
    if max_excess <= 0:
        raise TangencyInfeasibleError(
            "the maximum achievable cap-feasible expected excess return is "
            f"{max_excess:.4f} <= 0; the tangency portfolio is undefined for this window"
        )

    def sharpe_at(target: float) -> tuple[float, np.ndarray] | None:
        try:
            w = min_variance(mu, cov, max_weight=max_weight, target_return=target)
        except RuntimeError:
            return None
        vol = float(np.sqrt(max(w @ sigma @ w, 1e-18)))
        return (float(w @ excess) / vol, w)

    # Sharpe along the frontier is unimodal in the target return; locate the
    # peak with a dense scan and refine by ternary search.
    lo, hi = 0.0, max_excess
    best: tuple[float, np.ndarray] | None = None
    grid = np.linspace(lo, hi, 25)
    evaluated = [sharpe_at(t) for t in grid]
    pairs = [e for e in evaluated if e is not None]
    if not pairs:
        raise RuntimeError("max-Sharpe frontier scan failed")
    best = max(pairs, key=lambda p: p[0])
    a, b = lo, hi
    for _ in range(40):
        if b - a < 1e-7:
            break
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        s1 = sharpe_at(m1)
        s2 = sharpe_at(m2)
        if s1 is None:
            a = m1
            continue
        if s2 is None:
            b = m2
            continue
        if s1[0] < s2[0]:
            a = m1
            best = max(best, s2, key=lambda p: p[0])
        else:
            b = m2
            best = max(best, s1, key=lambda p: p[0])
    return best[1]


def risk_parity(
    cov: pd.DataFrame,
    max_weight: float | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Equal-risk-contribution portfolio via cyclical coordinate descent.

    Solves, for each coordinate ``i`` in turn, the fixed-point equation
    ``w_i (Sigma w)_i = b_i`` (``b_i = 1/n``) using the closed-form quadratic
    update of Griveau-Billion, Richard & Roncalli (2013). For a
    positive-definite covariance the iteration converges to the unique
    long-only ERC portfolio.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge or, when a ``max_weight`` cap is
        supplied, if the ERC solution violates it (the cap-feasible ERC
        problem would need a different formulation, which we refuse to
        silently approximate).
    """
    sigma = cov.to_numpy()
    n = sigma.shape[0]
    b = np.full(n, 1.0 / n)
    vol = np.sqrt(np.diag(sigma))
    w = (1.0 / vol) / (1.0 / vol).sum()  # canonical starting point

    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            cross = float(sigma[i] @ w) - sigma[i, i] * w[i]
            disc = cross * cross + 4.0 * sigma[i, i] * b[i]
            w[i] = (-cross + np.sqrt(disc)) / (2.0 * sigma[i, i])
        if float(np.max(np.abs(w - w_prev))) < tolerance:
            break
    else:
        raise RuntimeError("risk-parity CCD did not converge")

    w = w / w.sum()
    rc = risk_contributions(w, cov)
    if float(np.max(np.abs(rc - 1.0 / n))) > 1e-6:
        raise RuntimeError("risk-parity solution did not reach equal contributions")
    if max_weight is not None and (w > max_weight + 1e-9).any():
        raise RuntimeError(
            "the long-only ERC portfolio violates the max_weight cap; "
            "cap-feasible risk parity requires a different formulation"
        )
    return w


def risk_contributions(w: np.ndarray, cov: pd.DataFrame) -> np.ndarray:
    """Fractional risk contributions of each asset (sum to 1)."""
    w = np.asarray(w, dtype=float)
    sigma = cov.to_numpy()
    marginal = sigma @ w
    total = float(w @ marginal)
    return (w * marginal) / total


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float | None = None,
    n_points: int = 30,
    ret_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Trace the constrained efficient frontier.

    Returns a DataFrame with columns ``target_return`` (realized expected
    return of the solution), ``volatility`` and ``sharpe`` of the
    minimum-variance portfolio at each target return, plus the weight matrix.
    """
    m = mu.to_numpy()
    if ret_range is None:
        lo, hi = float(m.min()), float(m.max())
        ret_range = (lo, hi)
    rows: list[dict] = []
    weights = np.full((n_points, len(mu)), np.nan)
    for i, target in enumerate(np.linspace(*ret_range, n_points)):
        try:
            w = min_variance(mu, cov, max_weight=max_weight, target_return=target)
        except RuntimeError:
            # the frontier is empty beyond the constraint-feasible region
            rows.append({"target_return": target, "volatility": np.nan, "sharpe": np.nan})
            continue
        vol = float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
        rows.append({"target_return": float(w @ m), "volatility": vol, "sharpe": np.nan})
        weights[i] = w
    frontier = pd.DataFrame(rows)
    frontier["sharpe"] = frontier["target_return"] / frontier["volatility"]
    frontier["weights"] = list(weights)
    return frontier
