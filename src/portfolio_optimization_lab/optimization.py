"""Mean–variance and risk-parity portfolio construction.

Optimizers and their solvers:

- ``min_variance`` — quadratic program solved with ``scipy.optimize.minimize``
  (SLSQP) under long-only weights, full investment (weights sum to 1) and an
  optional per-asset upper bound. Objectives are rescaled internally
  (annualized variance entries are O(1e-2), which stalls SLSQP near the
  initial point).
- ``max_sharpe`` — closed-form unconstrained tangency ``Sigma^-1 (mu - rf)``
  when that vector already satisfies the constraint set; otherwise the
  constrained tangency is located on the efficient frontier traced with the
  min-variance solver (25-point target-return scan + ternary refinement).
- ``risk_parity`` — the dedicated cyclical coordinate descent algorithm of
  Griveau-Billion, Richard & Roncalli (2013), the standard robust method for
  the equal-risk-contribution problem (not SLSQP).

The min-variance solver applies a defensive clip-and-renormalize to the SLSQP
output; every time this safety net measurably alters a solution (and every
time each ``max_sharpe`` branch is taken) it is counted in
:func:`solver_diagnostics` so downstream runs can report solver behaviour
instead of hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_OBJECTIVE_SCALE = 1.0e3  # bring annualized-variance objectives to O(1)

# Tolerances for the defensive clip+renormalize safety net (see min_variance).
_SAFETY_NEG_TOL = 1e-9
_SAFETY_SUM_TOL = 1e-9

# Postcondition tolerance: after clip/renormalize the finished solution must
# satisfy every constraint to within this bound or the call FAILS LOUDLY
# (see _assert_solution_postconditions).
_POSTCONDITION_TOL = 1e-8

# Budget feasibility tolerance for the n * max_weight >= 1 precondition.
_BUDGET_TOL = 1e-9

# Counters for solver behaviour that would otherwise be invisible. Mutated in
# place (never rebound) so module-level state survives resets.
_SOLVER_DIAGNOSTICS: dict[str, int] = {
    "min_variance_clip_renorm": 0,
    "max_sharpe_closed_form": 0,
    "max_sharpe_frontier_scan": 0,
}


def solver_diagnostics() -> dict[str, int]:
    """Snapshot of solver fallback counters since the last reset.

    ``min_variance_clip_renorm`` counts how often the defensive
    clip-and-renormalize applied to the SLSQP output measurably changed the
    solution; ``max_sharpe_closed_form`` / ``max_sharpe_frontier_scan`` count
    which ``max_sharpe`` branch produced the returned weights.
    """
    return dict(_SOLVER_DIAGNOSTICS)


def reset_solver_diagnostics() -> None:
    """Zero all solver fallback counters (call at the start of a full run)."""
    _SOLVER_DIAGNOSTICS.update(dict.fromkeys(_SOLVER_DIAGNOSTICS, 0))


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


def _check_cap_budget(n: int, max_weight: float | None) -> None:
    """Fail fast on a constraint set no fully-invested portfolio can satisfy."""
    if max_weight is not None and n * max_weight < 1.0 - _BUDGET_TOL:
        raise ValueError(
            f"infeasible constraint set: {n} assets x max_weight {max_weight} "
            f"= {n * max_weight:.4f} < 1, so the weights can never sum to 1"
        )


def _validate_market_inputs(
    mu: pd.Series | None,
    cov: pd.DataFrame,
) -> np.ndarray:
    """Validate shapes, labels, finiteness and PSD-ness; return cov as ndarray.

    Raises ``ValueError`` (never silently mis-optimizes) when:
    - ``cov`` is not a square DataFrame, or its index and columns disagree;
    - ``mu`` is supplied and its index does not EXACTLY match ``cov.index``
      (silent positional alignment once produced an inverted portfolio);
    - any value of ``mu`` or ``cov`` is non-finite (NaN/inf);
    - any diagonal entry of ``cov`` is non-positive (zero-variance asset);
    - ``cov`` is materially asymmetric or not positive semidefinite.
    """
    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a pandas DataFrame")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if list(cov.index) != list(cov.columns):
        raise ValueError(
            "cov index and columns must be identical labels in the same order"
        )
    if mu is not None:
        if not isinstance(mu, pd.Series):
            raise ValueError("mu must be a pandas Series")
        if len(mu) != cov.shape[0]:
            raise ValueError(
                f"mu has {len(mu)} entries but cov is {cov.shape[0]}x{cov.shape[0]}"
            )
        if list(mu.index) != list(cov.index):
            raise ValueError(
                "mu.index does not match cov.index/labels; refusing to align "
                "silently — reindex mu and cov explicitly before optimizing"
            )
    mu_arr = mu.to_numpy(dtype=float) if mu is not None else None
    if mu_arr is not None and not np.isfinite(mu_arr).all():
        raise ValueError("mu contains non-finite values (NaN or inf)")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("cov contains non-finite values (NaN or inf)")
    if (np.diag(sigma) <= 0.0).any():
        raise ValueError(
            "covariance has a non-positive diagonal entry (zero-variance asset)"
        )
    scale = max(1.0, float(np.abs(sigma).max()))
    sym_err = float(np.abs(sigma - sigma.T).max())
    if sym_err > 1e-10 * scale:
        raise ValueError(
            f"covariance is not symmetric (max asymmetry {sym_err:.3e})"
        )
    min_eig = float(np.linalg.eigvalsh((sigma + sigma.T) / 2.0).min())
    if min_eig < -1e-10 * scale:
        raise ValueError(
            f"covariance is not positive semidefinite (min eigenvalue {min_eig:.3e})"
        )
    return sigma


def _assert_solution_postconditions(
    w: np.ndarray,
    mu_vec: np.ndarray,
    *,
    max_weight: float | None,
    target_return: float | None,
) -> None:
    """Raise RuntimeError if a finished solution violates the constraint set.

    Phase-1 hardening (item D): the clip-and-renormalize safety net must never
    silently return an invalid portfolio. Every constraint is re-checked on
    the FINAL weight vector; violations beyond tolerance are loud errors, not
    quiet corrections and not silent acceptance.
    """
    total = float(w.sum())
    if abs(total - 1.0) > _POSTCONDITION_TOL:
        raise RuntimeError(
            f"solution postcondition violated: weights sum to {total:.12f}, not 1"
        )
    w_min = float(w.min())
    if w_min < -_POSTCONDITION_TOL:
        raise RuntimeError(
            f"solution postcondition violated: min weight {w_min:.3e} < 0"
        )
    if max_weight is not None:
        w_max = float(w.max())
        if w_max > max_weight + _POSTCONDITION_TOL:
            raise RuntimeError(
                f"solution postcondition violated: max weight {w_max:.12f} "
                f"exceeds cap {max_weight}"
            )
    if target_return is not None:
        realized = float(w @ mu_vec)
        tol = _POSTCONDITION_TOL * max(1.0, abs(target_return))
        if abs(realized - target_return) > tol:
            raise RuntimeError(
                f"solution postcondition violated: realized return {realized:.12f} "
                f"off target {target_return:.12f} (tolerance {tol:.1e})"
            )


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
    """Minimum-variance portfolio, optionally at a target expected return.

    Inputs are validated (labels, finiteness, PSD covariance, feasible
    budget) and the finished solution is re-checked against EVERY constraint
    after the defensive clip-and-renormalize; a materially invalid solution
    raises instead of being returned.
    """
    sigma_full = _validate_market_inputs(mu, cov)
    m = mu.to_numpy(dtype=float)
    n = len(mu)
    _check_cap_budget(n, max_weight)

    sigma = sigma_full * _OBJECTIVE_SCALE

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
    # Defensive safety net: SLSQP can return tiny negative weights or a budget
    # off by float noise. Clip and renormalize, but COUNT every run where the
    # net measurably alters the solver's answer (thresholds above) — the
    # counter is exposed via solver_diagnostics() and persisted in the JSON
    # output so a systematically active safety net would be visible, not hidden.
    w = np.asarray(res.x, dtype=float)
    if (w < -_SAFETY_NEG_TOL).any() or abs(float(w.sum()) - 1.0) > _SAFETY_SUM_TOL:
        _SOLVER_DIAGNOSTICS["min_variance_clip_renorm"] += 1
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    # Postcondition: the FINAL vector must satisfy every constraint. This
    # turns any hypothetical silent invalid return into a loud error.
    _assert_solution_postconditions(
        w, m, max_weight=max_weight, target_return=target_return
    )
    return w


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf_annual: float = 0.0,
    max_weight: float | None = None,
) -> np.ndarray:
    """Maximum Sharpe ratio (tangency) portfolio under the constraint set.

    Algorithm (exactly what is implemented):

    1. **Closed-form candidate.** The unconstrained tangency portfolio has the
       closed form ``w \u221d Sigma^-1 (mu - rf)``. If that vector already
       satisfies long-only and cap constraints *without any modification*
       (``min(w) >= -1e-12`` and, when a cap is given, ``max(w) <= cap``), it
       IS the constrained optimum and is returned directly. The candidate is
       never clipped or renormalized into feasibility: if clipping would be
       required, the closed-form point is not the constrained optimum and we
       fall through to step 2 (returning a clipped candidate would be
       suboptimal whenever any weight is altered).
    2. **Constrained fallback: frontier scan + ternary search.** Otherwise the
       cap-feasible efficient frontier is traced with the (reliable)
       min-variance solver: the Sharpe ratio along the frontier is unimodal in
       the target return, so a 25-point scan over ``[0, max achievable excess
       return]`` (the latter from the greedy cap-filling portfolio) is refined
       by ternary search (40 iterations, interval tolerance 1e-7). Each scan
       target is an EXCESS return, mapped to the raw-return constraint as
       ``rf_annual + target``. The returned weights are the frontier point
       with the highest Sharpe.

    Raises
    ------
    TangencyInfeasibleError
        If no asset has positive excess return, or the maximum achievable
        cap-feasible excess return is non-positive (every risky portfolio is
        dominated by the risk-free asset).
    ValueError
        If the inputs are invalid (label mismatch, non-finite values,
        non-PSD covariance) or the constraint set is infeasible
        (``n_assets * max_weight < 1``).
    RuntimeError
        If the frontier scan fails entirely.
    """
    sigma = _validate_market_inputs(mu, cov)
    m = mu.to_numpy(dtype=float)
    excess = m - rf_annual
    n = len(mu)
    _check_cap_budget(n, max_weight)
    if (excess <= 0).all():
        raise TangencyInfeasibleError(
            "no asset has positive excess return; the tangency portfolio is undefined"
        )

    # 1) closed-form candidate: accept ONLY if feasible as-is (no clipping).
    # The cap is checked on the NORMALIZED vector: the raw solution of
    # Sigma^-1 (mu - rf) carries arbitrary units, and checking the cap before
    # normalization both rejected feasible portfolios and accepted infeasible
    # ones (whenever the raw sum < 1, normalization amplifies weights).
    try:
        w_cf = np.linalg.solve(sigma, excess)
        total = float(w_cf.sum())
        if total > 0 and float(w_cf.min()) >= -1e-12:
            w_norm = w_cf / total
            if max_weight is None or float(w_norm.max()) <= max_weight + 1e-12:
                _SOLVER_DIAGNOSTICS["max_sharpe_closed_form"] += 1
                return w_norm
    except np.linalg.LinAlgError:
        pass  # singular covariance: fall through to the frontier scan

    # 2) frontier scan: the tangency portfolio is the max-Sharpe point on the
    # constrained efficient frontier, which we can trace robustly with the
    # (reliable) min-variance solver, then refine by ternary search on the target
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
    if remaining > 1e-9:
        raise RuntimeError(
            "cap-filling greedy allocation could not invest the full budget; "
            "the constraint set is infeasible"
        )
    max_excess = float(excess @ w_greedy)
    if max_excess <= 0:
        raise TangencyInfeasibleError(
            "the maximum achievable cap-feasible expected excess return is "
            f"{max_excess:.4f} <= 0; the tangency portfolio is undefined for this window"
        )

    def sharpe_at(target: float) -> tuple[float, np.ndarray] | None:
        """Sharpe of the frontier point whose EXCESS return equals ``target``.

        The min-variance equality constraint lives in RAW return space, so the
        excess-space target is mapped to ``rf_annual + target``; the previous
        implementation passed the excess target directly, silently searching
        the wrong return interval whenever rf != 0 and missing the tangency.
        """
        try:
            w = min_variance(
                mu, cov, max_weight=max_weight, target_return=rf_annual + target
            )
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
    _SOLVER_DIAGNOSTICS["max_sharpe_frontier_scan"] += 1
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
        If the iteration fails to converge, if the solution does not reach
        equal contributions, or, when a ``max_weight`` cap is supplied, if
        the ERC solution violates it (the cap-feasible ERC problem would
        need a different formulation, which we refuse to silently
        approximate).
    ValueError
        If the covariance is invalid (non-finite, non-PSD, zero-variance
        asset, label mismatch) or ``n_assets * max_weight < 1`` makes the
        constraint set infeasible.
    """
    sigma = _validate_market_inputs(None, cov)
    n = sigma.shape[0]
    _check_cap_budget(n, max_weight)
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
