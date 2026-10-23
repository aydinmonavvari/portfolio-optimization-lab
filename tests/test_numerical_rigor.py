"""Phase-1 numerical-rigor regression tests (branch fix/numerical-rigor).

Every defect-oriented test here was demonstrated with a minimal hand-checked
example BEFORE the fix (red), and pins the corrected behavior (green):

- A: ``max_sharpe`` must apply ``max_weight`` to the NORMALIZED closed-form
  candidate; the cap decision must be invariant to the scale of the
  excess-return vector (it previously was not, returning weight vectors that
  violate the cap whenever ``sum(Sigma^-1 (mu - rf)) < 1``).
- B: the constrained-fallback frontier scan must evaluate target returns in
  the SAME units as the min-variance equality constraint (raw returns =
  rf + excess target); the previous grid silently mixed excess-space targets
  with raw-space constraints and missed the true tangency whenever rf > 0.
- C: the backtest consumes daily SIMPLE returns and annualizes geometrically
  via ``log1p`` (the previous engine summed simple returns and exponentiated,
  i.e. treated them as log returns); turnover is the pre-registered ONE-WAY
  convention ``sum(max(w - w_prev, 0))`` (initial build-up from cash = 1.0),
  matching docs/admissions/PHASE2-RESEARCH-PLAN.md.
- D: solver output postconditions (defensive hardening; no violating case was
  found in ~2400 adversarial searches, so the battery below is a guard, and
  the guard function itself is unit-tested directly).
- E: input validation (label alignment, finiteness, PSD covariance,
  positive variances, feasible n*max_weight budget) raises ValueError instead
  of silently mis-optimizing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_optimization import _exact_capped_tangency  # exact reference solver

from portfolio_optimization_lab.backtest import walk_forward_backtest
from portfolio_optimization_lab.data import simple_returns
from portfolio_optimization_lab.optimization import (
    max_sharpe,
    min_variance,
    risk_parity,
)

# ---------------------------------------------------------------------------
# A) cap must be enforced on normalized weights, scale-invariantly
# ---------------------------------------------------------------------------

# Hand-computed instance: Sigma = [[.04, .01], [.01, .09]], excess = [.02, .01]
# => w_cf = Sigma^-1 excess = [0.485714, 0.057143], sum = 0.542857
# => normalized = [0.894737, 0.105263] which VIOLATES cap 0.6 on asset 0.
_A_MU = pd.Series([0.03, 0.02], index=["A", "B"])  # rf = 0.01
_A_COV = pd.DataFrame(
    [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
)


def test_max_sharpe_cap_holds_on_normalized_closed_form():
    """The returned weights must respect the cap after normalization."""
    w = max_sharpe(_A_MU, _A_COV, rf_annual=0.01, max_weight=0.6)
    assert w.max() <= 0.6 + 1e-9, f"cap violated: {w}"
    # and it must be the exact constrained tangency, not merely feasible:
    excess = _A_MU.to_numpy() - 0.01
    exact_sharpe, exact_w = _exact_capped_tangency(excess, _A_COV.to_numpy(), 0.6)
    sigma = _A_COV.to_numpy()
    impl_sharpe = float(excess @ w) / float(np.sqrt(w @ sigma @ w))
    assert impl_sharpe == pytest.approx(exact_sharpe, rel=1e-8)
    np.testing.assert_allclose(w, exact_w, atol=1e-7)


@pytest.mark.parametrize("scale", [0.25, 0.5, 0.75, 1.0, 2.0, 4.0])
def test_max_sharpe_cap_decision_is_scale_invariant(scale: float):
    """Same normalized tangency for every positive scaling of excess returns.

    Pre-fix, scale=1.0 returned the cap-violating [0.8947, 0.1053] while
    scale=4.0 (raw candidate too large) fell through to the frontier scan and
    returned the correct [0.6, 0.4] — the cap decision depended on arbitrary
    input units. Post-fix every scale returns identical, feasible weights.
    """
    excess = np.array([0.02, 0.01]) * scale
    mu = pd.Series(excess + 0.01, index=["A", "B"])  # rf = 0.01
    w = max_sharpe(mu, _A_COV, rf_annual=0.01, max_weight=0.6)
    assert w.max() <= 0.6 + 1e-9
    np.testing.assert_allclose(w, [0.6, 0.4], atol=1e-6)


# ---------------------------------------------------------------------------
# B) fallback scan must be consistent with rf
# ---------------------------------------------------------------------------

# n=3, cap=0.5, rf=0.03: closed-form has negative weights -> fallback fires.
# Greedy max excess = 0.5*0.07 + 0.5*(-0.005) = 0.0325; the pre-fix scan
# searched RAW returns in [0, 0.0325] while the true tangency sits at raw
# return 0.0614 — it returned a ~zero-Sharpe portfolio (0.0199) against the
# exact optimum (0.2558).
_B_MU = pd.Series([0.10, 0.02, 0.025])
_B_SIGMA = np.diag([0.04, 0.04, 0.04])
_B_RF = 0.03
_B_CAP = 0.5


@pytest.mark.parametrize("rf", [0.0, 0.02, 0.03, 0.05])
def test_max_sharpe_fallback_matches_exact_tangency_with_rf(rf: float):
    w = max_sharpe(_B_MU, pd.DataFrame(_B_SIGMA), rf_annual=rf, max_weight=_B_CAP)
    excess = _B_MU.to_numpy() - rf
    exact_sharpe, _ = _exact_capped_tangency(excess, _B_SIGMA, _B_CAP)
    impl_sharpe = float(excess @ w) / float(np.sqrt(w @ _B_SIGMA @ w))
    assert impl_sharpe == pytest.approx(exact_sharpe, rel=1e-8), (
        f"rf={rf}: implementation {impl_sharpe:.8f} != exact {exact_sharpe:.8f}"
    )


# ---------------------------------------------------------------------------
# C) simple-return convention, geometric annualization, one-way turnover
# ---------------------------------------------------------------------------


def _two_block_frame() -> pd.DataFrame:
    """est=2, hold=2, two OOS blocks; block 1 is the hand-checked +10/-10%."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.10, -0.10, 0.0, 0.0],
            "B": [0.0, 0.0, 0.00, 0.00, 0.0, 0.0],
        },
        index=idx,
    )


def test_backtest_annualization_is_geometric_on_simple_returns():
    """Hand case: OOS gross growth must be 1.05 * 0.95 = 0.9975 exactly.

    Pre-fix the engine summed the daily simple portfolio returns and
    exponentiated (log-return annualization of simple data): sum = 0 =>
    reported ann_return_gross = 0.0 instead of the true -14.59%.
    """
    rets = _two_block_frame()
    daily, perf, _ = walk_forward_backtest(
        rets,
        strategies=("equal_weight",),
        estimation_window=2,
        holding_period=2,
        max_weight=1.0,
        cost_bps=0.0,
    )
    # per-day portfolio return is the exact simple weighted return
    np.testing.assert_allclose(
        daily["equal_weight"].to_numpy(), [0.05, -0.05, 0.0, 0.0], atol=1e-15
    )
    # geometric annualization from log1p
    years = 4 / 252
    expected = float(np.exp(np.log1p(np.array([0.05, -0.05, 0.0, 0.0])).sum() / years) - 1.0)
    assert perf.loc["equal_weight", "ann_return_gross"] == pytest.approx(expected, rel=1e-12)
    assert perf.loc["equal_weight", "ann_return_gross"] != pytest.approx(0.0, abs=1e-12)
    # cumulative growth of the net series (0 cost) equals the hand value
    growth = float(np.prod(1.0 + daily["equal_weight"].to_numpy()))
    assert growth == pytest.approx(0.9975, rel=1e-12)


def test_net_return_deducts_one_way_cost_once_per_rebalance():
    """cost_bps=100, initial build-up turnover 1.0 => day-1 net = 0.05 - 0.01."""
    rets = _two_block_frame()
    daily, perf, _ = walk_forward_backtest(
        rets,
        strategies=("equal_weight",),
        estimation_window=2,
        holding_period=2,
        max_weight=1.0,
        cost_bps=100.0,
    )
    np.testing.assert_allclose(
        daily["equal_weight"].to_numpy(), [0.04, -0.05, 0.0, 0.0], atol=1e-15
    )
    # one rebalance with one-way turnover 1.0 => total cost = 1.0 * 1% = 1%
    assert perf.loc["equal_weight", "total_costs_pct"] == pytest.approx(1.0, rel=1e-12)
    # mean over the 2 rebalances: initial build-up 1.0, then 0.0
    assert perf.loc["equal_weight", "avg_turnover"] == pytest.approx(0.5, rel=1e-12)
    assert perf.loc["equal_weight", "n_rebalances"] == 2


def test_turnover_is_one_way_sum_of_positive_weight_changes():
    """Registered one-way convention: turn = sum(max(w - w_prev, 0)).

    Engineered flip: max_sharpe goes all-in A, then all-in B, then back to A.
    Pre-fix turnover counted both sides (sum|dw| = 2.0 per flip); the
    registered convention counts the one-way traded fraction (1.0).
    """
    idx = pd.bdate_range("2024-01-01", periods=12)
    bull_a = [0.05, 0.06, 0.04]
    bear_a = [-0.01, 0.005, 0.0]
    rets = pd.DataFrame(
        {
            "A": bull_a + bear_a + bull_a + [0.0, 0.0, 0.0],
            "B": bear_a + bull_a + bear_a + [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    _, perf, weights = walk_forward_backtest(
        rets,
        strategies=("max_sharpe",),
        estimation_window=3,
        holding_period=3,
        max_weight=1.0,
        cost_bps=0.0,
    )
    w = weights.drop(columns=["strategy"]).to_numpy()
    # all-in A, then all-in B, then all-in A again
    np.testing.assert_allclose(w[0], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(w[1], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(w[2], [1.0, 0.0], atol=1e-9)
    # turnover: initial 1.0, two one-way flips of 1.0 each
    assert perf.loc["max_sharpe", "n_rebalances"] == 3
    assert perf.loc["max_sharpe", "avg_turnover"] == pytest.approx(1.0, rel=1e-9)


def test_simple_returns_definition():
    """simple_returns is the exact per-day fractional price change."""
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    out = simple_returns(prices)
    np.testing.assert_allclose(out["A"].to_numpy(), [0.10, 99.0 / 110.0 - 1.0], atol=1e-15)
    np.testing.assert_allclose(out["B"].to_numpy(), [0.0, 0.10], atol=1e-15)
    # exact daily-rebalanced portfolio algebra: growth = prod(1 + r @ w)
    port = 0.5 * out["A"] + 0.5 * out["B"]
    np.testing.assert_allclose(
        float(np.prod(1.0 + port)),
        (1.0 + 0.5 * 0.10) * (1.0 + 0.5 * (99.0 / 110.0 - 1.0) + 0.5 * 0.10),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# D) solver postconditions (defensive hardening)
# ---------------------------------------------------------------------------


def test_min_variance_postconditions_battery():
    """~200 seeded instances: every successful solution satisfies ALL constraints.

    Defensive guard: no violating case was found pre-fix (worst violation
    ~2e-9); this battery pins that the safety net never returns a materially
    invalid portfolio.
    """
    rng = np.random.default_rng(2026)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(2, 8))
        a = rng.normal(size=(60, n))
        cov = pd.DataFrame(np.cov(a.T) * 252)
        if np.linalg.eigvalsh(cov.to_numpy()).min() < 1e-8:
            continue
        mu = pd.Series(rng.uniform(0.0, 0.2, size=n))
        cap = float(rng.uniform(0.25, 1.0))
        if n * cap < 1.0:
            continue
        target = float(rng.uniform(mu.min(), mu.max()))
        try:
            w = min_variance(mu, cov, max_weight=cap, target_return=target)
        except RuntimeError:
            continue  # legitimately infeasible targets may fail loudly
        checked += 1
        assert abs(float(w.sum()) - 1.0) < 1e-8
        assert float(w.min()) >= -1e-8
        assert float(w.max()) <= cap + 1e-8
        realized = float(w @ mu.to_numpy())
        assert abs(realized - target) <= 1e-8 * max(1.0, abs(target))
    assert checked >= 50, f"battery too small: {checked}"


def test_postcondition_guard_raises_on_violating_solution():
    """The guard converts an invalid solver output into a loud error."""
    from portfolio_optimization_lab.optimization import _assert_solution_postconditions

    mu = np.array([0.10, 0.05, 0.02])
    good = np.array([0.5, 0.3, 0.2])
    _assert_solution_postconditions(good, mu, max_weight=0.6, target_return=None)

    cap_violation = np.array([0.7, 0.2, 0.1])
    with pytest.raises(RuntimeError, match="max weight"):
        _assert_solution_postconditions(cap_violation, mu, max_weight=0.6, target_return=None)

    budget_violation = np.array([0.6, 0.3, 0.2])  # sums to 1.1
    with pytest.raises(RuntimeError, match="sum"):
        _assert_solution_postconditions(budget_violation, mu, max_weight=None, target_return=None)

    negative_weight = np.array([1.2, -0.2, 0.0])
    with pytest.raises(RuntimeError, match="min weight"):
        _assert_solution_postconditions(negative_weight, mu, max_weight=None, target_return=None)

    target_violation = np.array([0.4, 0.4, 0.2])  # realized 0.066 vs target 0.10
    with pytest.raises(RuntimeError, match="target"):
        _assert_solution_postconditions(
            target_violation, mu, max_weight=None, target_return=0.10
        )


# ---------------------------------------------------------------------------
# E) input validation
# ---------------------------------------------------------------------------


def test_label_mismatch_raises_instead_of_silent_misalignment():
    """Pre-fix this silently returned the INVERTED portfolio (HIGH 99.75%)."""
    mu = pd.Series([0.20, 0.00], index=["HIGH", "LOW"])
    cov = pd.DataFrame(
        [[0.0001, 0.0], [0.0, 0.04]],
        index=["LOW", "HIGH"],  # deliberately inconsistent ordering
        columns=["LOW", "HIGH"],
    )
    with pytest.raises(ValueError, match="index"):
        min_variance(mu, cov)
    with pytest.raises(ValueError, match="index"):
        max_sharpe(mu, cov)
    # risk_parity takes no mu: the analogous defect is cov whose index and
    # columns disagree with each other
    cov_mis = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["B", "A"]
    )
    with pytest.raises(ValueError, match="index|labels"):
        risk_parity(cov_mis)


def test_nonfinite_inputs_raise():
    cov = pd.DataFrame(np.eye(3) * 0.04)
    with pytest.raises(ValueError, match="non-finite"):
        min_variance(pd.Series([np.inf, 0.05, 0.06]), cov)
    with pytest.raises(ValueError, match="non-finite"):
        min_variance(pd.Series([np.nan, 0.05, 0.06]), cov)
    with pytest.raises(ValueError, match="non-finite"):
        risk_parity(pd.DataFrame(np.diag([0.04, np.nan, 0.01])))


def test_non_psd_covariance_raises():
    # implied correlation 0.05 / (0.2 * 0.1) = 2.5 > 1 -> not PSD
    cov = pd.DataFrame([[0.04, 0.05], [0.05, 0.01]])
    with pytest.raises(ValueError, match="semidefinite|symmetric|eigenvalue"):
        min_variance(pd.Series([0.1, 0.1]), cov)
    with pytest.raises(ValueError, match="semidefinite|symmetric|eigenvalue"):
        max_sharpe(pd.Series([0.1, 0.1]), cov)


def test_zero_variance_asset_raises_cleanly():
    cov = pd.DataFrame(np.diag([0.04, 0.0, 0.01]))
    with pytest.raises(ValueError, match="variance|zero-vol"):
        risk_parity(cov)
    with pytest.raises(ValueError, match="variance|zero-vol"):
        min_variance(pd.Series([0.1, 0.1, 0.1]), cov)


def test_infeasible_budget_n_times_cap_below_one_raises():
    """3 assets x cap 0.3 = 0.9 < 1: no fully-invested feasible portfolio."""
    mu = pd.Series([0.1, 0.1, 0.1])
    cov = pd.DataFrame(np.eye(3) * 0.04)
    with pytest.raises(ValueError, match="[Ii]nfeasible"):
        min_variance(mu, cov, max_weight=0.3)
    with pytest.raises(ValueError, match="[Ii]nfeasible"):
        max_sharpe(mu, cov, 0.0, 0.3)
    with pytest.raises(ValueError, match="[Ii]nfeasible"):
        risk_parity(cov, max_weight=0.3)


def test_valid_inputs_still_optimize():
    """Validation must not reject the well-formed cases (no over-blocking)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 4))
    cov = pd.DataFrame(np.cov(a.T) * 252)
    mu = pd.Series([0.1, 0.08, 0.12, 0.05])
    for w in (
        min_variance(mu, cov, max_weight=0.4),
        max_sharpe(mu, cov, rf_annual=0.02, max_weight=0.4),
        risk_parity(cov, max_weight=0.4),
    ):
        assert abs(float(w.sum()) - 1.0) < 1e-9
        assert (w >= -1e-12).all()
        assert (w <= 0.4 + 1e-6).all()
