"""Walk-forward out-of-sample evaluation of portfolio strategies.

At each rebalance date ``t`` the estimators see ONLY the trailing
``estimation_window`` trading days; the resulting weights are then held for
the following ``holding_period`` days and scored on returns realized *after*
``t``. Estimation and evaluation windows are therefore strictly disjoint by
construction — a property covered by unit tests.

Transaction costs are deducted from the strategy returns themselves: on each
rebalance day the one-way turnover (fraction of the portfolio traded) times
``cost_bps / 1e4`` is subtracted from that day's return, so every reported
headline metric (annualized return, volatility, Sharpe, drawdowns, cumulative
growth) is a TRUE net-of-cost number. Gross-of-cost metrics are retained as
reference columns so the cost drag is always visible.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .estimates import estimate, portfolio_stats
from .optimization import (
    TangencyInfeasibleError,
    equal_weights,
    max_sharpe,
    min_variance,
    risk_parity,
)

logger = logging.getLogger(__name__)

STRATEGIES = ("equal_weight", "min_variance", "max_sharpe", "risk_parity")


def weights_for(
    strategy: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float | None,
) -> np.ndarray:
    """Dispatch one strategy name to its weight vector.

    The max-Sharpe strategy falls back to the minimum-variance portfolio in
    windows where the tangency portfolio is undefined (no cap-feasible mix
    has positive expected excess return). Such fallbacks are expected on
    short, noisy estimation windows and are logged.
    """
    n = len(mu)
    if strategy == "equal_weight":
        return equal_weights(n)
    if strategy == "min_variance":
        return min_variance(mu, cov, max_weight=max_weight)
    if strategy == "max_sharpe":
        try:
            return max_sharpe(mu, cov, max_weight=max_weight)
        except TangencyInfeasibleError:
            logger.warning(
                "tangency portfolio undefined in this window; falling back to "
                "minimum variance"
            )
            return min_variance(mu, cov, max_weight=max_weight)
    if strategy == "risk_parity":
        return risk_parity(cov, max_weight=max_weight)
    raise KeyError(f"unknown strategy: {strategy}")


def walk_forward_backtest(
    returns: pd.DataFrame,
    strategies: tuple[str, ...] = STRATEGIES,
    estimation_window: int = 252,
    holding_period: int = 63,
    max_weight: float | None = 0.40,
    cost_bps: float = 10.0,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    use_shrunk_cov: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the rolling estimation/holding study, net of transaction costs.

    On every rebalance day the one-way turnover ``t`` (fraction of the
    portfolio traded, as computed below) generates a cost ``t * cost_bps/1e4``
    which is subtracted from that day's strategy return. All headline metrics
    are computed on this NET daily series; gross metrics are kept for
    reference so the cost drag can be quantified.

    Returns
    -------
    daily_returns:
        DataFrame of out-of-sample daily NET log returns per strategy (indexed
        by the full out-of-sample date range). Cost drags are embedded in the
        rebalance-day entries.
    performance:
        Per-strategy summary computed on the NET series: annualized
        return/volatility, Sharpe, average rebalance turnover and total
        transaction costs, plus ``ann_return_gross`` / ``sharpe_gross``
        reference columns computed before cost deduction.
    weights_history:
        One row per (rebalance date, strategy) with the weight vector.
    """
    n_needed = estimation_window + holding_period + 1
    if len(returns) < n_needed:
        raise ValueError(f"need >= {n_needed} return rows, got {len(returns)}")

    r = returns.to_numpy()
    index = returns.index
    n_assets = returns.shape[1]
    columns = returns.columns

    daily_gross: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    daily_net: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    turnover: dict[str, list[float]] = {s: [] for s in strategies}
    weights_history_rows: list[pd.Series] = []

    prev_weights: dict[str, np.ndarray] = {s: np.zeros(n_assets) for s in strategies}
    start = estimation_window
    while start + holding_period <= len(returns):
        est = estimate(
            returns.iloc[start - estimation_window : start],
            periods_per_year=periods_per_year,
            use_shrunk=use_shrunk_cov,
        )
        hold_returns = r[start : start + holding_period]

        for strategy in strategies:
            w = weights_for(strategy, est.mu, est.cov, max_weight)
            gross_block = hold_returns @ w
            turn = (
                float(np.abs(w - prev_weights[strategy]).sum())
                if prev_weights[strategy].sum() > 0
                else float(np.abs(w).sum())  # initial build-up counts as turnover
            )
            # True cost deduction: the rebalance-day return pays turnover x bps.
            net_block = gross_block.copy()
            net_block[0] -= turn * cost_bps / 1e4
            daily_gross[strategy].append(gross_block)
            daily_net[strategy].append(net_block)
            turnover[strategy].append(turn)
            prev_weights[strategy] = w
            row = pd.Series(w, index=columns, name=index[start])
            row["strategy"] = strategy
            weights_history_rows.append(row)
        start += holding_period

    daily_returns = pd.DataFrame(
        {s: np.concatenate(daily_net[s]) for s in strategies},
        index=index[estimation_window:start],
    )
    gross_returns = pd.DataFrame(
        {s: np.concatenate(daily_gross[s]) for s in strategies},
        index=index[estimation_window:start],
    )
    weights_history = pd.DataFrame(weights_history_rows)

    rows = []
    oos_days = len(daily_returns)
    years = oos_days / periods_per_year
    for strategy in strategies:
        series = daily_returns[strategy]
        gross_series = gross_returns[strategy]
        total_log = float(series.sum())
        ann_return = float(np.exp(total_log / years) - 1.0)
        ann_vol = float(series.std(ddof=1) * np.sqrt(periods_per_year))
        ann_return_gross = float(np.exp(float(gross_series.sum()) / years) - 1.0)
        ann_vol_gross = float(gross_series.std(ddof=1) * np.sqrt(periods_per_year))
        costs_total = float(np.sum(turnover[strategy]) * cost_bps / 1e4)
        sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else float("nan")
        sharpe_gross = (
            (ann_return_gross - rf_annual) / ann_vol_gross if ann_vol_gross > 0 else float("nan")
        )
        rows.append(
            {
                "strategy": strategy,
                "oos_days": oos_days,
                "n_rebalances": len(turnover[strategy]),
                "ann_return": ann_return,
                "ann_volatility": ann_vol,
                "sharpe": sharpe,
                "ann_return_gross": ann_return_gross,
                "sharpe_gross": sharpe_gross,
                "avg_turnover": float(np.mean(turnover[strategy])),
                "total_costs_pct": costs_total * 100.0,
            }
        )
    performance = pd.DataFrame(rows).set_index("strategy")
    return daily_returns, performance, weights_history


def walk_forward_shrinkage(
    returns: pd.DataFrame,
    estimation_window: int = 252,
    holding_period: int = 63,
    periods_per_year: int = 252,
) -> list[float]:
    """Ledoit–Wolf shrinkage intensity of every walk-forward estimation window.

    Mirrors the window enumeration of :func:`walk_forward_backtest` exactly
    (same slices, same order) so callers can report the intensities the
    backtest actually used without re-running the optimization.
    """
    intensities: list[float] = []
    start = estimation_window
    while start + holding_period <= len(returns):
        est = estimate(
            returns.iloc[start - estimation_window : start],
            periods_per_year=periods_per_year,
        )
        intensities.append(est.shrinkage)
        start += holding_period
    return intensities


def in_sample_stats(
    returns: pd.DataFrame,
    strategies: tuple[str, ...] = STRATEGIES,
    max_weight: float | None = 0.40,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    use_shrunk_cov: bool = True,
) -> pd.DataFrame:
    """In-sample statistics of the same strategies on one estimation window.

    Comparing these numbers with the walk-forward output quantifies the
    estimation-error degradation this project is designed to demonstrate.
    """
    est = estimate(returns, periods_per_year, use_shrunk=use_shrunk_cov)
    rows = []
    for strategy in strategies:
        w = weights_for(strategy, est.mu, est.cov, max_weight)
        stats = portfolio_stats(w, est.mu, est.cov, rf_annual=rf_annual)
        rows.append({"strategy": strategy, **stats})
    return pd.DataFrame(rows).set_index("strategy")
