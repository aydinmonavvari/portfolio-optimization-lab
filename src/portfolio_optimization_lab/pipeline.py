"""End-to-end orchestration for the optimization study."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import plots
from .backtest import in_sample_stats, walk_forward_backtest
from .config import PROJECT_ROOT, OptConfig
from .data import load_price_frame, log_returns
from .estimates import estimate, portfolio_stats
from .optimization import (
    efficient_frontier,
    max_sharpe,
    min_variance,
    risk_parity,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: OptConfig | None = None, save_outputs: bool = True) -> dict:
    """Run the full study: estimates, frontier, walk-forward backtest, reports."""
    config = config or OptConfig()
    config.ensure_dirs()

    prices = load_price_frame(config.tickers, config.benchmark, config.raw_dir)
    # the benchmark is context only; optimization runs on the universe
    returns = log_returns(prices[list(config.tickers)])

    # ---- 1. in-sample picture on the full sample --------------------------
    full_est = estimate(returns, config.periods_per_year)
    asset_stats = pd.DataFrame(
        {
            t: portfolio_stats(
                _one_hot(len(config.tickers), i), full_est.mu, full_est.cov
            )
            for i, t in enumerate(config.tickers)
        }
    ).T

    portfolios = {
        "equal weight": portfolio_stats(
            _one_hot(len(config.tickers), -1) * 0 + 1 / len(config.tickers),
            full_est.mu,
            full_est.cov,
        ),
        "min variance": portfolio_stats(
            min_variance(full_est.mu, full_est.cov, config.max_weight),
            full_est.mu,
            full_est.cov,
        ),
        "max sharpe": portfolio_stats(
            max_sharpe(full_est.mu, full_est.cov, config.rf_annual, config.max_weight),
            full_est.mu,
            full_est.cov,
        ),
        "risk parity": portfolio_stats(
            risk_parity(full_est.cov, config.max_weight), full_est.mu, full_est.cov
        ),
    }

    frontier = efficient_frontier(
        full_est.mu,
        full_est.cov,
        max_weight=config.max_weight,
        n_points=config.frontier_points,
    )

    # ---- 2. walk-forward out-of-sample study ------------------------------
    daily, performance, weights_history = walk_forward_backtest(
        returns,
        estimation_window=config.estimation_window,
        holding_period=config.holding_period,
        max_weight=config.max_weight,
        cost_bps=config.cost_bps,
        periods_per_year=config.periods_per_year,
        rf_annual=config.rf_annual,
    )

    # ---- 3. in-sample vs out-of-sample comparison -------------------------
    in_sample = in_sample_stats(
        returns,
        max_weight=config.max_weight,
        periods_per_year=config.periods_per_year,
        rf_annual=config.rf_annual,
    )
    comparison = pd.DataFrame(
        {
            "in_sample_sharpe": in_sample["sharpe"],
            "oos_sharpe": performance["sharpe"],
        }
    )
    comparison["sharpe_degradation"] = comparison["oos_sharpe"] - comparison["in_sample_sharpe"]

    if save_outputs:
        _save_outputs(
            config,
            frontier,
            asset_stats,
            portfolios,
            daily,
            performance,
            comparison,
            weights_history,
        )

    return {
        "asset_stats": asset_stats,
        "portfolios": portfolios,
        "frontier": frontier,
        "performance": performance,
        "comparison": comparison,
        "weights_history": weights_history,
        "daily": daily,
    }


def _one_hot(n: int, i: int) -> object:
    """Unit vector factory used for asset rows (kept trivial on purpose)."""
    import numpy as np

    v = np.zeros(n)
    v[i] = 1.0
    return v


def _save_outputs(
    config: OptConfig,
    frontier: pd.DataFrame,
    asset_stats: pd.DataFrame,
    portfolios: dict,
    daily: pd.DataFrame,
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
    weights_history: pd.DataFrame,
) -> None:
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "portfolio-optimization-lab",
        "description": "Mean-variance and risk-parity optimization with shrinkage covariance "
        "and a leakage-free walk-forward evaluation (educational research project; "
        "not investment advice).",
        "config": {
            k: (str(v).replace(str(PROJECT_ROOT) + "/", "") if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "frontier": frontier[["target_return", "volatility"]].round(6).to_dict("records"),
        "asset_stats": asset_stats.round(6).to_dict("index"),
        "portfolios": {
            k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in portfolios.items()
        },
        "performance": performance.round(6).reset_index().to_dict("records"),
        "comparison": comparison.round(6).reset_index().to_dict("records"),
    }
    (config.reports_dir / "optimization_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    performance.round(6).to_csv(config.reports_dir / "oos_performance.csv")
    comparison.round(6).to_csv(config.reports_dir / "in_vs_out_of_sample.csv")
    _write_summary(config, payload, performance, comparison)

    figs = {
        "efficient_frontier.png": plots.plot_frontier(
            frontier,
            asset_stats,
            {k: (v["volatility"], v["return"]) for k, v in portfolios.items()},
        ),
        "oos_cumulative.png": plots.plot_oos_cumulative(daily),
        "oos_drawdowns.png": plots.plot_drawdowns(daily),
        "weights_history.png": plots.plot_weights_history(
            weights_history, "Rebalance-date weights per strategy"
        ),
    }
    for name, fig in figs.items():
        fig.savefig(config.figures_dir / name, dpi=150)
        plt_close(fig)


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _write_summary(
    config: OptConfig, payload: dict, performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    lines = [
        "# Portfolio Optimization Lab — Results Summary",
        "",
        "*Generated by the pipeline on real cached data; all numbers are actual outputs.*",
        "",
        "## In-sample vs out-of-sample Sharpe (the estimation-error story)",
        "",
        comparison.round(4).to_markdown(),
        "",
        "## Out-of-sample performance",
        "",
        performance.round(4).to_markdown(),
        "",
        "## Special portfolios (full-sample estimates, shrunk covariance)",
        "",
        "| portfolio | ann. return | ann. vol | sharpe |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in payload["portfolios"].items():
        lines.append(
            f"| {name} | {stats['return']:.4f} | {stats['volatility']:.4f} "
            f"| {stats['sharpe']:.4f} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Mean–variance inputs are estimated, not known: in-sample Sharpe ratios are",
        "  systematically optimistic and the walk-forward column is the honest estimate.",
        "- Ledoit–Wolf shrinkage is applied to stabilize the covariance; the shrinkage",
        "  intensity is reported in the JSON output.",
        "- Costs of 10 bps on turnover are charged at every rebalance.",
        "- Educational research project — not investment advice.",
        "",
    ]
    (config.reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
