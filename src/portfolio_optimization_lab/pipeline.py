"""End-to-end orchestration for the optimization study."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import in_sample_stats, walk_forward_backtest, walk_forward_shrinkage
from .bootstrap import bootstrap_sharpe_differences
from .config import PROJECT_ROOT, OptConfig
from .data import load_price_frame, log_returns
from .estimates import estimate, portfolio_stats
from .optimization import (
    efficient_frontier,
    equal_weights,
    max_sharpe,
    min_variance,
    reset_solver_diagnostics,
    risk_parity,
    solver_diagnostics,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: OptConfig | None = None, save_outputs: bool = True) -> dict:
    """Run the full study: estimates, frontier, walk-forward backtest, reports.

    Headline performance numbers are NET of transaction costs (the cost drag
    is deducted from rebalance-day returns inside the backtest); gross metrics
    are kept alongside for reference. Solver fallback counters are reset at
    the start of the run and persisted with the outputs.
    """
    config = config or OptConfig()
    config.ensure_dirs()
    reset_solver_diagnostics()

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

    full_sample_weights = {
        "equal_weight": equal_weights(len(config.tickers)),
        "min_variance": min_variance(full_est.mu, full_est.cov, config.max_weight),
        "max_sharpe": max_sharpe(
            full_est.mu, full_est.cov, config.rf_annual, config.max_weight
        ),
        "risk_parity": risk_parity(full_est.cov, config.max_weight),
    }
    portfolios = {
        name.replace("_", " "): portfolio_stats(
            w, full_est.mu, full_est.cov, rf_annual=config.rf_annual
        )
        for name, w in full_sample_weights.items()
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
    wf_shrinkage = walk_forward_shrinkage(
        returns,
        estimation_window=config.estimation_window,
        holding_period=config.holding_period,
        periods_per_year=config.periods_per_year,
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
            "oos_sharpe_net": performance["sharpe"],
            "oos_sharpe_gross": performance["sharpe_gross"],
        }
    )
    comparison["sharpe_degradation"] = comparison["oos_sharpe_net"] - comparison["in_sample_sharpe"]

    # ---- 4. bootstrap uncertainty on pairwise Sharpe differences ----------
    sharpe_bootstrap = bootstrap_sharpe_differences(
        daily,
        B=2000,
        block=config.holding_period,
        seed=config.seed,
        periods_per_year=config.periods_per_year,
        sensitivity_blocks=(21, 126),
    )

    diagnostics = solver_diagnostics()

    if save_outputs:
        _save_outputs(
            config,
            frontier,
            asset_stats,
            portfolios,
            full_sample_weights,
            daily,
            performance,
            comparison,
            weights_history,
            full_est.shrinkage,
            wf_shrinkage,
            diagnostics,
            sharpe_bootstrap,
        )

    return {
        "asset_stats": asset_stats,
        "portfolios": portfolios,
        "full_sample_weights": full_sample_weights,
        "frontier": frontier,
        "performance": performance,
        "comparison": comparison,
        "weights_history": weights_history,
        "daily": daily,
        "shrinkage_full_sample": full_est.shrinkage,
        "shrinkage_walk_forward": wf_shrinkage,
        "solver_diagnostics": diagnostics,
        "sharpe_diff_bootstrap": sharpe_bootstrap,
    }


def _one_hot(n: int, i: int) -> object:
    """Unit vector factory used for asset rows (kept trivial on purpose)."""
    import numpy as np

    v = np.zeros(n)
    v[i] = 1.0
    return v


def _sanitize(obj: object) -> object:
    """Recursively convert non-finite floats to None (JSON null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def strict_json_dumps(payload: object) -> str:
    """Serialize to strict JSON: no NaN/Infinity tokens, ever.

    Non-finite floats are converted to ``null`` first; ``allow_nan=False``
    guarantees any non-finite value that survived conversion raises instead of
    emitting invalid JSON.
    """
    return json.dumps(_sanitize(payload), indent=2, allow_nan=False)


def _save_outputs(
    config: OptConfig,
    frontier: pd.DataFrame,
    asset_stats: pd.DataFrame,
    portfolios: dict,
    full_sample_weights: dict,
    daily: pd.DataFrame,
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
    weights_history: pd.DataFrame,
    shrinkage_full_sample: float,
    shrinkage_walk_forward: list[float],
    diagnostics: dict[str, int],
    sharpe_bootstrap: dict,
) -> None:
    # Infeasible frontier targets carry NaN volatility internally; they are
    # emitted as JSON null by strict_json_dumps (never as NaN tokens).
    frontier_records = (
        frontier[["target_return", "volatility"]].round(6).to_dict("records")
    )
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "portfolio-optimization-lab",
        "description": "Mean-variance and risk-parity optimization with shrinkage covariance "
        "and a leakage-free walk-forward evaluation, net of transaction costs "
        "(educational research project; not investment advice).",
        "config": {
            k: (str(v).replace(str(PROJECT_ROOT) + "/", "") if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "frontier": frontier_records,
        "asset_stats": asset_stats.round(6).to_dict("index"),
        "portfolios": {
            k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in portfolios.items()
        },
        "weights_full_sample": {
            strategy: {
                ticker: round(float(w), 6)
                for ticker, w in zip(config.tickers, weights, strict=True)
            }
            for strategy, weights in full_sample_weights.items()
        },
        "shrinkage_intensity_full_sample": round(float(shrinkage_full_sample), 6),
        "shrinkage_intensity_walk_forward": {
            "mean": round(float(np.mean(shrinkage_walk_forward)), 6),
            "min": round(float(np.min(shrinkage_walk_forward)), 6),
            "max": round(float(np.max(shrinkage_walk_forward)), 6),
            "n_windows": len(shrinkage_walk_forward),
        },
        "solver_diagnostics": diagnostics,
        "performance": performance.round(6).reset_index().to_dict("records"),
        "comparison": comparison.round(6).reset_index().to_dict("records"),
        "sharpe_diff_bootstrap": sharpe_bootstrap,
    }
    (config.reports_dir / "optimization_results.json").write_text(
        strict_json_dumps(payload) + "\n", encoding="utf-8"
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
    config: OptConfig,
    payload: dict,
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
    sharpe_bootstrap: dict | None = None,
) -> None:
    lines = [
        "# Portfolio Optimization Lab — Results Summary",
        "",
        "*Generated by the pipeline on real cached data; all numbers are actual outputs.*",
        "",
        "## In-sample vs out-of-sample Sharpe (the estimation-error story)",
        "",
        "OOS Sharpes are net of transaction costs (10 bps on one-way turnover,",
        "deducted from each rebalance-day return); the gross column is shown for",
        "reference so the cost drag is visible.",
        "",
        comparison.round(4).to_markdown(),
        "",
        "## Out-of-sample performance (headline = net of costs)",
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
        "## Pairwise Sharpe differences (paired circular-block bootstrap)",
        "",
        "95% percentile CIs for Sharpe(a) − Sharpe(b) on OOS daily NET returns;",
        "B = 2000 circular blocks of 63 days, seed 42; sensitivity at block lengths",
        "21 and 126 is stored in the JSON output. 'significant' = the 95% CI",
        "excludes 0 (statistical, not economic, significance).",
        "",
        "| a | b | diff | ci_low | ci_high | significant_95 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in (sharpe_bootstrap or payload.get("sharpe_diff_bootstrap", {})).get("pairs", []):
        lines.append(
            f"| {pair['a']} | {pair['b']} | {pair['diff']:.4f} | {pair['ci_low']:.4f} "
            f"| {pair['ci_high']:.4f} | {pair['significant_95']} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Mean–variance inputs are estimated, not known: in-sample Sharpe ratios are",
        "  systematically optimistic and the walk-forward column is the honest estimate.",
        "- Ledoit–Wolf shrinkage is applied to stabilize the covariance; the shrinkage",
        "  intensities (full sample and per walk-forward window) are reported in the",
        "  JSON output.",
        "- Costs of 10 bps on turnover are charged AND deducted at every rebalance:",
        "  the Sharpe column is net of costs, with gross Sharpe reported alongside.",
        "- The trailing partial 63-day holding block is not used out of sample.",
        "- Block-bootstrap CIs quantify sampling uncertainty only; statistical",
        "  significance (CI excluding 0) is not economic significance.",
        "- Educational research project — not investment advice.",
        "",
    ]
    (config.reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
