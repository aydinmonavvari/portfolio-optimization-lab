"""Visualization layer for the optimization study."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_frontier(
    frontier: pd.DataFrame,
    assets_stats: pd.DataFrame,
    portfolios: dict[str, tuple[float, float]],
    title: str = "Constrained efficient frontier (shrunk covariance)",
):
    """Frontier curve, individual assets and special portfolios."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        frontier["volatility"],
        frontier["target_return"],
        color="tab:blue",
        lw=2,
        label="efficient frontier",
    )
    ax.scatter(
        assets_stats["volatility"],
        assets_stats["return"],
        color="grey",
        marker="o",
        s=45,
        label="individual assets",
    )
    for name, (vol, ret) in portfolios.items():
        ax.scatter(vol, ret, s=90, marker="*", zorder=5)
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(8, -4))
    for ticker, row in assets_stats.iterrows():
        ax.annotate(ticker, (row["volatility"], row["return"]), fontsize=8)
    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized expected return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weights_history(weights_history: pd.DataFrame, title: str):
    """Stacked area of one strategy's weights across rebalances."""
    strategies = weights_history["strategy"].unique()
    fig, axes = plt.subplots(len(strategies), 1, figsize=(10, 3 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies, strict=False):
        sub = weights_history[weights_history["strategy"] == strategy].drop(
            columns=["strategy"]
        )
        ax.stackplot(sub.index, sub.T.values, labels=sub.columns, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("weight")
        ax.set_title(strategy)
        ax.legend(loc="upper left", fontsize=7, ncol=4)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_oos_cumulative(daily_returns: pd.DataFrame, title: str = "Out-of-sample growth of 1"):
    """Cumulative growth of an initial investment across strategies."""
    growth = np.exp(daily_returns.cumsum())
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in growth.columns:
        ax.plot(growth.index, growth[col], lw=1.6, label=col)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drawdowns(daily_returns: pd.DataFrame, title: str = "Out-of-sample drawdowns"):
    """Drawdown curves computed from cumulative log returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in daily_returns.columns:
        cum = daily_returns[col].cumsum()
        dd = cum - cum.cummax()
        ax.plot(dd.index, dd, lw=1.2, label=col)
    ax.set_ylabel("drawdown (log)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PHASE 2 (pre-registered) figures
# ---------------------------------------------------------------------------


def plot_level_cis(levels: dict, title: str = "OOS Sharpe: point estimates and 95% CIs"):
    """Level CIs per strategy, one panel per block length (RQ1)."""
    strategies = list(levels["point_estimates"])
    blocks = [str(b) for b in levels["blocks"]]
    fig, axes = plt.subplots(1, len(blocks), figsize=(5 * len(blocks), 4.5), sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for ax, block in zip(axes, blocks, strict=False):
        points = [levels["point_estimates"][s] for s in strategies]
        lows = [levels["levels"][s][block]["ci_low"] for s in strategies]
        highs = [levels["levels"][s][block]["ci_high"] for s in strategies]
        y = np.arange(len(strategies))
        ax.errorbar(
            points, y, xerr=[np.array(points) - np.array(lows),
                             np.array(highs) - np.array(points)],
            fmt="o", capsize=4, lw=1.4, color="tab:purple",
        )
        ax.set_yticks(y, strategies, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0.0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("daily-convention OOS Sharpe")
        ax.set_title(f"block = {block}")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_rank_frequencies(
    rank_frequencies: dict, title: str = "Bootstrap rank frequencies (block 63)"
):
    """Pooled frequency of each rank per strategy (RQ2a)."""
    strategies = list(rank_frequencies)
    ranks = list(next(iter(rank_frequencies.values())))
    x = np.arange(len(strategies))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for r_i, rank in enumerate(ranks):
        vals = [rank_frequencies[s][rank] for s in strategies]
        ax.bar(x + (r_i - 1.5) * width, vals, width, label=f"rank {rank}")
    ax.set_xticks(x, strategies, fontsize=9)
    ax.set_ylabel("frequency over resamples")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cost_curves(
    cost_sensitivity: dict,
    title: str = "Cost-assumption sensitivity (true net engine)",
):
    """Net Sharpe and mean daily net return vs cost per strategy (RQ3)."""
    rows = pd.DataFrame(cost_sensitivity["rows"])
    strategies = list(rows["strategy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for s in strategies:
        sub = rows[rows["strategy"] == s].sort_values("cost_bps")
        axes[0].plot(sub["cost_bps"], sub["sharpe_net_dailyconv"], marker="o", label=s)
        axes[1].plot(
            sub["cost_bps"], sub["mean_daily_net_return"], marker="o", label=s
        )
    axes[0].set_xlabel("cost (bps, one-way on turnover)")
    axes[0].set_ylabel("OOS Sharpe (daily convention, net)")
    axes[0].set_title("Net Sharpe vs cost (NOT affine — no such claim)")
    axes[1].set_xlabel("cost (bps, one-way on turnover)")
    axes[1].set_ylabel("mean daily net return")
    axes[1].set_title("Mean daily net return vs cost (exactly affine)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_placebo_rank_dist(
    oracle_rank_counts: dict,
    m: int,
    records_present: bool = True,
    title: str = "Placebo: oracle rank distribution across worlds",
):
    """How often the by-construction-optimal oracle ranks k-th (RQ5)."""
    ranks = sorted(int(r) for r in oracle_rank_counts)
    counts = [oracle_rank_counts[str(r)] for r in ranks]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar([str(r) for r in ranks], counts, color="tab:green", alpha=0.8)
    for bar, count in zip(bars, counts, strict=True):
        ax.annotate(f"{count}/{m}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("oracle rank in the world (1 = best of 5)")
    ax.set_ylabel("number of worlds")
    ax.set_title(title + ("" if records_present else " (records not persisted)"))
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    return fig
