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
