"""Configuration for the portfolio-optimization laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OptConfig:
    """Static configuration for one full run of the optimization study.

    Attributes
    ----------
    tickers:
        Investable universe (daily adjusted closes are read from ``data/raw``).
    benchmark:
        Benchmark ticker used for beta/context (not part of the universe).
    rf_annual:
        Annualized risk-free rate assumption for Sharpe ratios (disclosed).
    periods_per_year:
        Annualization factor (252 trading days).
    estimation_window:
        Rolling in-sample window (trading days) used to estimate mu/cov.
    holding_period:
        Out-of-sample holding period between rebalances (trading days).
    max_weight:
        Upper bound on any single asset weight (constraint set).
    cost_bps:
        Transaction cost in basis points applied to turnover at rebalance.
    frontier_points:
        Number of target-return grid points for the efficient frontier.
    seed:
        Global random seed.
    """

    tickers: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "JNJ",
        "JPM",
        "XOM",
        "PG",
        "AMZN",
    )
    benchmark: str = "SPY"
    rf_annual: float = 0.0
    periods_per_year: int = 252
    estimation_window: int = 252
    holding_period: int = 63
    max_weight: float = 0.40
    cost_bps: float = 10.0
    frontier_points: int = 30
    seed: int = 42

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    figures_dir: Path = PROJECT_ROOT / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = OptConfig()
