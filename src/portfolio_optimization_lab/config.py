"""Configuration for the portfolio-optimization laboratory.

``configs/default.yaml`` is the shipped configuration file; it is read by
:func:`load_config` (used by ``scripts/run_optimization.py``). The loader is
strict: unknown keys raise ``ValueError`` so a typo in the YAML can never
silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


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


# Fields settable from YAML (the Path fields are derived locations, not config).
_YAML_FIELDS: dict[str, type] = {
    "tickers": tuple,
    "benchmark": str,
    "rf_annual": float,
    "periods_per_year": int,
    "estimation_window": int,
    "holding_period": int,
    "max_weight": float,
    "cost_bps": float,
    "frontier_points": int,
    "seed": int,
}


def _coerce(name: str, value: object) -> object:
    """Coerce one YAML value to the declared OptConfig field type."""
    target = _YAML_FIELDS[name]
    if target is tuple:  # tickers
        if not isinstance(value, list) or not value or not all(isinstance(t, str) for t in value):
            raise ValueError("config field 'tickers' must be a non-empty list of strings")
        return tuple(value)
    if target is str:
        if not isinstance(value, str):
            raise ValueError(f"config field '{name}' must be a string, got {value!r}")
        return value
    if target is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"config field '{name}' must be numeric, got {value!r}")
        return float(value)
    if target is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"config field '{name}' must be an integer, got {value!r}")
        return int(value)
    raise ValueError(f"config field '{name}' has unsupported type {target}")


def load_config(path: str | Path | None = None) -> OptConfig:
    """Load a strict YAML configuration file into an :class:`OptConfig`.

    - Unknown keys raise ``ValueError`` (typo protection — no silent defaults).
    - Known keys are coerced to their declared types and validated.
    - Fields absent from the YAML keep their OptConfig defaults (including the
      derived directory locations).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_YAML_FIELDS))
    if unknown:
        raise ValueError(
            f"{path}: unknown config key(s) {unknown}; "
            f"recognized keys are {sorted(_YAML_FIELDS)}"
        )
    values = {name: _coerce(name, raw[name]) for name in raw}
    cfg = OptConfig(**values)

    # Validation of cross-field invariants.
    positive_ints = ("periods_per_year", "estimation_window", "holding_period", "frontier_points")
    for name in positive_ints:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"config field '{name}' must be positive, got {getattr(cfg, name)}")
    if cfg.estimation_window <= cfg.holding_period:
        raise ValueError(
            "config field 'estimation_window' must exceed 'holding_period' "
            f"(got {cfg.estimation_window} <= {cfg.holding_period})"
        )
    if not 0.0 < cfg.max_weight <= 1.0:
        raise ValueError(f"config field 'max_weight' must be in (0, 1], got {cfg.max_weight}")
    if cfg.cost_bps < 0:
        raise ValueError(f"config field 'cost_bps' must be non-negative, got {cfg.cost_bps}")
    return cfg
