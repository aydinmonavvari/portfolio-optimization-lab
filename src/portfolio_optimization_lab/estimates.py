"""Return and covariance estimation, including shrinkage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class Estimates:
    """Mean-vector and covariance-matrix estimates for one estimation window.

    Attributes
    ----------
    mu:
        Annualized expected-return estimate (arithmetic, from daily simple
        returns, per asset).
    cov_sample:
        Annualized sample covariance matrix (unshrunk).
    cov_shrunk:
        Annualized Ledoit–Wolf shrunk covariance matrix.
    shrinkage:
        The Ledoit–Wolf shrinkage intensity in [0, 1] (1 = diagonal target).
    """

    mu: pd.Series
    cov_sample: pd.DataFrame
    cov_shrunk: pd.DataFrame
    shrinkage: float

    @property
    def cov(self) -> pd.DataFrame:
        """The covariance matrix actually used by default (shrunk)."""
        return self.cov_shrunk


def estimate(
    daily_returns: pd.DataFrame,
    periods_per_year: int = 252,
    use_shrunk: bool = True,
) -> Estimates:
    """Estimate annualized mu and covariance from a daily SIMPLE-return window.

    The engine consumes daily SIMPLE (fractional) returns end to end: the
    per-day portfolio return is then the exact weighted sum ``r @ w``, cost
    deductions live in the same units, and cumulative growth is
    ``(1 + r).prod()``. (Before Phase 1 this function consumed log returns;
    feeding log returns into a linear portfolio algebra mixed conventions
    and biased every reported return.)

    Notes
    -----
    - Sample covariance is scaled by ``periods_per_year`` for annualization.
    - Ledoit & Wolf (2004) shrinkage pulls the sample covariance toward a
      scaled identity target; the shrinkage intensity is fitted on the data.
    - The mean vector is annualized *arithmetically*
      (``mean_simple * periods``): the standard mean–variance input.
    """
    mean_simple = daily_returns.mean()
    mu = mean_simple * periods_per_year
    sample = daily_returns.cov() * periods_per_year
    lw = LedoitWolf().fit(daily_returns.to_numpy())
    shrunk = pd.DataFrame(
        lw.covariance_ * periods_per_year,
        index=daily_returns.columns,
        columns=daily_returns.columns,
    )
    return Estimates(
        mu=mu,
        cov_sample=sample,
        cov_shrunk=shrunk if use_shrunk else sample,
        shrinkage=float(lw.shrinkage_),
    )


def portfolio_stats(
    weights: np.ndarray | pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    """Annualized return, volatility and Sharpe of a weight vector."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
    sharpe = (ret - rf_annual) / vol if vol > 0 else float("nan")
    return {"return": ret, "volatility": vol, "sharpe": sharpe}
