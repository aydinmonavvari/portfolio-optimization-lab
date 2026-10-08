"""Circular block bootstrap for Sharpe-ratio differences.

Why not an IID bootstrap
------------------------
Out-of-sample daily strategy returns are serially dependent (volatility
clustering, cross-day momentum, and — most importantly here — the walk-forward
design holds a fixed weight vector for a whole 63-day holding period, so
consecutive returns share a common conditional distribution). An IID bootstrap
destroys exactly this dependence and therefore understates the sampling
variability of statistics like the Sharpe ratio, producing anti-conservative
confidence intervals. The circular block bootstrap (Politis & Romano 1994)
resamples contiguous blocks of length ``block`` drawn from the series with
wrap-around (circular) indexing: within-block dependence is preserved, the
wrap-around avoids edge loss, and the resampled series has the same length as
the original.

What is computed
----------------
For every unordered pair of strategies (a, b), the paired statistic

    d = Sharpe(a) - Sharpe(b),   Sharpe = sqrt(periods_per_year) * mean(r - rf/ppy) / std(r)

is evaluated on the OOS daily net returns with the SAME resampled index for
both legs (a paired bootstrap: it preserves the cross-sectional correlation
between strategies, which is what makes the difference well behaved). B
circular-block resamples give a percentile confidence interval [q2.5, q97.5].

Assumptions and limitations (stated, not hidden)
------------------------------------------------
- The OOS return series is treated as (approximately) stationary and its
  dependence is captured by blocks of length ``block``; longer-range
  dependence (e.g. regimes spanning years) is NOT captured.
- The block length is a modelling choice. The default (63 days) matches the
  walk-forward holding period; sensitivity to 21 and 126 days is reported
  alongside and conclusions should be read across all three.
- Percentile intervals have O(B^-1/2) accuracy and can undercover near the
  boundary of the parameter space; the Sharpe estimator itself is biased in
  small samples (the bias largely cancels in paired differences).
- Statistical significance (interval excluding 0 at the 95% level) is a
  statement about sampling noise only; it says nothing about economic
  significance, which must be judged against turnover and cost differences.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One circular-block resample of length ``n``.

    Draws ``ceil(n / block)`` block start positions uniformly from ``[0, n)``
    and lays out ``block`` consecutive (wrap-around) indices from each start,
    trimmed to length ``n``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, n={n}], got {block}")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]) % n
    return idx.reshape(-1)[:n]


def _daily_sharpes(returns: np.ndarray, idx: np.ndarray, periods_per_year: int) -> np.ndarray:
    """Sharpe of each resampled row of ``returns`` (rows = strategies)."""
    sampled = returns[:, idx]  # (n_strategies, n)
    mean = sampled.mean(axis=1)
    std = sampled.std(axis=1, ddof=1)
    ratio = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0)
    return ratio * np.sqrt(periods_per_year)


def bootstrap_sharpe_differences(
    daily_returns,
    strategies: Sequence[str] | None = None,
    B: int = 2000,
    block: int = 63,
    seed: int = 42,
    periods_per_year: int = 252,
    sensitivity_blocks: Sequence[int] = (21, 126),
    confidence_level: float = 0.95,
) -> dict:
    """Paired circular-block bootstrap CIs for all pairwise Sharpe differences.

    Parameters
    ----------
    daily_returns:
        DataFrame of out-of-sample daily net returns, one column per strategy.
    B:
        Number of bootstrap resamples for the headline (``block``) interval.
    sensitivity_blocks:
        Additional block lengths for which the same interval is recomputed
        (robustness of the conclusion to the block-length choice).

    Returns
    -------
    dict
        ``{"B", "block", "seed", "confidence_level", "statistic", "assumptions",
        "pairs": [{"a", "b", "diff", "ci_low", "ci_high", "significant_95",
        "sensitivity": {str(block): {"ci_low", "ci_high"}}}]}``.
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    frame = daily_returns
    if strategies is None:
        strategies = list(frame.columns)
    strategies = list(strategies)
    returns = frame[strategies].to_numpy(dtype=float).T  # (n_strategies, n)
    n = returns.shape[1]

    alpha = (1.0 - confidence_level) / 2.0

    def interval(
        block_len: int, rng: np.random.Generator, pair: tuple[int, int]
    ) -> tuple[float, float]:
        stats = np.empty(B)
        for b in range(B):
            idx = circular_block_indices(n, block_len, rng)
            s = _daily_sharpes(returns[[pair[0], pair[1]]], idx, periods_per_year)
            stats[b] = s[0] - s[1]
        lo, hi = np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi)

    pairs_out = []
    for i, j in combinations(range(len(strategies)), 2):
        a, b = strategies[i], strategies[j]
        point_sharpes = _daily_sharpes(returns[[i, j]], np.arange(n), periods_per_year)
        point = float(point_sharpes[0] - point_sharpes[1])
        rng = np.random.default_rng(seed)
        lo, hi = interval(block, rng, (i, j))
        sensitivity = {}
        for blk in sensitivity_blocks:
            rng_s = np.random.default_rng(seed)
            lo_s, hi_s = interval(blk, rng_s, (i, j))
            sensitivity[str(blk)] = {"ci_low": round(lo_s, 6), "ci_high": round(hi_s, 6)}
        pairs_out.append(
            {
                "a": a,
                "b": b,
                "diff": round(point, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant_95": bool(lo > 0 or hi < 0),
                "sensitivity": sensitivity,
            }
        )

    return {
        "B": B,
        "block": block,
        "seed": seed,
        "confidence_level": confidence_level,
        "statistic": (
            "paired circular-block bootstrap of Sharpe(a) - Sharpe(b) on OOS daily "
            "net returns; Sharpe = sqrt(periods_per_year) * mean(r) / std(r, ddof=1); "
            "identical resampled indices for both legs"
        ),
        "assumptions": (
            "approximately stationary OOS return series; within-block dependence "
            "captures the serial correlation; conclusions read across the "
            "sensitivity block lengths"
        ),
        "pairs": pairs_out,
    }
