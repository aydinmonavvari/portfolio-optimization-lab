"""PHASE 2 uncertainty-aware evaluation suite (pre-registered).

Implements exactly the registered methodology of
``docs/admissions/PHASE2-RESEARCH-PLAN.md`` (commit before implementation):
level confidence intervals for each strategy's OOS Sharpe, bootstrap rank
frequencies, a Holm-corrected family for the 6 pairwise Sharpe differences,
sub-window rank stability, a transaction-cost sensitivity grid with exact
algebraic properties, an exploratory chronological half split, and a placebo
simulation that calibrates evaluation-design noise against a by-construction
optimal (oracle) tangency portfolio.

Seeds (registered, never deviated from):
- ``42`` for every bootstrap engine (levels, ranks, pairwise, half split);
- ``43 + i`` for placebo world ``i``;
- ``2026`` is RESERVED-UNUSED by registration and must never be repurposed.

Honesty contract: null or unflattering results are kept and reported. No
document may claim that net Sharpe is affine in cost (it is NOT — only mean
and total net return are), and no unadjusted significance flag may survive in
any artifact or document (the registered family-wise procedure is Holm).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .backtest import STRATEGIES, walk_forward_backtest
from .bootstrap import circular_block_indices

# Registered seed constants. 2026 is registered as DEAD: it must never be
# silently repurposed for any random stream (plan Section 6, item 5).
SEED_BOOTSTRAP = 42
SEED_PLACEBO_BASE = 43
SEED_2026_RESERVED_UNUSED = 2026

DEFAULT_COST_GRID = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
STRATEGIES_DEFAULT = STRATEGIES


def daily_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Daily-convention annualized Sharpe: sqrt(ppy) * mean / std(ddof=1)."""
    r = np.asarray(returns, dtype=float)
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if not std > 0:
        return float("nan")
    return mean / std * np.sqrt(periods_per_year)


def _sharpes_matrix(returns: np.ndarray, idx: np.ndarray, ppy: int) -> np.ndarray:
    """Sharpe of every row (strategy) of ``returns`` on resampled ``idx``."""
    sampled = returns[:, idx]
    mean = sampled.mean(axis=1)
    std = sampled.std(axis=1, ddof=1)
    ratio = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0)
    return ratio * np.sqrt(ppy)


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment over a named family of p-values.

    Returns, per hypothesis, the adjusted p-value and the step-down rejection
    decision at ``alpha`` (family-wise error rate control). The adjusted value
    for the k-th smallest p is the running maximum of ``(m - k + 1) * p_k``,
    capped at 1 (monotonicity is enforced so adjusted p-values never decrease
    with the raw p-value).
    """
    names = sorted(p_values, key=lambda k: (p_values[k], k))
    m = len(names)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for k, name in enumerate(names):
        raw = float(p_values[name])
        running_max = max(running_max, (m - k) * raw)
        adjusted[name] = float(min(1.0, running_max))
    return {
        name: {
            "p_raw": round(p_values[name], 6),
            "p_holm": round(adjusted[name], 6),
            "reject_holm_05": bool(adjusted[name] <= alpha),
        }
        for name in p_values
    }


def level_cis_and_ranks(
    daily_returns: pd.DataFrame,
    B: int = 2000,
    blocks: tuple[int, ...] = (21, 63, 126),
    seed: int = SEED_BOOTSTRAP,
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
) -> dict:
    """Level CIs per strategy per block length + pooled rank frequencies.

    One fresh ``default_rng(seed)`` stream per block length (the SAME stream
    construction the v1.0.0 pairwise engine uses), B circular-block resamples,
    and per resample the daily-convention Sharpe of EVERY strategy computed on
    the shared resampled indices. Levels use percentile intervals; rank
    frequencies pool the per-resample ranks of the headline block (the LAST
    entry of ``blocks`` is treated as the headline, matching the registered
    block-63 headline).

    The rank-frequency stream is therefore the block-headline level stream —
    exactly "the same B resamples as the block-63 level engine" (plan §5.4).
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    strategies = list(daily_returns.columns)
    returns = daily_returns.to_numpy(dtype=float).T  # (n_strategies, n)
    n = returns.shape[1]
    point = _sharpes_matrix(returns, np.arange(n), periods_per_year)
    alpha = (1.0 - confidence_level) / 2.0

    headline_block = blocks[-1]
    levels: dict[str, dict] = {s: {} for s in strategies}
    rank_freq = np.zeros((len(strategies), len(strategies)))  # [strategy, rank-1]
    rank_counts_by_block: dict[str, list[int]] = {}

    for block in blocks:
        rng = np.random.default_rng(seed)
        all_sharpes = np.empty((B, len(strategies)))
        for b in range(B):
            idx = circular_block_indices(n, block, rng)
            all_sharpes[b] = _sharpes_matrix(returns, idx, periods_per_year)
        for i, s in enumerate(strategies):
            col = all_sharpes[:, i]
            lo, hi = np.nanpercentile(col, [100 * alpha, 100 * (1 - alpha)])
            levels[s][str(block)] = {
                "ci_low": round(float(lo), 6),
                "ci_high": round(float(hi), 6),
            }
        if block == headline_block:
            # ranks per resample: rank 1 = highest Sharpe (stable sort for
            # determinism; ties are measure-zero on continuous data)
            order = np.argsort(-all_sharpes, axis=1, kind="stable")
            ranks = np.empty_like(order)
            rows = np.arange(B)[:, None]
            ranks[rows, order] = np.arange(len(strategies))[None, :]
            for i in range(len(strategies)):
                for r_ in range(len(strategies)):
                    rank_freq[i, r_] = float(np.mean(ranks[:, i] == r_))
            rank1 = {s: int(np.sum(ranks[:, i] == 0)) for i, s in enumerate(strategies)}
            rank_counts_by_block[str(block)] = rank1

    frequencies = {
        s: {
            str(rank + 1): round(float(rank_freq[i, rank]), 6)
            for rank in range(len(strategies))
        }
        for i, s in enumerate(strategies)
    }
    return {
        "B": B,
        "blocks": [int(b) for b in blocks],
        "seed": seed,
        "confidence_level": confidence_level,
        "statistic": (
            "daily-convention Sharpe sqrt(252)*mean(r)/std(r, ddof=1) on OOS "
            "daily NET returns; circular-block bootstrap, percentile intervals; "
            "one fresh seed-42 stream per block length"
        ),
        "point_estimates": {
            s: round(float(point[i]), 6) for i, s in enumerate(strategies)
        },
        "levels": levels,
        "rank_frequencies": frequencies,
        "rank1_counts": rank_counts_by_block.get(str(headline_block), {}),
    }


def pairwise_suite_holm(
    daily_returns: pd.DataFrame,
    B: int = 2000,
    block: int = 63,
    seed: int = SEED_BOOTSTRAP,
    sensitivity_blocks: tuple[int, ...] = (21, 126),
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap per pair (v1.0.0-identical streams) + Holm family.

    Per pair a FRESH ``default_rng(seed)`` stream is used, exactly as the
    v1.0.0 ``bootstrap_sharpe_differences`` engine does, so the Phase-2 suite
    reproduces the v1.0.0 index stream and intervals bit-for-bit (A2). Adds:
    two-sided bootstrap p-value with add-one correction and Holm step-down
    adjustment across the 6-pair family (the registered multiplicity
    procedure). The raw per-pair ``significant_95`` flag is retained ONLY as
    the unadjusted CI statement and must never be reported without the Holm
    column next to it.
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    strategies = list(daily_returns.columns)
    returns = daily_returns[strategies].to_numpy(dtype=float).T
    n = returns.shape[1]
    alpha_ci = (1.0 - confidence_level) / 2.0

    def interval_and_pvalues(
        block_len: int, pair: tuple[int, int]
    ) -> tuple[float, float, np.ndarray]:
        stats = np.empty(B)
        leg = returns[[pair[0], pair[1]]]
        rng = np.random.default_rng(seed)
        for b in range(B):
            idx = circular_block_indices(n, block_len, rng)
            s = _sharpes_matrix(leg, idx, periods_per_year)
            stats[b] = s[0] - s[1]
        lo, hi = np.nanpercentile(stats, [100 * alpha_ci, 100 * (1 - alpha_ci)])
        return float(lo), float(hi), stats

    pairs_out = []
    raw_p: dict[str, float] = {}
    for i, j in combinations(range(len(strategies)), 2):
        a, b = strategies[i], strategies[j]
        leg = returns[[i, j]]
        point_sharpes = _sharpes_matrix(leg, np.arange(n), periods_per_year)
        point = float(point_sharpes[0] - point_sharpes[1])
        lo, hi, boot = interval_and_pvalues(block, (i, j))
        # two-sided bootstrap p-value with add-one correction (registered)
        n_le = int(np.sum(boot <= 0))
        n_ge = int(np.sum(boot >= 0))
        p_raw = 2.0 * min((n_le + 1) / (B + 1), (n_ge + 1) / (B + 1))
        p_raw = float(min(1.0, p_raw))
        key = f"{a}__vs__{b}"
        raw_p[key] = p_raw
        sensitivity = {}
        for blk in sensitivity_blocks:
            rng_s = np.random.default_rng(seed)
            leg_pair = returns[[i, j]]
            stats_s = np.empty(B)
            for b_i in range(B):
                idx = circular_block_indices(n, blk, rng_s)
                s = _sharpes_matrix(leg_pair, idx, periods_per_year)
                stats_s[b_i] = s[0] - s[1]
            lo_s, hi_s = np.nanpercentile(
                stats_s, [100 * alpha_ci, 100 * (1 - alpha_ci)]
            )
            sensitivity[str(blk)] = {
                "ci_low": round(float(lo_s), 6),
                "ci_high": round(float(hi_s), 6),
            }
        pairs_out.append(
            {
                "a": a,
                "b": b,
                "diff": round(point, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant_95_unadjusted": bool(lo > 0 or hi < 0),
                "sensitivity": sensitivity,
                "_boot": boot,
                "_key": key,
            }
        )

    holm = holm_adjust(raw_p, alpha=alpha)
    for pair in pairs_out:
        pair.pop("_boot")  # internal bootstrap draws are NOT persisted
        h = holm[pair.pop("_key")]
        pair["p_value_add_one"] = h["p_raw"]
        pair["p_holm_adjusted"] = h["p_holm"]
        pair["reject_holm_05"] = h["reject_holm_05"]

    n_reject = sum(1 for p in pairs_out if p["reject_holm_05"])
    return {
        "B": B,
        "block": block,
        "seed": seed,
        "alpha": alpha,
        "method": (
            "paired circular-block bootstrap per pair (fresh seed-42 stream per "
            "pair, identical to the v1.0.0 engine); two-sided add-one bootstrap "
            "p-values; Holm step-down across the 6-pair family (FWER 0.05)"
        ),
        "family_size": len(pairs_out),
        "n_reject_holm_05": n_reject,
        "pairs": pairs_out,
    }


def subwindow_stability(
    daily_returns: pd.DataFrame,
    window: int = 252,
    periods_per_year: int = 252,
) -> dict:
    """Rank stability across 7 contiguous 252-day sub-windows (RQ2b).

    Per window: daily-convention Sharpe of each strategy (the annualized
    geometric convention is ALSO reported per window for disclosure, since the
    two coexist in this project). Stability summaries: Kendall tau between the
    rankings of each ADJACENT pair of windows, and per-strategy rank-1 counts.
    """
    frame = daily_returns
    strategies = list(frame.columns)
    n_windows = len(frame) // window
    if n_windows < 2:
        raise ValueError(f"need >= {2 * window} days, got {len(frame)}")
    window_sharpes: list[dict[str, float]] = []
    window_sharpes_geom: list[dict[str, float]] = []
    for w in range(n_windows):
        seg = frame.iloc[w * window : (w + 1) * window]
        row = {s: daily_sharpe(seg[s].to_numpy(), periods_per_year) for s in strategies}
        window_sharpes.append(row)
        # annualized geometric convention (disclosed alongside): total log
        # return over the window annualized by window/ppy years, vol scaled
        row_g = {}
        for s in strategies:
            r = seg[s].to_numpy()
            years = window / periods_per_year
            # geometric annualization on the simple daily series
            ann_ret = float(np.exp(np.log1p(r).sum() / years) - 1.0)
            ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
            row_g[s] = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        window_sharpes_geom.append(row_g)

    taus = []
    for w in range(n_windows - 1):
        a = [window_sharpes[w][s] for s in strategies]
        b = [window_sharpes[w + 1][s] for s in strategies]
        tau = sps.kendalltau(a, b).statistic
        taus.append(round(float(tau), 6))
    rank1_counts = {
        s: sum(
            1
            for row in window_sharpes
            if s == max(strategies, key=lambda x: row[x])  # ties: first max wins
        )
        for s in strategies
    }
    return {
        "n_windows": n_windows,
        "window_days": window,
        "statistic": "daily-convention Sharpe per 252-day contiguous sub-window",
        "window_sharpes_daily": window_sharpes,
        "window_sharpes_geometric": window_sharpes_geom,
        "adjacent_kendall_tau": taus,
        "mean_adjacent_kendall_tau": round(float(np.mean(taus)), 6),
        "rank1_counts": rank1_counts,
    }


def cost_grid(
    returns: pd.DataFrame,
    grid: tuple[float, ...] = DEFAULT_COST_GRID,
    strategies: tuple[str, ...] | None = None,
    estimation_window: int = 252,
    holding_period: int = 63,
    max_weight: float | None = 0.40,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    use_shrunk_cov: bool = True,
) -> dict:
    """Re-run the verified engine per cost level on IDENTICAL weight paths (RQ3).

    Costs never enter the optimizers, so the weight and turnover paths are
    identical across cost levels by construction; this is asserted downstream
    (A6 turnover-invariance test). Registered exact claims pinned by tests:
    mean/total net return affine in cost_bps, bitwise net==gross at 0 bps,
    turnover invariance. NO affine-Sharpe claim exists anywhere.

    ``strategies`` restricts the strategy set (used by the fast synthetic
    tests); ``None`` means the full registered strategy tuple.
    """
    strategies = strategies or STRATEGIES_DEFAULT
    rows = []
    daily_by_cost: dict[float, pd.DataFrame] = {}
    for cost in grid:
        daily, perf, _ = walk_forward_backtest(
            returns,
            strategies=strategies,
            estimation_window=estimation_window,
            holding_period=holding_period,
            max_weight=max_weight,
            cost_bps=cost,
            periods_per_year=periods_per_year,
            rf_annual=rf_annual,
            use_shrunk_cov=use_shrunk_cov,
        )
        daily_by_cost[cost] = daily
        for strategy in daily.columns:
            rows.append(
                {
                    "cost_bps": cost,
                    "strategy": strategy,
                    "mean_daily_net_return": float(daily[strategy].mean()),
                    "sharpe_net_dailyconv": daily_sharpe(
                        daily[strategy].to_numpy(), periods_per_year
                    ),
                    "sharpe_net_engine": float(perf.loc[strategy, "sharpe"]),
                    "sharpe_gross_engine": float(perf.loc[strategy, "sharpe_gross"]),
                    "ann_return_net_engine": float(perf.loc[strategy, "ann_return"]),
                    "ann_return_gross_engine": float(
                        perf.loc[strategy, "ann_return_gross"]
                    ),
                    "avg_turnover": float(perf.loc[strategy, "avg_turnover"]),
                    "n_rebalances": int(perf.loc[strategy, "n_rebalances"]),
                    "total_costs_pct": float(perf.loc[strategy, "total_costs_pct"]),
                    "oos_days": int(perf.loc[strategy, "oos_days"]),
                }
            )
    table = pd.DataFrame(rows)
    # empirical two-point slope per strategy (robust; degenerate grid -> empty)
    slopes: dict[str, float] = {}
    if len(grid) >= 2:
        for strategy in daily_by_cost[next(iter(daily_by_cost))].columns:
            sub = table[table["strategy"] == strategy].sort_values("cost_bps")
            x = sub["cost_bps"].to_numpy(dtype=float)
            y = sub["mean_daily_net_return"].to_numpy(dtype=float)
            if x[-1] > x[0]:
                slopes[strategy] = float((y[-1] - y[0]) / (x[-1] - x[0]))
    return {
        "grid": [float(g) for g in grid],
        # 12 dp: the registered affine law is exact to ~1e-15 in floats and the
        # test pins it at 1e-11, so artifact rounding must stay far below that
        "rows": table.round(12).to_dict("records"),
        "mean_return_slope_per_bps": {k: round(v, 12) for k, v in slopes.items()},
        "registered_claims": (
            "mean/total net return affine in cost_bps with slope "
            "-sum(turnover)/(1e4*n); bitwise net==gross at 0 bps; turnover "
            "invariance across the grid; net Sharpe is NOT affine and no such "
            "claim is made"
        ),
    }


def half_split(
    daily_returns: pd.DataFrame,
    B: int = 2000,
    block: int = 63,
    seed: int = SEED_BOOTSTRAP,
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
) -> dict:
    """Chronological first-vs-second half comparison (RQ4, EXPLORATORY).

    Splits the OOS sample into two contiguous halves; per strategy reports the
    daily-convention Sharpe of each half with a block-bootstrap level CI per
    half. Registered label: EXPLORATORY — this is a fragility check only and
    supports NO regime-robustness claim.
    """
    strategies = list(daily_returns.columns)
    n = len(daily_returns)
    half1 = daily_returns.iloc[: n // 2]
    half2 = daily_returns.iloc[n // 2 :]
    alpha = (1.0 - confidence_level) / 2.0

    def ci_for(frame: pd.DataFrame, strategy: str) -> tuple[float, float]:
        r = frame[strategy].to_numpy(dtype=float)[None, :]
        m = r.shape[1]
        rng = np.random.default_rng(seed)
        stats = np.empty(B)
        for b in range(B):
            idx = circular_block_indices(m, block, rng)
            stats[b] = daily_sharpe(r[:, idx][0], periods_per_year)
        lo, hi = np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi)

    per_strategy = {}
    for s in strategies:
        lo1, hi1 = ci_for(half1, s)
        lo2, hi2 = ci_for(half2, s)
        per_strategy[s] = {
            "half1": {
                "sharpe": round(daily_sharpe(half1[s].to_numpy(), periods_per_year), 6),
                "ci_low": round(lo1, 6),
                "ci_high": round(hi1, 6),
            },
            "half2": {
                "sharpe": round(daily_sharpe(half2[s].to_numpy(), periods_per_year), 6),
                "ci_low": round(lo2, 6),
                "ci_high": round(hi2, 6),
            },
        }
    return {
        "label": "EXPLORATORY",
        "disclosure": (
            "chronological fragility check only; no regime definition, no "
            "regime-robustness claim is made from this output"
        ),
        "half1_days": int(len(half1)),
        "half2_days": int(len(half2)),
        "B": B,
        "block": block,
        "seed": seed,
        "per_strategy": per_strategy,
    }


def placebo_generate_world(
    seed: int,
    mu_daily: np.ndarray,
    cov_daily: np.ndarray,
    n_days: int,
    columns: list[str],
) -> pd.DataFrame:
    """One synthetic i.i.d. Gaussian world (registered placebo DGP).

    Parameters are the FULL-SAMPLE estimates (mu_hat, Sigma_LW) in DAILY units
    (annualized estimates divided by 252). Legitimate calibration use: the
    placebo's purpose is to measure the evaluation noise of a KNOWN-parameter
    world, not to produce an investable result; outputs never mix with the
    real-data headline table. Disclosure: the Gaussian i.i.d. DGP does not
    reproduce fat tails, volatility clustering, or regime persistence.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=mu_daily, cov=cov_daily, size=n_days)
    idx = pd.date_range("2000-01-01", periods=n_days, freq="B")
    return pd.DataFrame(draws, index=idx, columns=columns)


def placebo_world(
    world_index: int,
    world_returns: pd.DataFrame,
    oracle_weights: np.ndarray,
    estimation_window: int = 252,
    holding_period: int = 63,
    max_weight: float | None = 0.40,
    cost_bps: float = 10.0,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    use_shrunk_cov: bool = True,
) -> dict:
    """Run one placebo world: the 4 strategies through the standard engine.

    The oracle (by-construction optimal tangency portfolio for the DGP) is
    scored on the SAME world with FIXED full-sample weights: its daily net
    series is ``world_returns @ w`` with a one-time initial build-up cost
    (turnover 1.0 at the first day, matching the engine's initial build-up
    convention); it never rebalances, so no further cost applies.

    Returns a per-world record (strict-JSON safe after rounding/sanitizing)
    with per-strategy daily-convention Sharpes, the oracle Sharpe, ranks among
    all five portfolios, solver diagnostics deltas, and OOS dimensions.
    """
    from .optimization import solver_diagnostics  # local import: read counters

    diag0 = solver_diagnostics()
    daily, perf, _ = walk_forward_backtest(
        world_returns,
        estimation_window=estimation_window,
        holding_period=holding_period,
        max_weight=max_weight,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        rf_annual=rf_annual,
        use_shrunk_cov=use_shrunk_cov,
    )
    diag1 = solver_diagnostics()
    diag_delta = {k: int(diag1[k] - diag0[k]) for k in diag1}

    r = world_returns.to_numpy()
    w = np.asarray(oracle_weights, dtype=float)
    oos = r[estimation_window : estimation_window + len(daily)]
    oracle_net = oos @ w
    oracle_net[0] -= float(np.abs(w).sum()) * cost_bps / 1e4  # initial build-up
    oracle_sharpe = daily_sharpe(oracle_net, periods_per_year)

    sharpes = {
        s: daily_sharpe(daily[s].to_numpy(), periods_per_year) for s in daily.columns
    }
    sharpes["oracle"] = oracle_sharpe
    names = sorted(sharpes, key=lambda k: (-sharpes[k], k))  # stable, deterministic
    ranks = {name: i + 1 for i, name in enumerate(names)}
    best_among_strategies = min(
        daily.columns, key=lambda s: (-sharpes[s], s)
    )
    return {
        "world": int(world_index),
        "seed": SEED_PLACEBO_BASE + world_index,
        "oos_days": int(len(daily)),
        "n_rebalances": int(perf.loc[daily.columns[0], "n_rebalances"]),
        "sharpes_daily": {k: round(float(v), 6) for k, v in sharpes.items()},
        "ranks_all5": ranks,
        "oracle_rank": int(ranks["oracle"]),
        "best_among_4_strategies": best_among_strategies,
        "ew_beats_oracle": bool(sharpes["equal_weight"] > sharpes["oracle"]),
        "max_sharpe_diagnostics_delta": diag_delta,
    }


def placebo_summary(
    records: list[dict],
    real_diffs: dict[str, float],
    periods_per_year: int = 252,
) -> dict:
    """Summary statistics over per-world placebo records (RQ5 estimands).

    ``real_diffs`` maps the SAME statistic names used here (e.g. the keys of
    ``pairwise_diffs`` below) to their REAL-data values; each real value is
    located inside the placebo distribution of the same statistic as a
    percentile (0–100).
    """
    if not records:
        raise ValueError("no placebo records supplied")
    m = len(records)
    oracle_ranks = np.array([rec["oracle_rank"] for rec in records])
    ew_beats = np.array([rec["ew_beats_oracle"] for rec in records])
    best4 = [rec["best_among_4_strategies"] for rec in records]
    pairwise_diffs: dict[str, np.ndarray] = {}
    for rec in records:
        s = rec["sharpes_daily"]
        for a, b in (("equal_weight", "max_sharpe"), ("min_variance", "risk_parity")):
            key = f"{a}__minus__{b}"
            pairwise_diffs.setdefault(key, []).append(s[a] - s[b])
    pairwise_diffs = {k: np.array(v) for k, v in pairwise_diffs.items()}
    percentiles = {
        key: round(float(np.mean(real_diffs[key] <= vals) * 100.0), 2)
        for key, vals in pairwise_diffs.items()
        if key in real_diffs
    }
    rank_counts = {
        str(r): int(np.sum(oracle_ranks == r))
        for r in range(1, len(records[0]["ranks_all5"]) + 1)
    }
    best4_counts = {name: int(best4.count(name)) for name in set(best4)}
    return {
        "M": m,
        "seeds": f"43 + i for world i (i = 0..{m - 1})",
        "oracle_rank_counts": rank_counts,
        "P_oracle_rank1": round(float(np.mean(oracle_ranks == 1)), 4),
        "P_ew_beats_oracle": round(float(np.mean(ew_beats)), 4),
        "best_among_4_counts": best4_counts,
        "pairwise_diff_placebo_mean": {
            k: round(float(np.mean(v)), 6) for k, v in pairwise_diffs.items()
        },
        "pairwise_diff_placebo_sd": {
            k: round(float(np.std(v, ddof=1)), 6) for k, v in pairwise_diffs.items()
        },
        "real_diff_percentile_in_placebo": percentiles,
        "mc_se_proportion_p50": round(float(np.sqrt(0.25 / m)), 4),
        "interpretation": (
            "the oracle is optimal for the DGP by construction; every world in "
            "which it does not rank first is direct evidence of evaluation-"
            "design rank noise in a single OOS realization of this length"
        ),
    }


def full_suite(
    daily_returns: pd.DataFrame,
    universe_returns: pd.DataFrame,
    B: int = 2000,
    blocks: tuple[int, ...] = (21, 63, 126),
    seed: int = SEED_BOOTSTRAP,
    cost_grid_values: tuple[float, ...] = DEFAULT_COST_GRID,
    placebo_records: list[dict] | None = None,
    real_diffs: dict[str, float] | None = None,
    periods_per_year: int = 252,
    estimation_window: int = 252,
    holding_period: int = 63,
    max_weight: float | None = 0.40,
    rf_annual: float = 0.0,
    strategies: tuple[str, ...] | None = None,
    subwindow_window: int = 252,
) -> dict:
    """Assemble every registered Phase-2 statistic into one payload dict.

    Pure computation (no file I/O). ``placebo_records`` (if supplied) are
    summarized, not re-simulated; placebo simulation is executed chunk-wise by
    the caller via :func:`placebo_generate_world` + :func:`placebo_world` for
    resumability in the sandbox (foreground chunks over persisted per-world
    records).
    """
    levels = level_cis_and_ranks(
        daily_returns, B=B, blocks=blocks, seed=seed, periods_per_year=periods_per_year
    )
    pairs = pairwise_suite_holm(
        daily_returns, B=B, block=holding_period, seed=seed,
        sensitivity_blocks=tuple(b for b in blocks if b != holding_period),
        periods_per_year=periods_per_year,
    )
    subwin = subwindow_stability(
        daily_returns, window=subwindow_window, periods_per_year=periods_per_year
    )
    costs = cost_grid(
        universe_returns,
        grid=cost_grid_values,
        strategies=strategies,
        estimation_window=estimation_window,
        holding_period=holding_period,
        max_weight=max_weight,
        periods_per_year=periods_per_year,
        rf_annual=rf_annual,
    )
    halves = half_split(daily_returns, B=B, block=holding_period, seed=seed,
                        periods_per_year=periods_per_year)
    payload = {
        "levels_and_ranks": levels,
        "pairwise_holm": pairs,
        "subwindow_stability": subwin,
        "cost_sensitivity": costs,
        "half_split_exploratory": halves,
    }
    if placebo_records:
        real = real_diffs or {}
        payload["placebo_summary"] = placebo_summary(
            placebo_records, real, periods_per_year=periods_per_year
        )
    return payload
