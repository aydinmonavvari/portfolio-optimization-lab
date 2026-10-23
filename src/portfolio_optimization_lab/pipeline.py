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
from .data import load_price_frame, simple_returns
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
from .uncertainty import SEED_2026_RESERVED_UNUSED, full_suite  # noqa: F401

PLACEBO_RECORDS_FILENAME = "placebo_records.jsonl"
PLACEBO_ARTIFACT_FILENAME = "placebo_simulation.json"


def load_placebo_records(reports_dir: Path) -> list[dict] | None:
    """Load persisted per-world placebo records (one strict-JSON object/line).

    Returns ``None`` when the file does not exist so the caller can mark the
    placebo as NOT RUN instead of silently reporting an empty simulation.
    """
    path = Path(reports_dir) / PLACEBO_RECORDS_FILENAME
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records or None

logger = logging.getLogger(__name__)


def run_pipeline(
    config: OptConfig | None = None,
    save_outputs: bool = True,
    placebo_records: list[dict] | None = None,
) -> dict:
    """Run the full study: estimates, frontier, walk-forward backtest, reports.

    Headline performance numbers are NET of transaction costs (the cost drag
    is deducted from rebalance-day returns inside the backtest); gross metrics
    are kept alongside for reference. Solver fallback counters are reset at
    the start of the run and persisted with the outputs.

    PHASE 2 (pre-registered): the uncertainty-aware evaluation suite is
    computed on the OOS net daily returns and persisted to
    ``reports/uncertainty.json``, ``reports/cost_sensitivity.csv``,
    ``reports/rank_stability.csv``. The placebo simulation is NOT re-simulated
    here: pass pre-computed per-world records (or drop
    ``reports/placebo_records.jsonl`` next to the reports directory, one
    strict-JSON object per line, produced chunk-wise by the documented scratch
    runner) — otherwise the placebo is reported as NOT RUN.
    """
    config = config or OptConfig()
    config.ensure_dirs()
    reset_solver_diagnostics()

    prices = load_price_frame(config.tickers, config.benchmark, config.raw_dir)
    # the benchmark is context only; optimization runs on the universe.
    # Daily SIMPLE returns: the per-day portfolio return is then the exact
    # weighted sum r @ w (no log-space approximation anywhere downstream).
    returns = simple_returns(prices[list(config.tickers)])

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

    # ---- 5. PHASE 2 uncertainty-aware evaluation suite (pre-registered) ----
    uncertainty_suite = full_suite(
        daily,
        returns,
        B=2000,
        blocks=(21, config.holding_period, 126),
        seed=config.seed,
        cost_grid_values=(0.0, 10.0, 25.0, 50.0, 75.0, 100.0),
        placebo_records=placebo_records,
        real_diffs=_real_pairwise_diffs(sharpe_bootstrap),
        periods_per_year=config.periods_per_year,
        estimation_window=config.estimation_window,
        holding_period=config.holding_period,
        max_weight=config.max_weight,
        rf_annual=config.rf_annual,
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
            uncertainty_suite,
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
        "uncertainty_suite": uncertainty_suite,
    }


def _real_pairwise_diffs(sharpe_bootstrap: dict) -> dict[str, float]:
    """Real-data pairwise daily-convention diffs keyed for the placebo summary."""
    out: dict[str, float] = {}
    for pair in sharpe_bootstrap.get("pairs", []):
        key = f"{pair['a']}__minus__{pair['b']}"
        out[key] = float(pair["diff"])
    return out


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
    uncertainty_suite: dict | None = None,
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

    # ---- PHASE 2 artifacts (pre-registered file manifest) -----------------
    suite = uncertainty_suite or {}
    placebo_payload: dict = {"status": "not_run", "disclosure": (
        "per-world records were not supplied and "
        f"{PLACEBO_RECORDS_FILENAME} was not found; run the documented "
        "chunk-wise scratch runner to produce them"
    )}
    if suite:
        (config.reports_dir / "uncertainty.json").write_text(
            strict_json_dumps(
                {
                    k: v
                    for k, v in suite.items()
                    if k not in ("cost_sensitivity", "placebo_summary")
                }
            )
            + "\n",
            encoding="utf-8",
        )
        if "cost_sensitivity" in suite:
            pd.DataFrame(suite["cost_sensitivity"]["rows"]).round(8).to_csv(
                config.reports_dir / "cost_sensitivity.csv", index=False
            )
        # rank stability table: per-window daily-convention Sharpes
        if "subwindow_stability" in suite:
            sw = suite["subwindow_stability"]
            rank_table = pd.DataFrame(sw["window_sharpes_daily"])
            rank_table.index.name = "window"
            rank_table.round(6).to_csv(config.reports_dir / "rank_stability.csv")
        if "placebo_summary" in suite:
            placebo_payload = {
                "status": "complete",
                "M": suite["placebo_summary"]["M"],
                "summary": suite["placebo_summary"],
            }
            (config.reports_dir / PLACEBO_ARTIFACT_FILENAME).write_text(
                strict_json_dumps(placebo_payload) + "\n", encoding="utf-8"
            )
    _write_summary(config, payload, performance, comparison, sharpe_bootstrap, suite)

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
    if suite:
        if "levels_and_ranks" in suite:
            figs["sharpe_level_cis.png"] = plots.plot_level_cis(
                suite["levels_and_ranks"]
            )
            figs["rank_frequencies.png"] = plots.plot_rank_frequencies(
                suite["levels_and_ranks"]["rank_frequencies"]
            )
        if "cost_sensitivity" in suite:
            figs["cost_sensitivity.png"] = plots.plot_cost_curves(
                suite["cost_sensitivity"]
            )
    if placebo_payload.get("status") == "complete":
        records_path = config.reports_dir / PLACEBO_RECORDS_FILENAME
        records = load_placebo_records(config.reports_dir) or []
        if records:
            figs["placebo_rank_dist.png"] = plots.plot_placebo_rank_dist(
                placebo_payload["summary"]["oracle_rank_counts"],
                m=placebo_payload["summary"]["M"],
                records_present=records_path.exists(),
            )
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
    uncertainty_suite: dict | None = None,
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
        "NOTE on the flag column: `significant_95` is UNADJUSTED (per-pair CI "
        "statement only). The registered multiplicity-safe reading is the Holm "
        "family below — no pairwise claim may be quoted without its Holm column.",
        "",
    ]

    suite = uncertainty_suite or {}
    if "pairwise_holm" in suite:
        lines += [
            "## Pairwise differences, multiplicity-safe (Holm step-down, FWER 0.05)",
            "",
            "Same paired bootstrap streams as the v1.0.0 engine (per-pair fresh",
            "seed-42 streams); two-sided add-one bootstrap p-values; Holm-adjusted",
            "across the 6-pair family.",
            "",
            "| a | b | diff | p_raw | p_holm | reject_holm_05 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for pair in suite["pairwise_holm"]["pairs"]:
            lines.append(
                f"| {pair['a']} | {pair['b']} | {pair['diff']:.4f} "
                f"| {pair['p_value_add_one']:.4f} | {pair['p_holm_adjusted']:.4f} "
                f"| {pair['reject_holm_05']} |"
            )
        n_rej = suite["pairwise_holm"]["n_reject_holm_05"]
        lines += [
            "",
            f"**Holm verdict: {n_rej} of 6 pairwise differences survive family-wise"
            f" correction at 0.05.**",
            "",
        ]
    if "levels_and_ranks" in suite:
        lv = suite["levels_and_ranks"]
        lines += [
            "## Level CIs for each strategy's OOS Sharpe (daily convention)",
            "",
            f"Point estimate and {lv['B']}-resample circular-block percentile CIs",
            "(block 63 headline; blocks 21/126 in the JSON artifact), seed 42.",
            "",
            "| strategy | point | low (b63) | high (b63) | low (b21) | high (b21) "
            "| low (b126) | high (b126) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in lv["point_estimates"]:
            b63 = lv["levels"][s]["63"] if "63" in lv["levels"][s] else None
            b21 = lv["levels"][s].get("21")
            b126 = lv["levels"][s].get("126")
            lines.append(
                f"| {s} | {lv['point_estimates'][s]:.4f} | {b63['ci_low']:.4f} "
                f"| {b63['ci_high']:.4f} | {b21['ci_low']:.4f} | {b21['ci_high']:.4f} "
                f"| {b126['ci_low']:.4f} | {b126['ci_high']:.4f} |"
            )
        lines += [
            "",
            "Reading rule (pre-registered): an interval that spans another "
            "strategy's point estimate is evidence the ranking is not identified "
            "by this sample. Overlapping CIs do NOT imply equality.",
            "",
            "## Bootstrap rank frequencies (block 63, pooled over resamples)",
            "",
            "| strategy | rank 1 | rank 2 | rank 3 | rank 4 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s, freqs in lv["rank_frequencies"].items():
            lines.append(
                f"| {s} | {freqs['1']:.3f} | {freqs['2']:.3f} | {freqs['3']:.3f} "
                f"| {freqs['4']:.3f} |"
            )
        lines += [
            "",
            f"Rank-1 counts (of {lv['B']} resamples): "
            + ", ".join(f"{s} = {c}" for s, c in lv["rank1_counts"].items())
            + ".",
            "",
        ]
    if "subwindow_stability" in suite:
        sw = suite["subwindow_stability"]
        lines += [
            "## Rank stability across time (7 contiguous 252-day windows)",
            "",
            f"Adjacent-window Kendall tau: {sw['adjacent_kendall_tau']}",
            "",
            f"Mean adjacent tau = {sw['mean_adjacent_kendall_tau']}; rank-1 counts "
            f"per window: {sw['rank1_counts']}.",
            "",
        ]
    if "cost_sensitivity" in suite:
        cs = suite["cost_sensitivity"]
        lines += [
            "## Cost-assumption sensitivity (RQ3, true net-of-cost engine)",
            "",
            f"Grid: {cs['grid']} bps. Mean daily net return is exactly affine in "
            "cost with slope -sum(turnover)/(1e4*n) per strategy (pinned by "
            "tests); net Sharpe is NOT affine and no such claim is made.",
            "",
            "| strategy | Sharpe@0bps | Sharpe@10bps | Sharpe@50bps | Sharpe@100bps "
            "| turnover/reb |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        rows = cs["rows"]
        by_strategy: dict[str, dict[float, dict]] = {}
        for row in rows:
            by_strategy.setdefault(row["strategy"], {})[row["cost_bps"]] = row
        for s, per_cost in by_strategy.items():
            def _g(cost: float, _pc: dict = per_cost) -> dict:
                return _pc.get(cost, {})

            lines.append(
                f"| {s} | {_g(0.0).get('sharpe_net_dailyconv', float('nan')):.4f} "
                f"| {_g(10.0).get('sharpe_net_dailyconv', float('nan')):.4f} "
                f"| {_g(50.0).get('sharpe_net_dailyconv', float('nan')):.4f} "
                f"| {_g(100.0).get('sharpe_net_dailyconv', float('nan')):.4f} "
                f"| {_g(10.0).get('avg_turnover', float('nan')):.4f} |"
            )
        lines += [""]
    if "half_split_exploratory" in suite:
        hs = suite["half_split_exploratory"]
        lines += [
            "## First vs second half (EXPLORATORY — fragility check only)",
            "",
            "| strategy | H1 Sharpe | H1 95% CI | H2 Sharpe | H2 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s, halves in hs["per_strategy"].items():
            h1, h2 = halves["half1"], halves["half2"]
            lines.append(
                f"| {s} | {h1['sharpe']:.4f} | [{h1['ci_low']:.4f}, {h1['ci_high']:.4f}] "
                f"| {h2['sharpe']:.4f} | [{h2['ci_low']:.4f}, {h2['ci_high']:.4f}] |"
            )
        lines += [
            "",
            "Registered label: EXPLORATORY. No regime definition and no "
            "regime-robustness claim is made from this output.",
            "",
        ]
    if "placebo_summary" in suite:
        ps = suite["placebo_summary"]
        lines += [
            "## Placebo simulation (RQ5, evaluation-design noise calibration)",
            "",
            f"M = {ps['M']} synthetic worlds (registered DGP: i.i.d. Gaussian with "
            "the full-sample Ledoit–Wolf estimates; oracle = constrained tangency "
            f"optimal for this DGP by construction). MC SE on proportions ≈ "
            f"{ps['mc_se_proportion_p50']:.2f} at p = 0.5.",
            "",
            f"- P(oracle ranks first in a world) = {ps['P_oracle_rank1']}",
            f"- Oracle rank counts: {ps['oracle_rank_counts']}",
            f"- P(equal weight beats the oracle) = {ps['P_ew_beats_oracle']}",
            f"- Best among the 4 re-estimating strategies: {ps['best_among_4_counts']}",
            f"- Real pairwise diffs inside the placebo distribution (percentile): "
            f"{ps['real_diff_percentile_in_placebo']}",
            "",
        ]
    else:
        lines += [
            "## Placebo simulation (RQ5)",
            "",
            "NOT RUN in this invocation (no per-world records supplied). The",
            "registered engine and resumable chunk-wise runner are in",
            "`uncertainty.py`; `reports/placebo_simulation.json` is written only",
            "from persisted per-world records.",
            "",
        ]
    lines += [
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
