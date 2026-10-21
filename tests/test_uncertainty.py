"""PHASE 2 pre-registered tests for the uncertainty-aware evaluation suite.

Covers the acceptance criteria that are testable without data or network:
A2 (engine agreement with the v1.0.0 bootstrap), A6 (exact cost claims),
A8 (strict-JSON safety), plus determinism, Holm correctness, rank
accounting, sub-window stability, the exploratory half split, and the
placebo engine's smoke/determinism/JSON safety.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization_lab.bootstrap import bootstrap_sharpe_differences
from portfolio_optimization_lab.config import OptConfig
from portfolio_optimization_lab.estimates import estimate
from portfolio_optimization_lab.optimization import max_sharpe
from portfolio_optimization_lab.pipeline import strict_json_dumps
from portfolio_optimization_lab.uncertainty import (
    SEED_2026_RESERVED_UNUSED,
    SEED_BOOTSTRAP,
    SEED_PLACEBO_BASE,
    cost_grid,
    daily_sharpe,
    full_suite,
    half_split,
    holm_adjust,
    level_cis_and_ranks,
    pairwise_suite_holm,
    placebo_generate_world,
    placebo_summary,
    placebo_world,
    subwindow_stability,
)


def _synthetic_oos(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """AR(1)-ish 4-strategy daily return frame (data-independent)."""
    rng = np.random.default_rng(seed)
    cols = {}
    for k, name in enumerate(
        ("equal_weight", "min_variance", "max_sharpe", "risk_parity")
    ):
        shocks = rng.normal(0.0004 + 0.0002 * k, 0.006 + 0.001 * k, size=n)
        ar = np.empty(n)
        ar[0] = shocks[0]
        for t in range(1, n):
            ar[t] = 0.15 * ar[t - 1] + shocks[t]
        cols[name] = ar
    return pd.DataFrame(cols)


def _synthetic_universe(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """4-asset daily log-return frame for engine-level (cost grid) tests."""
    rng = np.random.default_rng(seed)
    cov = np.array(
        [
            [2.0e-4, 0.8e-4, 0.3e-4, 0.5e-4],
            [0.8e-4, 2.5e-4, 0.4e-4, 0.6e-4],
            [0.3e-4, 0.4e-4, 1.5e-4, 0.2e-4],
            [0.5e-4, 0.6e-4, 0.2e-4, 1.8e-4],
        ]
    )
    mu = np.array([0.0004, 0.0005, 0.0002, 0.0003])
    draws = rng.multivariate_normal(mu, cov, size=n)
    return pd.DataFrame(
        draws, columns=["A", "B", "C", "D"], index=pd.date_range("2020-01-01", periods=n)
    )


# ---------------------------------------------------------------------------
# A2: engine agreement with the v1.0.0 bootstrap
# ---------------------------------------------------------------------------


def test_engine_agreement_with_v1_engine():
    """Same seed -> identical paired index stream -> identical CIs (bitwise)."""
    daily = _synthetic_oos(400)
    v1 = bootstrap_sharpe_differences(
        daily, B=80, block=14, seed=SEED_BOOTSTRAP, sensitivity_blocks=(7, 28)
    )
    p2 = pairwise_suite_holm(
        daily, B=80, block=14, seed=SEED_BOOTSTRAP, sensitivity_blocks=(7, 28)
    )
    assert len(v1["pairs"]) == len(p2["pairs"]) == 6
    for v1_pair, p2_pair in zip(v1["pairs"], p2["pairs"], strict=True):
        assert v1_pair["a"] == p2_pair["a"]
        assert v1_pair["b"] == p2_pair["b"]
        assert v1_pair["diff"] == p2_pair["diff"]
        assert v1_pair["ci_low"] == p2_pair["ci_low"]
        assert v1_pair["ci_high"] == p2_pair["ci_high"]
        for blk in ("7", "28"):
            assert (
                v1_pair["sensitivity"][blk]["ci_low"]
                == p2_pair["sensitivity"][blk]["ci_low"]
            )
            assert (
                v1_pair["sensitivity"][blk]["ci_high"]
                == p2_pair["sensitivity"][blk]["ci_high"]
            )


def test_pairwise_suite_is_deterministic_and_p_values_valid():
    daily = _synthetic_oos(300)
    a = pairwise_suite_holm(daily, B=50, block=21, sensitivity_blocks=())
    b = pairwise_suite_holm(daily, B=50, block=21, sensitivity_blocks=())
    assert a == b
    for pair in a["pairs"]:
        assert 0.0 <= pair["p_value_add_one"] <= 1.0
        assert pair["p_holm_adjusted"] >= pair["p_value_add_one"]
        assert pair["p_holm_adjusted"] <= 1.0
        assert pair["ci_low"] <= pair["ci_high"]
        assert a["family_size"] == 6


# ---------------------------------------------------------------------------
# Level CIs and rank frequencies (RQ1/RQ2)
# ---------------------------------------------------------------------------


def test_level_cis_determinism_and_validity():
    daily = _synthetic_oos(300)
    one = level_cis_and_ranks(daily, B=60, blocks=(10, 21, 42))
    two = level_cis_and_ranks(daily, B=60, blocks=(10, 21, 42))
    assert one == two
    for s in daily.columns:
        point = one["point_estimates"][s]
        # point estimates are stored rounded to 6 dp
        assert point == pytest.approx(daily_sharpe(daily[s].to_numpy()), abs=1e-6)
        for block in ("10", "21", "42"):
            ci = one["levels"][s][block]
            assert ci["ci_low"] <= ci["ci_high"]
            assert np.isfinite(ci["ci_low"]) and np.isfinite(ci["ci_high"])
        freqs = one["rank_frequencies"][s]
        assert set(freqs) == {"1", "2", "3", "4"}
        assert sum(freqs.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(one["rank1_counts"].values()) == 60


def test_rank_frequencies_dominant_strategy():
    """A strategy with strictly dominant returns must rank 1 with frequency 1."""
    n = 200
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.01, size=n)
    daily = pd.DataFrame(
        {
            "equal_weight": base,
            "min_variance": base + 0.005,
            "max_sharpe": base + 0.010,
            "risk_parity": base + 0.020,  # strictly dominant
        }
    )
    out = level_cis_and_ranks(daily, B=50, blocks=(20, 40))
    assert out["rank_frequencies"]["risk_parity"]["1"] == 1.0
    assert out["rank_frequencies"]["equal_weight"]["4"] == 1.0
    assert out["rank1_counts"]["risk_parity"] == 50


# ---------------------------------------------------------------------------
# Holm adjustment
# ---------------------------------------------------------------------------


def test_holm_known_example():
    out = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["a"]["reject_holm_05"] is True
    assert out["b"]["reject_holm_05"] is False
    assert out["c"]["reject_holm_05"] is False


def test_holm_monotone_and_capped():
    out = holm_adjust({f"p{i}": p for i, p in enumerate([0.001, 0.9, 0.5, 0.2])})
    values = [out[k]["p_holm"] for k in out]
    assert all(0.0 <= v <= 1.0 for v in values)
    raw_order = sorted(out, key=lambda k: out[k]["p_raw"])
    adj_sorted = [out[k]["p_holm"] for k in raw_order]
    assert adj_sorted == sorted(adj_sorted)


# ---------------------------------------------------------------------------
# A6: exact cost claims on the verified engine
# ---------------------------------------------------------------------------


def test_cost_grid_exact_registered_claims():
    returns = _synthetic_universe(300)
    grid = (0.0, 10.0, 25.0, 50.0)
    out = cost_grid(
        returns,
        grid=grid,
        estimation_window=100,
        holding_period=20,
        max_weight=0.4,
    )
    rows = pd.DataFrame(out["rows"])
    for strategy in rows["strategy"].unique():
        sub = rows[rows["strategy"] == strategy].sort_values("cost_bps")
        x = sub["cost_bps"].to_numpy(dtype=float)
        y = sub["mean_daily_net_return"].to_numpy(dtype=float)
        # (i) mean net return obeys the REGISTERED affine law directly:
        #     mean_net(c) = mean_net(0) - c * sum(turnover) / (1e4 * n)
        n_oos = int(sub["oos_days"].iloc[0])
        n_reb = int(sub["n_rebalances"].iloc[0])
        sum_turnover = float(sub["avg_turnover"].iloc[0]) * n_reb
        expected = y[x == 0.0][0] - x * sum_turnover / (1e4 * n_oos)
        residual = float(np.abs(y - expected).max())
        assert residual < 1e-12, (strategy, residual)
        # empirical two-point slope sanity check (loose, vs registered slope)
        empirical_slope = (y[-1] - y[0]) / (x[-1] - x[0])
        registered_slope = -sum_turnover / (1e4 * n_oos)
        assert empirical_slope == pytest.approx(registered_slope, rel=1e-6)
        # (ii) bitwise net == gross at 0 bps (observable via engine columns)
        zero = sub[sub["cost_bps"] == 0.0].iloc[0]
        assert zero["sharpe_net_engine"] == zero["sharpe_gross_engine"]
        assert zero["ann_return_net_engine"] == zero["ann_return_gross_engine"]
        assert zero["total_costs_pct"] == 0.0
        # (iii) turnover invariance across the grid
        assert sub["avg_turnover"].nunique() == 1
        assert sub["n_rebalances"].nunique() == 1


def test_cost_grid_bitwise_net_gross_at_zero_bps():
    """At cost 0 the engine's net and gross daily series are bitwise equal."""
    from portfolio_optimization_lab.backtest import walk_forward_backtest

    returns = _synthetic_universe(260)
    daily0, perf0, _ = walk_forward_backtest(
        returns, strategies=("min_variance", "risk_parity"),
        estimation_window=100, holding_period=20, cost_bps=0.0,
    )
    # the gross series is not returned; at 0 bps the engine skips deduction,
    # so net == gross bitwise -> any cost>0 run must differ on rebalance days
    daily10, _, _ = walk_forward_backtest(
        returns, strategies=("min_variance", "risk_parity"),
        estimation_window=100, holding_period=20, cost_bps=10.0,
    )
    diff = (daily0["min_variance"] - daily10["min_variance"]).to_numpy()
    nonzero = np.where(np.abs(diff) > 0)[0]
    assert len(nonzero) > 0  # costs bite somewhere
    assert perf0["sharpe"].iloc[0] == perf0["sharpe_gross"].iloc[0]
    assert perf0["ann_return"].iloc[0] == perf0["ann_return_gross"].iloc[0]


# ---------------------------------------------------------------------------
# Sub-window stability and half split (RQ2b/RQ4)
# ---------------------------------------------------------------------------


def test_subwindow_stability_structure():
    daily = _synthetic_oos(360)
    out = subwindow_stability(daily, window=60)
    assert out["n_windows"] == 6
    assert len(out["adjacent_kendall_tau"]) == 5
    assert all(-1.0 <= t <= 1.0 for t in out["adjacent_kendall_tau"])
    assert sum(out["rank1_counts"].values()) == 6
    again = subwindow_stability(daily, window=60)
    assert out == again


def test_half_split_exploratory_and_deterministic():
    daily = _synthetic_oos(320)
    out = half_split(daily, B=40, block=20)
    assert out["label"] == "EXPLORATORY"
    assert out["half1_days"] == 160 and out["half2_days"] == 160
    for s in daily.columns:
        h1, h2 = out["per_strategy"][s]["half1"], out["per_strategy"][s]["half2"]
        assert h1["ci_low"] <= h1["ci_high"]
        assert h2["ci_low"] <= h2["ci_high"]
    again = half_split(daily, B=40, block=20)
    assert out == again


# ---------------------------------------------------------------------------
# Placebo engine (RQ5)
# ---------------------------------------------------------------------------


def _placebo_setup():
    returns = _synthetic_universe(220)
    est = estimate(returns)
    oracle = max_sharpe(est.mu, est.cov, 0.0, 0.40)
    world = placebo_generate_world(
        SEED_PLACEBO_BASE,
        est.mu.to_numpy() / 252.0,
        est.cov.to_numpy() / 252.0,
        n_days=220,
        columns=list(returns.columns),
    )
    return returns, est, oracle, world


def test_placebo_world_deterministic_and_json_safe():
    _, est, oracle, world = _placebo_setup()
    rec1 = placebo_world(
        0, world, oracle, estimation_window=80, holding_period=20, max_weight=0.40
    )
    rec2 = placebo_world(
        0, world, oracle, estimation_window=80, holding_period=20, max_weight=0.40
    )
    assert rec1 == rec2
    assert rec1["seed"] == SEED_PLACEBO_BASE
    assert rec1["oos_days"] > 0
    assert np.isfinite(rec1["sharpes_daily"]["oracle"])
    assert np.isfinite(rec1["sharpes_daily"]["equal_weight"])
    assert set(rec1["ranks_all5"]) == {
        "equal_weight", "min_variance", "max_sharpe", "risk_parity", "oracle"
    }
    assert sorted(rec1["ranks_all5"].values()) == [1, 2, 3, 4, 5]
    # strict JSON safety: no NaN/Infinity tokens anywhere in the record
    text = strict_json_dumps(rec1)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    # oracle weights are a valid portfolio
    assert oracle.sum() == pytest.approx(1.0, abs=1e-9)
    assert oracle.max() <= 0.40 + 1e-9


def test_placebo_generate_world_deterministic():
    returns = _synthetic_universe(120)
    est = estimate(returns)
    w1 = placebo_generate_world(
        SEED_PLACEBO_BASE + 3, est.mu.to_numpy() / 252.0,
        est.cov.to_numpy() / 252.0, 120, list(returns.columns),
    )
    w2 = placebo_generate_world(
        SEED_PLACEBO_BASE + 3, est.mu.to_numpy() / 252.0,
        est.cov.to_numpy() / 252.0, 120, list(returns.columns),
    )
    pd.testing.assert_frame_equal(w1, w2)
    # world shape and columns
    assert w1.shape == (120, 4)
    assert list(w1.columns) == list(returns.columns)


def test_placebo_summary_percentiles():
    records = []
    for i in range(4):
        sharps = {
            "equal_weight": 0.10 * i,
            "min_variance": 0.05 * i,
            "max_sharpe": 0.30 + 0.01 * i,
            "risk_parity": 0.20,
            "oracle": 0.50 if i < 3 else 0.005,  # oracle loses in world 3
        }
        names = sorted(sharps, key=lambda k: (-sharps[k], k))
        ranks = {name: j + 1 for j, name in enumerate(names)}
        records.append(
            {
                "world": i,
                "seed": SEED_PLACEBO_BASE + i,
                "oos_days": 100,
                "n_rebalances": 5,
                "sharpes_daily": sharps,
                "ranks_all5": ranks,
                "oracle_rank": ranks["oracle"],
                "best_among_4_strategies": "max_sharpe",
                "ew_beats_oracle": sharps["equal_weight"] > sharps["oracle"],
                "max_sharpe_diagnostics_delta": {"max_sharpe_closed_form": 5},
            }
        )
    out = placebo_summary(
        records,
        real_diffs={"equal_weight__minus__max_sharpe": -0.35},
    )
    assert out["M"] == 4
    assert out["P_oracle_rank1"] == 0.75
    assert out["oracle_rank_counts"]["1"] == 3
    # in world 3 the oracle (0.005) ranks LAST of 5 (max_sharpe 0.33 wins)
    assert out["oracle_rank_counts"]["5"] == 1
    # world 3: EW Sharpe 0.30 > oracle 0.005 -> EW beats the oracle there
    assert out["P_ew_beats_oracle"] == 0.25
    assert out["best_among_4_counts"] == {"max_sharpe": 4}
    assert 0.0 <= out["real_diff_percentile_in_placebo"][
        "equal_weight__minus__max_sharpe"
    ] <= 100.0
    assert out["mc_se_proportion_p50"] == pytest.approx(np.sqrt(0.25 / 4), abs=1e-6)
    text = strict_json_dumps(out)
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# End-to-end suite wiring (A8) and registered seeds
# ---------------------------------------------------------------------------


def test_full_suite_end_to_end_wiring_strict_json():
    daily = _synthetic_oos(340)
    universe = _synthetic_universe(300)
    suite = full_suite(
        daily,
        universe,
        B=30,
        blocks=(10, 21, 42),
        cost_grid_values=(0.0, 10.0),
        strategies=("equal_weight", "min_variance", "risk_parity"),
        estimation_window=100,
        holding_period=21,
        subwindow_window=60,
    )
    for key in (
        "levels_and_ranks",
        "pairwise_holm",
        "subwindow_stability",
        "cost_sensitivity",
        "half_split_exploratory",
    ):
        assert key in suite, key
    # strict JSON round-trip (A8): no NaN/Infinity tokens survive
    text = strict_json_dumps(suite)
    assert "NaN" not in text and "Infinity" not in text
    reparsed = json.loads(text)
    assert reparsed["pairwise_holm"]["family_size"] == 6
    # without placebo records, no placebo summary is claimed
    assert "placebo_summary" not in suite


def test_full_suite_with_placebo_records():
    daily = _synthetic_oos(340)
    universe = _synthetic_universe(300)
    records = [
        {
            "world": 0,
            "seed": SEED_PLACEBO_BASE,
            "oos_days": 100,
            "n_rebalances": 5,
            "sharpes_daily": {
                "equal_weight": 0.2, "min_variance": 0.3, "max_sharpe": 0.1,
                "risk_parity": 0.25, "oracle": 0.6,
            },
            "ranks_all5": {
                "oracle": 1, "min_variance": 2, "risk_parity": 3,
                "equal_weight": 4, "max_sharpe": 5,
            },
            "oracle_rank": 1,
            "best_among_4_strategies": "min_variance",
            "ew_beats_oracle": False,
            "max_sharpe_diagnostics_delta": {"max_sharpe_closed_form": 5},
        }
    ]
    suite = full_suite(
        daily,
        universe,
        B=30,
        blocks=(10, 21, 42),
        cost_grid_values=(0.0,),
        strategies=("equal_weight",),
        estimation_window=100,
        holding_period=21,
        subwindow_window=60,
        placebo_records=records,
        real_diffs={"equal_weight__minus__max_sharpe": 0.1},
    )
    assert suite["placebo_summary"]["M"] == 1
    assert suite["placebo_summary"]["P_oracle_rank1"] == 1.0
    text = strict_json_dumps(suite)
    assert "NaN" not in text and "Infinity" not in text


def test_registered_seed_constants():
    """The registered seeds are exactly as pre-registered (plan Section 6)."""
    assert SEED_BOOTSTRAP == 42
    assert SEED_PLACEBO_BASE == 43
    assert SEED_2026_RESERVED_UNUSED == 2026  # registered DEAD; never used


def test_config_seed_default_is_registered_seed():
    assert OptConfig().seed == SEED_BOOTSTRAP
