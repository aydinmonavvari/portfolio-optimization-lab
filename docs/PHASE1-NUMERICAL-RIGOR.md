# Phase 1 — Numerical-Rigor Corrections (branch `fix/numerical-rigor`)

Status: applied on the isolated branch `fix/numerical-rigor` (base: v1.1.0 `b7586d2`).
No push, no PR, no merge, no tag, no release without separate owner approval.
Shipped v1.1.0 artifacts in `reports/` and `figures/` are untouched.

## What was wrong, and how each defect was demonstrated first

Every defect below was reproduced with a minimal hand-checked example BEFORE any
code change (red test), then fixed, then pinned (green test).

### A) `max_weight` was checked on the UNNORMALIZED closed-form candidate

`max_sharpe` accepted the closed-form tangency `Σ⁻¹(μ−rf)` when the RAW vector
satisfied the cap. The raw vector carries arbitrary units: whenever
`sum(Σ⁻¹(μ−rf)) < 1`, normalization amplifies weights and the returned portfolio
violated the cap.

- Demonstration: `Σ = [[.04,.01],[.01,.09]]`, `μ−rf = [.02,.01]`, cap `0.6` →
  returned `[0.8947, 0.1053]` (cap 0.6 violated). Hand check:
  `Σ⁻¹(μ−rf) = [0.485714, 0.057143]`, normalized max `0.8947 > 0.6`.
- Production impact: NONE on the shipped v1.1.0 artifacts — the run's
  `solver_diagnostics` shows `max_sharpe_closed_form = 0` (the scan branch served
  every window) and the shipped weights respect the cap.
- Fix: normalize first, apply the cap test to the normalized vector.
- Tests: `test_max_sharpe_cap_holds_on_normalized_closed_form`,
  `test_max_sharpe_cap_decision_is_scale_invariant` (6 scales).

### B) Fallback frontier scan mixed excess-return targets with a raw-return constraint

The scan enumerated targets in `[0, max_excess]` (excess space) but passed them
directly as RAW-return equality constraints to `min_variance`. With `rf > 0` the
searched interval sits far below the true optimum.

- Demonstration: `μ = [.10,.02,.025]`, `Σ = diag(.04)`, `rf = 0.03`, cap `0.5` →
  implementation Sharpe `0.0199` vs exact constrained tangency `0.2558`
  (activity-pattern enumeration, rel. gap ≈ 92%). With `rf = 0` the same
  instance is solved exactly (control).
- Fix: map each excess target to the raw constraint as `rf + target`.
- Tests: `test_max_sharpe_fallback_matches_exact_tangency_with_rf` (rf ∈
  {0, .02, .03, .05} vs the exact reference).
- Production impact: NONE on shipped artifacts (`rf_annual = 0.0` in the config;
  the bug requires `rf ≠ 0`).

### C) Return-convention mixing and turnover mislabeling

The engine consumed daily LOG returns but (i) formed portfolio returns as the
linear `r @ w` (a weighted average of log returns ≠ the exact portfolio return),
(ii) annualized by summing the daily series and exponentiating (treating simple
figures as log), and (iii) computed turnover as `Σ|Δw|` (both sides) while
documenting it as one-way.

- Demonstration (hand case): asset +10% then −10%, flat second asset, equal
  weights → true growth `1.05 × 0.95 = 0.9975` (−0.25%); engine convention
  reported −0.50% (2× error). End-to-end on synthetic data the annualized gross
  return error was +0.72pp. Turnover: a full flip `[1,0] → [0,1]` reports 2.0
  under `Σ|Δw|` but one-way traded value is 1.0.
- Fix: daily SIMPLE returns end to end (`data.simple_returns`, pipeline switch,
  `estimate()` semantics), geometric annualization `exp(Σ log1p(r)/years) − 1`
  (net and gross), turnover = the registered ONE-WAY convention
  `Σ max(Δw, 0)` (initial build from cash = 1.0) per
  `docs/admissions/PHASE2-RESEARCH-PLAN.md`, cumulative-growth and drawdown
  plots recomputed on growth `(1+r).cumprod()`.
- Tests: `test_backtest_annualization_is_geometric_on_simple_returns`,
  `test_net_return_deducts_one_way_cost_once_per_rebalance`,
  `test_turnover_is_one_way_sum_of_positive_weight_changes`,
  `test_simple_returns_definition`.

### D) Post-constraint re-check after clip/renormalize (defensive hardening)

The clip-and-renormalize safety net was not followed by a constraint re-check.

- Adversarial search: ~2,400 randomized near-singular/boundary instances — worst
  observed post-renormalization violation ≈ 2.1e-9 (below materiality); NO
  materially violating case found. The shipped run's counter is
  `min_variance_clip_renorm = 0`.
- Fix: `_assert_solution_postconditions` re-checks budget, long-only, cap and
  target on the FINAL vector (tolerance 1e-8) and raises loudly; the frontier
  scan treats such a raise as an infeasible point (existing `RuntimeError`
  handling).
- Tests: `test_min_variance_postconditions_battery` (200 instances),
  `test_postcondition_guard_raises_on_violating_solution` (guard unit test).

### E) Input validation

Previously silent: label mismatches (positional alignment once produced an
INVERTED portfolio in a 2-asset demo), non-finite `mu` (accepted, returned 1/n),
zero-variance assets (divide-by-zero warnings then a cryptic failure),
`n·max_weight < 1` (cryptic SLSQP errors), non-PSD covariance.

- Fix: `_validate_market_inputs` enforces exact label alignment, finiteness,
  symmetry, PSD (tolerance 1e-10 relative), positive variances; `_check_cap_budget`
  rejects infeasible budgets; `walk_forward_backtest` validates finite returns
  > −1. All raise `ValueError` with actionable messages.
- Tests: 5 dedicated tests incl. the inverted-portfolio case.

## Reproduction under the corrected engine (item F)

Data: re-downloaded from Yahoo Finance for the IDENTICAL window
(2018-01-02 → 2026-10-07; all 8 symbols = 2,203 aligned rows). Caveat: Yahoo
revises adjusted closes, so inputs are the same window but not guaranteed
bit-identical to the original download.

Outputs were written to NEW directories only (`reports_phase1/`,
`figures_phase1/`); nothing under `reports/` or `figures/` was modified.
The placebo simulation was NOT re-run (its persisted per-world records were
generated under the v1.1.0 convention; re-simulation is out of scope for this
branch) and is reported as NOT RUN in `reports_phase1/summary.md`.

Headline comparison (v1.1.0 engine → corrected engine):

| strategy     | OOS Sharpe net | avg turnover | total costs % |
|--------------|---------------:|-------------:|--------------:|
| equal_weight | 1.0055 → 1.1495 | 0.0333 → 0.0333 | 0.10 → 0.10 |
| min_variance | 0.6905 → 0.8063 | 0.2189 → 0.1259 | 0.66 → 0.38 |
| max_sharpe   | 0.9569 → 1.0569 | 0.6192 → 0.3226 | 1.86 → 0.97 |
| risk_parity  | 0.9723 → 1.1165 | 0.0888 → 0.0610 | 0.27 → 0.18 |

Every change is mechanically explained: simple-vs-log feeding + `log1p`
annualization shifts all returns; the one-way turnover convention roughly halves
turnover/costs (initial build-up unchanged); cap decisions now made on
normalized weights (diagnostics: `max_sharpe_closed_form` 0 → 2 of 212 windows).
Registered conclusion unchanged: **0 of 6 pairwise gaps survive Holm correction
under both engines**; equal weight remains the best OOS net Sharpe.

## Verification battery on this branch

- `pytest tests/` → 73 passed (50 pre-existing + 23 new), exit 0
- `ruff check .` → clean
- `scripts/verify_docs.py` → 170/170 (README/report vs committed artifacts)
