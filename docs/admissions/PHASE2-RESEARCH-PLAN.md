# PHASE 2 — Research Plan (Pre-Registered)

**Title:** Beyond Point Estimates: Uncertainty-Aware Evaluation of Portfolio
Strategies Under Estimation Error and Transaction Costs

**Repository:** `portfolio-optimization-lab` (branch `phase2-uncertainty`, off `main` @ `9eb7b3d` = v1.0.0)
**Status of this document:** PRE-REGISTERED before any implementation of the
uncertainty suite. It is committed to the branch *before* the first
implementation commit, and its acceptance criteria (Section 9) are enforced by
tests where testable. Any deviation discovered during implementation MUST be
recorded in the amendment log (Section 10) with its rationale, before results
are looked at whenever possible.

**Honesty contract:** the goal of this phase is research that is *real,
accurate, reproducible, and honest* — not impressive. Null or unflattering
results are kept, reported, and made visible in the README.

---

## 1. Existing methodology (what v1.0.0 already does — verified in source, not from memory)

The v1.0.0 pipeline (`src/portfolio_optimization_lab/`) implements:

- **Universe & sample:** 7 US large caps (AAPL, MSFT, JNJ, JPM, XOM, PG, AMZN),
  adjusted closes 2018-01-01 → 2026-10-08 (2,203 trading days at the v1.0.0
  vintage), benchmark SPY for context only.
- **Walk-forward design:** at each rebalance date `t`, estimators see ONLY the
  trailing 252 trading days; weights are held for the following 63 days and
  scored on returns realized after `t`. Estimation and evaluation windows are
  strictly disjoint by construction (unit-tested).
- **Strategies:** equal weight, min-variance (SLSQP QP, long-only, budget,
  40% cap), max-Sharpe (closed-form tangency accepted ONLY if feasible as-is;
  otherwise constrained frontier scan + ternary search), risk parity (CCD of
  Griveau-Billion, Richard & Roncalli 2013, with equal-contribution assertion).
- **Covariance:** Ledoit–Wolf shrinkage.
- **Costs:** one-way turnover × `cost_bps/1e4` (10 bps) deducted from the
  rebalance-day return itself; the daily NET series is the headline, and the
  gross series is retained for reference. This was verified line-by-line in
  `backtest.py` (cost is subtracted exactly once per rebalance, on the correct
  day; `net_block[0] -= turn * cost_bps / 1e4`).
- **Uncertainty (v1.0.0 scope):** paired circular-block bootstrap (Politis &
  Romano 1994) CIs for all 6 pairwise Sharpe differences, B = 2000,
  block = 63 (holding-period-matched), sensitivity at blocks 21 and 126,
  seed 42, on OOS daily net returns.
- **Known v1.0.0 defects (found in audit, unfixed on main):** (i)
  `docs/research_report.md` is stale — its abstract and comparison table quote
  max-Sharpe OOS Sharpe 0.876 whereas every regenerated artifact says 0.957
  net; (ii) a README rounding slip (−0.027 vs artifact −0.0265); (iii) the
  v1.0.0 pairwise table reports unadjusted "significant_95" flags, so the one
  nominal rejection (min-variance vs risk parity) is **not** multiplicity-safe.

## 2. Genuine novelty (scoped claim — no overstatement)

What this phase adds is **evaluation methodology**, not a new portfolio
algorithm and not a new market anomaly:

1. Estimation-uncertainty quantification for the *evaluation itself*: level
   CIs for each strategy's OOS Sharpe (v1.0.0 only had pairwise differences),
   bootstrap rank frequencies, and Holm-corrected inference across the 6
   pairwise comparisons (v1.0.0's per-pair "significant" flags are not a
   family-wise-error-controlled procedure).
2. A **placebo simulation** that separates "the strategy is good" from "the
   evaluation window happened to favor it": on synthetic worlds generated from
   the model's own DGP, the by-construction-optimal (oracle) tangency
   portfolio is scored with the same walk-forward machinery. How often the
   oracle fails to rank first in a single 1,890-day OOS realization calibrates
   how much rank noise the evaluation design itself carries.
3. Cost-assumption sensitivity computed on the true net-of-cost engine, with
   the exact algebraic claims (Section 5, RQ3) pinned by tests.

We do NOT claim: new optimization theory, a strategy that beats 1/n, regime
forecasting, or any out-of-universe generalization. DeMiguel, Garlappi & Uppal
(2009) and Ledoit & Wolf (2008) are the reference points; the contribution is
a faithful, uncertainty-aware *application* and evaluation, which is the
honest scope of a one-author final-year BSc research project.

## 3. Research questions, estimands, and falsifiable hypotheses

Estimands are registered here, before looking at any Phase-2 result. All
bootstrap statistics are computed on the 1,890-day OOS daily NET return series
produced by the verified v1.0.0 engine.

| ID | Question (plain language) | Estimand (registered) | Test / statistic | Falsifiable hypothesis (H0) |
|----|---------------------------|------------------------|------------------|------------------------------|
| RQ1 | How uncertain is each strategy's OOS Sharpe? | Per-strategy annualized daily-convention Sharpe `sqrt(252)·mean(r)/std(r,ddof=1)` on OOS net returns; its sampling distribution under the circular-block bootstrap | Level CIs, percentile method, B=2000, blocks 21/63/126, seed 42 | H0: none (estimation, not a hypothesis test). *Reading rule*: an interval that spans another strategy's point estimate is evidence the ranking is not identified by this sample |
| RQ2 | Does the strategy ranking survive resampling and time? | (a) bootstrap rank frequencies per strategy (pooled over B resamples, seed 42, block 63); (b) rank correlation between adjacent 252-day sub-window Sharpes (7 contiguous windows, Kendall τ); (c) rank-1 counts per sub-window | Descriptive: frequencies, τ, counts | H0: ranking is time-invariant → adjacent-window τ = 0 is consistent with H0; τ near 1 rejects H0 |
| RQ3 | How sensitive are conclusions to the cost assumption? | Net OOS Sharpe (and mean net return) as a function of `cost_bps ∈ {0, 10, 25, 50, 75, 100}`; turnover paths (identical across costs by construction — weights do not depend on cost) | Re-run the verified backtest engine per cost level; exact properties pinned by unit tests (Section 5, A6) | H0: ordering of strategies is invariant on [0,100] bps. Rejection = any crossing of any two net-Sharpe curves in the grid |
| RQ4 | Are conclusions stable over calendar time? (period robustness — deliberately NOT a regime claim) | Per-strategy net Sharpe in the first vs second chronological half of the OOS sample (945 + 945 days) | Descriptive halves + level CIs; labeled **exploratory** | H0: no period effect. We pre-commit to *not* claiming regime robustness from this; it is a fragility check only |
| RQ5 | How much of the observed cross-strategy performance gap is compatible with pure estimation/evaluation noise? | Under M synthetic worlds from the registered DGP (below), the distribution of (i) the oracle's rank, (ii) P(EW Sharpe > oracle Sharpe), (iii) the percentile at which the REAL pairwise diffs (EW−MS, MV−RP) fall inside the placebo distribution of the same statistic | Placebo Monte Carlo; M per decision rule (Section 7); per-world seed `43 + i` | H0: the oracle ranks first in every world. Any world where it does not is direct evidence of evaluation-design noise |

**Placebo DGP (registered):** multivariate Gaussian, i.i.d. across days,
parameters = full-sample estimates `(μ̂, Σ̂_LW)` from the project's own
`estimate()` (Ledoit–Wolf). The oracle = the constrained tangency portfolio
computed from the SAME `(μ̂, Σ̂_LW)` with the same 40% cap, i.e. exactly
optimal *for this DGP by construction*. All other strategies re-estimate
inputs from each world's simulated walk-forward windows through the standard
v1.0.0 pipeline. Disclosure (registered): the Gaussian i.i.d. DGP does not
reproduce fat tails, volatility clustering, or regime persistence; the placebo
therefore isolates **evaluation-design noise**, not all real-world noise.

**Known interpretation limits (registered now):**
- Two Sharpe conventions coexist in the project: the backtest table uses
  annualized-geometric-return/volatility Sharpe; the bootstrap uses the
  daily-convention `sqrt(252)·mean/std`. They differ numerically; both are
  disclosed, and RQ1–RQ2 statistics use the daily convention consistently.
- Overlapping-CI ≠ equality test: non-overlapping CIs imply a difference, but
  overlapping CIs do NOT imply equality; the formal check is the Holm-adjusted
  pairwise test.
- Economic vs statistical significance are reported separately (turnover and
  total cost drag vs CI position).

## 4. Data and limitations

- **Data:** the project's cached adjusted closes (yfinance), 2018-01-01 →
  2026-10-08 window as documented in the README reproduction section; 2,203
  trading days at download; 1,890 OOS days after the 252-day burn-in;
  30 rebalances. Datasets are NEVER committed (`.gitignore` enforces this).
- **Vintage:** this re-run downloads fresh data (current sandbox date
  2026-10-16) with the same documented window. Any drift vs committed
  artifacts is disclosed in the validation report; at the v1.0.0 vintage check
  all four headline net Sharpes matched to ≥3 decimal places, and any
  remaining differences are expected at the 6th decimal (provider-side
  revisions of adjusted closes).
- **Limitations (registered):** one universe, one market, one period; 7 assets
  (estimation of a 7×7 covariance is well-conditioned but μ is barely
  estimable — that is precisely the object of study); yfinance adjusted-close
  methodology is provider-controlled; survivorship of today's mega-caps is
  acknowledged (the universe is a pedagogical, not investable, selection);
  results do not generalize beyond the sample; no multiple-testing protection
  is claimed beyond the registered Holm family (the 6 pairwise Sharpe
  differences).

## 5. Statistical methodology (registered)

1. **Block bootstrap engine:** circular block bootstrap (Politis & Romano
   1994), wrap-around indexing, resample length = n. The Phase-2 engine MUST
   consume the identical index stream as the v1.0.0
   `bootstrap.circular_block_indices` for the same seed (exact-agreement
   regression test, A2). Block length 63 (holding-period-matched) is the
   headline; 21 and 126 are the pre-registered sensitivity set. Block-length
   justification: 63 = the design's dependence scale (fixed weights per
   holding period); 21/126 bracket it at monthly and half-year scales.
2. **Level CIs (RQ1):** B = 2000 percentile intervals per strategy per block
   length, seed 42, on the OOS net daily returns.
3. **Pairwise diffs + multiplicity (RQ1/RQ2):** for each of the 6 unordered
   pairs, the paired bootstrap diff (same resampled indices for both legs);
   two-sided bootstrap p-value with add-one correction
   `p = 2·min((#{d_b ≤ 0}+1)/(B+1), (#{d_b ≥ 0}+1)/(B+1))`; Holm step-down
   adjustment across the 6-test family; family-wise α = 0.05.
4. **Rank frequencies (RQ2):** from the SAME B resamples as the block-63
   level engine, rank the 4 strategies per resample; report pooled frequency
   of each rank 1–4 per strategy.
5. **Sub-window stability (RQ2):** 7 contiguous 252-day windows; net daily
   Sharpes per window (both conventions disclosed); Kendall τ between adjacent
   rankings (6 adjacent pairs); rank-1 counts.
6. **Cost grid (RQ3):** `cost_bps ∈ {0, 10, 25, 50, 75, 100}` re-run through
   the verified engine. Registered *exact* claims (tested, not asserted):
   (i) mean/total net return is affine in `cost_bps` with slope
   `−Σ turnover/n`; (ii) at 0 bps the net series equals the gross series
   bitwise; (iii) turnover per strategy is invariant to `cost_bps`; and the
   registered *non-claim*: net Sharpe is **not** exactly affine in
   `cost_bps` (the denominator changes with the rebalance-day adjustments), so
   no such claim is made anywhere.
7. **Half split (RQ4):** chronological halves; per-strategy Sharpes + level
   CIs; exploratory label enforced in prose.
8. **Placebo (RQ5):** M worlds; per-world seed `default_rng(43 + i)`; each
   world runs the full walk-forward backtest (same config) plus the oracle;
   per-world records persisted (strategy Sharpes, oracle Sharpe, ranks,
   fallback count) so the run is resumable and auditable. Summary statistics
   as in RQ5's estimand row. The real pairwise diffs are compared as
   percentiles against the placebo distribution of the same statistic.
9. **Reporting discipline:** every artifact emits strict JSON
   (`allow_nan=False`, non-finite → null); every README/report number that is
   artifact-derived is checked programmatically (A9).

## 6. Evaluation protocol and leakage risk

- The OOS evaluation protocol is unchanged from v1.0.0 (strictly disjoint
  estimation/evaluation windows, unit-tested; costs deducted inside the
  engine; no look-ahead in estimation).
- **Phase-2-specific leakage risks and controls:**
  1. *Placebo DGP calibration* uses full-sample `(μ̂, Σ̂_LW)` — this is
     legitimate because the placebo's purpose is to calibrate evaluation noise
     of a KNOWN-parameter world, not to produce an investable result; it is
     disclosed as such and its outputs are never mixed with the real-data
     headline table.
  2. *Sub-windows* are chronological and contiguous — no overlapping mixed
     periods in a single ranking; adjacent-τ comparisons only.
  3. *Cost grid* re-uses the identical weight paths (costs do not enter
     optimizers) — asserted by the turnover-invariance test (A6), preventing
     silent re-optimization across cost levels.
  4. *No validation/test split gaming:* the sample is fully OOS after
     burn-in; no model selection is performed on the OOS segment. Phase 2
     introduces no tunable hyperparameters fit on OOS data (B, block lengths,
     cost grid, M are fixed here, before results are seen).
  5. *Seeds:* 42 (all bootstrap engines), `43 + i` (placebo world i), and 2026
     is RESERVED-UNUSED (registered as dead so it can never be silently
     repurposed).

## 7. Computational cost and the M decision rule

- Baseline pipeline (data → artifacts): ≈ 2 min on this sandbox.
- Uncertainty suite (levels + pairs + rank frequencies, B=2000 × 3 block
  lengths): ≈ 3 min.
- Cost grid (6 backtest runs): ≈ 2 min.
- Placebo: ≈ 22 s per world (full walk-forward + oracle).
- **Registered decision rule for M (pre-committed before the run):** measure
  per-world wall time on 2 pilot worlds; if > 15 s, run M = 50 (Monte Carlo
  SE on a proportion ≈ 7 percentage points at p ≈ 0.5 — disclosed in every
  artifact and document); else M = 100. Hard budget cap: 30 minutes of
  placebo compute. Sandbox constraint: background processes are killed when a
  tool call ends, so the placebo run is executed as resumable foreground
  chunks over persisted per-world records (deterministic per world; byte-safe
  by seed).

## 8. Exactly which files change (registered file manifest)

New:
- `src/portfolio_optimization_lab/uncertainty.py` — level CIs, pairwise suite
  + Holm, rank frequencies, sub-window stability, cost grid, half split,
  placebo engine (`placebo_run_paths` / `placebo_summary` split for
  resumability)
- `tests/test_uncertainty.py` — determinism, v1.0.0 engine-agreement
  regression (A2), interval validity, Holm known-example, rank accounting,
  cost exactness (A6), placebo smoke/determinism/JSON-safety, half split,
  end-to-end suite wiring
- `scripts/verify_docs.py` — programmatic README/report vs artifact checks
- `tests/test_docs_match_artifacts.py` — CI-enforced doc-drift checks (A9)
- `docs/admissions/PHASE2-RESEARCH-PLAN.md` (this file),
  `docs/admissions/PHASE2-VALIDATION-REPORT.md`,
  `docs/admissions/PROFESSOR-SUMMARY.md`, `docs/admissions/INTERVIEW-NOTES.md`

Modified:
- `src/portfolio_optimization_lab/pipeline.py` — suite wiring; new strict-JSON
  artifacts: `reports/uncertainty.json`, `reports/placebo_simulation.json`,
  `reports/cost_sensitivity.csv`, `reports/rank_stability.csv`; extended
  `reports/summary.md`
- `src/portfolio_optimization_lab/plots.py` — +4 figures (rank frequencies,
  cost curves, level CIs, placebo rank distribution)
- `docs/research_report.md` — fix stale 0.876 → regenerated net figure; add
  Phase-2 sections; add "What this study does NOT establish"
- `README.md` — methodology item 7 (uncertainty suite), Holm-aware reading of
  the old unadjusted claim, rounding-slip fix, new results section,
  limitations (two Sharpe conventions, placebo circularity, multiplicity),
  section renumbering as needed

Regenerated (pipeline outputs, committed as artifacts):
- `reports/summary.md`, `reports/optimization_results.json`,
  `reports/oos_performance.csv`, `reports/in_vs_out_of_sample.csv` (vintage
  refresh at the 6th decimal, disclosed), `figures/*.png` (+4 new)

Untouched: `main` branch, tag `v1.0.0`, all other repositories, the release
metadata, `data/raw/*.csv` (gitignored, never committed).

## 9. Acceptance criteria and rollback

| ID | Criterion | Verification |
|----|-----------|--------------|
| A1 | This plan is committed to the branch BEFORE any implementation commit | git history order |
| A2 | Phase-2 bootstrap engine reproduces the v1.0.0 index stream exactly (same seed → identical resample indices → identical intervals on a fixed input) | `tests/test_uncertainty.py::test_engine_agreement_with_v1_engine` |
| A3 | (i) Data-independent engine test green; (ii) fresh-vintage baseline run reproduces committed headline net Sharpes to ≥3 dp, with any residual disclosed | vintage check log in the validation report |
| A4 | No unadjusted significance claim survives anywhere in docs; every pairwise claim is Holm-qualified | `verify_docs.py` + manual diff read |
| A5 | Level CIs for all 4 strategies × 3 block lengths present in artifacts; block sensitivity visible in summary | artifact inspection |
| A6 | Exact cost claims tested: affine mean/total net return; bitwise net==gross at 0 bps; turnover invariance; NO affine-Sharpe claim in any doc | unit tests + grep of docs |
| A7 | Placebo M decision rule followed exactly and disclosed (including pilot timing) | validation report |
| A8 | All new JSON artifacts parse with `allow_nan=False` strictness (no NaN/Infinity tokens) | strict-JSON tests |
| A9 | `scripts/verify_docs.py` exits 0: every artifact-derived README/report number matches its artifact | CI + local run |
| A10 | Full suite green (v1.0.0's 30 tests + new tests), `ruff check` clean, `main` and `v1.0.0` untouched (work on branch only) | pytest + git |

**Rollback:** all work lives on `phase2-uncertainty`. If any criterion fails
and cannot be fixed within scope, the branch is deleted; `main` and v1.0.0 are
never force-pushed, re-written, or re-tagged. Release discipline: no new tag
until the branch is pushed and CI is observed green; the next release (only
after explicit approval) would be v1.1.0 with `CITATION.cff` updated.

---

## 10. Amendment log

- **A-1 (registered at plan-writing, carried from the earlier session's
  pre-registration):** A3 is split into a *data-independent* engine-agreement
  test (runs in CI on synthetic input, no network/data dependency) plus a
  *disclosed* vintage comparison of the regenerated headline numbers.
- **A-2 (registered at plan-writing):** RQ3/A6 corrected from an earlier
  draft that claimed net Sharpe is affine in cost. It is NOT (the Sharpe
  denominator also changes). The exact registered claims are: affine
  total/mean net return, bitwise net==gross at 0 bps, and turnover
  invariance. No affine-Sharpe statement may appear in any document.
- No further amendments. Any later amendment must be appended here with a
  timestamp and rationale, and flagged in the validation report.
