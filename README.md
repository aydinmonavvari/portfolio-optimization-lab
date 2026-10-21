# portfolio-optimization-lab

![CI](https://github.com/aydinmonavvari/portfolio-optimization-lab/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

Mean–variance and risk-parity portfolio construction under realistic constraints —
with Ledoit–Wolf covariance shrinkage, explicit transaction costs, and a strictly
chronological walk-forward evaluation that quantifies how much of the promised
in-sample performance survives out of sample.

*Educational research project. Not investment advice.*

## 2 · Research question

> How large is the gap between the in-sample promise of optimized portfolios and their
> out-of-sample delivery, and do sophisticated optimizers (minimum variance, maximum
> Sharpe, risk parity) actually beat a naive equal-weight benchmark once estimation
> error and transaction costs are accounted for?

## 3 · Motivation

Markowitz mean–variance optimization is elegant in theory and notoriously fragile in
practice: it consumes *estimated* expected returns and covariances as if they were known
constants, and its output weights are extremely sensitive to input noise. This project
builds the standard optimizer toolkit properly — shrinkage, constraints, costs — and
then does the part most tutorials skip: a leakage-free walk-forward evaluation that
measures what the optimizers actually deliver.

## 4 · Why this matters

- **For quantitative finance:** the estimation-error problem is the central practical
  flaw of classical portfolio theory; measuring it honestly is a rite of passage for
  anyone entering quantitative asset management.
- **For the portfolio narrative:** this project consumes the covariance analytics built
  in Project 01 and turns them into decisions, connecting descriptive statistics to
  optimization and backtesting.
- **For ML practice:** the in-sample/out-of-sample gap demonstrated here is the same
  phenomenon as overfitting in machine learning — the finance-native way to internalize
  why validation discipline matters.

## 5 · Methodology

1. **Estimation** — from daily log returns: annualized mean vector, sample covariance,
   and Ledoit–Wolf (2004) shrunk covariance (the shrinkage intensity is fitted on the
   data and reported).
2. **Portfolio construction** — all portfolios solved under long-only, full-investment
   (`Σw = 1`) and a 40% per-asset cap:
   - *minimum variance*: quadratic program (SLSQP), with a defensive
     clip-and-renormalize safety net whose use is counted and reported;
   - *maximum Sharpe*: closed-form tangency `Σ⁻¹(μ−rf)` accepted only when it is
     already long-only and cap-feasible **without clipping**; otherwise located on
     the constrained efficient frontier by a 25-point scan + ternary refinement
     (a clipped candidate is never returned — it is suboptimal whenever clipping
     changes the solution); the branch taken per solve is counted and reported;
   - *risk parity*: equal risk contributions via cyclical coordinate descent
     (Griveau-Billion, Richard & Roncalli 2013);
   - *equal weight*: the naive benchmark.
3. **Efficient frontier** — minimum-variance portfolios across a target-return grid
   under the same constraints (infeasible targets are reported as `null`).
4. **Walk-forward backtest** — every `holding_period = 63` trading days, μ and Σ are
   re-estimated on the trailing 252 days only; weights are built and held for the next
   63 days. Estimation and evaluation windows are disjoint *by construction* (verified
   by a dedicated unit test that intercepts the estimator and asserts window
   boundaries). The trailing partial 63-day block (61 trading days here) is not used
   out of sample.
5. **Costs** — 10 bps applied to one-way turnover at each rebalance and **deducted
   from that day's strategy return**; every headline metric (return, volatility,
   Sharpe, drawdowns, cumulative growth) is therefore net of costs, with gross
   metrics reported alongside for reference.
6. **Uncertainty** — paired circular-block bootstrap (Politis & Romano 1994) 95%
   percentile CIs for every pairwise Sharpe *difference* on OOS daily net returns:
   B = 2000, block = 63 days (the holding period), seed 42, with block-length
   sensitivity at 21 and 126 days. Blocks are used because daily OOS returns are
   serially dependent (fixed 63-day weights, volatility clustering), which invalidates
   an IID bootstrap.
7. **Uncertainty-aware evaluation (Phase 2, pre-registered)** — beyond pairwise
   differences: *level* CIs for each strategy's OOS Sharpe (daily convention),
   bootstrap rank frequencies, **Holm-corrected** inference across the 6 pairwise
   comparisons (family-wise α = 0.05), rank stability across 7 contiguous 252-day
   windows (Kendall τ), a transaction-cost sensitivity grid on the true net engine
   (0/10/25/50/75/100 bps), an exploratory first-vs-second-half split, and a
   100-world placebo simulation that scores a by-construction-optimal *oracle*
   tangency portfolio through the same walk-forward machinery to calibrate how much
   rank noise the evaluation design itself carries. Registered in
   `docs/admissions/PHASE2-RESEARCH-PLAN.md` BEFORE any implementation (seeds:
   bootstrap 42, placebo world i = 43 + i; the plan's exactness claims — affine mean
   net return in cost, bitwise net == gross at 0 bps, turnover invariance — are
   pinned by unit tests; net Sharpe is explicitly NOT claimed affine).

## 6 · Dataset

| Property | Value (actual run) |
| --- | --- |
| Universe | AAPL, MSFT, JNJ, JPM, XOM, PG, AMZN (benchmark: SPY for context) |
| Source | Yahoo Finance daily adjusted closes via `yfinance`, cached CSVs |
| Window | 2018-01-02 → 2026-10-07 (2,203 aligned trading days) |
| Returns | Daily log returns, annualized with a 252-day convention |

## 7 · Data sources

Yahoo Finance (`yfinance`) — delayed adjusted bars; cached locally under `data/raw/`
(git-ignored) so the study reruns offline. Usage terms respected; no credentials.

## 8 · Architecture

```
cached adjusted closes ──► validation ──► daily log returns
                                │
              estimate (trailing 252d only): mu, Σ_sample, Σ_LW
                                │
        ┌───────────────┬───────┴────────┬──────────────┐
   equal weight   min variance (QP)  max Sharpe     risk parity
  (benchmark)        SLSQP          (frontier scan)    (CCD)
        └───────────────┬───────┬────────┴──────────────┘
                        │  hold 63d out-of-sample  │
            turnover × 10 bps costs (deducted)
                        │
     in-sample vs OOS Sharpe comparison · frontier · bootstrap CIs · figures
```

## 9 · Experimental design

**Estimation-error isolation.** The same four strategies are evaluated twice: (i)
in-sample — weights optimized and scored on the full sample (the "promise"); (ii)
walk-forward out-of-sample — weights optimized on trailing windows and scored on
subsequent data net of costs (the "delivery"). The difference is the estimation-error
degradation this project measures.

**Leakage controls.** Estimation windows end exactly one trading day before their
holding period begins (unit-tested); no statistic ever crosses the boundary; costs are
charged AND deducted on realized turnover, not ignored. The trailing partial 63-day
holding block is excluded from the OOS sample rather than partially evaluated.

**Known limitations of the design** (stated up front): seven assets, one period, one
cost model; μ estimated as annualized mean log returns (no Black–Litterman or
shrinkage on the mean vector — a deliberate scope choice discussed in §14).

## 10 · Models

| Strategy | Solver | Constraints |
| --- | --- | --- |
| Equal weight | — (benchmark) | full investment |
| Minimum variance | SLSQP quadratic program | long-only, budget, 40% cap |
| Maximum Sharpe | closed-form tangency / frontier scan + ternary refinement | same |
| Risk parity | cyclical coordinate descent | long-only (cap-checked) |
| Efficient frontier | SLSQP at 30 target returns | same |

`configs/default.yaml` is loaded by a strict loader (unknown keys raise); the data
window itself is controlled by `scripts/download_data.py`.

## 11 · Evaluation metrics

- **Annualized return, volatility, Sharpe** — in-sample and OOS, **net of costs** as
  headline (`rf = 0`, disclosed); gross Sharpe/return reported for reference.
- **Sharpe degradation** = OOS net Sharpe − in-sample Sharpe per strategy.
- **Average turnover and total transaction costs** per strategy.
- **Ledoit–Wolf shrinkage intensity** (full sample and per walk-forward window).
- **Bootstrap 95% CIs for pairwise Sharpe differences** (circular blocks) with
  **Holm step-down adjustment** across the 6-pair family (the registered
  multiplicity-safe procedure; per-pair flags are unadjusted statements only).
- **Phase 2 additions:** per-strategy level CIs, bootstrap rank frequencies,
  adjacent-window Kendall τ, cost-grid sensitivity, exploratory half split,
  placebo oracle-rank distribution (all in `reports/uncertainty.json`,
  `reports/placebo_simulation.json`, `reports/cost_sensitivity.csv`,
  `reports/rank_stability.csv`).

## 12 · Results

Actual outputs of the committed run (`reports/`, `figures/`). All OOS numbers are
net of transaction costs (10 bps on one-way turnover, deducted from each
rebalance-day return); gross figures are shown for reference.

**In-sample vs out-of-sample Sharpe (30 quarterly rebalances, 1,890 OOS days):**

| Strategy | In-sample Sharpe | OOS Sharpe (net) | OOS Sharpe (gross) | Degradation (net − in-sample) |
| --- | --- | --- | --- | --- |
| Equal weight | 0.844 | **1.005** | 1.006 | +0.162 |
| Risk parity | 0.831 | 0.972 | 0.975 | +0.142 |
| Max Sharpe | **0.906** | 0.957 | 0.971 | +0.051 |
| Min variance | 0.695 | 0.691 | 0.696 | −0.004 |

**Out-of-sample performance (headline = net of 10 bps costs):**

| Strategy | Ann. return (net) | Ann. vol | Sharpe (net) | Sharpe (gross) | Avg turnover | Total costs |
| --- | --- | --- | --- | --- | --- | --- |
| Equal weight | 18.7% | 18.6% | **1.005** | 1.006 | 0.033 | 0.10% |
| Risk parity | 16.9% | 17.4% | 0.972 | 0.975 | 0.089 | 0.27% |
| Max Sharpe | 20.1% | 21.0% | 0.957 | 0.971 | 0.619 | 1.86% |
| Min variance | 11.5% | 16.6% | 0.691 | 0.696 | 0.219 | 0.66% |

**Full-sample frontier portfolios (shrunk covariance):** min variance 15.8% vol at
11.0% return; max Sharpe 0.906 at 18.5% return / 20.4% vol; risk parity 0.831 at
14.3% return / 17.2% vol. Full-sample shrinkage intensity 0.015 (walk-forward
windows: mean 0.072, range 0.031–0.158).

**Pairwise Sharpe differences — paired circular-block bootstrap 95% CIs, Holm-corrected**
(OOS daily net returns; B = 2000, block = 63 days, seed 42; two-sided add-one
bootstrap p-values; Holm step-down across the 6-pair family, FWER α = 0.05):¹

| a − b | diff | 95% CI | p (raw) | p (Holm-adj) | Reject at FWER 0.05? |
| --- | --- | --- | --- | --- | --- |
| Equal weight − min variance | +0.268 | [−0.058, +0.610] | 0.096 | 0.480 | No |
| Equal weight − max Sharpe | +0.050 | [−0.294, +0.497] | 0.812 | 1.000 | No |
| Equal weight − risk parity | +0.023 | [−0.105, +0.133] | 0.734 | 1.000 | No |
| Min variance − max Sharpe | −0.218 | [−0.495, +0.080] | 0.153 | 0.612 | No |
| Min variance − risk parity | −0.245 | [−0.499, −0.034] | 0.025 | 0.150 | **No** |
| Max Sharpe − risk parity | −0.026 | [−0.426, +0.253] | 0.873 | 1.000 | No |

¹ **Multiplicity correction changes the headline.** The v1.0.0 release flagged
min-variance − risk parity as "significant at 95%" — that was an UNADJUSTED
per-pair CI statement. Under the registered Holm family (6 comparisons),
**0 of 6** pairwise differences reject at family-wise α = 0.05 (the smallest
adjusted p is 0.150 for min-variance − risk parity). Block-length sensitivity
for that pair: the raw interval stays negative at block 21 ([−0.503, −0.011])
and block 126 ([−0.509, −0.072]), but raw-interval negativity is not a
multiplicity-safe claim and none is made.

### Phase 2 results: uncertainty-aware evaluation (pre-registered)

All statistics below are computed on the same OOS daily NET series; Sharpes in
this subsection use the **daily convention** `√252·mean/std`, which differs
numerically from the annualized-geometric convention in the tables above (both
are disclosed; see §14).

**Level CIs per strategy (B = 2000, block 63, seed 42)** — an interval that
spans another strategy's point estimate is evidence the ranking is not
identified by this sample (pre-registered reading rule):

| Strategy | Point | 95% CI (block 63) | 95% CI (block 21) | 95% CI (block 126) |
| --- | --- | --- | --- | --- |
| Equal weight | +0.922 | [+0.336, +1.723] | [+0.259, +1.730] | [+0.406, +1.703] |
| Min variance | +0.654 | [+0.074, +1.399] | [−0.011, +1.461] | [+0.199, +1.287] |
| Max Sharpe | +0.872 | [+0.334, +1.582] | [+0.273, +1.625] | [+0.429, +1.446] |
| Risk parity | +0.898 | [+0.309, +1.699] | [+0.232, +1.733] | [+0.396, +1.665] |

Every interval spans every other strategy's point estimate: the OOS ranking is
**not identified** by this sample.

**Bootstrap rank frequencies (block 63, pooled over B = 2000 resamples):**
rank-1 shares — equal weight 0.460, max Sharpe 0.381, risk parity 0.159,
min variance 0.000. No strategy ranks first in even half of the resamples;
even the naive benchmark fails to rank first 54% of the time, and min
variance is essentially never first.

**Rank stability across time (7 contiguous 252-day windows):** adjacent-window
Kendall τ = [0.667, 0.000, −1.000, −0.667, 1.000, 0.000]; mean τ = 0.000.
Rank-1 counts: equal weight 3/7, max Sharpe 3/7, risk parity 1/7, min variance
0/7. The cross-sectional ranking does NOT persist through time.

**Cost-assumption sensitivity (0/10/25/50/75/100 bps, true net engine):**
mean daily net return is exactly affine in cost with the registered slope
`−Σturnover/(10⁴·n)` per strategy (unit-tested); turnover paths are invariant
to cost (weights never depend on cost — unit-tested); at 0 bps the net series
equals the gross series bitwise (unit-tested). The ordering equal weight >
risk parity > max Sharpe > min variance (daily-convention net Sharpe) holds at
every grid point — no two curves cross anywhere on [0, 100] bps. Max Sharpe,
the highest-turnover strategy, loses ≈0.12 Sharpe across the full 100 bps
range (≈0.03 per 25 bps), roughly 6× risk parity's ≈0.02 and ~19× equal
weight's ≈0.006. Net Sharpe is NOT affine in cost and no such claim is made
anywhere.

**First vs second OOS half (EXPLORATORY — fragility check only):** every
strategy's daily-convention Sharpe is HIGHER in the second half (equal weight
+0.771 → +1.230; max Sharpe +0.810 → +1.054; risk parity +0.754 → +1.230;
min variance +0.529 → +0.933), with heavily overlapping CIs across halves.
Registered label: exploratory; this is a fragility check only and supports no
regime-robustness claim in either direction.

**Placebo simulation (100 worlds, seeds 43 + i):** synthetic i.i.d. Gaussian
worlds generated from the project's own full-sample Ledoit–Wolf estimates, in
which a constrained *oracle* tangency portfolio is optimal **by construction**
and is scored through the identical walk-forward machinery alongside the four
re-estimating strategies. Result: the oracle ranks first in only **53 of 100**
worlds (rank 2: 19, rank 3: 13, rank 4: 11, rank 5: 4); equal weight beats the
oracle in 28% of worlds; the best of the four re-estimating strategies is
equal weight in 46, min variance in 19, risk parity in 18, max Sharpe in 17
worlds. The real pairwise diffs fall at the 68th (EW − MS) and 80th (MV − RP)
percentiles of the placebo distributions of the same statistic — well inside
the noise a single 1,890-day OOS realization can produce. Monte Carlo SE on
proportions ≈ 0.05 at p = 0.5. Disclosed circularity: the DGP is Gaussian i.i.d.
by registration — it isolates evaluation-design rank noise, not all real-world
noise (fat tails, clustering, regimes are not reproduced).

Figures: [`efficient_frontier.png`](figures/efficient_frontier.png) (frontier, assets
and special portfolios), [`oos_cumulative.png`](figures/oos_cumulative.png) (growth of
$1, log scale, net of costs), [`oos_drawdowns.png`](figures/oos_drawdowns.png),
[`weights_history.png`](figures/weights_history.png) (weights at every rebalance),
plus the Phase-2 figures [`sharpe_level_cis.png`](figures/sharpe_level_cis.png),
[`rank_frequencies.png`](figures/rank_frequencies.png),
[`cost_sensitivity.png`](figures/cost_sensitivity.png),
[`placebo_rank_dist.png`](figures/placebo_rank_dist.png).

## 13 · Interpretation

1. **The naive benchmark still finishes first — and no gap is measurable.**
   Equal weight delivers the best OOS net Sharpe (1.005), replicating in this sample
   the central result of DeMiguel, Garlappi & Uppal (2009). Under the registered
   Holm family, **0 of 6** pairwise differences reject at family-wise α = 0.05 —
   including the min-variance vs risk-parity gap that the v1.0.0 release flagged
   as significant on an UNADJUSTED per-pair interval (raw p = 0.025, Holm-adjusted
   p = 0.150). The defensible reading is that this sample identifies no ranking.
2. **Fixing the optimizer changed the max-Sharpe story.** The previous release
   accepted a clipped closed-form tangency (suboptimal whenever clipping changed the
   solution). With the corrected solver — closed form only when feasible as-is,
   otherwise frontier scan + ternary refinement (used in all real-data solves) —
   max-Sharpe's gross OOS Sharpe rises 0.876 → 0.971 and net of its 1.86% cost drag
   it delivers 0.957, close to its 0.906 in-sample promise. This is a period effect
   (the OOS window favored equities — every strategy's Sharpe is higher in the
   second half), not evidence of robust skill; and its edge over the naive
   benchmark is not statistically distinguishable (Holm-adjusted p = 1.000).
3. **Min-variance does its job but is not free.** It realizes the lowest OOS
   volatility (16.6%) exactly as designed, at the cost of the lowest return; its
   net Sharpe slips just below its in-sample value (0.695 → 0.691). The v1.0.0
   release's unadjusted flag on the min-variance vs risk-parity gap (raw p =
   0.025) does not survive the registered Holm correction (adjusted p = 0.150);
   the Phase-2 level CIs and rank frequencies (min variance ranks first in ~0.1%
   of resamples, last in 94%) describe its position without a significance claim.
4. **Risk parity is the robust middle — described, not "significant".** OOS net
   Sharpe 0.972 with low turnover (0.089) and costs (0.27%) and no reliance on the
   mean vector. Under Holm it is not separable from any other strategy in this
   sample; the honest statement is about its cost/turnover profile and mean
   independence, not a significance ranking.
5. **Costs matter at rebalancing frequency — now measurably.** Max-Sharpe's 0.619
   average one-way turnover per quarter costs 1.86% cumulatively (≈0.30%/yr including
   lost compounding), versus 0.10% for equal weight. That drag is the difference
   between gross 0.971 and net 0.957 — material relative to the (Holm-insignificant)
   gaps between strategies, and visible across the whole cost grid: max Sharpe's
   net Sharpe falls ≈0.12 from 0 to 100 bps while equal weight loses ≈0.006, with
   no ordering change anywhere on the grid.
6. **Statistical significance, economic significance, and uncertainty are
   reported separately.** No pairwise claim survives multiplicity correction; the
   level CIs are as wide as the quoted gaps; the placebo shows a by-construction
   optimal portfolio ranks first in only 53% of synthetic realizations of this
   exact design. The practical conclusion stands but is now properly calibrated:
   nothing beats 1/n convincingly in this sample, cheaper strategies do not lose
   anything measurable, and single-realization rankings of this design carry
   enough rank noise to overturn even a genuinely optimal strategy.

## 14 · Limitations

- **Mean-vector estimation.** Annualized sample means have enormous standard errors
  even with 252 observations; no shrinkage or prior is applied to μ (only to Σ). This
  is the main reason max-Sharpe is fragile — stated, not hidden.
- **Seven-asset single-period study.** Results are sample-specific; no cross-sectional
  breadth, no multiple-asset-class panel, no regime conditioning.
- **Cost model.** 10 bps linear on turnover ignores market impact, spreads per asset,
  and taxes.
- **No risk-free asset in the opportunity set.** The tangency portfolio assumes
  borrowing/lending at `rf`; the cap constrains it further.
- **Bootstrap assumptions.** The circular-block bootstrap treats each OOS series as
  approximately stationary and captures dependence up to the block length (63 days,
  with 21/126-day sensitivity); longer-range dependence (multi-year regimes) is not
  captured, percentile CIs can undercover near boundaries, and conclusions are read
  across block lengths rather than from a single choice.
- **Trailing partial block.** The final 61 trading days (an incomplete 63-day holding
  block) are not used out of sample; the study window effectively ends 61 days before
  the data does.
- **CAPM-era assumptions.** No factor models (Black–Litterman, robust Bayes, HRP) —
  future work.
- **Log-return annualization** approximates arithmetic compounding.
- **Two Sharpe conventions coexist (disclosed).** The performance tables use the
  annualized geometric convention (exp-growth return / scaled vol); the Phase-2
  bootstrap statistics use the daily convention `√252·mean/std`. They differ
  numerically (e.g. equal weight 1.005 vs 0.922); each is used consistently within
  its own section and never mixed.
- **Overlapping CIs ≠ equality.** Non-overlapping CIs would imply a difference,
  but overlapping CIs do not imply equality; the registered formal check is the
  Holm-adjusted pairwise family, which rejects 0 of 6.
- **Placebo DGP circularity (registered).** The placebo's synthetic worlds are
  Gaussian i.i.d. draws from the project's own full-sample Ledoit–Wolf estimates:
  they deliberately do NOT reproduce fat tails, volatility clustering, or regime
  persistence. The resulting oracle-rank distribution calibrates **evaluation-design
  rank noise**, not all real-world noise; its outputs never mix with the real-data
  headline table.
- **Multiplicity.** No multiple-testing protection is claimed beyond the registered
  Holm family over the 6 pairwise Sharpe differences; exploratory outputs (the
  half split) are labeled as such and support no claims.

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) data (network, then cached)
python scripts/download_data.py

# 3) full study + uncertainty suite (~5 min; deterministic)
python scripts/run_optimization.py

# 3b) placebo (registered engine; resumable chunk-wise, ~9 s/world)
#     see docs/admissions/PHASE2-VALIDATION-REPORT.md §8 for the exact runner

# 4) verify: lint + 47 offline tests + doc/artifact consistency checks
ruff check .
pytest -q
python scripts/verify_docs.py
```

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/portfolio-optimization-lab.git
cd portfolio-optimization-lab
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 17 · Usage

```bash
python scripts/download_data.py      # fetch + cache prices
python scripts/run_optimization.py   # estimates, frontier, walk-forward backtest
pytest -q                            # offline test suite
```

As a library:

```python
from portfolio_optimization_lab.config import DEFAULT_CONFIG
from portfolio_optimization_lab.data import load_price_frame, log_returns
from portfolio_optimization_lab.optimization import risk_parity

prices = load_price_frame(DEFAULT_CONFIG.tickers, DEFAULT_CONFIG.benchmark, DEFAULT_CONFIG.raw_dir)
returns = log_returns(prices[list(DEFAULT_CONFIG.tickers)])
w_rp = risk_parity(returns.cov() * 252)
print(dict(zip(returns.columns, w_rp.round(3))))
```

## 18 · Example

Full-sample weights under the shrunk covariance (exact values from the committed
run, also stored in `optimization_results.json` → `weights_full_sample`):

| Ticker | Equal weight | Min variance | Max Sharpe | Risk parity |
| --- | --- | --- | --- | --- |
| AAPL | 14.29% | 0.00% | 33.07% | 10.93% |
| MSFT | 14.29% | 4.26% | 27.59% | 11.59% |
| JNJ | 14.29% | 35.51% | 24.86% | 20.07% |
| JPM | 14.29% | 3.60% | 8.22% | 12.36% |
| XOM | 14.29% | 12.79% | 6.26% | 14.04% |
| PG | 14.29% | 33.65% | 0.00% | 19.61% |
| AMZN | 14.29% | 10.19% | 0.00% | 11.40% |

Reading the risk-parity column (the worked example):

- the lowest-volatility defensive names (JNJ 20.1%, PG 19.6%) receive the largest
  capital weights;
- high-volatility names (AAPL 10.9%, AMZN 11.4%) receive the smallest weights
  despite strong returns — equalizing *risk contribution*, not capital;
- the resulting OOS net Sharpe (0.972) is within touching distance of equal weight
  with one-seventh of max-Sharpe's costs — a descriptive cost/robustness advantage;
  under the registered Holm family no pairwise gap in this study is statistically
  separable (see §12–§13).

## 19 · Project structure

```
portfolio-optimization-lab/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/portfolio_optimization_lab/
│   ├── config.py         # dataclass config + strict YAML loader (unknown keys raise)
│   ├── data.py           # cached price loading + validation
│   ├── estimates.py      # mu, sample/LW covariance, portfolio stats
│   ├── optimization.py   # min-var, max-Sharpe, risk parity, frontier + solver counters
│   ├── backtest.py       # walk-forward loop, net-of-cost deduction, in-sample comparison
│   ├── bootstrap.py      # circular block bootstrap for Sharpe differences
│   ├── uncertainty.py    # PHASE 2: level CIs, Holm family, ranks, cost grid, half split, placebo
│   ├── plots.py          # frontier, cumulative growth, drawdowns, weights + 4 Phase-2 figures
│   └── pipeline.py       # orchestration + strict-JSON reports
├── tests/                # 47 offline tests incl. engine-agreement (A2) + exact cost claims (A6)
├── notebooks/            # frontier & estimation-error walkthrough
├── scripts/              # download_data.py, run_optimization.py, verify_docs.py
├── configs/default.yaml
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/              # JSON + CSV results (generated, committed)
├── figures/              # 8 generated figures (committed)
├── docs/research_report.md
├── docs/admissions/      # PHASE2-RESEARCH-PLAN (pre-registered), VALIDATION-REPORT, summaries
└── .github/workflows/ci.yml
```

## 20 · Future work

- Mean-vector shrinkage (James–Stein / Black–Litterman) and its effect on max-Sharpe.
- Higher-frequency rebalancing with realistic impact cost models.
- Factor-model covariance (fundamental + statistical factors) vs shrunk sample Σ.
- Hierarchical risk parity (Lopez de Prado 2016) as a structurally different robustness
  benchmark.
- Multi-period optimization with turnover penalties.

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026portfoliooptimizationlab,
  author  = {Monavvari, Aydin},
  title   = {portfolio-optimization-lab: mean-variance and risk-parity optimization with leakage-free walk-forward evaluation},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/portfolio-optimization-lab}
}
```

## 22 · License

MIT — see [`LICENSE`](LICENSE).

## 23 · Acknowledgments

Market data via Yahoo Finance (`yfinance`). Built with pandas, NumPy, SciPy,
scikit-learn and matplotlib.
