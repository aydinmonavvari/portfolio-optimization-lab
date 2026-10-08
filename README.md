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
- **Bootstrap 95% CIs for pairwise Sharpe differences** (circular blocks).

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

**Pairwise Sharpe differences — paired circular-block bootstrap 95% CIs** (OOS daily
net returns; B = 2000, block = 63 days, seed 42):¹

| a − b | diff | 95% CI | Significant at 95%? |
| --- | --- | --- | --- |
| Equal weight − min variance | +0.268 | [−0.058, +0.610] | No |
| Equal weight − max Sharpe | +0.050 | [−0.294, +0.497] | No |
| Equal weight − risk parity | +0.023 | [−0.105, +0.133] | No |
| Min variance − max Sharpe | −0.218 | [−0.495, +0.080] | No |
| **Min variance − risk parity** | **−0.245** | **[−0.499, −0.034]** | **Yes** |
| Max Sharpe − risk parity | −0.027 | [−0.426, +0.253] | No |

¹ Sensitivity to the block length: the min-variance − risk-parity interval stays
negative at block 21 ([−0.503, −0.011]) and block 126 ([−0.509, −0.072]); the
equal-weight − min-variance interval turns (barely) positive only at block 126
([+0.017, +0.601]) — read conclusions across all three block lengths, stored in
`reports/optimization_results.json` under `sharpe_diff_bootstrap`.

Figures: [`efficient_frontier.png`](figures/efficient_frontier.png) (frontier, assets
and special portfolios), [`oos_cumulative.png`](figures/oos_cumulative.png) (growth of
$1, log scale, net of costs), [`oos_drawdowns.png`](figures/oos_drawdowns.png),
[`weights_history.png`](figures/weights_history.png) (weights at every rebalance).

## 13 · Interpretation

1. **The naive benchmark still finishes first — but not by a measurable margin.**
   Equal weight delivers the best OOS net Sharpe (1.005), replicating in this sample
   the central result of DeMiguel, Garlappi & Uppal (2009). Yet the block-bootstrap
   CIs say its edge over max-Sharpe (+0.050) and risk parity (+0.023) is NOT
   statistically distinguishable from zero; only risk parity vs min-variance is a
   significant gap.
2. **Fixing the optimizer changed the max-Sharpe story.** The previous release
   accepted a clipped closed-form tangency (suboptimal whenever clipping changed the
   solution). With the corrected solver — closed form only when feasible as-is,
   otherwise frontier scan + ternary refinement (used in all 32 real-data solves;
   the closed form was never cap-feasible) — max-Sharpe's gross OOS Sharpe rises
   0.876 → 0.971 and net of its 1.86% cost drag it delivers 0.957, close to its
   0.906 in-sample promise. This is a period effect (the OOS window favored equities),
   not evidence of robust skill: the estimation-error warning stands, but the old
   numbers understate what a correct optimizer delivers.
3. **Min-variance does its job but is not free.** It realizes the lowest OOS
   volatility (16.6%) exactly as designed, at the cost of the lowest return; its net
   Sharpe slips just below its in-sample value (0.695 → 0.691) — the covariances it
   relies on are estimable, but its Sharpe is the only one that degrades, and the
   bootstrap marks it significantly worse than risk parity.
4. **Risk parity is the robust middle.** OOS net Sharpe 0.972 with low turnover
   (0.089) and costs (0.27%), no reliance on the mean vector, and — per the bootstrap
   — the only strategy significantly ahead of min-variance across all block lengths.
5. **Costs matter at rebalancing frequency — now measurably.** Max-Sharpe's 0.619
   average one-way turnover per quarter costs 1.86% cumulatively (≈0.30%/yr including
   lost compounding), versus 0.10% for equal weight. That drag is the difference
   between gross 0.971 and net 0.957 — material relative to the (statistically
   insignificant) gaps between strategies.
6. **Statistical vs economic significance.** A CI excluding zero is a statement
   about sampling noise only. The one significant difference (risk parity over
   min-variance, ≈0.25 Sharpe) is also economically meaningful (lower vol, higher
   return, lower turnover); the insignificant equal-weight lead is achieved with the
   least turnover, so the defensible practical conclusion is “nothing beats 1/n
   convincingly, and cheaper strategies do not lose anything measurable”.

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

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) data (network, then cached)
python scripts/download_data.py

# 3) full study (~1 min; deterministic)
python scripts/run_optimization.py

# 4) verify: lint + 30 offline tests (incl. window-disjointness and an
#    optimality regression vs the exact capped tangency)
ruff check .
pytest -q
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
  with one-seventh of max-Sharpe's costs — and, per the bootstrap, significantly
  ahead of min-variance.

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
│   ├── plots.py          # frontier, cumulative growth, drawdowns, weights
│   └── pipeline.py       # orchestration + strict-JSON reports
├── tests/                # 30 offline tests incl. window-disjointness + optimality regression
├── notebooks/            # frontier & estimation-error walkthrough
├── scripts/              # download_data.py, run_optimization.py
├── configs/default.yaml
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/              # JSON + CSV results (generated, committed)
├── figures/              # 4 generated figures (committed)
├── docs/research_report.md
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
