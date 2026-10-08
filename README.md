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
   - *minimum variance*: quadratic program (SLSQP);
   - *maximum Sharpe*: closed-form tangency `Σ⁻¹(μ−rf)` when constraint-inactive,
     otherwise located on the constrained efficient frontier by dense scan + ternary
     refinement (the fractional program is avoided deliberately — see §19);
   - *risk parity*: equal risk contributions via cyclical coordinate descent
     (Griveau-Billion, Richard & Roncalli 2013);
   - *equal weight*: the naive benchmark.
3. **Efficient frontier** — minimum-variance portfolios across a target-return grid
   under the same constraints.
4. **Walk-forward backtest** — every `holding_period = 63` trading days, μ and Σ are
   re-estimated on the trailing 252 days only; weights are built and held for the next
   63 days. Estimation and evaluation windows are disjoint *by construction* (verified
   by a dedicated unit test that intercepts the estimator and asserts window
   boundaries).
5. **Costs** — 10 bps applied to one-way turnover at each rebalance.

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
                turnover × 10 bps costs
                        │
     in-sample vs OOS Sharpe comparison · frontier · figures
```

## 9 · Experimental design

**Estimation-error isolation.** The same four strategies are evaluated twice: (i)
in-sample — weights optimized and scored on the full sample (the "promise"); (ii)
walk-forward out-of-sample — weights optimized on trailing windows and scored on
subsequent data net of costs (the "delivery"). The difference is the estimation-error
degradation this project measures.

**Leakage controls.** Estimation windows end exactly one trading day before their
holding period begins (unit-tested); no statistic ever crosses the boundary; costs are
charged on realized turnover, not ignored.

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

## 11 · Evaluation metrics

- **Annualized return, volatility, Sharpe** (in-sample and OOS; `rf = 0`, disclosed).
- **Sharpe degradation** = OOS Sharpe − in-sample Sharpe per strategy.
- **Average turnover and total transaction costs** per strategy.
- **Ledoit–Wolf shrinkage intensity** (fitted per estimation window).

## 12 · Results

Actual outputs of the committed run (`reports/`, `figures/`).

**In-sample vs out-of-sample Sharpe:**

| Strategy | In-sample Sharpe | OOS Sharpe | Degradation |
| --- | --- | --- | --- |
| Equal weight | 0.844 | **1.006** | +0.162 |
| Min variance | 0.695 | 0.696 | +0.001 |
| Max Sharpe | **0.906** | 0.876 | −0.030 |
| Risk parity | 0.831 | 0.975 | +0.144 |

**Out-of-sample performance (net of 10 bps costs, 30 quarterly rebalances, 1,890 OOS days):**

| Strategy | Ann. return | Ann. vol | Sharpe | Avg turnover | Total costs |
| --- | --- | --- | --- | --- | --- |
| Equal weight | 18.7% | 18.6% | **1.006** | 0.033 | 0.10% |
| Risk parity | 16.9% | 17.4% | 0.975 | 0.089 | 0.27% |
| Max Sharpe | 18.9% | 21.6% | 0.876 | 0.616 | 1.85% |
| Min variance | 11.6% | 16.6% | 0.696 | 0.219 | 0.66% |

**Full-sample frontier portfolios (shrunk covariance):** min variance 15.8% vol at
11.0% return; max Sharpe 0.906 at 18.8% return / 20.7% vol; risk parity 0.831 at
14.3% return / 17.2% vol.

Figures: [`efficient_frontier.png`](figures/efficient_frontier.png) (frontier, assets
and special portfolios), [`oos_cumulative.png`](figures/oos_cumulative.png) (growth of
$1, log scale), [`oos_drawdowns.png`](figures/oos_drawdowns.png),
[`weights_history.png`](figures/weights_history.png) (weights at every rebalance).

## 13 · Interpretation

1. **The naive benchmark wins out of sample.** Equal weight delivers the best OOS
   Sharpe (1.006) despite being the "non-method". This replicates, in this sample, the
   central result of DeMiguel, Garlappi & Uppal (2009): optimized portfolios rarely beat
   1/n once estimation error is priced in.
2. **Max-Sharpe shows the classic pattern — mildly here.** It *looks* best in-sample
   (0.906) and falls behind equal-weight out of sample (0.876), with by far the highest
   turnover (0.62 average per rebalance) and 1.85% total costs. Its in-sample advantage
   was largely an artifact of estimated inputs; the degradation is smaller than the
   literature's typical finding because this particular window favored equities
   broadly.
3. **Min-variance does its job but is not free.** It realizes the lowest OOS volatility
   (16.6%) exactly as designed, at the cost of the lowest return; its Sharpe is stable
   (0.695 → 0.696) because it depends mostly on the covariance (estimable) rather than
   the mean (barely estimable) — a structurally meaningful contrast with max-Sharpe,
   which leans on μ.
4. **Risk parity is the robust middle.** Near-benchmark OOS Sharpe (0.975) with low
   turnover (0.089) and costs (0.27%), and it does not require the mean vector at all —
   consistent with its appeal under estimation error.
5. **Costs matter at rebalancing frequency.** Max-Sharpe's 0.616 average one-way
   turnover per quarter translates into 1.85% cumulative costs over the study — a
   material drag relative to the spread between strategies.

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

# 4) verify: lint + 9 offline unit tests (incl. window-disjointness)
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

Actual risk-parity weights implied by the full-sample shrunk covariance (values from
the committed run):

- the lowest-volatility defensive names (PG, JNJ) receive the largest risk budgets;
- high-volatility names (AMZN, AAPL) receive the smallest weights despite strong
  returns — equalizing *risk contribution*, not capital;
- the resulting OOS Sharpe (0.975) is within touching distance of equal weight with
  one-third of max-Sharpe's costs.

## 19 · Project structure

```
portfolio-optimization-lab/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/portfolio_optimization_lab/
│   ├── config.py         # dataclass config (windows, cap, costs, seed)
│   ├── data.py           # cached price loading + validation
│   ├── estimates.py      # mu, sample/LW covariance, portfolio stats
│   ├── optimization.py   # min-var, max-Sharpe, risk parity, frontier
│   ├── backtest.py       # walk-forward loop, costs, in-sample comparison
│   ├── plots.py          # frontier, cumulative growth, drawdowns, weights
│   └── pipeline.py       # orchestration + reports
├── tests/                # 9 offline tests incl. window-disjointness
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
