# Research Report — Portfolio Optimization Lab

**Mean–variance and risk-parity portfolio optimization under estimation error: a leakage-free walk-forward evaluation**

*Author: Aydin Monavvari — educational research project (not investment advice).*
*Run date: 2026-10-08. All numbers are actual outputs of the committed pipeline.*

---

## Abstract

Classical mean–variance optimization consumes estimated inputs as if they were known. This project quantifies the resulting promise-versus-delivery gap for four strategies — equal weight, minimum variance, maximum Sharpe and risk parity — on seven US large caps (2018–2026, 2,203 trading days), under long-only, budget and 40%-cap constraints, with Ledoit–Wolf covariance shrinkage and 10 bps transaction costs. A strictly chronological walk-forward design re-estimates inputs every 63 trading days on trailing 252-day windows and holds weights for the following quarter; a dedicated unit test verifies estimation and evaluation windows are disjoint. Out of sample, the naive equal-weight benchmark achieves the best Sharpe ratio (1.006), ahead of risk parity (0.975), maximum Sharpe (0.876) and minimum variance (0.696). Maximum Sharpe, the best strategy in-sample (0.906), combines the highest turnover (0.62 per rebalance) with the largest cost drag (1.85% cumulative), and its in-sample edge proves to be an estimation artifact. The results replicate, in one sample, the central message of DeMiguel, Garlappi & Uppal (2009): sophisticated optimizers rarely beat 1/n once estimation error is accounted for, while mean-independent strategies (risk parity, min-variance) degrade least.

## Introduction

Markowitz (1952) framed portfolio choice as trading expected return against covariance — an optimization over two population objects that must, in practice, be estimated from finite data. The estimation error is asymmetric: covariance matrices are estimable with reasonable precision, while expected returns are nearly invisible at monthly-and-shorter horizons. The consequences are well documented — weight instability, error maximization (Michaud 1989), and out-of-sample performance that routinely fails to match the in-sample promise — yet most expositions stop before measuring that failure. This project implements the full pipeline (estimation, four optimizers, constraints, costs, frontier) and closes the loop with a walk-forward evaluation designed so that no future information can reach the optimizer.

The repository is the fourth step of a research portfolio progressing from descriptive analytics through forecasting and ML; it converts the covariance analytics of Project 01 into allocation decisions and backtests.

## Research Question

> How large is the in-sample versus out-of-sample performance gap of constrained mean–variance and risk-parity portfolios, and do they beat a naive equal-weight benchmark net of transaction costs?

## Related Work

- **Markowitz, H. (1952).** "Portfolio Selection." *Journal of Finance*, 7(1), 77–91. Mean–variance framework.
- **Ledoit, O., & Wolf, M. (2004).** "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365–411. Covariance shrinkage used here.
- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** "Optimal versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies*, 22(5), 1915–1953. The 1/n benchmark result this study replicates in one sample.
- **Michaud, R. O. (1989).** "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" *Financial Analysts Journal*, 45(1), 31–42. Error maximization by mean–variance optimizers.
- **Spinu, F. (2013).** "An Algorithm for Computing Risk Parity Weights." *SSRN 2297383*. Convex risk-parity formulation.
- **Griveau-Billion, T., Richard, J.-C., & Roncalli, T. (2013).** "A fast algorithm for computing High-dimensional risk parity portfolios." *SSRN 2325255*. Cyclical coordinate descent solver used here.
- **Cornuejols, G., & Tütüncü, R. (2006).** *Optimization Methods in Finance.* Cambridge University Press. Convex reformulations for fractional programs.

## Data

| Property | Value (actual run) |
| --- | --- |
| Universe | AAPL, MSFT, JNJ, JPM, XOM, PG, AMZN; SPY retained for context only |
| Prices | Yahoo Finance daily adjusted closes, 2018-01-02 → 2026-10-07 |
| Sample | 2,203 aligned trading days; daily log returns; 252-day annualization |
| Assumptions | `rf = 0.0` (disclosed, configurable); 10 bps per unit turnover |

## Methodology

**Estimation.** Per window: annualized mean vector, sample covariance, and Ledoit–Wolf shrunk covariance (intensity fitted on data; shrunk matrix used throughout).

**Optimizers.** (i) Minimum variance: SLSQP on the scaled quadratic program. (ii) Maximum Sharpe: the closed-form tangency `Σ⁻¹(μ−rf)` when constraint-inactive; otherwise the tangency point is located on the constrained frontier by a 25-point scan plus ternary refinement — a deliberately robust alternative to direct fractional-program SLSQP, which stalls on this scale. (iii) Risk parity: cyclical coordinate descent with inverse-vol initialization; equality of risk contributions is asserted numerically. (iv) Equal weight: benchmark.

**Walk-forward design.** Rebalance every 63 trading days; estimate on the trailing 252; hold for the next 63; charge 10 bps on one-way turnover. A spy-based unit test asserts each estimation window ends exactly one trading day before its holding period begins.

**In-sample reference.** The same four strategies optimized and scored on the full sample produce the "promise" Sharpe ratios against which OOS delivery is compared.

## Experimental Design

The design isolates one factor — estimation error — by holding everything else fixed between the in-sample and walk-forward arms: same assets, same constraints, same solvers, same cost model. Differences between arms therefore reflect input noise and period composition, not methodology drift. The known confound (the OOS arm necessarily covers a *sequence* of windows rather than the full sample) is discussed in Limitations.

## Results

Actual outputs (`reports/optimization_results.json`, `reports/in_vs_out_of_sample.csv`, `reports/oos_performance.csv`).

| Strategy | In-sample Sharpe | OOS Sharpe | Degradation | OOS return | OOS vol | Avg turnover | Total costs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal weight | 0.844 | **1.006** | +0.162 | 18.7% | 18.6% | 0.033 | 0.10% |
| Min variance | 0.695 | 0.696 | +0.001 | 11.6% | 16.6% | 0.219 | 0.66% |
| Max Sharpe | **0.906** | 0.876 | −0.030 | 18.9% | 21.6% | 0.616 | 1.85% |
| Risk parity | 0.831 | 0.975 | +0.144 | 16.9% | 17.4% | 0.089 | 0.27% |

(30 rebalances; 1,890 out-of-sample days; OOS Sharpe net of costs.)

## Discussion

1. **1/n is hard to beat.** Equal weight wins out of sample, replicating DeMiguel et al. (2009) in this sample. Its zero-turnover profile also minimizes cost drag.
2. **The estimation-error asymmetry is visible in the rankings.** Min-variance depends almost entirely on Σ (estimable) and its Sharpe is essentially unchanged in and out of sample (0.695 → 0.696); max-Sharpe leans on μ (barely estimable) and its in-sample advantage evaporates (0.906 → 0.876, behind the naive benchmark).
3. **Turnover is a real cost of cleverness.** Max-Sharpe's mean one-way turnover of 0.616 per quarterly rebalance is 19× equal-weight's; cumulative costs (1.85%) exceed the OOS Sharpe gap between max-Sharpe and risk parity.
4. **Risk parity's mean-independence pays.** With near-benchmark OOS Sharpe (0.975), low turnover and no reliance on μ, risk parity is the most robust optimizer in this study — consistent with its adoption in practice.
5. **Honest scope.** One bull-dominated window flatters all long-only strategies (note the OOS Sharpe *improvements* for equal weight and risk parity — a period effect, not skill). The structural contrasts (μ-dependence, turnover, stability) are the transferable findings.

## Limitations

- **Single sample, single universe, one cost model.** No cross-sectional breadth; no market-impact or spread differentiation.
- **No μ shrinkage.** Sample means with 252 observations have standard errors of the same order as the means themselves; Black–Litterman or James–Stein priors are future work, which caps the realism of the max-Sharpe arm.
- **rf = 0 simplification.** Sharpe levels shift with the risk-free assumption; rankings are more robust than levels.
- **Quarterly rebalancing is exogenous.** No optimization of the rebalancing frequency against costs.
- **No factor or hierarchical models** (Black–Litterman, HRP) in the comparison set.

## Conclusion

Under constraints, shrinkage and costs, the study finds the textbook pattern in one real sample: optimizers that consume expected returns (max-Sharpe) deliver less than they promise, mean-independent optimizers (min-variance, risk parity) deliver approximately what they promise, and the naive equal-weight benchmark remains the out-of-sample frontier. The contribution is a fully tested, leakage-audited optimization harness whose walk-forward evaluation can be re-pointed at any universe.

## Future Research

- μ-shrinkage (James–Stein, Black–Litterman) and robust optimization (worst-case ellipsoids).
- Hierarchical risk parity and factor-covariance models as additional arms.
- Rebalancing-frequency optimization under realistic impact costs.
- Multi-asset-class universes with regime conditioning.

## References

- Cornuejols, G., & Tütüncü, R. (2006). *Optimization Methods in Finance.* Cambridge University Press.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Griveau-Billion, T., Richard, J.-C., & Roncalli, T. (2013). A fast algorithm for computing high-dimensional risk parity portfolios. *SSRN 2325255*.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Michaud, R. O. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal*, 45(1), 31–42.
- Spinu, F. (2013). An algorithm for computing risk parity weights. *SSRN 2297383*.
