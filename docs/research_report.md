# Research Report — Portfolio Optimization Lab

**Mean–variance and risk-parity portfolio optimization under estimation error: a leakage-free walk-forward evaluation**

*Author: Aydin Monavvari — educational research project (not investment advice).*
*Run date: 2026-10-08 (v1.0.0 baseline); Phase-2 uncertainty suite re-run 2026-10-21 on the same documented window. All numbers are actual outputs of the committed pipeline.*

---

## Abstract

Classical mean–variance optimization consumes estimated inputs as if they were known. This project quantifies the resulting promise-versus-delivery gap for four strategies — equal weight, minimum variance, maximum Sharpe and risk parity — on seven US large caps (2018–2026, 2,203 trading days), under long-only, budget and 40%-cap constraints, with Ledoit–Wolf covariance shrinkage and 10 bps transaction costs. A strictly chronological walk-forward design re-estimates inputs every 63 trading days on trailing 252-day windows and holds weights for the following quarter; a dedicated unit test verifies estimation and evaluation windows are disjoint. Out of sample (net of costs), the naive equal-weight benchmark achieves the best Sharpe ratio (1.005), ahead of risk parity (0.972), maximum Sharpe (0.957) and minimum variance (0.691). Maximum Sharpe, the best strategy in-sample (0.906), combines the highest turnover (0.619 per rebalance) with the largest cost drag (1.86% cumulative). A pre-registered uncertainty suite then shows that none of the cross-strategy gaps survives multiplicity correction (0 of 6 Holm-adjusted rejections), the per-strategy level confidence intervals are as wide as the quoted gaps, the ranking does not persist across sub-windows (mean adjacent Kendall τ = 0.000), and a by-construction-optimal oracle tangency portfolio ranks first in only 53 of 100 placebo worlds — calibrating how much rank noise a single out-of-sample realization of this design carries. The results replicate, in one sample, the central message of DeMiguel, Garlappi & Uppal (2009): sophisticated optimizers rarely beat 1/n once estimation error is accounted for — and, once evaluation noise is quantified, even the apparent gaps among the sophisticated strategies are not identified by this sample.

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

**Uncertainty-aware evaluation (Phase 2, pre-registered in `docs/admissions/PHASE2-RESEARCH-PLAN.md` before implementation).** On the same OOS daily net series: per-strategy level confidence intervals and bootstrap rank frequencies (circular blocks, B = 2000, blocks 21/63/126, seed 42); a Holm step-down family over the 6 pairwise Sharpe differences with two-sided add-one bootstrap p-values (FWER α = 0.05); adjacent-252-day-window rank stability (Kendall τ); a transaction-cost grid (0/10/25/50/75/100 bps) re-run through the verified net-of-cost engine with exact registered claims pinned by unit tests (mean net return affine in cost, bitwise net == gross at 0 bps, turnover invariance; net Sharpe is NOT affine and is never claimed to be); an exploratory chronological half split; and a 100-world placebo simulation (per-world seed 43 + i) in which a constrained oracle tangency portfolio — optimal for the simulated data-generating process by construction — is scored through the identical walk-forward machinery to measure the evaluation design's own rank noise.

**In-sample reference.** The same four strategies optimized and scored on the full sample produce the "promise" Sharpe ratios against which OOS delivery is compared.

## Experimental Design

The design isolates one factor — estimation error — by holding everything else fixed between the in-sample and walk-forward arms: same assets, same constraints, same solvers, same cost model. Differences between arms therefore reflect input noise and period composition, not methodology drift. The known confound (the OOS arm necessarily covers a *sequence* of windows rather than the full sample) is discussed in Limitations.

## Results

Actual outputs (`reports/optimization_results.json`, `reports/in_vs_out_of_sample.csv`, `reports/oos_performance.csv`).

| Strategy | In-sample Sharpe | OOS Sharpe (net) | Degradation | OOS return (net) | OOS vol | Avg turnover | Total costs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal weight | 0.844 | **1.005** | +0.162 | 18.7% | 18.6% | 0.033 | 0.10% |
| Min variance | 0.695 | 0.691 | −0.004 | 11.5% | 16.6% | 0.219 | 0.66% |
| Max Sharpe | **0.906** | 0.957 | +0.051 | 20.1% | 21.0% | 0.619 | 1.86% |
| Risk parity | 0.831 | 0.972 | +0.142 | 16.9% | 17.4% | 0.089 | 0.27% |

(30 rebalances; 1,890 out-of-sample days; OOS Sharpe net of costs; gross Sharpes 1.006 / 0.696 / 0.971 / 0.975 respectively.)

### Phase-2 uncertainty results (pre-registered)

Computed on the same OOS daily net series; Sharpes below use the daily convention √252·mean/std (disclosed; differs numerically from the geometric convention above).

- **Multiplicity (Holm, 6 pairwise differences, FWER α = 0.05): 0 of 6 rejections.** The smallest adjusted p is 0.150 (min variance − risk parity, raw p = 0.025). The v1.0.0 release's unadjusted "significant" flag on that pair does not survive its own registered correction.
- **Level CIs (B = 2000, block 63):** equal weight +0.922 [+0.336, +1.723]; min variance +0.654 [+0.074, +1.399]; max Sharpe +0.872 [+0.334, +1.582]; risk parity +0.898 [+0.309, +1.699]. Every interval spans every other strategy's point estimate — the ranking is not identified by this sample.
- **Rank frequencies (block 63):** rank-1 shares 0.460 / 0.000 / 0.381 / 0.159 (equal weight / min variance / max Sharpe / risk parity); min variance ranks last in 94% of resamples.
- **Sub-window stability (7 × 252 days):** adjacent Kendall τ = [0.667, 0.000, −1.000, −0.667, 1.000, 0.000], mean 0.000; rank-1 counts equal weight 3, max Sharpe 3, risk parity 1, min variance 0 of 7.
- **Cost grid (0–100 bps):** daily-convention net Sharpe at 0 bps: equal weight +0.9224, min variance +0.6587, max Sharpe +0.8834, risk parity +0.9002. The ordering equal weight > risk parity > max Sharpe > min variance holds at every grid point; max Sharpe loses ≈0.12 across the full range (≈19× equal weight's ≈0.006). Exact registered claims (affine mean net return, bitwise net == gross at 0 bps, turnover invariance) are unit-tested.
- **Half split (EXPLORATORY):** every strategy's daily-convention Sharpe is higher in the second half (e.g. equal weight +0.771 → +1.230) with heavily overlapping CIs — a fragility check, not a regime claim.
- **Placebo (M = 100, seeds 43 + i):** the by-construction optimal oracle tangency portfolio ranks first in 53/100 worlds (rank 5 in 4); equal weight beats the oracle in 28% of worlds; the real pairwise diffs sit at the 68th (EW − MS) and 80th (MV − RP) percentiles of the placebo distributions. Monte Carlo SE on proportions ≈ 0.05. Registered circularity: the Gaussian i.i.d. DGP isolates evaluation-design rank noise, not all real-world noise.

## Discussion

1. **1/n is hard to beat — and the gaps are not identified.** Equal weight wins out of sample, replicating DeMiguel et al. (2009) in this sample. Its zero-turnover profile also minimizes cost drag. Under the registered Holm family no pairwise difference is statistically separable, and the level CIs are as wide as the gaps themselves.
2. **The estimation-error asymmetry is visible — with a corrected solver.** Min-variance depends almost entirely on Σ (estimable) and its Sharpe is essentially unchanged in and out of sample (0.695 → 0.691). Max-Sharpe leans on μ (barely estimable); with the v1.0.0 constrained solver it delivers 0.957 net — close to its 0.906 in-sample promise — but its edge over the naive benchmark is not statistically distinguishable, and the second-half improvement of every strategy marks the period, not skill, as the driver.
3. **Turnover is a real cost of cleverness.** Max-Sharpe's mean one-way turnover of 0.619 per quarterly rebalance is 19× equal-weight's; cumulative costs (1.86%) exceed the OOS Sharpe gap between max-Sharpe and risk parity, and the cost grid shows max Sharpe losing ≈0.12 net Sharpe from 0 to 100 bps while the ordering of strategies never changes.
4. **Risk parity's mean-independence pays.** With near-benchmark OOS net Sharpe (0.972), low turnover and no reliance on μ, risk parity is the most robust optimizer in this study — described by its cost and turnover profile rather than by a significance claim, since no pairwise gap survives Holm correction.
5. **Evaluation noise is itself measurable — and large.** The placebo simulation ranks a by-construction optimal portfolio first in only 53% of synthetic realizations of this exact design. Any single-sample ranking of these strategies — including the headline table above — carries that much rank noise.
6. **Honest scope.** One bull-dominated window flatters all long-only strategies (note the OOS Sharpe *improvements* for equal weight and risk parity — a period effect, not skill). The structural contrasts (μ-dependence, turnover, stability) are the transferable findings.

## What this study does NOT establish

- **No strategy ranking.** 0 of 6 pairwise differences survive Holm correction; every level CI spans every other strategy's point estimate; sub-window rank persistence is zero (mean adjacent τ = 0.000).
- **No out-performance claim for any strategy, including vs 1/n.** The equal-weight first place is a point estimate whose sampling uncertainty contains every alternative.
- **No regime robustness.** The half split is registered exploratory and supports no claim in either direction.
- **No out-of-universe or out-of-period generalization.** One universe, one market, one bull-dominated decade, one cost model.
- **No real-time/vintage claims.** Results use contemporaneous revised adjusted closes; provider-side revisions are disclosed as a 6th-decimal vintage effect.
- **No affine-in-cost claim for net Sharpe.** Only mean/total net return is exactly affine (registered and unit-tested); the Sharpe denominator changes with cost.

## Limitations

- **Single sample, single universe, one cost model.** No cross-sectional breadth; no market-impact or spread differentiation.
- **Two Sharpe conventions coexist.** The geometric convention (performance tables) and the daily convention (Phase-2 statistics) differ numerically; each is used consistently within its own section.
- **Placebo DGP circularity (registered).** Synthetic worlds are Gaussian i.i.d. draws from the project's own full-sample Ledoit–Wolf estimates; they deliberately exclude fat tails, volatility clustering and regimes, so the placebo calibrates evaluation-design rank noise only.
- **Multiplicity.** No multiple-testing protection beyond the registered Holm family; the half split is exploratory.
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
