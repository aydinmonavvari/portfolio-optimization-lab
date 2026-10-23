# Professor Summary — PHASE 2

**Project:** portfolio-optimization-lab · **Phase:** uncertainty-aware
evaluation of portfolio strategies (pre-registered) · ~250 words

---

My v1.0.0 project walked forward four portfolio strategies (equal weight,
min-variance, max-Sharpe, risk parity) over seven US large caps, 2018–2026,
net of transaction costs. Equal weight finished first (net Sharpe 1.005). For
Phase 2 I pre-registered — committed before implementation — an evaluation
methodology study asking a different question: *how much of the observed
cross-strategy performance structure survives uncertainty quantification?*

Five registered questions with estimands fixed in advance (RQ1–RQ5): level
uncertainty of each OOS Sharpe; rank stability under resampling and across
time; sensitivity to the transaction-cost assumption; calendar-time fragility;
and a placebo simulation separating "the strategy is good" from "the
evaluation window favored it". Bootstrap machinery reproduces the v1.0.0
engine's index streams bitwise (regression-tested); the engine's cost model
and constrained tangency solver were verified line-by-line, and the plan's
exact claims (mean net return affine in cost; net equals gross bitwise at 0
bps; turnover invariant to cost) are unit-tested. Net Sharpe is explicitly
*not* claimed affine.

Findings are mostly null, and I report them as such: 0 of 6 pairwise Sharpe
differences survive Holm correction (the smallest adjusted p is 0.150); every
level CI spans every other strategy's point estimate; the ranking does not
persist across sub-windows (mean adjacent Kendall τ = 0.000); and a
by-construction-optimal oracle tangency portfolio ranks first in only 53 of
100 placebo worlds — equal weight beats it in 28%. A single 1,890-day
out-of-sample realization of this design carries enough rank noise to
overturn even a genuinely optimal strategy.

Everything is reproducible: 73 offline test cases (59 test functions),
170 CI-enforced documentation-versus-artifact checks, seeds fixed in the plan.
