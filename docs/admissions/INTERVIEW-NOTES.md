# Interview Notes — PHASE 2 (portfolio-optimization-lab)

**Purpose:** honest, concise talking points for admissions interviews about the
pre-registered uncertainty-aware evaluation phase. No claims beyond what the
artifacts support.

---

## The 60-second version

- v1.0.0 asked "how do strategies deliver out of sample, net of costs?"
  Phase 2 asks "how much of that answer is real?" — and I pre-registered the
  entire methodology before implementing it.
- Headline finding is null: after Holm correction, no strategy pair is
  statistically separable; a portfolio that is optimal by construction ranks
  first in only 53% of placebo worlds. I kept and highlighted the null.

## Likely questions and honest answers

**Q: Why pre-register for a solo BSc project?**
A: Because the biggest risk in a one-author project is fooling myself. The
plan (estimands, seeds, block lengths, decision rules, acceptance criteria,
rollback) was committed before any implementation commit — the git history
proves the order. It also forced me to fix analysis choices I would otherwise
have tuned after seeing results.

**Q: What exactly does the placebo show?**
A: In synthetic worlds where I *know* the optimal portfolio (the constrained
tangency for the data-generating process), running it through the same
walk-forward machinery, it ranks first in only 53 of 100 worlds. That
calibrates the rank noise of the evaluation design itself: one 1,890-day OOS
realization cannot reliably identify even a genuinely optimal strategy.

**Q: Is the Gaussian i.i.d. DGP not circular?**
A: Deliberately, and registered as such. Its purpose is to isolate
*evaluation-design* noise in a known-parameter world, not to mimic markets.
It never mixes with the real-data headline numbers.

**Q: What changed vs your v1.0.0 release?**
A: Two things, both documented in place. (1) The v1.0.0 "significant"
min-variance vs risk-parity flag was an unadjusted per-pair interval statement;
under the registered Holm family it does not survive (raw p 0.025 → adjusted
0.150). (2) A stale 0.876 number from the pre-fix optimizer survived in the
report abstract; the committed artifacts say 0.957 net. Both are now guarded
by 170 CI-enforced doc-versus-artifact checks.

**Q: What did you verify in the source rather than assume?**
A: The cost model (single deduction on the rebalance day, gross series kept
separately — so "net of costs" is a true claim), the constrained tangency
solver (closed form only if feasible as-is, else frontier scan + ternary
search; clipping never returned), and the strict-JSON serialization
(allow_nan=False).

**Q: Weaknesses?**
A: One universe, one decade, seven assets, one linear cost model; the placebo
DGP excludes fat tails and clustering; the half split is exploratory. The
level CIs are wide — that is the finding, not a bug.

**Q: What next?**
A: Bootstrap-t (studentized) intervals and the Ledoit–Wolf (2008) Sharpe-equality
test as a cross-check of my bootstrap p-values; a second universe; heavier-tailed
placebo DGPs (multivariate t / Gaussian copula with GARCH margins).

## One-line takeaway

"Be suspicious of any ranking produced by a single backtest — including mine;
I built the machinery that quantifies exactly how suspicious to be."
