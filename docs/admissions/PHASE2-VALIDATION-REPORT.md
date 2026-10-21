# PHASE 2 — Validation Report

**Branch:** `phase2-uncertainty` · **Plan:** `PHASE2-RESEARCH-PLAN.md` (committed
before implementation) · **Environment:** Python 3.12, Linux sandbox, run date
2026-10-21 (fresh data download, documented window 2018-01-01 → 2026-10-08,
2,203 aligned price rows → 2,202 return rows → 1,890 OOS days, 30 rebalances).

---

## 1. Provenance note (honest)

This is the third establishment of this phase in this sandbox. Two earlier
complete implementations were lost to sandbox resets before any push (they
were never published; no remote branch ever existed). The pre-registered plan
survived as a git bundle and was restored from
`bundles/portfolio-optimization-lab-phase2.bundle` (heads verified: `main` @
`9eb7b3d` = v1.0.0, `phase2-uncertainty` @ the plan commit). The implementation
recorded here was produced in a single session from the restored plan. Nothing
was pushed at the time of writing; `main` and the `v1.0.0` tag on GitHub are
untouched (verified via `git ls-remote`).

## 2. Vintage check (A3-ii)

Fresh `scripts/download_data.py --start 2018-01-01 --end 2026-10-08` (yfinance
end-exclusive → prices through 2026-10-07, exactly the documented v1.0.0
window). Full pipeline re-run and compared to the committed v1.0.0 artifacts:

| Strategy | Committed net Sharpe | Re-run | Residual | Verdict |
| --- | --- | --- | --- | --- |
| Equal weight | 1.005450 | 1.005451 | 6.7e-07 | PASS |
| Min variance | 0.690546 | 0.690548 | 1.7e-06 | PASS |
| Max Sharpe | 0.956938 | 0.956937 | 1.0e-06 | PASS |
| Risk parity | 0.972329 | 0.972329 | 8.6e-08 | PASS |

All four reproduce to ≥3 decimal places; residuals (≤1.7e-06) are the
registered provider-side vintage refresh at the 6th decimal, disclosed in the
committed `optimization_results.json` (102 changed digits, all at the 6th
decimal of per-window values).

## 3. Baseline and suite verification (A2, A3-i, A8, A10)

```
ruff check .                      → All checks passed!
python -m pytest                  → 47 passed  (30 v1.0.0 + 17 new)
python scripts/verify_docs.py     → all 170 checks passed
```

Key new regression: `test_uncertainty.py::test_engine_agreement_with_v1_engine`
(A2) — the Phase-2 pairwise suite reproduces the v1.0.0
`bootstrap_sharpe_differences` CIs **bitwise** (same per-pair seed-42 streams)
at B=80, block 14, sensitivities 7/28, on synthetic input, in CI.

## 4. Placebo decision rule (A7) — followed exactly

- Registered rule: pilot 2 worlds; if per-world wall time > 15 s run M = 50,
  else M = 100; hard cap 30 minutes of placebo compute.
- **Measured pilot (this session, real data, seeds 43/44): 8.1 s and 9.7 s**
  → ≤ 15 s → **M = 100**. (An earlier session measured 22.1 s/world on a
  slower sandbox and would have used M = 50; the rule is applied to the
  pilot actually measured in the executing session, as registered.)
- Total placebo compute: 100 worlds × ≈9 s ≈ 15 min < 30 min cap. ✓
- Execution mode: resumable foreground chunks (5 × ~20 worlds) through the
  scratch runner in §8 — the sandbox kills background processes when a tool
  call ends, exactly as the plan anticipated.

## 5. Headline results (all null-or-negative claims preserved)

- **Holm (6-pair family, FWER 0.05): 0 of 6 rejections.** Smallest adjusted
  p = 0.150 (min variance − risk parity; raw p = 0.025 — the v1.0.0
  unadjusted "significant" flag does not survive).
- **Level CIs (block 63):** every interval spans every other strategy's point
  estimate → the OOS ranking is not identified by this sample.
- **Rank stability:** mean adjacent Kendall τ = 0.000 across 7 sub-windows.
- **Cost grid:** ordering equal weight > risk parity > max Sharpe > min
  variance at every grid point on [0, 100] bps; max Sharpe loses ≈0.12 net
  Sharpe over the full range; registered exact claims unit-tested.
- **Half split (EXPLORATORY):** all strategies higher in H2 (period effect);
  no regime claim.
- **Placebo (M = 100):** the by-construction optimal oracle ranks first in
  **53/100** worlds (rank 2: 19, rank 3: 13, rank 4: 11, rank 5: 4);
  P(equal weight > oracle) = 0.28; real pairwise diffs at the 68th (EW−MS)
  and 80th (MV−RP) percentiles of the placebo distributions; MC SE on
  proportions ≈ 0.05.

## 6. Artifacts produced (committed)

- `reports/uncertainty.json` (levels, ranks, Holm family, sub-windows, half split)
- `reports/cost_sensitivity.csv` (6 costs × 4 strategies, 12-dp rows)
- `reports/rank_stability.csv` (per-window daily-convention Sharpes)
- `reports/placebo_records.jsonl` (100 per-world strict-JSON records)
- `reports/placebo_simulation.json` (status complete + summary)
- `reports/optimization_results.json`, `oos_performance.csv`,
  `in_vs_out_of_sample.csv`, `summary.md` (extended with Phase-2 sections)
- `figures/`: +4 new (`sharpe_level_cis.png`, `rank_frequencies.png`,
  `cost_sensitivity.png`, `placebo_rank_dist.png`), 4 regenerated
- `scripts/verify_docs.py` + `tests/test_docs_match_artifacts.py` (A9: 170
  artifact-derived doc checks, CI-enforced)
- README.md and docs/research_report.md reconciled to the artifacts
  (stale 0.876 fixed → 0.957 net; unadjusted significance flags replaced by
  the Holm family; rounding slip −0.027 → −0.026)

## 7. Known limitations of this run

- Data provider vintage: adjusted closes are provider-revised; results carry
  the disclosed ≤1.7e-06 drift vs committed v1.0.0 numbers.
- The placebo DGP is Gaussian i.i.d. by registration (evaluation-design noise
  only, not all real-world noise).
- The half split is exploratory and supports no regime claim.
- No multiple-testing protection beyond the registered Holm family.
- The 61-day trailing partial holding block remains excluded from the OOS
  sample (unchanged from v1.0.0).

## 8. Placebo runner (exact code used, kept outside the repo per plan §7)

Saved at `/tmp/p2scratch/run_placebo_chunk.py` in the executing sandbox; its
complete logic is reproduced here so the run is auditable and reproducible:

```python
"""Scratch runner (outside the repo per plan Section 7): resumable placebo chunks.

Usage: python run_placebo_chunk.py <start_world> <end_world_exclusive>
Appends one strict-JSON record per world to reports/placebo_records.jsonl.
Deterministic per world (seed 43 + i); safe to re-run: worlds already present
in the file are skipped.
"""
import json, sys, time
from pathlib import Path

REPO = Path("/path/to/local/portfolio-optimization-lab")  # set to your checkout root
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from portfolio_optimization_lab.config import OptConfig  # noqa: E402
from portfolio_optimization_lab.data import load_price_frame, log_returns  # noqa: E402
from portfolio_optimization_lab.estimates import estimate  # noqa: E402
from portfolio_optimization_lab.optimization import (  # noqa: E402
    max_sharpe, reset_solver_diagnostics, solver_diagnostics)
from portfolio_optimization_lab.uncertainty import (  # noqa: E402
    SEED_PLACEBO_BASE, placebo_generate_world, placebo_world)

RECORDS = REPO / "reports" / "placebo_records.jsonl"

def existing_worlds() -> set[int]:
    if not RECORDS.exists():
        return set()
    return {json.loads(l)["world"] for l in RECORDS.read_text().splitlines() if l.strip()}

def main() -> None:
    start, end = int(sys.argv[1]), int(sys.argv[2])
    cfg = OptConfig()
    cfg.ensure_dirs()
    reset_solver_diagnostics()
    prices = load_price_frame(cfg.tickers, cfg.benchmark, cfg.raw_dir)
    returns = log_returns(prices[list(cfg.tickers)])
    est = estimate(returns, cfg.periods_per_year)
    oracle = max_sharpe(est.mu, est.cov, cfg.rf_annual, cfg.max_weight)
    done = existing_worlds()
    for i in range(start, end):
        if i in done:
            print(f"world {i}: already present, skipped", flush=True)
            continue
        world = placebo_generate_world(
            SEED_PLACEBO_BASE + i,
            est.mu.to_numpy() / cfg.periods_per_year,
            est.cov.to_numpy() / cfg.periods_per_year,
            n_days=len(returns), columns=list(returns.columns))
        t0 = time.time()
        rec = placebo_world(
            i, world, oracle,
            estimation_window=cfg.estimation_window,
            holding_period=cfg.holding_period,
            max_weight=cfg.max_weight, cost_bps=cfg.cost_bps,
            periods_per_year=cfg.periods_per_year, rf_annual=cfg.rf_annual)
        rec["n_days_sim"] = int(len(returns))
        rec["shrinkage_full_sample"] = round(float(est.shrinkage), 6)
        elapsed = time.time() - t0
        line = json.dumps(rec, allow_nan=False)
        with RECORDS.open("a") as fh:
            fh.write(line + "\n")
        print(f"world {i}: {elapsed:.1f}s oracle_rank={rec['oracle_rank']} "
              f"best4={rec['best_among_4_strategies']}", flush=True)

if __name__ == "__main__":
    main()
```

Chunk invocations: `0 2` (pilot), `2 22`, `22 42`, `42 62`, `62 82`, `82 100`.
The final pipeline run loads `reports/placebo_records.jsonl` via
`load_placebo_records()` and writes `reports/placebo_simulation.json`.

## 9. Push instructions (when credentials are available)

All work lives on branch `phase2-uncertainty` (nothing pushed; `main` and
v1.0.0 untouched). From a machine with push rights to
`github.com/aydinmonavvari/portfolio-optimization-lab`:

```bash
git fetch origin && git checkout phase2-uncertainty
git push origin phase2-uncertainty
# observe CI (pytest incl. the 47-test suite + verify_docs battery) → must be green
# only then, and only with explicit approval, tag v1.1.0 and update CITATION.cff
```

Rollback (registered): delete the branch; never force-push or re-tag `main`.
