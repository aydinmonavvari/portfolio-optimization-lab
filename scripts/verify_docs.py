"""Programmatic README/report vs artifact consistency checks (Phase 2, A9).

Every artifact-derived number quoted in ``README.md`` and
``docs/research_report.md`` is rebuilt here FROM THE ARTIFACT and checked for
presence in the (minus-normalized) document text. The script also enforces the
A4 honesty rule: no unadjusted significance claim and no affine-Sharpe claim
may appear in any document.

Exit code 0 = all checks pass; 1 = list of failures printed.

Run: ``python scripts/verify_docs.py`` (also enforced by
``tests/test_docs_match_artifacts.py`` in CI).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
README = PROJECT_ROOT / "README.md"
REPORT = PROJECT_ROOT / "docs" / "research_report.md"

STRATEGIES = ("equal_weight", "min_variance", "max_sharpe", "risk_parity")
PRETTY = {
    "equal_weight": "Equal weight",
    "min_variance": "Min variance",
    "max_sharpe": "Max Sharpe",
    "risk_parity": "Risk parity",
}

# Forbidden substrings (A4): unadjusted significance claims and the forbidden
# affine-Sharpe claim must not appear anywhere in the documents.
FORBIDDEN = [
    "Significant at 95%?",
    "significantly worse than risk parity",
    "significantly ahead of min-variance",
    "only strategy significantly",
    "is a significant gap",
    "Sharpe is affine",
    "Sharpe is exactly affine",
]


def _load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _load_csv(name: str) -> list[dict]:
    with (REPORTS / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _norm(text: str) -> str:
    """Normalize unicode minus so artifact-built ASCII strings can match."""
    return text.replace("\u2212", "-")


def build_checks() -> list[tuple[str, str, str]]:
    """Return (doc, description, expected_substring) triples built from artifacts."""
    checks: list[tuple[str, str, str]] = []

    def add(doc: str, description: str, expected: str) -> None:
        checks.append((doc, description, expected))

    opt = _load_json("optimization_results.json")
    unc = _load_json("uncertainty.json")
    plac = _load_json("placebo_simulation.json")
    perf = {r["strategy"]: r for r in _load_csv("oos_performance.csv")}
    comp = {r["strategy"]: r for r in _load_csv("in_vs_out_of_sample.csv")}
    costs = _load_csv("cost_sensitivity.csv")

    # ---- README §12 headline tables ----------------------------------------
    for s in STRATEGIES:
        p = perf[s]
        add("README", f"{s} OOS net Sharpe", f"{float(p['sharpe']):.3f}")
        add("README", f"{s} OOS gross Sharpe", f"{float(p['sharpe_gross']):.3f}")
        add("README", f"{s} avg turnover", f"{float(p['avg_turnover']):.3f}")
        add("README", f"{s} total costs", f"{float(p['total_costs_pct']):.2f}%")
        add("README", f"{s} ann return", f"{float(p['ann_return']) * 100:.1f}%")
        add("README", f"{s} ann vol", f"{float(p['ann_volatility']) * 100:.1f}%")
        c = comp[s]
        add("README", f"{s} in-sample Sharpe", f"{float(c['in_sample_sharpe']):.3f}")
        add(
            "README",
            f"{s} degradation",
            f"{float(c['sharpe_degradation']):+.3f}",
        )
        add("report", f"{s} OOS net Sharpe", f"{float(p['sharpe']):.3f}")
        add("report", f"{s} in-sample Sharpe", f"{float(c['in_sample_sharpe']):.3f}")

    # gross-Sharpe summary line in the report
    gross_line = " / ".join(f"{float(perf[s]['sharpe_gross']):.3f}" for s in STRATEGIES)
    add("report", "gross Sharpes line", gross_line)

    # ---- README pairwise Holm table -----------------------------------------
    pairs = unc["pairwise_holm"]["pairs"]
    for pr in pairs:
        a, b = PRETTY[pr["a"]], PRETTY[pr["b"]]
        add("README", f"{a}-{b} diff", f"{pr['diff']:+.3f}")
        add("README", f"{a}-{b} ci_low", f"{pr['ci_low']:+.3f}")
        add("README", f"{a}-{b} ci_high", f"{pr['ci_high']:+.3f}")
        add("README", f"{a}-{b} p_raw", f"{pr['p_value_add_one']:.3f}")
        add("README", f"{a}-{b} p_holm", f"{pr['p_holm_adjusted']:.3f}")
    n_rej = unc["pairwise_holm"]["n_reject_holm_05"]
    min_p = min(pr["p_holm_adjusted"] for pr in pairs)
    add("README", "holm verdict count", f"**{n_rej} of 6**")
    add("README", "holm min adjusted p", f"{min_p:.3f}")
    add("report", "holm verdict count", f"{n_rej} of 6 rejections.**")
    add("report", "holm min adjusted p", f"{min_p:.3f}")
    mv_rp = next(pr for pr in pairs if pr["a"] == "min_variance" and pr["b"] == "risk_parity")
    add("report", "MV-RP raw p", f"{mv_rp['p_value_add_one']:.3f}")

    # ---- README level CIs -----------------------------------------------------
    lv = unc["levels_and_ranks"]
    for s in STRATEGIES:
        add("README", f"{s} level point", f"{lv['point_estimates'][s]:+.3f}")
        for blk in ("21", "63", "126"):
            ci = lv["levels"][s][blk]
            add("README", f"{s} level CI b{blk} low", f"{ci['ci_low']:+.3f}")
            add("README", f"{s} level CI b{blk} high", f"{ci['ci_high']:+.3f}")
    for s in STRATEGIES:
        add("report", f"{s} level point", f"{lv['point_estimates'][s]:+.3f}")
        ci = lv["levels"][s]["63"]
        add("report", f"{s} level CI b63", f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
    for s in STRATEGIES:
        add("README", f"{s} rank-1 share", f"{lv['rank_frequencies'][s]['1']:.3f}")
    add("report", "rank-1 shares", " / ".join(
        f"{lv['rank_frequencies'][s]['1']:.3f}" for s in STRATEGIES
    ))
    mv_last = f"{lv['rank_frequencies']['min_variance']['4'] * 100:.0f}%"
    add("report", "min variance last share", mv_last)

    # ---- sub-window stability --------------------------------------------------
    sw = unc["subwindow_stability"]
    tau_str = ", ".join(f"{t:.3f}" for t in sw["adjacent_kendall_tau"])
    add("README", "adjacent taus", f"[{tau_str}]")
    add("README", "mean tau", f"{sw['mean_adjacent_kendall_tau']:.3f}")
    add("report", "adjacent taus", f"[{tau_str}]")
    for s, cnt in sw["rank1_counts"].items():
        add("README", f"subwindow rank1 {s}", str(cnt))

    # ---- cost grid ---------------------------------------------------------------
    by_sc: dict[str, dict[float, dict]] = {}
    for row in costs:
        by_sc.setdefault(row["strategy"], {})[float(row["cost_bps"])] = row
    for s in STRATEGIES:
        sharpe0 = float(by_sc[s][0.0]["sharpe_net_dailyconv"])
        add("report", f"{s} cost sharpe @0", f"{sharpe0:+.4f}")
    ms_loss = (
        float(by_sc["max_sharpe"][0.0]["sharpe_net_dailyconv"])
        - float(by_sc["max_sharpe"][100.0]["sharpe_net_dailyconv"])
    )
    ew_loss = (
        float(by_sc["equal_weight"][0.0]["sharpe_net_dailyconv"])
        - float(by_sc["equal_weight"][100.0]["sharpe_net_dailyconv"])
    )
    add("README", "max sharpe cost loss", f"\u2248{ms_loss:.2f}")
    add("README", "ew cost loss", f"\u2248{ew_loss:.3f}")
    add("report", "max sharpe cost loss", f"\u2248{ms_loss:.2f}")
    # logical check behind the "ordering never changes" claim
    order_ok = True
    for c in by_sc["equal_weight"]:
        seq = [float(by_sc[s][c]["sharpe_net_dailyconv"]) for s in STRATEGIES]
        if not (seq[0] > seq[3] > seq[2] > seq[1]):
            order_ok = False
    if not order_ok:
        add("README", "cost ordering claim (LOGICAL FAILURE)", "ORDERING CLAIM FALSE")

    # ---- half split ------------------------------------------------------------------
    hs = unc["half_split_exploratory"]
    for s in STRATEGIES:
        h1 = hs["per_strategy"][s]["half1"]["sharpe"]
        h2 = hs["per_strategy"][s]["half2"]["sharpe"]
        add("README", f"{s} half split", f"{h1:+.3f} \u2192 {h2:+.3f}")
    ew1 = hs["per_strategy"]["equal_weight"]
    add(
        "report",
        "EW half split",
        f"+{ew1['half1']['sharpe']:.3f} \u2192 +{ew1['half2']['sharpe']:.3f}",
    )

    # ---- placebo ------------------------------------------------------------------------
    summ = plac["summary"]
    assert plac["status"] == "complete", "placebo artifact must be complete"
    m = summ["M"]
    r1 = summ["oracle_rank_counts"]["1"]
    add("README", "oracle rank1 count", f"**{r1} of {m}**")
    add("README", "P_ew_beats_oracle", f"{summ['P_ew_beats_oracle'] * 100:.0f}%")
    for rank in ("2", "3", "4", "5"):
        add(
            "README",
            f"oracle rank {rank} count",
            f"rank {rank}: {summ['oracle_rank_counts'][rank]}",
        )
    b4 = summ["best_among_4_counts"]
    add("README", "best4 EW", f"equal weight in {b4.get('equal_weight', 0)}")
    add("README", "best4 MV", f"min variance in {b4.get('min_variance', 0)}")
    add("README", "best4 RP", f"risk parity in {b4.get('risk_parity', 0)}")
    add("README", "best4 MS", f"max Sharpe in {b4.get('max_sharpe', 0)}")
    pct = summ["real_diff_percentile_in_placebo"]
    add("README", "real diff percentile EW-MS", f"{pct['equal_weight__minus__max_sharpe']:.0f}th")
    add("README", "real diff percentile MV-RP", f"{pct['min_variance__minus__risk_parity']:.0f}th")
    add("report", "oracle rank1", f"{r1}/100" if m == 100 else f"{r1}/{m}")
    add("report", "P_ew_beats_oracle", f"{summ['P_ew_beats_oracle'] * 100:.0f}%")
    add("report", "EW-MS percentile", f"{pct['equal_weight__minus__max_sharpe']:.0f}th")
    add("report", "MV-RP percentile", f"{pct['min_variance__minus__risk_parity']:.0f}th")
    add("report", "rank5 count", f"rank 5 in {summ['oracle_rank_counts']['5']}")
    add(
        "README",
        "oracle rank5 in report-consistent count",
        f"rank 5: {summ['oracle_rank_counts']['5']}",
    )

    # ---- full-sample portfolios + shrinkage -----------------------------------------
    for name, key in (
        ("min variance", "min variance"),
        ("max Sharpe", "max sharpe"),
        ("risk parity", "risk parity"),
    ):
        st = opt["portfolios"][key]
        add("README", f"{name} full-sample sharpe", f"{st['sharpe']:.3f}")
        add("README", f"{name} full-sample vol", f"{st['volatility'] * 100:.1f}%")
        add("README", f"{name} full-sample return", f"{st['return'] * 100:.1f}%")
    shr = opt["shrinkage_intensity_full_sample"]
    add("README", "full-sample shrinkage", f"{shr:.3f}")
    wf = opt["shrinkage_intensity_walk_forward"]
    add("README", "wf shrinkage mean", f"{wf['mean']:.3f}")
    add("README", "wf shrinkage min", f"{wf['min']:.3f}")
    add("README", "wf shrinkage max", f"{wf['max']:.3f}")

    # ---- report abstract turnover/costs -------------------------------------------------
    ms = perf["max_sharpe"]
    add("report", "abstract turnover", f"{float(ms['avg_turnover']):.3f}")
    add("report", "abstract total costs", f"{float(ms['total_costs_pct']):.2f}%")

    return checks


def run_checks() -> list[str]:
    """Execute all checks; return a list of human-readable failures."""
    checks = build_checks()
    docs = {
        "README": _norm(README.read_text(encoding="utf-8")),
        "report": _norm(REPORT.read_text(encoding="utf-8")),
    }
    failures = []
    for doc, description, expected in checks:
        if expected not in docs[doc]:
            failures.append(
                f"[{doc}] {description}: expected substring not found: {expected!r}"
            )
    for doc_name, text in docs.items():
        for pattern in FORBIDDEN:
            if pattern.lower() in text.lower():
                failures.append(f"[{doc_name}] FORBIDDEN claim present: {pattern!r}")
    return failures


def main() -> int:
    failures = run_checks()
    total = len(build_checks())
    if failures:
        print(f"verify_docs: {len(failures)} FAILURE(S) of {total} checks:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"verify_docs: all {total} checks passed (README + research report vs artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
