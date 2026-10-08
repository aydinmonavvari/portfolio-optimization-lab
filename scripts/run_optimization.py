"""CLI: run the full portfolio-optimization study."""

from __future__ import annotations

import logging

from portfolio_optimization_lab.config import DEFAULT_CONFIG
from portfolio_optimization_lab.pipeline import run_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("== portfolio-optimization-lab ==")
    results = run_pipeline(DEFAULT_CONFIG)
    print("\n=== in-sample vs out-of-sample Sharpe ===")
    print(results["comparison"].round(3).to_string())
    print("\n=== out-of-sample performance ===")
    print(results["performance"].round(3).to_string())


if __name__ == "__main__":
    main()
