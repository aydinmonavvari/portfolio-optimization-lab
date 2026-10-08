"""CLI: fetch and cache adjusted closes for the configured universe."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from portfolio_optimization_lab.config import DEFAULT_CONFIG


def download_one(symbol: str, start: str, end: str | None, retries: int = 3) -> pd.DataFrame:
    """Download adjusted closes for one symbol with simple retry logic."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                raise ValueError(f"empty frame for {symbol}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance multi-ticker layout
                close = close.iloc[:, 0]
            out = pd.DataFrame({"Date": close.index, symbol: close.to_numpy()})
            return out.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - deliberate broad retry
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed for {symbol}: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None, help="defaults to today")
    parser.add_argument("--refresh", action="store_true", help="re-download cached symbols")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    for symbol in (*cfg.tickers, cfg.benchmark):
        cache_path = Path(cfg.raw_dir) / f"{symbol}.csv"
        if cache_path.exists() and not args.refresh:
            print(f"{symbol}: cached at {cache_path}")
            continue
        print(f"{symbol}: downloading ...")
        frame = download_one(symbol, args.start, args.end)
        frame.to_csv(cache_path, index=False)
        print(f"     cached {len(frame)} rows -> {cache_path}")


if __name__ == "__main__":
    main()
