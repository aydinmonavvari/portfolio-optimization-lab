"""Data loading and validation for the optimization study."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_price_frame(
    tickers: tuple[str, ...],
    benchmark: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """Load cached per-ticker CSVs and return an aligned adjusted-close frame.

    Each ``<TICKER>.csv`` in ``raw_dir`` must have the two-column layout
    written by ``scripts/download_data.py``: a header row ``Date,<TICKER>``
    with ISO dates in the first column and adjusted closes in the second (the
    loader reads the first column as the date index and the LAST column as the
    value column). The function returns a DataFrame indexed by date with one
    column per ticker + the benchmark.

    Raises
    ------
    FileNotFoundError
        If any requested ticker has no cached file.
    ValueError
        If the frame fails validation (negative prices, duplicate dates).
    """
    frames: dict[str, pd.Series] = {}
    for symbol in (*tickers, benchmark):
        path = Path(raw_dir) / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing cached prices for {symbol}: expected {path}. "
                "Run scripts/download_data.py first."
            )
        df = pd.read_csv(path, parse_dates=[0])
        value_col = df.columns[-1]
        series = df.set_index(df.columns[0])[value_col].astype(float)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.name = symbol
        frames[symbol] = series
    prices = pd.DataFrame(frames).dropna(how="any")
    _validate_prices(prices)
    logger.info(
        "Loaded %d symbols, %d aligned trading days (%s -> %s)",
        prices.shape[1],
        len(prices),
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return prices


def _validate_prices(prices: pd.DataFrame) -> None:
    if (prices <= 0).any().any():
        raise ValueError("non-positive prices found in the price frame")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be sorted")
    if prices.isna().any().any():
        raise ValueError("aligned price frame must not contain missing values")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of a price frame."""
    return np.log(prices).diff().dropna(how="any")
