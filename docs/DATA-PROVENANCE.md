# Data Provenance — `data/raw/*.csv`

Integrity manifest for the raw input dataset. This manifest is committed; the
CSVs themselves are intentionally **not** in git (`data/raw/*` is git-ignored
by policy — see `.gitignore`: "Data: never commit datasets").

## Dataset description

- **Files:** 8 CSVs, one per symbol; each holds two columns (`Date`, `<SYMBOL>`)
  of Yahoo Finance **adjusted daily closes** (fetched with
  `scripts/download_data.py`, which calls `yf.download(..., auto_adjust=True)`).
- **Ticker universe:** AAPL, MSFT, JNJ, JPM, XOM, PG, AMZN (the 7 strategy
  tickers from `configs/default.yaml`) + SPY (benchmark, context only).
- **Date window:** 2018-01-02 → 2026-10-07 — 2,203 trading rows per file,
  date-aligned across all 8 symbols.
- **Fetch date:** not recorded at download time — the downloader writes no
  fetch log and the CSVs contain only `Date,PRICE` columns. The surviving
  acquisition record is `docs/PHASE1-NUMERICAL-RIGOR.md` (§ "Reproduction under
  the corrected engine (item F)"), which documents a re-download of this exact
  window during the Phase-1 reproduction (after the v1.1.0 release of
  2026-10-21). The hashes below were computed and verified on **2026-10-23**.
- **Known caveat:** Yahoo Finance revises adjusted closes retroactively, so a
  future re-download of the same window is not guaranteed to be bit-identical.
  Verify any fresh download against the hashes below before re-running the
  pipeline.

## SHA-256 manifest

Regenerate the hashes with `sha256sum data/raw/*.csv` and compare:

| File      | Rows | First date | Last date  | SHA-256                                                          |
| --------- | ----:| ---------- | ---------- | ---------------------------------------------------------------- |
| AAPL.csv  | 2203 | 2018-01-02 | 2026-10-07 | `bdcbbf046a5eadce1f62057b629e1ee8efdb2330f47f181ba66ccb3be14f2303` |
| AMZN.csv  | 2203 | 2018-01-02 | 2026-10-07 | `067670163e5567df5ef47dfc5e443cb4513c23dc4c823a550439ed30122555a6` |
| JNJ.csv   | 2203 | 2018-01-02 | 2026-10-07 | `f9e0fda2c3378c9c9867f44fad9f0816c8cc74268d449035b5c59318ef136b8f` |
| JPM.csv   | 2203 | 2018-01-02 | 2026-10-07 | `925edacd437204786708b13ec0de74ddf318ab0478f7c8a7969aed6369496d5a` |
| MSFT.csv  | 2203 | 2018-01-02 | 2026-10-07 | `266b9436744d08f687c3a8360a40681a8f30cb1260248d2216ed97f574943616` |
| PG.csv    | 2203 | 2018-01-02 | 2026-10-07 | `de07b8ac699ca3f792b795a96f031b4658d4b48c17355b967247703515b1f9ad` |
| SPY.csv   | 2203 | 2018-01-02 | 2026-10-07 | `11676303b1e8d2c11ece5d7b8aa68bafefd8cb4f603a36de71a96d07a7496fb5` |
| XOM.csv   | 2203 | 2018-01-02 | 2026-10-07 | `a1eb1e9ff2984591c8907b7587c021c71b394c0e001de918c35bef27f43ae8dc` |

## Verify locally

```bash
sha256sum data/raw/*.csv            # must match the table above
head -2 data/raw/AAPL.csv           # first data row must be 2018-01-02
tail -1 data/raw/AAPL.csv           # last data row must be 2026-10-07
python scripts/download_data.py     # re-fetch (network); compare hashes afterwards
```
