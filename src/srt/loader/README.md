# `srt.loader` — Return/PnL Matrix Loader

Loads per-strategy trade logs from disk and aligns them into a single time-indexed
`T×N` matrix (rows = periods, columns = strategies), ready to feed the downstream
overfitting (CSCV/PBO) and portfolio (HRP) modules.

> **Note:** This README was generated with the assistance of an AI tool (Claude) and
> reviewed by the author.

---

## What it does

Given a directory of per-strategy CSV files, the loader:

1. Reads each strategy's trade log.
2. Buckets each trade into a calendar period (daily / weekly / monthly) by its
   **close time**.
3. Sums the per-trade `Profit/Loss` within each period.
4. Reindexes every strategy onto a **common period grid**, filling empty periods
   with `0.00`, so all strategies share one aligned index.
5. Concatenates the strategies column-wise into a single `T×N` `DataFrame`.

The result is a dense, gap-free matrix suitable for statistical testing, where a
period with no closed trades contributes a flat `0.00`.

## What the values represent

Each cell is the **per-period Profit/Loss** for that strategy — the sum of the
`Profit/Loss` column of all trades whose close time falls in that period. These are
**not** arithmetic returns (there is no division by account equity or capital); they
are additive per-period PnL. Because PnL is additive, summing within a period and the
`0.00` fill for empty periods are both well-defined.

If a downstream method requires arithmetic returns rather than PnL, that conversion is
**not** performed here and must be applied by the caller.

## Attribution convention

Trades are bucketed by **close time**, so a trade contributes entirely to the period
in which it was closed, regardless of when it was opened. This is the deliberate
choice for weekly/monthly bucketing.

---

## Usage

```python
from srt.loader.returns import Loader

# Directory is resolved under ./Data/<stratsDir>/
ld = Loader("myStrategySet")

matrix = ld.readStrats(
    startDate="2003-01-01 00:00:00",
    endDate="2025-12-31 23:59:59",
    returns="Weekly",          # "Daily" | "Weekly" | "Monthly"
)

matrix.shape   # (n_periods, n_strategies)
```

### Constructor — `Loader(stratsDir)`

| Argument     | Type  | Description                                                        |
|--------------|-------|-------------------------------------------------------------------|
| `stratsDir`  | `str` | Sub-directory name under `./Data/`. Each `*.csv` inside is one strategy. |

Raises if the resolved directory does not exist.

### `readStrats(startDate, endDate, returns)`

| Argument    | Type  | Default                   | Description                                     |
|-------------|-------|---------------------------|-------------------------------------------------|
| `startDate` | `str` | `"2003-01-01 00:00:00"`   | Inclusive start of the period grid.             |
| `endDate`   | `str` | `"2025-12-31 23:59:59"`   | Inclusive end of the period grid.               |
| `returns`   | `str` | `"Weekly"`                | Aggregation frequency: `"Daily"`, `"Weekly"`, or `"Monthly"`. |

**Returns:** `pandas.DataFrame` of shape `T×N`, indexed by period, one column per
strategy (column name = CSV filename without the `.csv` suffix).

**Raises:**
- `ValueError` if `returns` is not one of the three accepted values.
- `Exception` if the number of output columns does not match the number of input files
  (guards against a silently dropped or duplicated strategy).

---

## Expected input format

Each strategy is a semicolon-delimited CSV under `./Data/<stratsDir>/`:

- Separator: `;`
- Required columns: `Close time`, `Profit/Loss`
- `Close time` format: `%Y.%m.%d %H:%M:%S` (e.g. `2021.03.15 14:30:00`)
- The output column name is taken from the filename (`EURUSD_ea1.csv` → `EURUSD_ea1`).

---

## Frequency codes

| `returns`   | pandas period freq |
|-------------|--------------------|
| `"Daily"`   | `D`                |
| `"Weekly"`  | `W`                |
| `"Monthly"` | `ME`               |

`ME` (month-end) is used rather than the deprecated `M` for pandas ≥ 2.2 compatibility.

---

## Design notes & conventions

- **Alignment by shared index.** All strategies are reindexed onto the same
  `pd.period_range`, so the column-wise concat aligns cleanly and every column has the
  same length. Empty periods become `0.00`.
- **Dense, gap-free output** by construction — no `NaN` leakage into downstream
  statistical tests.
- **Column-count guard.** After the concat, the loader asserts the output column count
  equals the number of input files.

## Known limitations

- **PnL, not returns.** Values are per-period PnL, not arithmetic returns (see *What the
  values represent*). Downstream methods that assume returns must convert upstream.
- **Duplicate strategy names.** If two files reduce to the same column name, the
  column-count guard may not catch the collision cleanly; strategy filenames should be
  unique.
- **No reconciliation check.** Cross-source agreement of two independent return streams
  is not verified here.

## Dependencies

`pandas`. Standard library `os`.
