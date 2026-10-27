# Validation against R EGRET

A translation is only useful if it gives the same answers as the original.
PyEGRET is checked against results computed by **R EGRET itself**, not
against its own expectations. The reference results come from EGRET's own
repository and vignettes (EGRET 3.0.12) and are stored in `tests/reference/`.
The `rdata` package reads them, so R does not need to be installed to rerun
any of these checks.

## The comparisons

| what R ran | data | agreement |
|---|---|---|
| `modelEstimation`, current defaults (`windowY = 7`, `edgeAdjust = TRUE`) | Potomac River at Chain Bridge, total phosphorus, 1984-2018 (`ChainBridge.TP.RData`, from EGRET's WRTDS-Kalman vignette) | daily `ConcDay` and `FNConc` median difference 0.02 %; annual flow-normalized concentration and flux within 0.004 %; cross-validated `yHat` within 2e-5 log units |
| `modelEstimation`, older defaults (`windowY = 10`, no edge adjustment) | Choptank River, nitrate plus nitrite, 1979-2011 (`Choptank_eList.RData`, EGRET's example data) | 99.6 % of surface cells to 1e-4; daily estimates median 4e-8; flow-normalized change over the record +35.5 % v. +35.5 % |
| `runPairs(eList, 1985, 2010, windowSide = 7)` | Choptank (`pairResults.rds`) | x10, x11, x20 and x22 within 0.05 %; total change +38.5 % v. +38.6 % |
| `runPairs` with `flowBreak`, `wall`, `QStartDate` / `QEndDate`, `paStart = 4`, `paLong = 5` | Choptank (`pairResults2.rds`) | within 0.06 % |
| `runSeries(windowSide = 12, flowBreak = TRUE, wall = TRUE)` | Green River near Greendale, UT, chloride, 1950-2018 (`eListOut.rds`) | annual flow-normalized values median 1e-6, worst year 0.1 % |

## Where the small differences come from

At each point of the WRTDS grid, R fits a weighted censored regression with
`survival::survreg`; PyEGRET fits the same likelihood with SciPy's L-BFGS-B
optimizer. Where the data constrain the regression well the two agree to
many decimals. They drift apart, by a few parts in a thousand, only at grid
points far from any data (very low or very high discharge in years with few
samples), which carry little weight in the daily estimates.

Two further differences are known and expected:

- **February 28/29 and December 30/31 in the Choptank comparison.** The
  shipped `Choptank_eList` was fitted by an older EGRET, before it pooled the
  reference flows of February 28 and 29 and before its leap-adjusted day
  numbering. PyEGRET follows current EGRET, so those few days differ from
  that old file; Chain Bridge, fitted by current EGRET, agrees on every day.
- **WRTDS-Kalman** uses random simulation, and NumPy and R have different
  random-number generators, so `wrtds_kalman` agrees with R in distribution
  but not digit for digit. It is tested against the properties the method
  guarantees: it reproduces the sampled concentration exactly on sampled days,
  and its conditional simulation matches the AR(1) bridge mean.

## Rerunning the checks

```
python -m pytest tests/test_egret_reference.py tests/test_egret_trend_reference.py
python diagnostics/validate_against_egret.py     # prints the Choptank comparison
```

Other tests check PyEGRET against synthetic records whose true trend is
known (`pyegret.testing.make_synthetic_elist`), and check individual
functions against the formulas in the R source.
