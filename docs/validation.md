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

## EGRETci: bootstrap uncertainty

`pyegret.ci` was checked against EGRETci 2.0.5 itself. The R sources in
`reference/` were run in R 4.3.3 (survival 3.5-8) on the Choptank record, and
the results were saved in `tests/reference/egretci/` with the scripts that
made them.

**Same resamples as R.** EGRETci draws each bootstrap resample after
`RNGversion("3.5.0"); set.seed(startSeed + i)`. PyEGRET reproduces R's
Mersenne-Twister, its seeding, and its pre-3.6 `sample()` exactly
(`pyegret.ci.r_random`), so replicate *i* in PyEGRET resamples the same
samples, in the same order, as replicate *i* in R. That makes the comparison
one for one rather than only in distribution: any difference between a
PyEGRET replicate and R's is the model fit, not the resampling.

| what R ran | agreement |
|---|---|
| `blockSample` (two seeds, two block lengths) | identical rows |
| `binom.bayes` (central), `pVal`, `wordLike` | identical |
| `runPairsBoot`, wall, flow break, April-August, 100 replicates | replicates median 0.04 %; same p-values (0.0198) and likelihoods; interval ends within 0.3 % |
| `runPairsBoot`, `windowSide = 7`, 30 replicates | replicates within 0.5 % |
| `wBT`, 1985-2005, nBoot 100, bootBreak 39 | stops at the same replicate (39); replicates within 0.6 %; same p-values and likelihoods |
| `runGroupsBoot`, wall, April-May, 20 replicates | replicates within 0.0016 mg/L and 0.0005 x 10^6 kg/yr; same likelihoods; p-values 0.095 v. 0.095 and 0.564 v. 0.562 |
| `bootAnnual` and `ciBands`, `modelEstimation` baseline | replicates typically 0.1 %; bands median 0.01-0.03 %, worst 0.16 % |
| the same with a `runSeries` baseline (`windowSide = 7`) | bands median 0.01-0.03 %, worst 1.3 % |

About one replicate in a hundred differs by more: in the pairs run,
replicate 80's flux change differs by 30 % while its concentration change
agrees to 0.1 %. In that replicate R's `survreg` failed to converge at five
grid points of one surface; R then jitters the data at random and refits, a
path PyEGRET does not follow, and the two surfaces differ by a few percent in
11 of their 126 cells, mostly at extrapolated low discharges near the end of
the record. The intervals built from all the replicates are hardly affected,
as the table shows.

**EGRETci's own vignette file.** `bootPairOut2.rds` in EGRETci's vignettes
agrees with PyEGRET replicate for replicate up to replicate 54. In the R
setup that wrote it, replicate 55's fit failed, so every later replicate is
shifted by one. With current R that replicate fits (R gives 0.4237, PyEGRET
0.4239) and the shift disappears, which is why the references above were
recomputed.

### Deliberate differences from EGRETci

- **`gen_daily_boot` uses each bootstrap refit.** In EGRETci 2.0.5,
  `genDailyBoot` fits a surface to each resample but `WRTDSKalman` takes
  `yHat` and `SE` from `Daily`, which still holds the original model's
  values, so the refits have no effect: run in R with `nKalman = 1` and a
  fixed seed, its two bootstrap replicates are identical to the last digit.
  EGRETci's documentation says each replicate "is used to estimate the WRTDS
  model (we can think of this as the estimation error)", and PyEGRET does
  that, recomputing `Daily` and the sample predictions from each refit before
  the Kalman step. `boot_surfaces=False` reproduces EGRETci 2.0.5. The
  prediction intervals are therefore somewhat wider than EGRETci's.
- **`run_pairs_boot` passes the windows `run_pairs` used.** R's
  `runPairsBoot` leaves `windowY`, `windowQ` and `windowS` at their defaults
  inside each replicate, whatever `runPairs` was given. The results are the
  same unless the windows were changed.
- **No extra replicate.** R's loop fits one replicate more than it needs and
  discards it; PyEGRET stops at `n_boot`. The results are the same.
- **`jitterOn`** is not translated (it needs `jitterSam`).

Translating EGRETci also brought one fix to the core: like R's `runSurvReg`,
fitting a surface now refuses, with an error, a sample with no more
observations than `min_num_obs` (or uncensored values than `min_num_uncen`),
where PyEGRET used to carry on with the windows widened. R's bootstrap relies
on this to reject a resample.

## Rerunning the checks

```
python -m pytest tests/test_egret_reference.py tests/test_egret_trend_reference.py
python -m pytest tests/test_egretci.py
python diagnostics/validate_against_egret.py     # prints the Choptank comparison
```

Other tests check PyEGRET against synthetic records whose true trend is
known (`pyegret.testing.make_synthetic_elist`), and check individual
functions against the formulas in the R source.
