# PyEGRET - status and roadmap

Last updated 2026-10-29.

PyEGRET translates the USGS R packages EGRET (3.0.12) and EGRETci (2.0.5). The R sources are in `reference/`. How the code must be
written is in `docs/dev/CLAUDE_CODE_INSTRUCTIONS.md`; this file records what exists,
how well it matches R, and what is left.

462 tests pass offline (3 skipped need the live USGS service).
`python diagnostics/check_style.py` reports 0 problems: no classes, no
chained calls, and the package imports straight from the folder.

## How to use it

```python
import sys
sys.path.insert(0, r"C:\path\to\PyEGRET")
import pyegret
```

`examples/choptank_wrtds.py` runs the whole workflow offline on the
Choptank River - data overview, model, error statistics, tables, contour
plots, run_pairs with generalized flow normalization, run_series, and
WRTDS-Kalman - and writes the figures to `examples/output/`.

## Agreement with R EGRET

Every number below compares PyEGRET with output R EGRET itself produced,
read from `tests/reference/` without needing R.

| check | reference | agreement |
|---|---|---|
| surface, daily estimates, FN trend | Choptank_eList (older EGRET, edgeAdjust off) | surface 99.6 % of cells to 1e-4; daily median 4e-8; FN change +35.5 % v. +35.5 % |
| model with **current defaults** (windowY 7, edgeAdjust on) | Chain Bridge TP, WRTDSK vignette | daily ConcDay / FNConc median 2e-4; annual FN within 4e-5; CV yHat within 2e-5 log units |
| `runPairs(1985, 2010, windowSide = 7)` | pairResults.rds | x10-x22 within 0.05 %; +38.5 % v. R's +38.6 % |
| `runPairs` with wall, flowBreak, Q dates, Apr-Aug season | pairResults2.rds | within 0.06 % |
| `runSeries` with wall, flowBreak, windowSide 12, 68-year record | eListOut.rds (Green River) | annual FN median 1e-6, worst year 0.1 % |

The residual differences sit where the weighted censored regression is
least constrained (grid corners outside the data), where scipy's L-BFGS-B
and R's `survreg` settle a few decimals apart.

## What has been ported

95 of EGRET's 131 exported functions, by area:

- **Data**: readUserDaily / readUserSample / readDataFromFile (cfs default,
  multi-column remark/value summing, zero-result removal, as in R),
  populateDaily (calendar fill, zero-flow shift, Q7/Q30), mergeReport (exact
  date merge), compressData, removeDuplicates, populateConcentrations,
  processQWData, fixSampleFrame, dataOverview, makeAugmentedSample, USGS
  readers on the modern Water Data API (`read_waterdata_*`; the legacy
  `read_nwis_*` remain).
- **Model**: modelEstimation, estSurfaces (full or partial span),
  estCrossVal, estDailyFromSurfaces (yHat, SE, ConcDay, FluxDay, FNConc,
  FNFlux; Feb 28/29 pooling), runSurvReg (interval censoring; single point and
  vectorised), triCube and the adaptive windows, flexFN, makeDateInfo, the
  wall (`stitch`), surfaceStartEnd, decimalHighLow.
- **WRTDS-Kalman**: WRTDSKalman with cleanUp, randomSubset,
  populateDailySamp, genmissing, precompute_L.
- **Trends**: runPairs, runGroups, runSeries (all four flow-normalization
  cases, wall, Q and sample date limits), printPairs, printGroups.
- **Annual and summary**: setupYears (complete years, 90 % rule),
  calculateMonthlyResults, tableResults, tableChange, tableChangeSingle,
  errorStats, fluxBiasStat, setPA, startEnd, blankTime, nObservations,
  nCensoredVals, nDischarge.
- **Plots** (24): plotConcQ, plotConcTime, boxConcMonth, boxQTwice,
  plotQTimeDaily, multiPlotDataOverview, plotConcHist, plotFluxHist,
  plotConcTimeDaily, plotFluxTimeDaily, plotWRTDSKalman, plotTimeSlice,
  plotResidPred, plotResidQ, plotResidTime, boxResidMonth, boxConcThree,
  plotConcPred, plotFluxPred, fluxBiasMulti, plotContours, plotDiffContours,
  plotConcQSmooth, plotConcTimeSmooth.
- **flowHistory**: makeAnnualSeries, flowDuration, cumQdate, and the
  Quantile-Kendall analysis from EGRET's streamflow-trend vignette.
- **Units and labels**: EGRET's qConst / fluxConst tables with EGRET's
  numbering, setSeasonLabel wording.

## Remaining work, in order

### Done since 0.1.0

- **EGRETci** (October 2026): `pyegret.ci`, checked replicate for replicate
  against EGRETci 2.0.5 run in R (see docs/validation.md). Not translated:
  `jitterOn` (needs `jitterSam`) and `saveEGRETci`.
- **Live USGS check** (October 2026): `read_waterdata_*` verified against the
  live service; the sample reader was rebuilt on R's steps, and
  `read_wqp_sample` fixed (it dropped non-detects).
- `read_nwis_sample` is now a wrapper of `read_waterdata_sample`.

### 1. Report upstream

- EGRETci 2.0.5's `genDailyBoot` ignores its bootstrap refits (WRTDSKalman
  reads `yHat` and `SE` from `Daily`). Worth an issue on
  https://github.com/DOI-USGS/EGRETci with the R check in
  `tests/reference/egretci/r/make_references.R`.

### 2. Distribution

- Publish on PyPI; link the repository to Zenodo for a DOI per release; a
  short paper in the Journal of Open Source Software.

### 3. The rest of EGRET

- flowHistory plots in EGRET's own form: plotFlowSingle, plotFour,
  plotFourStats, plotSDLogQ (PyEGRET has its own `plot_flow_trend` and the
  Quantile-Kendall figure), and tableFlowChange, printSeries.
- plotFluxQ, plotMonthTrend (uses run_pairs' `byMonth`), plot15 / plot1of15.
- fill_missing_daily (populateDaily's optional gap filling).
- readUserInfo, readWQPInfo, populateSiteINFO / populateParameterINFO,
  saveResults, run_WRTDS, checkSurfaceSpan, jitterSam.
- Internal plot helpers (generalAxis, genericEGRETDotPlot, yPretty,
  flexPlotAddOn) are not needed as such; matplotlib does that work.

### 4. Tidy-ups

- `read_nwis_daily` and `read_nwis_info` could become wrappers too.
- `jitterSam`, which would make EGRETci's `jitterOn` option possible.

## Deliberate differences from R

- `start_end` uses the real calendar; R's month-length table gives August
  30 days.
- Left-censored samples carry `ConcLow = 0` (R: NA); equivalent everywhere.
- `surfaces` is `[year, logQ, slice]` (R: `[logQ, year, slice]`); the grid
  axes live in `elist["vector_year"]` / `elist["vector_logQ"]` rather than
  as array attributes.
- Results R returns as attributes (`runPairs`, `runGroups`) come back as
  dicts with the same names.
- `wrtds_kalman` agrees with R in distribution, not digit for digit: NumPy
  and R use different random generators.

## Running things

```
python -m pytest                                  # 420 tests, offline
python diagnostics/check_style.py                 # style rules
python diagnostics/validate_against_egret.py      # Choptank v. R
python diagnostics/wrtds_truth_check.py           # synthetic truth
python diagnostics/check_waterdata.py             # live USGS API
python examples/choptank_wrtds.py                 # end-to-end example
```

Needs numpy, pandas, scipy, matplotlib; `dataretrieval` for the USGS readers,
`pytest` for tests, `rdata` for the comparisons with R.
