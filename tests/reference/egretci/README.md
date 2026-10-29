# EGRETci reference results

These files hold results computed by R EGRETci itself, which
`tests/test_egretci.py` compares PyEGRET's `pyegret.ci` against.

They were produced in October 2026 with R 4.3.3 and survival 3.5-8, from the
R sources of EGRET 3.0.12 and EGRETci 2.0.5 kept in `reference/` of this
repository, on EGRET's `Choptank_eList` (`tests/reference/Choptank_eList.RData`).
The scripts are in `r/`: `make_references.R` runs everything (about an hour
and a half, mostly `runGroupsBoot`) and writes `.rds` files, which were
converted to the JSON files here keeping only what the tests compare.

| file | R call |
|---|---|
| `small.json` | `blockSample` (two seeds), `binom::binom.bayes(..., "central")`, `pVal`, `wordLike` |
| `pairs_wall_season.json` | `runPairs(Choptank_eList, 1985, 2010, windowSide = 7, flowBreak = TRUE, Q1EndDate = "1995-05-31", wall = TRUE, sample1EndDate = "1995-05-31", QStartDate = "1979-10-01", QEndDate = "2010-09-30", paStart = 4, paLong = 5)`, then `runPairsBoot(nBoot = 100)` |
| `pairs_simple.json` | `runPairs(Choptank_eList, 1985, 2010, windowSide = 7)`, then `runPairsBoot(nBoot = 30)` |
| `wbt.json` | `wBT(Choptank_eList, Choptank_caseSetUp)`: 1985 to 2005, nBoot 100, bootBreak 39 |
| `ci_model.json` | `modelEstimation`, then `bootAnnual(eList, 200, 494817 + n)` for n = 1 to 5 and `ciBands` |
| `ci_series.json` | `runSeries(windowSide = 7)`, then `bootAnnual` for n = 1 to 3 and `ciBands` |
| `groups_wall.json` | `runGroups(Choptank_eList, 1995, 2004, 2005, 2010, windowSide = 7, wall = TRUE, sample1EndDate = "2004-10-30", paStart = 4, paLong = 2)`, then `runGroupsBoot(nBoot = 20)` |

`r/Choptank_caseSetUp.RData` is copied unchanged from EGRETci's `data/`
folder.

Why not the files in EGRETci's own vignettes? Its `bootPairOut2.rds` agrees
with PyEGRET replicate for replicate up to replicate 54, but in the R setup
that wrote it, replicate 55's model fit failed and every later replicate
shifted by one. With current R the same replicate fits (R gives 0.4237,
PyEGRET 0.4239), and the shift disappears. Results computed from the same
sources with current R are the fairer reference.
