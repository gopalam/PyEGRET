# Reference data

`Choptank_eList.RData` is the `Choptank_eList` dataset shipped with the USGS
EGRET R package, downloaded unmodified from
<https://github.com/DOI-USGS/EGRET/blob/main/data/Choptank_eList.RData>.

EGRET is a USGS product released to the public domain (CC0 1.0); see
<https://github.com/DOI-USGS/EGRET/blob/main/LICENSE.md>.

It is a *fitted* eList — INFO, Daily, Sample **and** the WRTDS `surfaces`
array as EGRET itself computed them — for the Choptank River at Greensboro,
MD (USGS site 01491000), nitrate plus nitrite (parameter 00631), water years
1980–2011. 11,688 daily values, 606 samples, 1 censored.

That makes it a gold-standard reference: `tests/test_egret_reference.py`
re-fits the same Daily and Sample with PyEGRET and compares against EGRET's
own output, so the port is checked against the reference implementation
rather than against itself.

`INFO` carries the settings EGRET used, and the test reads them from there
rather than hard-coding: `windowY` 10, `windowQ` 2, `windowS` 0.5,
`minNumObs` 100, `minNumUncen` 50, `stepYear` 1/16, `nVectorYear` 529,
`nVectorLogQ` 14.

**Edge adjustment is off for this comparison.** `windowY = 10` was EGRET's
default before it changed to 7, and the stored surface matches PyEGRET
exactly only with `edge_adjust=False` — with it on, the interior still agrees
to machine precision but the outer ten years differ, which is what you would
expect from a surface computed before `edgeAdjust` existed (EGRET 1.3.0, July
2014). PyEGRET keeps `edge_adjust=True` as its default, matching current
EGRET.

Reading `.RData` needs the `rdata` package (`pip install rdata`); no R
installation is required. The test skips if it is not installed.

## Results computed by current EGRET (3.0.12)

Three files saved by EGRET's own vignette, `vignettes/Enhancements.Rmd`,
copied unmodified from <https://github.com/DOI-USGS/EGRET/tree/main/vignettes>
(commit b5eb22c, July 2026). They check the trend functions against what
current R EGRET computes, with edge adjustment on:

| file | what R ran |
|---|---|
| `pairResults.rds` | `runPairs(Choptank_eList, 1985, 2010, windowSide = 7)` |
| `pairResults2.rds` | the same with `flowBreak`, `Q1EndDate = "1995-05-31"`, `wall`, `sample1EndDate = "1995-05-31"`, `QStartDate`, `QEndDate`, `paStart = 4`, `paLong = 5` |
| `ChainBridge.TP.RData` | a fitted eList (total phosphorus, Potomac River at Chain Bridge, 1984-2018) from the WRTDSK vignette, fitted with current defaults: `windowY = 7`, `edgeAdjust = TRUE` |
| `eListOut.rds` | `runSeries(eList_Green, windowSide = 12, flowBreak = TRUE, Q1EndDate = "1963-03-31", wall = TRUE, sample1EndDate = "1963-03-01")` for chloride in the Green River near Greendale, UT (09234500), 1950-2018 |

`tests/test_egret_trend_reference.py` re-runs each one in PyEGRET from the
same Daily and Sample. Agreement at the time of writing: the x10/x11/x20/x22
values within 0.05 %; the Green River annual flow-normalized series within a
median of 1e-6 (worst year 0.1 %); and for Chain Bridge, daily ConcDay and
FNConc within a median of 2e-4, annual flow-normalized values within 4e-5, and
cross-validated yHat within 2e-5 log units. Chain Bridge is the check that
the current defaults, including edge adjustment, match.
