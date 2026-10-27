# Notices and credits

## What PyEGRET is

PyEGRET is an independent Python translation of the R packages **EGRET**
(Exploration and Graphics for RivEr Trends) and **EGRETci**, and of the
**WRTDS** method (Weighted Regressions on Time, Discharge, and Season) they
implement. The method, the R packages and the design of almost every function
here are the work of Robert M. Hirsch, Laura A. De Cicco, Jennifer C. Murphy,
Stacey A. Archfield, Qian Zhang and their colleagues at the **U.S. Geological
Survey**. PyEGRET's contribution is the translation, its tests and its
validation against R; the science is theirs.

**PyEGRET is not a USGS product.** It is not developed, reviewed, approved or
endorsed by the U.S. Geological Survey, and any errors in it are its own.
For the reference implementation, use EGRET:
<https://github.com/DOI-USGS/EGRET>.

## Licenses

- The PyEGRET source code and documentation (`pyegret/`, `tests/*.py`,
  `diagnostics/`, `examples/`, `docs/`, and the notebooks in `tutorials/`) are
  released under the BSD 3-Clause License; see
  `LICENSE`.
- `reference/EGRET_R/` and `reference/EGRETci_R/` are unmodified copies of the
  R sources of EGRET 3.0.12 and EGRETci 2.0.5, kept for line-by-line
  translation. They are U.S. Government works released by the USGS under
  CC0 1.0 (public domain), and remain so.
- The data files in `tests/reference/` (`Choptank_eList.RData`,
  `ChainBridge.TP.RData`, `pairResults.rds`, `pairResults2.rds`,
  `eListOut.rds`) come unmodified from the EGRET repository and are likewise
  CC0 1.0 (public domain). They are used only to check PyEGRET against R.
- `tutorials/data/choptank_daily.csv` and `choptank_sample.csv` are the
  Daily and Sample of the same `Choptank_eList`, written out as CSV (discharge
  converted back to cubic feet per second), and are likewise public domain.

The BSD license applies to the Python translation only; it adds no
restriction to the public-domain material above.

## How to cite

If you use PyEGRET in published work, please cite the method and the R
package it translates, as well as PyEGRET itself (see `CITATION.cff`):

- Hirsch, R.M., Moyer, D.L., and Archfield, S.A., 2010, Weighted Regressions
  on Time, Discharge, and Season (WRTDS), with an application to Chesapeake
  Bay river inputs: Journal of the American Water Resources Association,
  v. 46, no. 5, p. 857-880. <https://doi.org/10.1111/j.1752-1688.2010.00482.x>
- Hirsch, R.M., and De Cicco, L.A., 2015, User guide to Exploration and
  Graphics for RivEr Trends (EGRET) and dataRetrieval: R packages for
  hydrologic data: U.S. Geological Survey Techniques and Methods, book 4,
  chap. A10, 93 p. <https://doi.org/10.3133/tm4A10>
- Hirsch, R.M., De Cicco, L.A., and Murphy, J., 2025, Exploration and
  Graphics for RivEr Trends (EGRET), version 3.0.12.
  <https://doi.org/10.5066/P9CC9JEX>

and, for the parts of the method you use:

- Generalized flow normalization (`run_pairs`, `run_groups`, `run_series`
  with `window_side > 0` or a flow break): Choquette, A.F., Hirsch, R.M.,
  Murphy, J.C., Johnson, L.T., and Confesor, R.B., Jr., 2019, Tracking changes
  in nutrient delivery to western Lake Erie: Approaches to compensate for
  variability and trends in streamflow: Journal of Great Lakes Research,
  v. 45, no. 1, p. 21-39. <https://doi.org/10.1016/j.jglr.2018.11.012>
- WRTDS-Kalman (`wrtds_kalman`): Zhang, Q., and Hirsch, R.M., 2019, River
  water-quality concentration and flux estimation can be improved by
  accounting for serial correlation through an autoregressive model: Water
  Resources Research, v. 55, no. 11, p. 9705-9723.
  <https://doi.org/10.1029/2019WR025338>
- Bootstrap uncertainty (EGRETci; planned for PyEGRET): Hirsch, R.M.,
  Archfield, S.A., and De Cicco, L.A., 2015, A bootstrap method for estimating
  uncertainty of water quality trends: Environmental Modelling & Software,
  v. 73, p. 148-166. <https://doi.org/10.1016/j.envsoft.2015.07.017>
- EGRETci itself: Hirsch, R.M., De Cicco, L.A., Archfield, S.A., and Murphy,
  J.C., 2025, Exploration and Graphics for RivEr Trends (EGRET) Uncertainty and
  Confidence Intervals, version 2.0.5. <https://doi.org/10.5066/P9CC9JEX>

The Quantile-Kendall analysis in `pyegret.flow` follows the streamflow-trend
vignette distributed with EGRET (`vignettes/streamflow_trend.Rmd`).
