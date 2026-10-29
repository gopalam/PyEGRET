"""
Uncertainty for WRTDS results: a translation of the USGS R package EGRETci.

Mirrors EGRETci 2.0.5 (Hirsch, De Cicco and others; https://github.com/DOI-USGS/EGRETci).
The method is described in Hirsch, R.M., Archfield, S.A., and De Cicco, L.A.,
2015, A bootstrap method for estimating uncertainty of water quality trends:
Environmental Modelling & Software, v. 73, p. 148-166,
https://doi.org/10.1016/j.envsoft.2015.07.017.

Trends between two years or two groups of years:

- :func:`run_pairs_boot`, :func:`run_groups_boot`: confidence intervals,
  two-sided p-values and likelihoods for ``run_pairs`` and ``run_groups``
  results (the recommended route).
- :func:`wbt`: the original WRTDS Bootstrap Test.
- :func:`plot_histogram_trend`: the distribution of the bootstrap trends.

The annual series:

- :func:`ci_calculations`: confidence bands on flow-normalized
  concentration and flux; :func:`plot_conc_hist_boot`,
  :func:`plot_flux_hist_boot` draw them.

WRTDS-Kalman estimates:

- :func:`gen_daily_boot` then :func:`make_daily_pi`, :func:`make_month_pi`,
  :func:`make_annual_pi`: prediction intervals.

Bootstrap resamples are drawn with R's own random-number generator, seeded as
EGRETci seeds it, so PyEGRET's replicate *i* resamples the same samples as
EGRETci's.
"""
from pyegret.ci.bootstrap import (block_sample, p_val, set_for_boot, trend_set_up,
                                  word_like)
from pyegret.ci.boot_trends import run_groups_boot, run_pairs_boot
from pyegret.ci.ci_bands import boot_annual, ci_bands, ci_calculations
from pyegret.ci.plots import plot_conc_hist_boot, plot_flux_hist_boot, plot_histogram_trend
from pyegret.ci.prediction import (gen_daily_boot, make_annual_pi, make_daily_pi,
                                   make_month_pi, month_seq_to_dec)
from pyegret.ci.wbt import wbt

__all__ = [
    "run_pairs_boot",
    "run_groups_boot",
    "wbt",
    "trend_set_up",
    "set_for_boot",
    "block_sample",
    "p_val",
    "word_like",
    "ci_calculations",
    "boot_annual",
    "ci_bands",
    "plot_conc_hist_boot",
    "plot_flux_hist_boot",
    "plot_histogram_trend",
    "gen_daily_boot",
    "make_daily_pi",
    "make_month_pi",
    "make_annual_pi",
    "month_seq_to_dec",
]
