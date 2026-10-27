"""
Report how well the WRTDS engine recovers a known answer.

The same checks run as assertions in ``tests/test_wrtds_truth.py``; this
script prints the numbers instead, which is what you want when tuning windows
or grid resolution and you need to see how much a change actually bought.

Run:  python diagnostics/wrtds_truth_check.py
No network needed. Takes a few seconds.

Reference values, on the default synthetic record, before the four
correctness fixes of September 2026 (coarse surface grid, chord-distance
season metric, no window expansion, season-blind flow normalization):

    ConcDay median rel err   +4.4 %   IQR [-28.5 %, +37.2 %]
    FN trend                 -0.0066 /yr   (truth -0.0200)
    FN change over record    +33.0 %       (truth -31.6 %)   <- wrong sign
    NaN grid cells           1 of 224
    runtime                  ~190 s
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyegret.model.model_estimation import model_estimation           # noqa: E402
from pyegret.testing import (                                          # noqa: E402
    DEFAULT_TRUTH,
    annual_mean_by_calendar_year,
    make_synthetic_elist,
    truth_change_over,
    truth_conc,
)

_TWO_PI = 2.0 * np.pi


def seasonal_amplitude(values, dec_year):
    """Amplitude of the annual cycle in log(values), in log units."""
    design = np.column_stack([
        np.ones_like(dec_year),
        np.sin(_TWO_PI * dec_year),
        np.cos(_TWO_PI * dec_year),
    ])
    beta = np.linalg.lstsq(design, np.log(values), rcond=None)[0]
    return float(np.hypot(beta[1], beta[2]))


def main() -> None:
    truth = DEFAULT_TRUTH
    print("Synthetic record: 20 years, 240 samples, "
          f"true trend {100 * truth['trend']:+.0f} %/yr")

    elist = make_synthetic_elist(truth=truth)
    started = time.time()
    fitted = model_estimation(elist, verbose=False)
    elapsed = time.time() - started

    daily = fitted["Daily"]
    surfaces = fitted["surfaces"]
    dec_year = daily["DecYear"].to_numpy()
    log_q = daily["LogQ"].to_numpy()

    n_yr, n_lq = surfaces.shape[0], surfaces.shape[1]
    n_nan = int(np.sum(np.isnan(surfaces[:, :, 0])))
    step = float(np.diff(fitted["vector_year"])[0])

    expected = truth_conc(dec_year, log_q, truth)
    rel = (daily["ConcDay"].to_numpy() - expected) / expected

    annual = annual_mean_by_calendar_year(daily, "FNConc")
    years = annual.index.to_numpy(dtype=float)
    fn = annual.to_numpy(dtype=float)
    slope = float(np.polyfit(years, np.log(fn), 1)[0])
    change = float(fn[-1] / fn[0] - 1.0)
    truth_change = truth_change_over(truth, years[-1] - years[0])

    print()
    print(f"  grid                     {n_yr} x {n_lq} "
          f"({n_yr * n_lq} points, {step:.4f} yr step)")
    print(f"  unestimated grid points  {n_nan}")
    print(f"  runtime                  {elapsed:.1f} s")
    print()
    print(f"  ConcDay median rel err   {100 * np.nanmedian(rel):+.1f} %"
          f"   IQR [{100 * np.nanpercentile(rel, 25):+.1f} %,"
          f" {100 * np.nanpercentile(rel, 75):+.1f} %]")
    print(f"  seasonal amplitude       "
          f"{seasonal_amplitude(daily['ConcDay'].to_numpy(), dec_year):.3f}"
          f"        truth {seasonal_amplitude(expected, dec_year):.3f}")
    print(f"  FN trend (log units/yr)  {slope:+.4f}       truth {truth['trend']:+.4f}")
    print(f"  FN change over record    {100 * change:+.1f} %        "
          f"truth {100 * truth_change:+.1f} %")

    print()
    print("=" * 62)
    if np.sign(change) != np.sign(truth_change):
        print("  TREND SIGN IS WRONG - the engine has regressed.")
    else:
        print(f"  Trend recovered: {100 * change:+.1f} % "
              f"against a truth of {100 * truth_change:+.1f} %.")
    print("=" * 62)


if __name__ == "__main__":
    main()
