"""
End-to-end WRTDS workflow on the Choptank River, the dataset EGRET ships.

Runs offline: the Daily and Sample come from EGRET's own Choptank_eList,
read from tests/reference with the ``rdata`` package (pip install rdata).
To use your own data instead, build ``daily`` and ``sample`` with
``read_user_daily`` / ``read_user_sample`` (or the USGS readers) and skip
``load_choptank``.

Run from anywhere:  python examples/choptank_wrtds.py
Figures are written to examples/output/.
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # import pyegret from this folder; nothing installed

from pyegret import merge_report, model_estimation, wrtds_kalman  # noqa: E402
from pyegret.analysis import (  # noqa: E402
    error_stats, flux_bias_stat, run_pairs, run_series, set_pa, table_change, table_results,
)
from pyegret.data.populate import populate_date_columns, populate_sample_columns  # noqa: E402
import pyegret.plots as plots  # noqa: E402


def load_choptank():
    """Daily, Sample and INFO from EGRET's Choptank_eList.RData."""
    import rdata

    path = os.path.join(ROOT, "tests", "reference", "Choptank_eList.RData")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = rdata.conversion.convert(rdata.parser.parse_file(path))
    elist_r = converted["Choptank_eList"]

    # rdata gives R Dates as days since 1970-01-01.
    daily = elist_r["Daily"][["Date", "Q"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], unit="D", origin="1970-01-01")
    daily["LogQ"] = np.log(daily["Q"].to_numpy(dtype=float))
    daily = populate_date_columns(daily)

    sample = elist_r["Sample"][["Date", "ConcLow", "ConcHigh", "Uncen"]].copy()
    sample["Date"] = pd.to_datetime(sample["Date"], unit="D", origin="1970-01-01")
    sample["ConcLow"] = sample["ConcLow"].fillna(0.0)
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)

    info = pd.DataFrame([{
        "shortName": "Choptank River",
        "paramShortName": "Inorganic nitrogen (nitrate and nitrite)",
        "drainSqKm": 292.67,
    }])
    return info, daily, sample


def save(fig, name):
    out = os.path.join(HERE, "output")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=120)
    plt.close(fig)


def main():
    info, daily, sample = load_choptank()
    elist = merge_report(info, daily, sample)
    elist = set_pa(elist, pa_start=10, pa_long=12)

    fig, _ = plots.multi_plot_data_overview(elist)
    save(fig, "01_data_overview")

    elist = model_estimation(elist)
    error_stats(elist)
    print(flux_bias_stat(elist["Sample"]))

    table_results(elist)
    table_change(elist, year_points=[1985, 1995, 2005, 2010])

    fig, _ = plots.plot_conc_hist(elist)
    save(fig, "02_conc_hist")
    fig, _ = plots.plot_flux_hist(elist)
    save(fig, "03_flux_hist")
    # As in EGRET's vignette, choose the contour levels: the default spans
    # the whole surface, including extrapolated corners the river never visits.
    levels = np.linspace(0.0, 2.0, 11)
    fig, _ = plots.plot_contours(elist, 1980, 2011, q_bottom=0.5, q_top=20,
                                 contour_levels=levels)
    save(fig, "04_contours")
    fig, _ = plots.plot_diff_contours(elist, 1985, 2010, q_bottom=0.5, q_top=20, max_diff=0.6)
    save(fig, "05_diff_contours")
    fig, _ = plots.flux_bias_multi(elist)
    save(fig, "06_flux_bias_multi")

    # Generalized flow normalization and the trend decomposition.
    pairs = run_pairs(elist, 1985, 2010, window_side=7)
    print(pairs["pairResults"])
    series = run_series(elist, window_side=7, verbose=False)
    fig, _ = plots.plot_conc_hist(series)
    save(fig, "07_conc_hist_gfn")

    # WRTDS-Kalman.
    kalman = wrtds_kalman(elist, niter=200, seed=1)
    fig, _ = plots.plot_wrtds_kalman(kalman, side_by_side=True)
    save(fig, "08_wrtds_kalman")
    fig, _ = plots.plot_conc_time_daily(kalman, 2000, 2002)
    save(fig, "09_conc_time_daily")


if __name__ == "__main__":
    main()
