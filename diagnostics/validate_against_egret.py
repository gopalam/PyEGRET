"""
Report how closely PyEGRET reproduces R EGRET on the Choptank reference.

The same comparison runs as assertions in ``tests/test_egret_reference.py``;
this prints the numbers, which is what you want when changing the optimiser,
the kernel, or the grid and you need to see what moved.

Run:  python diagnostics/validate_against_egret.py
Needs ``rdata`` (pip install rdata). No R installation, no network.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import rdata                                                        # noqa: E402

from pyegret.data.populate import populate_sample_columns             # noqa: E402
from pyegret.elist import make_elist                         # noqa: E402
from pyegret.model.est_daily import est_daily_from_surfaces           # noqa: E402
from pyegret.model.est_surfaces import est_surfaces                   # noqa: E402
from pyegret.testing import annual_mean_by_calendar_year              # noqa: E402

REFERENCE = os.path.join(_ROOT, "tests", "reference", "Choptank_eList.RData")

# EGRET's stored surface for this dataset predates edgeAdjust; with it on, the
# interior still matches to machine precision but the outer decade does not.
EDGE_ADJUST = False


def main() -> None:
    converted = rdata.conversion.convert(rdata.parser.parse_file(REFERENCE))
    elist_r = converted["Choptank_eList"]
    info, daily_r, sample_r = elist_r["INFO"], elist_r["Daily"], elist_r["Sample"]
    for frame in (info, daily_r, sample_r):
        frame.columns = [str(c) for c in frame.columns]

    keys = ["windowY", "windowQ", "windowS", "minNumObs", "minNumUncen",
            "nVectorYear", "nVectorLogQ", "stepYear"]
    cfg = {k: float(info[k].iloc[0]) for k in keys}
    print("EGRET settings from the reference INFO row:")
    print("   windowY %g  windowQ %g  windowS %g  minNumObs %d  minNumUncen %d"
          % (cfg["windowY"], cfg["windowQ"], cfg["windowS"],
             cfg["minNumObs"], cfg["minNumUncen"]))
    print("   grid %d x %d, stepYear %g, edge_adjust=%s"
          % (cfg["nVectorYear"], cfg["nVectorLogQ"], cfg["stepYear"], EDGE_ADJUST))

    daily = daily_r[["Date", "Q", "LogQ", "DecYear", "Month", "Day",
                     "MonthSeq", "Julian"]].copy()
    sample = populate_sample_columns(
        sample_r[["Date", "ConcLow", "ConcHigh", "Uncen", "DecYear", "Month",
                  "Day", "MonthSeq", "Julian", "SinDY", "CosDY", "Q", "LogQ"]].copy()
    )
    elist = make_elist(INFO=pd.DataFrame([{"shortName": "Choptank River"}]),
                  Daily=daily, Sample=sample)

    started = time.time()
    elist["surfaces"] = est_surfaces(
        elist, window_y=cfg["windowY"], window_q=cfg["windowQ"],
        window_s=cfg["windowS"], min_num_obs=int(cfg["minNumObs"]),
        min_num_uncen=int(cfg["minNumUncen"]), edge_adjust=EDGE_ADJUST,
        n_vector_logQ=int(cfg["nVectorLogQ"]), verbose=False,
    )
    elapsed = time.time() - started
    elist["Daily"] = est_daily_from_surfaces(elist)

    surfaces_r = np.transpose(elist_r["surfaces"], (1, 0, 2))
    print("\nPyEGRET fitted %s in %.1f s\n" % (str(elist["surfaces"].shape), elapsed))

    print("Surface (7,406 grid points):")
    for index, name in [(0, "yHat"), (1, "SE"), (2, "ConcHat")]:
        ours = elist["surfaces"][:, :, index]
        theirs = surfaces_r[:, :, index]
        absdiff = np.abs(ours - theirs)
        share = float(np.sum((absdiff <= 1e-4))) / absdiff.size
        print("   %-8s %6.2f %% of cells agree to 1e-4    worst cell %.6f"
              % (name, 100 * share, absdiff.max()))

    print("\nDaily outputs (11,688 days; for FNConc / FNFlux, February 28 and 29")
    print("are left out: current EGRET pools their reference flows, and this")
    print("reference was fitted before it did. The remaining worst days are")
    print("December 30-31, for the same reason - the reference predates EGRET's")
    print("leap-adjusted day numbering; Chain Bridge, fitted by current EGRET,")
    print("agrees on every day; see tests/test_egret_trend_reference.py):")
    day = elist["Daily"]["Day"].to_numpy()
    for column in ["ConcDay", "FluxDay", "FNConc", "FNFlux"]:
        ours = elist["Daily"][column].to_numpy(dtype=float)
        theirs = daily_r[column].to_numpy(dtype=float)
        if column.startswith("FN"):
            keep = (day != 59) & (day != 60)
            ours = ours[keep]
            theirs = theirs[keep]
        rel = np.abs((ours - theirs) / theirs)
        print("   %-8s median %.2e   99th pct %.2e   worst %.2e"
              % (column, np.median(rel), np.percentile(rel, 99), rel.max()))

    def annual(frame):
        series = annual_mean_by_calendar_year(frame, "FNConc")
        return 100 * float(series.iloc[-1] / series.iloc[0] - 1.0)

    print("\nFlow-normalized concentration change over the record:")
    print("   PyEGRET %+.1f %%     EGRET %+.1f %%"
          % (annual(elist["Daily"]), annual(daily_r)))


if __name__ == "__main__":
    main()
