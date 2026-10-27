"""
Write EGRET's Choptank River data as the two plain CSV files the tutorials use.

The data are the Daily and Sample of ``Choptank_eList``, the example dataset
that ships with R EGRET (USGS site 01491000, Choptank River near Greensboro,
Maryland; nitrate plus nitrite, parameter 00631; water years 1980-2011). They
are written in the format ``read_user_daily`` and ``read_user_sample`` read,
which is also the format the tutorials ask users to put their own data in:

- ``tutorials/data/choptank_daily.csv``: date, discharge in cubic feet per
  second (as USGS reports it);
- ``tutorials/data/choptank_sample.csv``: date, remark, value in mg/L as N,
  with ``<`` in the remark column for a value below the reporting limit.

Needs the ``rdata`` package.  Run:  python diagnostics/export_choptank_csv.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tutorials", "data")
CFS_PER_CMS = 35.314667


def main() -> int:
    import rdata

    path = os.path.join(ROOT, "tests", "reference", "Choptank_eList.RData")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)
    elist = converted["Choptank_eList"]
    daily = elist["Daily"]
    sample = elist["Sample"]

    # rdata gives R Dates as days since 1970-01-01.
    daily_dates = pd.to_datetime(daily["Date"], unit="D", origin="1970-01-01")
    sample_dates = pd.to_datetime(sample["Date"], unit="D", origin="1970-01-01")

    # The file stored discharge in m3/s converted from whole or two-decimal cfs;
    # convert back and round to the precision USGS publishes.
    q_cfs = daily["Q"].to_numpy(dtype=float) * CFS_PER_CMS
    q_cfs = np.where(q_cfs >= 10.0, np.round(q_cfs, 0), np.round(q_cfs, 2))

    os.makedirs(OUT, exist_ok=True)
    daily_out = pd.DataFrame({
        "date": daily_dates.dt.strftime("%Y-%m-%d"),
        "discharge_cfs": q_cfs,
    })
    daily_out.to_csv(os.path.join(OUT, "choptank_daily.csv"), index=False, float_format="%g")

    uncen = sample["Uncen"].to_numpy(dtype=int)
    remark = np.where(uncen == 1, "", "<")
    value = sample["ConcHigh"].to_numpy(dtype=float)
    sample_out = pd.DataFrame({
        "date": sample_dates.dt.strftime("%Y-%m-%d"),
        "remark": remark,
        "nitrate_nitrite_mgL": value,
    })
    sample_out.to_csv(os.path.join(OUT, "choptank_sample.csv"), index=False)

    print(f"{len(daily_out)} days and {len(sample_out)} samples written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
