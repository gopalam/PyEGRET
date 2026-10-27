"""
Annual results for a period of analysis. Mirrors R ``setupYears()``,
``setPA()`` and ``startEnd()``.

The period of analysis (PA) is a start month ``paStart`` and a length
``paLong`` in months. The default, 10 and 12, is the water year (October
through September). A year is used only if the Daily record covers all of
its months, and its WRTDS values only if more than 90 % of its days have a
``ConcDay`` estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["setup_years", "set_pa", "start_end", "get_pa"]

_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def get_pa(elist: dict, pa_start: int | None = None, pa_long: int | None = None) -> tuple:
    """The period of analysis to use: explicit values, else INFO, else 10/12.

    R functions read ``INFO$paStart`` / ``INFO$paLong`` when set by
    ``setPA()`` and fall back to the water year; this does the same.

    Returns:
        ``(pa_start, pa_long)``.
    """
    info = elist.get("INFO")
    if pa_start is None:
        pa_start = 10
        if info is not None and "paStart" in info.columns:
            pa_start = int(info["paStart"].iloc[0])
    if pa_long is None:
        pa_long = 12
        if info is not None and "paLong" in info.columns:
            pa_long = int(info["paLong"].iloc[0])
    return pa_start, pa_long


def setup_years(daily, pa_long: int = 12, pa_start: int = 10) -> pd.DataFrame:
    """Annual means of the daily WRTDS results. Mirrors R ``setupYears()``.

    Args:
        daily: Daily DataFrame (R's ``localDaily``). An eList dict is also
            accepted, in which case its Daily is used.
        pa_long: Length of the period of analysis in months. Default 12.
        pa_start: First month of the period of analysis. Default 10.

    Returns:
        DataFrame with one row per period of analysis fully inside the
        record: ``DecYear``, ``Q``, ``Conc``, ``Flux``, ``FNConc``,
        ``FNFlux``, plus ``GenConc`` / ``GenFlux`` when the Daily has them
        (WRTDS-Kalman), and ``PeriodLong`` / ``PeriodStart``. Conc, Flux and
        the rest are NaN for a year with 90 % or fewer of its days estimated.
        A Daily without model columns gives only DecYear and Q.

    Raises:
        ValueError: If the Daily record has gaps; EGRET cannot use it.
    """
    if isinstance(daily, dict):
        daily = daily["Daily"]

    dates = pd.to_datetime(daily["Date"])
    dates = dates.reset_index(drop=True)
    n_days = len(daily)
    span = (dates.iloc[-1] - dates.iloc[0]).days
    if n_days - 1 != span:
        raise ValueError("Daily dataframe cannot have gaps in the data.")

    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    month = daily["Month"].to_numpy(dtype=int)
    dec_year = daily["DecYear"].to_numpy(dtype=float)
    q = daily["Q"].to_numpy(dtype=float)

    has_model = "ConcDay" in daily.columns
    wrtds_k = "GenConc" in daily.columns and "GenFlux" in daily.columns
    columns = {}
    for name in ("ConcDay", "FluxDay", "FNConc", "FNFlux", "GenConc", "GenFlux"):
        if name in daily.columns:
            columns[name] = daily[name].to_numpy(dtype=float)
        else:
            columns[name] = np.full(n_days, np.nan)

    first_month_seq = month_seq[0]
    last_month_seq = month_seq[-1]
    starts = np.arange(pa_start, last_month_seq + 1, 12)
    ends = starts + pa_long - 1
    inside = (starts >= first_month_seq) & (ends <= last_month_seq)
    starts = starts[inside]
    ends = ends[inside]

    n_years = len(starts)
    out = {}
    for name in ("DecYear", "Q", "Conc", "Flux", "FNConc", "FNFlux", "GenConc", "GenFlux"):
        out[name] = np.full(n_years, np.nan)

    for i in range(n_years):
        in_start = month_seq == starts[i]
        in_stop = month_seq == ends[i]
        first_day = dates[in_start].iloc[0]
        last_month = month[in_stop][0]
        last_year = int(np.floor(dec_year[in_stop][0]))
        last_day = _DAYS_IN_MONTH[last_month - 1]
        if last_month == 2 and _is_leap(last_year):
            last_day = 29
        last_date = pd.Timestamp(year=last_year, month=last_month, day=last_day)
        days_in_year = (last_date - first_day).days + 1

        in_year = (month_seq >= starts[i]) & (month_seq <= ends[i])
        if not np.any(in_year):
            continue

        out["DecYear"][i] = np.mean(dec_year[in_year])
        out["Q"][i] = np.mean(q[in_year])

        n_estimated = np.sum(~np.isnan(columns["ConcDay"][in_year]))
        good = n_estimated / days_in_year > 0.9
        if good:
            out["Conc"][i] = np.nanmean(columns["ConcDay"][in_year])
            out["Flux"][i] = np.nanmean(columns["FluxDay"][in_year])
            out["FNConc"][i] = np.nanmean(columns["FNConc"][in_year])
            out["FNFlux"][i] = np.nanmean(columns["FNFlux"][in_year])
            if wrtds_k:
                out["GenConc"][i] = np.nanmean(columns["GenConc"][in_year])
                out["GenFlux"][i] = np.nanmean(columns["GenFlux"][in_year])

    names = ["DecYear", "Q"]
    if has_model:
        names = names + ["Conc", "Flux", "FNConc", "FNFlux"]
    if wrtds_k:
        names = names + ["GenConc", "GenFlux"]
    result = pd.DataFrame({name: out[name] for name in names})
    result["PeriodLong"] = pa_long
    result["PeriodStart"] = pa_start
    result = result[~np.isnan(result["DecYear"].to_numpy())]
    result = result.reset_index(drop=True)
    return result


def set_pa(elist: dict, pa_start: int = 10, pa_long: int = 12, window: float = 20) -> dict:
    """Record the period of analysis in INFO. Mirrors R ``setPA()``.

    Functions that take a period of analysis read it from INFO when it is
    not passed explicitly.

    Args:
        elist: eList dict.
        pa_start: First month of the period of analysis. Default 10.
        pa_long: Length in months. Default 12.
        window: Smoothing half-window in years, used by the flow-history
            functions. Default 20.

    Returns:
        A new eList whose INFO has ``paStart``, ``paLong`` and ``window``.
        The input eList is not changed.
    """
    elist = dict(elist)
    info = elist["INFO"].copy()
    info["paStart"] = pa_start
    info["paLong"] = pa_long
    info["window"] = window
    elist["INFO"] = info
    return elist


def start_end(pa_start: int, pa_long: int, year: int) -> dict:
    """First and last date of one period of analysis. Mirrors R ``startEnd()``.

    The period is labelled by the year it ends in: water year 2001 runs
    from 2000-10-01 to 2001-09-30.

    Note: R's table of month lengths gives August 30 days, so a period
    ending in August ends on the 30th in R. This uses the real calendar.

    Args:
        pa_start: First month of the period.
        pa_long: Length in months.
        year: The year the period ends in.

    Returns:
        Dict with ``startDate`` and ``endDate`` (pandas Timestamps).
    """
    end_month = pa_start + pa_long - 1
    start_year = year
    if end_month > 12:
        start_year = year - 1
        end_month = end_month - 12
    start_date = pd.Timestamp(year=start_year, month=pa_start, day=1)
    end_day = _DAYS_IN_MONTH[end_month - 1]
    if end_month == 2 and _is_leap(year):
        end_day = 29
    end_date = pd.Timestamp(year=year, month=end_month, day=end_day)
    return {"startDate": start_date, "endDate": end_date}
