"""
The Quantile-Kendall analysis: trends across the whole flow distribution.

A trend in mean annual discharge is one number describing a river's entire
hydrograph, and it routinely hides the changes that matter. Urbanising
catchments push flood peaks up while baseflow falls; a new reservoir does the
opposite; a warming snowpack can leave the annual mean untouched while moving
every quantile in the year.

The Quantile-Kendall analysis computes a separate trend for *every order
statistic* of the year: the annual minimum daily discharge, the second lowest,
and so on up to the annual maximum. Each gets a Theil-Sen slope in percent per
year and a Mann-Kendall p-value corrected for serial correlation. Plotted
against the normal quantile of non-exceedance probability, the result shows at
a glance whether low flows and high flows are moving together or apart.

The method is Hirsch's, published as a workflow in EGRET's *Daily Streamflow
Trend Analysis* vignette rather than as package functions. This is a
translation of that workflow.

Only complete years are used. A year missing even one day is dropped outright,
because an order statistic is defined by its position in a full ranking and a
short year would shift every rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyegret.utils.convert import as_float_array
from pyegret.flow.trend_tests import (
    mann_kendall,
    theil_sen_slope,
    zhang_prewhitened_test,
)

__all__ = ["sort_annual_flows", "quantile_kendall"]


def _first_julian(month_seq: int) -> int:
    """First Julian day (1850 origin) of the month with this MonthSeq."""
    year = 1850 + (month_seq - 1) // 12
    month = month_seq - 12 * ((month_seq - 1) // 12)
    return int((pd.Timestamp(year=year, month=month, day=1)
                - pd.Timestamp("1850-01-01")).days)


def _drop_one_middle(values: np.ndarray) -> np.ndarray:
    """Remove one value from the middle of a sorted array.

    Mirrors EGRET's ``leapOdd`` / ``leapEven``. A leap year contributes one
    more daily value than a common year, so its sorted vector has to lose one
    element before the years can be stacked into a matrix. Taking it from the
    middle disturbs the tails least, and the tails are what the analysis is
    about. An odd-length vector drops its exact median; an even-length one
    replaces its two central values with their mean.
    """
    n = values.size
    if n % 2 == 1:
        middle = (n - 1) // 2
        return np.concatenate([values[:middle], values[middle + 1:]])
    middle = n // 2 - 1
    merged = (values[middle] + values[middle + 1]) / 2.0
    return np.concatenate([values[:middle], [merged], values[middle + 2:]])


def sort_annual_flows(
    elist: dict,
    pa_start: int = 10,
    pa_long: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack each complete year's sorted daily discharges into a matrix.

    Mirrors ``makeSortQ`` from EGRET's streamflow trend vignette.

    Args:
        elist: eList dict with a Daily DataFrame.
        pa_start: First month of the period of analysis. Default 10.
        pa_long: Length of the period in months. Default 12.

    Returns:
        ``(sorted_flows, years)`` where ``sorted_flows`` has one column per
        complete year and one row per rank, ascending, and ``years`` holds
        the midpoint decimal year of each column.

    Raises:
        ValueError: If Daily is missing, or no complete year is available.
    """
    daily = elist["Daily"]
    if daily is None or daily.empty:
        raise ValueError("eList has no Daily data.")

    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    dec_year = daily["DecYear"].to_numpy(dtype=float)
    q = as_float_array(daily["Q"])

    first, last = month_seq[0], month_seq[-1]
    starts = np.arange(pa_start, last + 1, 12)
    ends = starts + pa_long - 1
    keep = (ends >= first) & (starts <= last)

    columns, midpoints = [], []
    for start, end in zip(starts[keep], ends[keep]):
        mask = (month_seq >= start) & (month_seq <= end)
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        # A complete period spans exactly the days between the first of its
        # first month and the first of the month after its last.
        expected = _first_julian(int(end) + 1) - _first_julian(int(start))
        if n_days != expected:
            continue
        values = q[mask]
        if not np.all(np.isfinite(values)):
            continue
        columns.append(np.sort(values))
        midpoints.append((dec_year[mask][0] + dec_year[mask][-1]) / 2.0)

    if not columns:
        raise ValueError(
            "No complete years found. The Quantile-Kendall analysis needs "
            "years with no missing days."
        )

    shortest = min(column.size for column in columns)
    trimmed = []
    for column in columns:
        while column.size > shortest:
            column = _drop_one_middle(column)
        trimmed.append(column)

    return np.column_stack(trimmed), np.asarray(midpoints, dtype=float)


def quantile_kendall(
    elist: dict,
    pa_start: int = 10,
    pa_long: int = 12,
    adjust_for_serial_correlation: bool = True,
) -> pd.DataFrame:
    """Trend in every order statistic of the annual flow distribution.

    Mirrors ``trendSortQ`` from EGRET's streamflow trend vignette.

    Slopes are fit to log discharge, so they express proportional change and
    can be read as percent per year regardless of the size of the river.

    Args:
        elist: eList dict with a Daily DataFrame.
        pa_start: First month of the period of analysis. Default 10 (water
            year). Use ``pa_start=3, pa_long=4`` for a March-June melt season.
        pa_long: Length of the period in months. Default 12.
        adjust_for_serial_correlation: Also compute Zhang-prewhitened
            p-values. This is the column EGRET plots. Default True; setting
            False is markedly faster and useful when screening many sites.

    Returns:
        DataFrame with one row per rank:

        - ``rank`` - 1 is the annual minimum daily discharge
        - ``freq`` - non-exceedance probability, ``rank / (n + 1)``
        - ``z`` - standard normal quantile of ``freq``, the plotting position
        - ``slope_log`` - Theil-Sen slope of log discharge, per year
        - ``slope_pct`` - the same as percent per year
        - ``tau`` - Kendall's tau
        - ``p_value`` - Mann-Kendall, unadjusted
        - ``p_value_adj`` - Mann-Kendall after Zhang prewhitening
        - ``autocorrelation`` - lag-1 value that drove the prewhitening

    Raises:
        ValueError: If no complete year is available.
    """
    sorted_flows, years = sort_annual_flows(elist, pa_start, pa_long)
    n_ranks, n_years = sorted_flows.shape

    if n_years < 4:
        raise ValueError(
            f"Only {n_years} complete years available; a trend test needs at least 4."
        )

    rows = []
    for rank in range(n_ranks):
        log_q = np.log(sorted_flows[rank, :])
        slope = theil_sen_slope(years, log_q)
        result = mann_kendall(years, log_q)

        row = {
            "rank": rank + 1,
            "freq": (rank + 1) / (n_ranks + 1),
            "slope_log": slope,
            "slope_pct": 100.0 * (np.exp(slope) - 1.0) if np.isfinite(slope) else np.nan,
            "tau": result["tau"],
            "p_value": result["p_value"],
        }
        if adjust_for_serial_correlation:
            adjusted = zhang_prewhitened_test(log_q, years)
            row["p_value_adj"] = adjusted["p_value"]
            row["autocorrelation"] = adjusted["autocorrelation"]
        else:
            row["p_value_adj"] = np.nan
            row["autocorrelation"] = np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["z"] = norm.ppf(frame["freq"])
    frame.attrs["n_years"] = n_years
    frame.attrs["year_range"] = (float(years[0]), float(years[-1]))
    return frame[[
        "rank", "freq", "z", "slope_log", "slope_pct", "tau",
        "p_value", "p_value_adj", "autocorrelation",
    ]]
