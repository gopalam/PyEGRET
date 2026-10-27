"""
Pieces shared by ``run_pairs``, ``run_groups`` and ``run_series``.

Mirrors the set-up code those three R functions repeat, plus R
``surfaceStartEnd()`` and the partial-surface branch of ``estSurfaces()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.model.est_surfaces import build_grid_axes, est_surfaces_on_grid
from pyegret.model.flex_fn import _sliding_segments
from pyegret.analysis.setup_years import start_end
from pyegret.utils.date_utils import to_decimal_year

__all__ = ["surface_start_end", "flow_date_info", "estimate_span_surface",
           "record_dates", "decimal_date"]


def decimal_date(date) -> float:
    """Decimal year of one date. Mirrors R ``decimalDate()`` for a scalar."""
    values = to_decimal_year(pd.Series([pd.Timestamp(date)]))
    return float(values.iloc[0])


def surface_start_end(pa_start: int, pa_long: int, date1, date2) -> dict:
    """Whole periods of analysis covering two dates. Mirrors R ``surfaceStartEnd()``.

    Returns the start of the first period of analysis that begins on or
    after ``date1`` (backing off one period, as R does) and the end of the
    last one that begins on or before ``date2``.

    Args:
        pa_start: First month of the period of analysis.
        pa_long: Length in months.
        date1: First date, usually the first sample.
        date2: Last date, usually the last sample.

    Returns:
        Dict with ``surfaceStart`` and ``surfaceEnd`` Timestamps.
    """
    date1 = pd.Timestamp(date1)
    date2 = pd.Timestamp(date2)
    year1_temp = int(np.trunc(decimal_date(date1))) - 2
    year2_temp = int(np.trunc(decimal_date(date2))) + 2

    year = year1_temp
    for year in range(year1_temp, year2_temp + 1):
        pair = start_end(pa_start, pa_long, year)
        if date1 <= pair["startDate"]:
            break
    surface_start = start_end(pa_start, pa_long, year - 1)["startDate"]

    for year in range(year2_temp, year1_temp - 1, -1):
        pair = start_end(pa_start, pa_long, year)
        if date2 >= pair["startDate"]:
            break
    surface_end = pair["endDate"]
    return {"surfaceStart": surface_start, "surfaceEnd": surface_end}


def record_dates(elist: dict, sample_start_date=None, sample_end_date=None,
                 q_start_date=None, q_end_date=None) -> dict:
    """First and last sample and discharge dates, defaulting to the record's ends."""
    sample_dates = pd.to_datetime(elist["Sample"]["Date"])
    q_dates = pd.to_datetime(elist["Daily"]["Date"])
    dates = {
        "sampleStartDate": sample_dates.iloc[0],
        "sampleEndDate": sample_dates.iloc[-1],
        "QStartDate": q_dates.iloc[0],
        "QEndDate": q_dates.iloc[-1],
    }
    if sample_start_date is not None:
        dates["sampleStartDate"] = pd.Timestamp(sample_start_date)
    if sample_end_date is not None:
        dates["sampleEndDate"] = pd.Timestamp(sample_end_date)
    if q_start_date is not None:
        dates["QStartDate"] = pd.Timestamp(q_start_date)
    if q_end_date is not None:
        dates["QEndDate"] = pd.Timestamp(q_end_date)
    return dates


def flow_date_info(window_side: float, flow_break: bool, q1_end_date,
                   surface_start, surface_end, q_start, q_end) -> pd.DataFrame:
    """The flow-normalization segments ``runGroups`` and ``runSeries`` build.

    Four cases, as in R: stationary (``window_side <= 0``, no break), a
    moving window (``window_side > 0``), a single break in the flow record
    (``flow_break`` with ``q1_end_date``, the last day before it), or both.

    Returns:
        DataFrame with ``flowNormStart``, ``flowNormEnd``, ``flowStart`` and
        ``flowEnd``.
    """
    surface_start = pd.Timestamp(surface_start)
    surface_end = pd.Timestamp(surface_end)
    q_start = pd.Timestamp(q_start)
    q_end = pd.Timestamp(q_end)

    if flow_break:
        if q1_end_date is None:
            raise ValueError("if there is a flow_break you must provide q1_end_date")
        q1_end = pd.Timestamp(q1_end_date)
        q2_start = q1_end + pd.Timedelta(days=1)

    if window_side <= 0 and not flow_break:
        info = pd.DataFrame({
            "flowNormStart": [q_start], "flowNormEnd": [q_end],
            "flowStart": [surface_start], "flowEnd": [surface_end],
        })
    elif window_side > 0 and not flow_break:
        info = _sliding_segments(surface_start, surface_end, q_start, q_end,
                                 int(window_side))
    elif window_side <= 0 and flow_break:
        info = pd.DataFrame({
            "flowNormStart": [q_start, q2_start],
            "flowNormEnd": [q1_end, q_end],
            "flowStart": [surface_start, q2_start],
            "flowEnd": [q1_end, surface_end],
        })
    else:
        first = _sliding_segments(surface_start, q1_end, q_start, q1_end, int(window_side))
        second = _sliding_segments(q2_start, surface_end, q2_start, q_end, int(window_side))
        info = pd.concat([first, second], ignore_index=True)
    return info


def estimate_span_surface(elist: dict, sample: pd.DataFrame, surface_start,
                          surface_end, window_y=7.0, window_q=2.0, window_s=0.5,
                          min_num_obs=100, min_num_uncen=50, edge_adjust=True,
                          verbose=False) -> dict:
    """Fit the surface only for the years between two dates.

    Mirrors R ``estSurfaces(eList, surfaceStart, surfaceEnd, localSample)``:
    the grid is the eList's full grid (built from its Daily), cut down to
    the year steps between the two dates plus one step either side.

    Args:
        elist: eList dict whose Daily defines the grid.
        sample: The samples to fit from (R ``localSample``).
        surface_start: First date to cover.
        surface_end: Last date to cover.
        window_y, window_q, window_s: Starting half-windows.
        min_num_obs, min_num_uncen: Minimum weighted observations.
        edge_adjust: Widen the time window near the sample record's ends.
        verbose: Print progress.

    Returns:
        Dict with ``surfaces``, ``vector_year`` and ``vector_logQ``.
    """
    vector_year, vector_logQ = build_grid_axes(elist)
    low = decimal_date(surface_start)
    high = decimal_date(surface_end)
    inside = np.flatnonzero((vector_year >= low) & (vector_year <= high))
    first = max(inside[0] - 1, 0)
    last = min(inside[-1] + 1, len(vector_year) - 1)
    year_axis = vector_year[first:last + 1]

    surfaces, _ = est_surfaces_on_grid(
        sample, year_axis, vector_logQ,
        window_y=window_y, window_q=window_q, window_s=window_s,
        min_num_obs=min_num_obs, min_num_uncen=min_num_uncen,
        edge_adjust=edge_adjust, verbose=verbose,
    )
    return {"surfaces": surfaces, "vector_year": year_axis, "vector_logQ": vector_logQ}


def add_plus(x: float) -> str:
    """Signed percent with two decimals, as R's internal ``add_plus()``."""
    text = f"{x:+.2f} %"
    if text in ("+0.00 %", "-0.00 %"):
        text = " 0 %"
    return text
