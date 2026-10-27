"""
Change between two groups of years, split into its two causes.

Mirrors R ``runGroups()`` and ``printGroups()``. The decomposition is the
one :func:`~pyegret.analysis.run_pairs.run_pairs` uses (``x11``, ``x22``,
``x10``, ``x20``; CQTC and QTC), but each "year" is the average of a group of
years, and one surface is fitted over the whole span.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.analysis.trend_setup import (
    add_plus,
    estimate_span_surface,
    flow_date_info,
    record_dates,
    surface_start_end,
)
from pyegret.model.flex_fn import flex_fn
from pyegret.model.wall import stitch_surfaces
from pyegret.utils.season import set_season_label

__all__ = ["run_groups", "print_groups", "span_surface_elist"]

KG_PER_DAY_TO_MILLION_KG_PER_YEAR = 0.00036525


def _between(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = pd.to_datetime(frame["Date"])
    inside = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    subset = frame[inside.to_numpy()]
    subset = subset.reset_index(drop=True)
    return subset


def _group_mean(annual: pd.DataFrame, first: float, last: float, column: str) -> float:
    """Mean of an annual column over the years whose DecYear is in [first, last]."""
    dec_year = annual["DecYear"].to_numpy(dtype=float)
    in_group = (dec_year >= first) & (dec_year <= last)
    values = annual[column].to_numpy(dtype=float)
    values = values[in_group]
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def span_surface_elist(elist: dict, window_side, surface_start=None, surface_end=None,
                       flow_break=False, q1_end_date=None, q_start_date=None,
                       q_end_date=None, wall=False, old_surface=False,
                       sample1_end_date=None, sample_start_date=None,
                       sample_end_date=None, pa_start=None, pa_long=None,
                       fract_min=0.75, min_num_obs=100, min_num_uncen=50,
                       window_y=7, window_q=2, window_s=0.5, edge_adjust=True,
                       verbose=True) -> dict:
    """The set-up R's runGroups and runSeries share.

    Trims Daily and Sample to the requested dates, fits the surface over
    ``surface_start``-``surface_end`` (with a wall if asked), and builds the
    flow-normalization segments.

    Returns:
        Dict with ``elist`` (the trimmed eList carrying the fitted surface and
        its axes), ``dateInfo``, ``dates`` (sample and Q date limits),
        ``surfaceStart``, ``surfaceEnd``, ``paStart``, ``paLong``,
        ``minNumObs``, ``minNumUncen`` and ``sample1EndDate``.
    """
    pa_start, pa_long = get_pa(elist, pa_start, pa_long)
    dates = record_dates(elist, sample_start_date, sample_end_date,
                         q_start_date, q_end_date)
    daily = _between(elist["Daily"], dates["QStartDate"], dates["QEndDate"])
    sample = _between(elist["Sample"], dates["sampleStartDate"], dates["sampleEndDate"])

    span = surface_start_end(pa_start, pa_long, dates["sampleStartDate"],
                             dates["sampleEndDate"])
    if surface_start is None:
        surface_start = span["surfaceStart"]
    if surface_end is None:
        surface_end = span["surfaceEnd"]
    surface_start = pd.Timestamp(surface_start)
    surface_end = pd.Timestamp(surface_end)

    local = dict(elist)
    local["Daily"] = daily
    local["Sample"] = sample
    info = local["INFO"].copy()
    info["paStart"] = pa_start
    info["paLong"] = pa_long
    local["INFO"] = info

    if wall:
        if sample1_end_date is None:
            raise ValueError("if there is a wall, the user must specify sample1_end_date")
        surfaces = stitch_surfaces(
            local, sample1_end_date, surface_start=surface_start, surface_end=surface_end,
            window_y=window_y, window_q=window_q, window_s=window_s,
            min_num_obs=min_num_obs, min_num_uncen=min_num_uncen,
            fract_min=fract_min, edge_adjust=edge_adjust, verbose=verbose,
        )
        local["surfaces"] = surfaces
    else:
        if old_surface and elist.get("surfaces") is None:
            old_surface = False
        if not old_surface:
            fitted = estimate_span_surface(
                local, sample, surface_start, surface_end,
                window_y=window_y, window_q=window_q, window_s=window_s,
                min_num_obs=min_num_obs, min_num_uncen=min_num_uncen,
                edge_adjust=edge_adjust, verbose=verbose,
            )
            local["surfaces"] = fitted["surfaces"]
            local["vector_year"] = fitted["vector_year"]
            local["vector_logQ"] = fitted["vector_logQ"]

    date_info = flow_date_info(window_side, flow_break, q1_end_date, surface_start,
                               surface_end, dates["QStartDate"], dates["QEndDate"])
    return {
        "elist": local, "dateInfo": date_info, "dates": dates,
        "surfaceStart": surface_start, "surfaceEnd": surface_end,
        "paStart": pa_start, "paLong": pa_long, "oldSurface": old_surface,
        "sample1EndDate": sample1_end_date,
    }


def run_groups(
    elist: dict,
    window_side: float,
    group1_first_year: float,
    group1_last_year: float,
    group2_first_year: float,
    group2_last_year: float,
    surface_start=None,
    surface_end=None,
    flow_break: bool = False,
    q1_end_date=None,
    q_start_date=None,
    q_end_date=None,
    wall: bool = False,
    old_surface: bool = False,
    fract_min: float = 0.75,
    sample1_end_date=None,
    sample_start_date=None,
    sample_end_date=None,
    pa_start: int | None = None,
    pa_long: int | None = None,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    window_y: float = 7,
    window_q: float = 2,
    window_s: float = 0.5,
    edge_adjust: bool = True,
    verbose: bool = True,
) -> dict:
    """Change between the averages of two groups of years. Mirrors R ``runGroups()``.

    A year belongs to a group when the mean decimal year of its period of
    analysis lies between the group's first and last year, inclusive. That is
    R's rule: for water years, whose mean decimal year is about Y + 0.24, a
    group 1990-1995 holds water years 1990 through 1994.

    Args:
        elist: eList dict with Daily and Sample.
        window_side: Half-width in years of the flow-normalization window. 0
            is stationary. Required, as in R.
        group1_first_year, group1_last_year: The first group.
        group2_first_year, group2_last_year: The second group.
        surface_start, surface_end: Span of the fitted surface. Default:
            whole periods of analysis covering the samples.
        flow_break, q1_end_date: An abrupt change in the discharge record,
            and the last day before it.
        q_start_date, q_end_date: Limit the discharge record used.
        wall, sample1_end_date: Fit separate surfaces either side of
            ``sample1_end_date`` (see :func:`~pyegret.model.wall.stitch_surfaces`).
        old_surface: Use the eList's existing surface.
        fract_min: Used with the wall to relax the minimum observations.
        sample_start_date, sample_end_date: Limit the samples used.
        pa_start, pa_long: Period of analysis. Default: INFO, else 10/12.
        min_num_obs, min_num_uncen: Minimum weighted observations.
        window_y, window_q, window_s: Starting half-windows.
        edge_adjust: Widen the time window near the record ends.
        verbose: Print the summary R prints.

    Returns:
        Dict with ``"groupResults"`` (rows ``Conc`` in mg/L and ``Flux`` in
        10^6 kg/yr; columns ``TotalChange``, ``CQTC``, ``QTC``, ``x10``,
        ``x11``, ``x20``, ``x22``), ``"groupInfo"``, ``"dateInfo"``,
        ``"SampleBlocks"`` and ``"Other"``.
    """
    setup = span_surface_elist(
        elist, window_side, surface_start, surface_end, flow_break, q1_end_date,
        q_start_date, q_end_date, wall, old_surface, sample1_end_date,
        sample_start_date, sample_end_date, pa_start, pa_long, fract_min,
        min_num_obs, min_num_uncen, window_y, window_q, window_s, edge_adjust,
        verbose,
    )
    local = setup["elist"]
    pa_start = setup["paStart"]
    pa_long = setup["paLong"]
    dates = setup["dates"]

    stationary_info = pd.DataFrame({
        "flowNormStart": [dates["QStartDate"]], "flowNormEnd": [dates["QEndDate"]],
        "flowStart": [setup["surfaceStart"]], "flowEnd": [setup["surfaceEnd"]],
    })
    daily_flex = flex_fn(local, date_info=setup["dateInfo"])
    daily_stationary = flex_fn(local, date_info=stationary_info)

    ann_flex = setup_years(daily_flex, pa_long=pa_long, pa_start=pa_start)
    ann_sta = setup_years(daily_stationary, pa_long=pa_long, pa_start=pa_start)

    c11 = _group_mean(ann_flex, group1_first_year, group1_last_year, "FNConc")
    f11 = _group_mean(ann_flex, group1_first_year, group1_last_year, "FNFlux")
    c22 = _group_mean(ann_flex, group2_first_year, group2_last_year, "FNConc")
    f22 = _group_mean(ann_flex, group2_first_year, group2_last_year, "FNFlux")
    c10 = _group_mean(ann_sta, group1_first_year, group1_last_year, "FNConc")
    f10 = _group_mean(ann_sta, group1_first_year, group1_last_year, "FNFlux")
    c20 = _group_mean(ann_sta, group2_first_year, group2_last_year, "FNConc")
    f20 = _group_mean(ann_sta, group2_first_year, group2_last_year, "FNFlux")

    c_total = c22 - c11
    c_cqtc = c20 - c10
    c_qtc = c_total - c_cqtc
    f_total = f22 - f11
    f_cqtc = f20 - f10
    f_qtc = f_total - f_cqtc

    columns = ["TotalChange", "CQTC", "QTC", "x10", "x11", "x20", "x22"]
    conc_row = [c_total, c_cqtc, c_qtc, c10, c11, c20, c22]
    flux_row = []
    for value in [f_total, f_cqtc, f_qtc, f10, f11, f20, f22]:
        flux_row.append(KG_PER_DAY_TO_MILLION_KG_PER_YEAR * value)
    group_results = pd.DataFrame([conc_row, flux_row], index=["Conc", "Flux"],
                                 columns=columns)

    if wall:
        sample1_end = pd.Timestamp(sample1_end_date)
        blocks = {"sample1StartDate": dates["sampleStartDate"],
                  "sample1EndDate": sample1_end,
                  "sample2StartDate": sample1_end + pd.Timedelta(days=1),
                  "sample2EndDate": dates["sampleEndDate"]}
    else:
        blocks = {"sample1StartDate": dates["sampleStartDate"],
                  "sample1EndDate": dates["sampleEndDate"],
                  "sample2StartDate": dates["sampleStartDate"],
                  "sample2EndDate": dates["sampleEndDate"]}
    blocks["surfaceStart"] = setup["surfaceStart"]
    blocks["surfaceEnd"] = setup["surfaceEnd"]

    info = elist["INFO"]
    result = {
        "groupResults": group_results,
        "groupInfo": {"paStart": pa_start, "paLong": pa_long,
                      "group1firstYear": group1_first_year,
                      "group1lastYear": group1_last_year,
                      "group2firstYear": group2_first_year,
                      "group2lastYear": group2_last_year},
        "dateInfo": setup["dateInfo"],
        "SampleBlocks": blocks,
        "Other": {"minNumObs": min_num_obs, "minNumUncen": min_num_uncen,
                  "fractMin": fract_min, "windowY": window_y, "windowQ": window_q,
                  "windowS": window_s, "wall": wall, "edgeAdjust": edge_adjust,
                  "QStartDate": dates["QStartDate"], "QEndDate": dates["QEndDate"],
                  "siteName": str(info["shortName"].iloc[0]) if "shortName" in info.columns else "",
                  "paramShortName": (str(info["paramShortName"].iloc[0])
                                     if "paramShortName" in info.columns else "")},
    }
    if verbose:
        print_groups(result)
    return result


def print_groups(groups: dict) -> None:
    """Print a run_groups result as R does. Mirrors R ``printGroups()``."""
    other = groups["Other"]
    info = groups["groupInfo"]
    table = groups["groupResults"]
    print(f"\n   {other['siteName']} \n   {other['paramShortName']}")
    print(f"\n   {set_season_label(info['paStart'], info['paLong'])} ")
    if other["wall"]:
        print("\n Sample data set was partitioned with a wall right after ",
              groups["SampleBlocks"]["sample1EndDate"].date())
    print(f"\n Change estimates for\n average of {info['group2firstYear']}  through "
          f"{info['group2lastYear']}  minus average of {info['group1firstYear']}  through "
          f"{info['group1lastYear']} ")
    for label, name, unit in (("Conc", "concentration", "mg/L"),
                              ("Flux", "flux", "million kg/year")):
        x11 = table.loc[label, "x11"]
        x22 = table.loc[label, "x22"]
        print(f"\n For {name}: total change is  {table.loc[label, 'TotalChange']:.3g} {unit}")
        print(f" expressed as Percent Change is  {add_plus(100 * (x22 - x11) / x11)}")
        print(f"\n Concentration v. Q Trend Component  {add_plus(100 * table.loc[label, 'CQTC'] / x11)}")
        print(f"       Q Trend Component             {add_plus(100 * table.loc[label, 'QTC'] / x11)} \n")
    print(table.to_string(float_format=lambda v: f"{v:.2g}"))
