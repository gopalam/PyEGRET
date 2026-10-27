"""
Change between two years, split into its two causes. Mirrors R ``runPairs()``.

Comparing flow-normalized concentration between two years tells you *that*
something changed, not what. A watershed that started leaking more nitrogen
and a wetter decade give the same number. ``run_pairs`` separates them.

It fits two surfaces, one for each year, each from the samples around it, and
evaluates each one twice:

============  ==========================================================
term          surface x flow distribution
============  ==========================================================
``x11``       year 1's surface, year 1's flow-normalization window
``x22``       year 2's surface, year 2's flow-normalization window
``x10``       year 1's surface, the whole discharge record
``x20``       year 2's surface, the whole discharge record
============  ==========================================================

- **TotalChange** = ``x22 - x11``
- **CQTC** = ``x20 - x10``, the concentration-versus-discharge trend
  component: same flows, different surfaces, so this is the watershed.
- **QTC** = TotalChange - CQTC, the discharge trend component.

Flux is reported in millions of kg per year, as in R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.monthly_results import calculate_monthly_results
from pyegret.analysis.setup_years import get_pa, setup_years, start_end
from pyegret.analysis.trend_setup import (
    add_plus,
    estimate_span_surface,
    record_dates,
)
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import build_grid_axes
from pyegret.model.flex_fn import _sliding_segments
from pyegret.utils.season import set_season_label
from pyegret.utils.date_utils import to_decimal_year

__all__ = ["run_pairs", "print_pairs"]

# kg/day -> million kg/year, EGRET's reporting unit for flux change.
KG_PER_DAY_TO_MILLION_KG_PER_YEAR = 0.00036525

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _pair_date_info(window_side, flow_break, q1_end_date, span1, span2,
                    q_start, q_end) -> pd.DataFrame:
    """dateInfo for the two years: row 0 is year 1, row 1 is year 2."""
    if flow_break and q1_end_date is None:
        raise ValueError("if there is a flow_break you must provide q1_end_date")

    if window_side <= 0 and not flow_break:
        return pd.DataFrame({
            "flowNormStart": [q_start, q_start],
            "flowNormEnd": [q_end, q_end],
            "flowStart": [span1["startDate"], span2["startDate"]],
            "flowEnd": [span1["endDate"], span2["endDate"]],
        })
    if window_side > 0 and not flow_break:
        first = _sliding_segments(span1["startDate"], span1["endDate"],
                                  q_start, q_end, int(window_side))
        second = _sliding_segments(span2["startDate"], span2["endDate"],
                                   q_start, q_end, int(window_side))
        return pd.concat([first, second], ignore_index=True)

    q1_end = pd.Timestamp(q1_end_date)
    q2_start = q1_end + pd.Timedelta(days=1)
    if window_side <= 0:
        return pd.DataFrame({
            "flowNormStart": [q_start, q2_start],
            "flowNormEnd": [q1_end, q_end],
            "flowStart": [span1["startDate"], span2["startDate"]],
            "flowEnd": [span1["endDate"], span2["endDate"]],
        })
    first = _sliding_segments(span1["startDate"], span1["endDate"],
                              q_start, q1_end, int(window_side))
    second = _sliding_segments(span2["startDate"], span2["endDate"],
                               q2_start, q_end, int(window_side))
    return pd.concat([first, second], ignore_index=True)


def _between(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = pd.to_datetime(frame["Date"])
    inside = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    subset = frame[inside.to_numpy()]
    subset = subset.reset_index(drop=True)
    return subset


def _mean_fn(daily: pd.DataFrame, pa_long: int, pa_start: int) -> tuple:
    """Mean annual FNConc and FNFlux over the years setupYears can use."""
    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)
    conc = annual["FNConc"].to_numpy(dtype=float)
    flux = annual["FNFlux"].to_numpy(dtype=float)
    if np.all(np.isnan(conc)):
        return np.nan, np.nan
    return float(np.nanmean(conc)), float(np.nanmean(flux))


def _old_surface_slice(elist: dict, span: dict) -> dict:
    """The eList's own surface, cut to one year plus a grid step each side."""
    vector_year = np.asarray(elist["vector_year"], dtype=float)
    low = float(to_decimal_year(pd.Series([span["startDate"]])).iloc[0])
    high = float(to_decimal_year(pd.Series([span["endDate"]])).iloc[0])
    inside = np.flatnonzero((vector_year >= low) & (vector_year <= high))
    first = max(inside[0] - 1, 0)
    last = min(inside[-1] + 1, len(vector_year) - 1)
    return {"surfaces": elist["surfaces"][first:last + 1],
            "vector_year": vector_year[first:last + 1],
            "vector_logQ": elist["vector_logQ"]}


def _by_month(elist, daily1, daily2, year1, year2, pa_start, pa_long) -> pd.DataFrame:
    """Monthly flow-normalized results for the two years (R's byMonth)."""
    info = elist["INFO"]
    drain = np.nan
    if "drainSqKm" in info.columns:
        drain = float(info["drainSqKm"].iloc[0])

    table = pd.DataFrame({"Year": [year1, year1, year2, year2],
                          "Type": ["Flux", "Conc", "Flux", "Conc"]})
    for name in _MONTHS:
        table[name] = np.nan

    rows = ((daily1, 0), (daily2, 2))
    months = []
    for k in range(pa_long):
        month = pa_start + k
        if month > 12:
            month = month - 12
        months.append(month)

    for daily, row in rows:
        monthly = calculate_monthly_results({"Daily": daily})
        monthly = monthly.dropna()
        month_flux = monthly["nDays"].to_numpy() * monthly["FNFlux"].to_numpy() / drain
        month_conc = monthly["FNConc"].to_numpy()
        month_index = monthly["Month"].to_numpy()
        for month in months:
            hit = np.flatnonzero(month_index == month)
            if len(hit) > 0:
                table.loc[row, _MONTHS[month - 1]] = month_flux[hit[0]]
                table.loc[row + 1, _MONTHS[month - 1]] = month_conc[hit[0]]
    return table


def run_pairs(
    elist: dict,
    year1: int,
    year2: int,
    window_side: float,
    flow_break: bool = False,
    q1_end_date=None,
    q_start_date=None,
    q_end_date=None,
    wall: bool = False,
    old_surface: bool = False,
    sample1_end_date=None,
    sample_start_date=None,
    sample_end_date=None,
    pa_start: int | None = None,
    pa_long: int | None = None,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    fract_min: float = 0.75,
    window_y: float = 7,
    window_q: float = 2,
    window_s: float = 0.5,
    edge_adjust: bool = True,
    verbose: bool = True,
) -> dict:
    """Change in flow-normalized concentration and flux between two years.

    Mirrors R ``runPairs()``, argument for argument (in snake_case).

    Args:
        elist: eList dict with Daily and Sample. A fitted surface is needed
            only for ``old_surface=True``.
        year1: First year, labelled by the year its period of analysis ends.
        year2: Second year.
        window_side: Half-width in years of the flow-normalization window.
            Required, as in R. 0 is stationary flow normalization (QTC is then
            zero by construction); EGRET's guidance suggests 7.
        flow_break: The discharge record has an abrupt change.
        q1_end_date: Last day before the flow break.
        q_start_date, q_end_date: Limit the discharge record used.
        wall: Fit year 1 only from samples up to ``sample1_end_date`` and
            year 2 only from samples after it.
        old_surface: Use the eList's existing surface instead of fitting two.
        sample1_end_date: Last sample date before the wall.
        sample_start_date, sample_end_date: Limit the samples used.
        pa_start, pa_long: Period of analysis. Default: from INFO, else 10/12.
        min_num_obs, min_num_uncen, fract_min: Minimum observations; reduced
            to ``fract_min`` of the smaller sample set when that is smaller.
        window_y, window_q, window_s: Starting half-windows.
        edge_adjust: Widen the time window near the record ends.
        verbose: Print the summary R prints.

    Returns:
        Dict holding what R attaches to its result:

        - ``"pairResults"``: DataFrame with rows ``Conc`` (mg/L) and
          ``Flux`` (10^6 kg/yr) and columns ``TotalChange``, ``CQTC``,
          ``QTC``, ``x10``, ``x11``, ``x20``, ``x22``.
        - ``"yearPair"``: paStart, paLong, year1, year2.
        - ``"dateInfo"``: the flow-normalization segments.
        - ``"SampleBlocks"``: the sample date ranges used for each year.
        - ``"Other"``: settings and the percent changes
          (``PercentChangeConc``, ``PercentChangeFlux``).
        - ``"byMonth"``: monthly flow-normalized flux (per km2, needs
          ``INFO["drainSqKm"]``) and concentration for both years.

    Raises:
        ValueError: If a year is outside the sample record, or the samples
            are too few for the minimums.
    """
    if wall and old_surface:
        if verbose:
            print("Setting both arguments wall and old_surface to True is not allowed.")
            print("Re-calculating surface.")
        old_surface = False

    pa_start, pa_long = get_pa(elist, pa_start, pa_long)
    sample = elist["Sample"]
    sample_dates = pd.to_datetime(sample["Date"])
    span1 = start_end(pa_start, pa_long, year1)
    span2 = start_end(pa_start, pa_long, year2)
    if span2["startDate"] > sample_dates.iloc[-1]:
        raise ValueError("year2 is outside the Sample range")
    if span1["endDate"] < sample_dates.iloc[0]:
        raise ValueError("year1 is outside the Sample range")

    dates = record_dates(elist, sample_start_date, sample_end_date,
                         q_start_date, q_end_date)
    if dates["sampleStartDate"] > span1["endDate"]:
        raise ValueError("Sample start is later than year2")
    if dates["sampleEndDate"] < span2["startDate"]:
        raise ValueError("Sample end is earlier than year1")

    if old_surface and elist.get("surfaces") is None:
        if verbose:
            print("No surface included in eList, running est_surfaces")
        old_surface = False

    date_info = _pair_date_info(window_side, flow_break, q1_end_date, span1, span2,
                                dates["QStartDate"], dates["QEndDate"])

    if wall:
        if sample1_end_date is None:
            raise ValueError("When using the wall option, please specify sample1_end_date")
        sample1_end = pd.Timestamp(sample1_end_date)
        blocks = {
            "sample1StartDate": dates["sampleStartDate"],
            "sample1EndDate": sample1_end,
            "sample2StartDate": sample1_end + pd.Timedelta(days=1),
            "sample2EndDate": dates["sampleEndDate"],
        }
    else:
        blocks = {
            "sample1StartDate": dates["sampleStartDate"],
            "sample1EndDate": dates["sampleEndDate"],
            "sample2StartDate": dates["sampleStartDate"],
            "sample2EndDate": dates["sampleEndDate"],
        }
    sample1 = _between(sample, blocks["sample1StartDate"], blocks["sample1EndDate"])
    sample2 = _between(sample, blocks["sample2StartDate"], blocks["sample2EndDate"])

    fract_min = min(fract_min, 1.0)
    min_num_obs = int(np.ceil(min(min_num_obs, fract_min * len(sample1),
                                  fract_min * len(sample2))))
    min_num_uncen = int(np.ceil(min(0.5 * min_num_obs, min_num_uncen)))
    n_uncen1 = int(np.sum(sample1["Uncen"].to_numpy()))
    n_uncen2 = int(np.sum(sample2["Uncen"].to_numpy()))
    if verbose:
        print(f"Sample1 has {len(sample1)} Samples and {n_uncen1} are uncensored")
        print(f"Sample2 has {len(sample2)} Samples and {n_uncen2} are uncensored")
        print(f"minNumObs has been set to {min_num_obs} minNumUncen has been set to {min_num_uncen}")
    too_small = (min_num_obs > len(sample1) or min_num_obs > len(sample2)
                 or min_num_uncen > n_uncen1 or min_num_uncen > n_uncen2)
    if too_small:
        raise ValueError("Data set too small for minNumObs or minNumUncen")

    daily = elist["Daily"]
    daily1 = _between(daily, date_info["flowNormStart"].iloc[0], date_info["flowNormEnd"].iloc[0])
    daily2 = _between(daily, date_info["flowNormStart"].iloc[1], date_info["flowNormEnd"].iloc[1])

    if old_surface:
        if elist.get("vector_year") is None:
            vector_year, vector_logQ = build_grid_axes(elist)
            elist = dict(elist)
            elist["vector_year"] = vector_year
            elist["vector_logQ"] = vector_logQ
        surface1 = _old_surface_slice(elist, span1)
        surface2 = _old_surface_slice(elist, span2)
    else:
        settings = {"window_y": window_y, "window_q": window_q, "window_s": window_s,
                    "min_num_obs": min_num_obs, "min_num_uncen": min_num_uncen,
                    "edge_adjust": edge_adjust, "verbose": False}
        surface1 = estimate_span_surface(elist, sample1, span1["startDate"],
                                         span1["endDate"], **settings)
        surface2 = estimate_span_surface(elist, sample2, span2["startDate"],
                                         span2["endDate"], **settings)

    est1 = est_daily_from_surfaces(elist, surfaces=surface1["surfaces"],
                                   vector_year=surface1["vector_year"],
                                   vector_logQ=surface1["vector_logQ"], daily=daily1)
    c11, f11 = _mean_fn(est1, pa_long, pa_start)
    if np.isnan(c11):
        raise ValueError("year1 does not have adequate Daily data (missing >10%).")

    est2 = est_daily_from_surfaces(elist, surfaces=surface2["surfaces"],
                                   vector_year=surface2["vector_year"],
                                   vector_logQ=surface2["vector_logQ"], daily=daily2)
    c22, f22 = _mean_fn(est2, pa_long, pa_start)
    if np.isnan(c22):
        raise ValueError("year2 does not have adequate Daily data (missing >10%).")

    daily0 = _between(daily, dates["QStartDate"], dates["QEndDate"])
    est10 = est_daily_from_surfaces(elist, surfaces=surface1["surfaces"],
                                    vector_year=surface1["vector_year"],
                                    vector_logQ=surface1["vector_logQ"], daily=daily0)
    c10, f10 = _mean_fn(est10, pa_long, pa_start)
    est20 = est_daily_from_surfaces(elist, surfaces=surface2["surfaces"],
                                    vector_year=surface2["vector_year"],
                                    vector_logQ=surface2["vector_logQ"], daily=daily0)
    c20, f20 = _mean_fn(est20, pa_long, pa_start)

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
    pair_results = pd.DataFrame([conc_row, flux_row], index=["Conc", "Flux"],
                                columns=columns)

    info = elist["INFO"]
    other = {
        "minNumObs": min_num_obs, "minNumUncen": min_num_uncen,
        "windowY": window_y, "windowQ": window_q, "windowS": window_s,
        "wall": wall, "edgeAdjust": edge_adjust,
        "QStartDate": dates["QStartDate"], "QEndDate": dates["QEndDate"],
        "siteName": str(info["shortName"].iloc[0]) if "shortName" in info.columns else "",
        "paramShortName": (str(info["paramShortName"].iloc[0])
                           if "paramShortName" in info.columns else ""),
        "PercentChangeConc": {
            "Total Percent Change": 100 * c_total / c11,
            "CQTC Percent": 100 * c_cqtc / c11,
            "QTC Percent": 100 * c_qtc / c11,
        },
        "PercentChangeFlux": {
            "Total Percent Change": 100 * f_total / f11,
            "CQTC Percent": 100 * f_cqtc / f11,
            "QTC Percent": 100 * f_qtc / f11,
        },
    }
    result = {
        "pairResults": pair_results,
        "yearPair": {"paStart": pa_start, "paLong": pa_long,
                     "year1": year1, "year2": year2},
        "dateInfo": date_info,
        "SampleBlocks": blocks,
        "Other": other,
        "byMonth": _by_month(elist, est1, est2, year1, year2, pa_start, pa_long),
    }
    if verbose:
        print_pairs(elist, result)
    return result


def print_pairs(elist: dict, pairs: dict) -> None:
    """Print a run_pairs result as R does. Mirrors R ``printPairs()``."""
    other = pairs["Other"]
    year_pair = pairs["yearPair"]
    table = pairs["pairResults"]
    print(f"\n   {other['siteName']} \n   {other['paramShortName']}")
    print(f"\n   {set_season_label(year_pair['paStart'], year_pair['paLong'])} ")
    if other["wall"]:
        print("\n Sample data set was partitioned with a wall right after ",
              pairs["SampleBlocks"]["sample1EndDate"].date())
    print(f"\n Change estimates  {year_pair['year2']}  minus  {year_pair['year1']} ")

    for label, unit, percents in (("Conc", "mg/L", other["PercentChangeConc"]),
                                  ("Flux", "million kg/year", other["PercentChangeFlux"])):
        name = "concentration"
        if label == "Flux":
            name = "flux"
        total = table.loc[label, "TotalChange"]
        print(f"\n For {name}: total change is  {total:.3g} {unit}")
        print(f" expressed as Percent Change is  {add_plus(percents['Total Percent Change'])}")
        print(f"\n Concentration v. Q Trend Component  {add_plus(percents['CQTC Percent'])}")
        print(f"       Q Trend Component             {add_plus(percents['QTC Percent'])} \n")
    print(table.to_string(float_format=lambda v: f"{v:.2g}"))
