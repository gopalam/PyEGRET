"""
Printed tables of annual results and of change between years.

Mirrors R ``tableResults()``, ``tableChange()`` and ``tableChangeSingle()``.

These describe what the flow-normalized series did between years, by simple
arithmetic on the annual values. :func:`~pyegret.analysis.run_pairs.run_pairs`
answers the harder question of how much of a change came from the watershed
and how much from the flow distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.utils.season import set_season_label
from pyegret.utils.units import get_flux_unit, get_q_unit

__all__ = ["table_results", "table_change", "table_change_single"]


def _annual_results(elist: dict, daily=None) -> tuple:
    """setupYears for the eList's period of analysis, empty years removed."""
    pa_start, pa_long = get_pa(elist)
    if daily is None:
        daily = elist["Daily"]
    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)
    model_cols = annual[["Conc", "Flux", "FNConc", "FNFlux"]]
    all_missing = np.all(np.isnan(model_cols.to_numpy(dtype=float)), axis=1)
    annual = annual[~all_missing]
    annual = annual.reset_index(drop=True)
    return annual, pa_start, pa_long


def _info_text(elist: dict, column: str) -> str:
    info = elist["INFO"]
    if column in info.columns:
        return str(info[column].iloc[0])
    return ""


def _fmt(value: float, width: int) -> str:
    """R's format(x, digits = 3, width = w), closely enough for a table."""
    if np.isnan(value):
        return "NA".rjust(width)
    return f"{value:.3g}".rjust(width)


def table_results(elist: dict, q_unit=2, flux_unit=9, daily=None,
                  verbose: bool = True) -> pd.DataFrame:
    """Table of annual discharge, concentration and flux.

    Mirrors R ``tableResults()``: one row per year of the period of analysis,
    with mean discharge, WRTDS and flow-normalized concentration and flux,
    and the WRTDS-Kalman values when present.

    Args:
        elist: eList dict with a fitted model.
        q_unit: Discharge unit, EGRET number or name. Default 2 (cms).
        flux_unit: Flux unit, EGRET number or name. Default 9 (10^6 kg/yr).
        daily: Use this Daily instead of the eList's (R's ``localDaily``).
        verbose: Print the table, as R does. Default True.

    Returns:
        DataFrame with the printed values, numeric, with R's column names
        (e.g. ``"Discharge [cms]"``, ``"FN Flux [10^6kg/yr]"``).
    """
    annual, pa_start, pa_long = _annual_results(elist, daily)
    q_info = get_q_unit(q_unit)
    f_info = get_flux_unit(flux_unit)
    q_name = q_info["short_name"]
    f_name = f_info["short_name"]
    q_tag = q_name.replace(" ", "")
    f_tag = f_name.replace(" ", "")

    period = pa_long / 12
    year = np.trunc(annual["DecYear"].to_numpy() + period / 2)
    q = annual["Q"].to_numpy() * q_info["factor"]
    f_factor = f_info["factor"]
    kalman = "GenConc" in annual.columns

    table = {"Year": year, f"Discharge [{q_tag}]": q,
             "Conc [mg/L]": annual["Conc"].to_numpy(),
             "FN Conc [mg/L]": annual["FNConc"].to_numpy()}
    if kalman:
        table["GenConc [mg/L]"] = annual["GenConc"].to_numpy()
    table[f"Flux [{f_tag}]"] = annual["Flux"].to_numpy() * f_factor
    table[f"FN Flux [{f_tag}]"] = annual["FNFlux"].to_numpy() * f_factor
    if kalman:
        table[f"GenFlux [{f_tag}]"] = annual["GenFlux"].to_numpy() * f_factor
    result = pd.DataFrame(table)

    if verbose:
        period_name = set_season_label(pa_start, pa_long)
        print(f"\n   {_info_text(elist, 'shortName')} \n   {_info_text(elist, 'paramShortName')}")
        print(f"\n   {period_name} \n")
        if kalman:
            print("   Year   Discharge    Conc    FN_Conc   GenConc     Flux    FN_Flux   GenFlux")
        else:
            print("   Year   Discharge    Conc    FN_Conc     Flux    FN_Flux")
        print(f"            {q_name}         mg/L             {f_name}\n")
        for i in range(len(result)):
            row = result.iloc[i].to_numpy(dtype=float)
            text = f"{int(row[0]):7d}"
            for value in row[1:]:
                text = text + " " + _fmt(value, 9)
            print(text)
    return result


def _default_year_points(first_year: int, last_year: int) -> list:
    """Every fifth year counting back from the last, in increasing order."""
    points = list(range(last_year, first_year - 1, -5))
    points.reverse()
    return points


def table_change_single(elist: dict, flux_unit=9, year_points=None,
                        flux: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Change in flow-normalized concentration or flux between pairs of years.

    Mirrors R ``tableChangeSingle()``. For every pair of years in
    ``year_points`` (earlier first), reports the change, the change per year,
    and both as percentages of the earlier year.

    Args:
        elist: eList dict with a fitted model.
        flux_unit: Flux unit for the flux table. Default 9 (10^6 kg/yr).
        year_points: Years to compare, e.g. ``[1990, 2000, 2010]``. Default:
            every fifth year counting back from the last year of the record.
            Points outside the record are replaced by the default, as in R.
        flux: Report flux (True) or concentration (False, default).
        verbose: Print the table. Default True.

    Returns:
        DataFrame with ``Year1``, ``Year2``, and the change, slope, change
        in percent and slope in percent per year, with R's column names.

    Raises:
        ValueError: If the model has not been run.
    """
    daily = elist["Daily"]
    if "FNFlux" not in daily.columns or "FNConc" not in daily.columns:
        raise ValueError("Must run model_estimation on eList first.")

    annual, pa_start, pa_long = _annual_results(elist)
    f_info = get_flux_unit(flux_unit)
    f_tag = f_info["short_name"].replace(" ", "")
    period = pa_long / 12
    first_year = int(np.trunc(annual["DecYear"].iloc[0] + period / 2))
    last_year = int(np.trunc(annual["DecYear"].iloc[-1] + period / 2))

    default_points = _default_year_points(first_year, last_year)
    if year_points is None:
        year_points = default_points
    year_points = list(year_points)
    if year_points[-1] > last_year or year_points[0] < first_year:
        year_points = default_points

    if flux:
        values = annual["FNFlux"].to_numpy() * f_info["factor"]
        header = ["Year1", "Year2", f"change [{f_tag}]", f"slope [{f_tag}/yr]",
                  "change[percent]", "slope [percent/yr]"]
    else:
        values = annual["FNConc"].to_numpy()
        header = ["Year1", "Year2", "change[mg/L]", "slope[mg/L/yr]",
                  "change[%]", "slope [%/yr]"]

    rows = []
    for i in range(len(year_points) - 1):
        for j in range(i + 1, len(year_points)):
            index_first = year_points[i] - first_year
            index_last = year_points[j] - first_year
            y_first = values[index_first]
            y_last = values[index_last]
            x_dif = index_last - index_first
            y_dif = y_last - y_first
            rows.append([year_points[i], year_points[j], y_dif, y_dif / x_dif,
                         100 * y_dif / y_first, 100 * y_dif / y_first / x_dif])
    result = pd.DataFrame(rows, columns=header)

    if verbose:
        if flux:
            print("\n                 Flux Trends")
            print("   time span          change        slope       change        slope")
            print(f"              {f_info['short_name']} {f_info['short_name']}/yr      %         %/yr")
            width = 12
        else:
            print("\n           Concentration trends")
            print("   time span       change     slope    change     slope")
            print("                     mg/L   mg/L/yr        %       %/yr")
            width = 9
        for i in range(len(result)):
            row = result.iloc[i].to_numpy(dtype=float)
            text = f" {int(row[0])}  to  {int(row[1])}"
            for value in row[2:]:
                if np.isnan(value):
                    text = text + "       ---"
                else:
                    text = text + " " + f"{value:.2g}".rjust(width)
            print(text)
    return result


def table_change(elist: dict, flux_unit=9, year_points=None,
                 verbose: bool = True) -> dict:
    """Concentration and flux change tables. Mirrors R ``tableChange()``.

    Args:
        elist: eList dict with a fitted model.
        flux_unit: Flux unit for the flux table. Default 9 (10^6 kg/yr).
        year_points: Years to compare; see :func:`table_change_single`.
        verbose: Print both tables, as R does. Default True.

    Returns:
        Dict with ``"concentration"`` and ``"flux"`` DataFrames.
    """
    if verbose:
        pa_start, pa_long = get_pa(elist)
        print(f"\n   {_info_text(elist, 'shortName')} \n   {_info_text(elist, 'paramShortName')}")
        print(f"\n   {set_season_label(pa_start, pa_long)} ")
    conc = table_change_single(elist, flux_unit, year_points, flux=False, verbose=verbose)
    flux = table_change_single(elist, flux_unit, year_points, flux=True, verbose=verbose)
    return {"concentration": conc, "flux": flux}
