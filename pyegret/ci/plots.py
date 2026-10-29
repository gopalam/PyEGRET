"""
Plots of bootstrap results.

Mirrors R EGRETci 2.0.5 ``plotConcHistBoot()``, ``plotFluxHistBoot()`` and
``plotHistogramTrend()``. Like PyEGRET's other plots they return
``(fig, ax)``, accept ``ax=`` to draw into a figure of your own, and never call
``plt.show()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.plots.history_plots import plot_conc_hist, plot_flux_hist
from pyegret.plots.plot_utils import new_axes
from pyegret.utils.season import set_season_label
from pyegret.utils.units import get_flux_unit

__all__ = ["plot_conc_hist_boot", "plot_flux_hist_boot", "plot_histogram_trend"]


def _period_name(elist: dict, pa_start: int, pa_long: int, starred: bool) -> str:
    """Season label, with R's asterisk for generalized flow normalization."""
    name = set_season_label(pa_start, pa_long)
    if starred:
        name = name + " *"
    return name


def _band_years(elist: dict, ci_annual: pd.DataFrame, year_start, year_end) -> pd.DataFrame:
    """The band rows inside the plotted range (the sample record +/- half a year)."""
    sample_years = elist["Sample"]["DecYear"].to_numpy(dtype=float)
    low = np.nanmin(sample_years) - 0.5
    high = np.nanmax(sample_years) + 0.5
    if year_start is not None:
        low = max(year_start, low)
    if year_end is not None:
        high = min(year_end, high)
    years = ci_annual["Year"].to_numpy(dtype=float)
    keep = (years >= low) & (years <= high)
    band = ci_annual[keep]
    return band


def _ci_title(elist: dict, what: str, ci_annual: pd.DataFrame) -> str:
    info = elist["INFO"]
    pa_start, pa_long = get_pa(elist)
    probs = ci_annual.attrs.get("probs", (0.05, 0.95))
    width = round((max(probs) - min(probs)) * 100)
    n_boot = ci_annual.attrs.get("nBoot")
    block = ci_annual.attrs.get("blockLength")
    short_name = str(info["shortName"].iloc[0]) if "shortName" in info.columns else ""
    param = str(info["paramShortName"].iloc[0]) if "paramShortName" in info.columns else ""
    period = _period_name(elist, pa_start, pa_long, bool(elist.get("runSeries", False)))
    return (f"{short_name}  {param}\n{period}\n"
            f"{width} % CI on FN {what}, Replicates = {n_boot}, Block = {block} days")


def plot_conc_hist_boot(elist: dict, ci_annual: pd.DataFrame, year_start=None,
                        year_end=None, conc_max=None, plot_flow_norm: bool = True,
                        plot_annual: bool = True, plot_gen_conc: bool = False,
                        col_pred: str = "green", print_title: bool = True, lw: float = 2,
                        ax=None):
    """Concentration history with its bootstrap confidence band. Mirrors R ``plotConcHistBoot()``.

    The plot of :func:`~pyegret.plots.plot_conc_hist`, with the band from
    :func:`~pyegret.ci.ci_bands.ci_calculations` as dashed lines.

    Returns:
        ``(fig, ax)``.
    """
    band = _band_years(elist, ci_annual, year_start, year_end)
    if conc_max is None:
        pa_start, pa_long = get_pa(elist)
        annual = setup_years(elist["Daily"], pa_long=pa_long, pa_start=pa_start)
        tops = [np.nanmax(band["FNConcHigh"].to_numpy(dtype=float)),
                np.nanmax(annual["Conc"].to_numpy(dtype=float))]
        if plot_gen_conc and "GenConc" in annual.columns:
            tops.append(np.nanmax(annual["GenConc"].to_numpy(dtype=float)))
        conc_max = 1.05 * max(tops)
    fig, ax = plot_conc_hist(elist, year_start=year_start, year_end=year_end,
                             conc_max=conc_max, print_title=False,
                             plot_flow_norm=plot_flow_norm, plot_annual=plot_annual,
                             plot_gen_conc=plot_gen_conc, col_pred=col_pred, lw=lw, ax=ax)
    years = band["Year"].to_numpy(dtype=float)
    ax.plot(years, band["FNConcLow"].to_numpy(dtype=float), "--", color=col_pred, linewidth=1)
    ax.plot(years, band["FNConcHigh"].to_numpy(dtype=float), "--", color=col_pred, linewidth=1)
    if print_title:
        ax.set_title(_ci_title(elist, "Concentration", ci_annual), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_flux_hist_boot(elist: dict, ci_annual: pd.DataFrame, year_start=None,
                        year_end=None, flux_unit=9, flux_max=None,
                        plot_flow_norm: bool = True, plot_annual: bool = True,
                        plot_gen_flux: bool = False, col_pred: str = "green",
                        print_title: bool = True, lw: float = 2, ax=None):
    """Flux history with its bootstrap confidence band. Mirrors R ``plotFluxHistBoot()``.

    Args:
        flux_unit: EGRET flux unit number or name. Default 9 (10^6 kg/yr).

    Returns:
        ``(fig, ax)``.
    """
    factor = get_flux_unit(flux_unit)["factor"]
    band = _band_years(elist, ci_annual, year_start, year_end)
    if flux_max is None:
        pa_start, pa_long = get_pa(elist)
        annual = setup_years(elist["Daily"], pa_long=pa_long, pa_start=pa_start)
        tops = [np.nanmax(band["FNFluxHigh"].to_numpy(dtype=float)) * factor,
                np.nanmax(annual["Flux"].to_numpy(dtype=float)) * factor]
        if plot_gen_flux and "GenFlux" in annual.columns:
            tops.append(np.nanmax(annual["GenFlux"].to_numpy(dtype=float)) * factor)
        flux_max = 1.05 * max(tops)
    fig, ax = plot_flux_hist(elist, year_start=year_start, year_end=year_end,
                             flux_unit=flux_unit, flux_max=flux_max, print_title=False,
                             plot_flow_norm=plot_flow_norm, plot_annual=plot_annual,
                             plot_gen_flux=plot_gen_flux, col_pred=col_pred, lw=lw, ax=ax)
    years = band["Year"].to_numpy(dtype=float)
    ax.plot(years, band["FNFluxLow"].to_numpy(dtype=float) * factor, "--",
            color=col_pred, linewidth=1)
    ax.plot(years, band["FNFluxHigh"].to_numpy(dtype=float) * factor, "--",
            color=col_pred, linewidth=1)
    if print_title:
        ax.set_title(_ci_title(elist, "Flux", ci_annual), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_histogram_trend(elist: dict, boot: dict, flux: bool = True, x_min=None,
                         x_max=None, x_step=None, print_title: bool = True,
                         col_fill: str = "grey", ax=None):
    """Histogram of the bootstrap trend replicates, in percent. Mirrors R ``plotHistogramTrend()``.

    The dashed line is the estimated change; the solid line is zero.

    Args:
        elist: eList dict.
        boot: Result of ``run_pairs_boot``, ``run_groups_boot`` or ``wbt``.
        flux: Flux (True, default) or concentration.
        x_min, x_max, x_step: Histogram range and bin width, in percent.
            Default: at least -10 to +10, in ten bins.
        print_title: Add EGRETci's title.
        col_fill: Bar colour.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    out = boot["bootOut"].iloc[0]
    info = elist["INFO"]
    pa_start = boot.get("paStart")
    pa_long = boot.get("paLong")
    if pa_start is None or pa_long is None:
        pa_start, pa_long = get_pa(elist)
    period = _period_name(elist, pa_start, pa_long, False)

    if flux:
        change = 100 * out["estF"] / out["baseFlux"]
        reps = np.asarray(boot["pFlux"], dtype=float)
        x_label = "Flux trend, in %"
        word = "Flux"
    else:
        change = 100 * out["estC"] / out["baseConc"]
        reps = np.asarray(boot["pConc"], dtype=float)
        x_label = "Concentration trend, in %"
        word = "Concentration"
    reps = reps[~np.isnan(reps)]

    if x_min is None:
        x_min = min(-10.0, float(np.min(reps)))
    if x_max is None:
        x_max = max(10.0, float(np.max(reps)))
    if x_step is None:
        x_step = (x_max - x_min) / 10
    edges = np.arange(x_min, x_max + 0.5 * x_step, x_step)

    fig, ax = new_axes(ax)
    ax.hist(reps, bins=edges, density=True, color=col_fill, edgecolor="black")
    ax.axvline(change, linewidth=3, linestyle="--", color="black")
    ax.axvline(0, linewidth=3, color="black")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.tick_params(direction="in", top=True, right=True)
    if print_title:
        short_name = str(info["shortName"].iloc[0]) if "shortName" in info.columns else ""
        param = str(info["paramShortName"].iloc[0]) if "paramShortName" in info.columns else ""
        if "group1firstYear" in boot:
            span = (f"{boot['group2firstYear']} to {boot['group2lastYear']} minus "
                    f"{boot['group1firstYear']} to {boot['group1lastYear']}")
        else:
            span = f"{boot['year1']} to {boot['year2']}"
        ax.set_title(f"Trend magnitude in {param}\nFlow Normalized {word} {span}\n"
                     f"{short_name} {period}", fontsize=10)
    fig.tight_layout()
    return fig, ax
