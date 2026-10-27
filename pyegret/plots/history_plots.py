"""
Plots of the fitted record over time.

Mirrors R ``plotConcHist()``, ``plotFluxHist()``, ``plotConcTimeDaily()``,
``plotFluxTimeDaily()`` and ``plotWRTDSKalman()``. Every function returns
``(fig, ax)`` (``(fig, axes)`` for two panels) and never calls ``plt.show()``.
"""
from __future__ import annotations

import numpy as np

from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.data.quality import make_augmented_sample
from pyegret.plots.plot_utils import censored_segments, date_ticks, new_axes, select_days, site_title
from pyegret.utils.season import set_season_label
from pyegret.utils.units import get_flux_unit

__all__ = [
    "plot_conc_hist",
    "plot_flux_hist",
    "plot_conc_time_daily",
    "plot_flux_time_daily",
    "plot_wrtds_kalman",
    "plot_time_slice",
]


def _history_title(plot_annual: bool, plot_gen: bool, plot_flow_norm: bool, what: str) -> str:
    """Third title line of plotConcHist / plotFluxHist, as R words it."""
    if plot_annual and plot_gen and plot_flow_norm:
        text = "Mean (red = Kalman, black = WRTDS) & Flow-Normalized (line)"
    elif plot_annual and plot_gen:
        text = "Mean (red = Kalman, black = WRTDS)"
    elif not plot_annual and not plot_gen and plot_flow_norm:
        text = "Flow Normalized"
    elif plot_flow_norm and (plot_gen or plot_annual):
        text = "Mean (dots) & Flow-Normalized (line)"
    elif plot_annual:
        text = "Mean"
    elif plot_gen:
        text = "Mean Kalman"
    else:
        return ""
    return text + " " + what


def _history(elist, column, gen_column, fn_column, factor, year_start, year_end,
             y_max, print_title, plot_flow_norm, plot_annual, plot_gen, what,
             y_label, col, col_pred, col_gen, lw, ax):
    daily = elist["Daily"]
    if fn_column not in daily.columns:
        raise ValueError("This function requires running model_estimation on eList")
    if plot_gen and gen_column not in daily.columns:
        raise ValueError("This option requires running wrtds_kalman on eList")

    pa_start, pa_long = get_pa(elist)
    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)

    sample_years = elist["Sample"]["DecYear"].to_numpy(dtype=float)
    data_start = np.nanmin(sample_years) - 0.5
    data_end = np.nanmax(sample_years) + 0.5
    if year_start is None:
        year_start = data_start
    else:
        year_start = max(year_start, data_start)
    if year_end is None:
        year_end = data_end
    else:
        year_end = min(year_end, data_end)

    dec_year = annual["DecYear"].to_numpy(dtype=float)
    keep = (dec_year >= year_start) & (dec_year <= year_end)
    x = dec_year[keep]
    mean_values = annual[column].to_numpy(dtype=float)[keep] * factor
    fn_values = annual[fn_column].to_numpy(dtype=float)[keep] * factor

    fig, ax = new_axes(ax)
    tops = [np.nanmax(fn_values)]
    if plot_annual:
        ax.plot(x, mean_values, "o", color=col, markersize=4, label="Mean")
        tops.append(np.nanmax(mean_values))
    if plot_gen:
        gen_values = annual[gen_column].to_numpy(dtype=float)[keep] * factor
        ax.plot(x, gen_values, "o", color=col_gen, markersize=4, label="Kalman mean")
        tops.append(np.nanmax(gen_values))
    if plot_flow_norm:
        ax.plot(x, fn_values, "-", color=col_pred, linewidth=lw, label="Flow-normalized")

    if y_max is None:
        y_max = 1.05 * np.nanmax(tops)
    pad = 0.05 * (year_end - year_start)
    date_ticks(ax, year_start - pad, year_end + pad)
    ax.set_ylim(0, y_max)
    ax.set_ylabel(y_label)
    if print_title:
        period = set_season_label(pa_start, pa_long)
        third = _history_title(plot_annual, plot_gen, plot_flow_norm, what)
        ax.set_title(site_title(elist, period + "\n" + third), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_conc_hist(elist: dict, year_start=None, year_end=None, conc_max=None,
                   print_title: bool = True, plot_flow_norm: bool = True,
                   plot_annual: bool = True, plot_gen_conc: bool = False,
                   col: str = "black", col_pred: str = "green", col_gen: str = "red",
                   lw: float = 2, ax=None):
    """Annual mean and flow-normalized concentration. Mirrors R ``plotConcHist()``.

    Dots are the annual means of the daily WRTDS estimates (red ones the
    WRTDS-Kalman means when ``plot_gen_conc``); the line is flow-normalized
    concentration. The x range is the sample record padded by half a year.

    Args:
        elist: eList dict after model_estimation.
        year_start, year_end: Limit the x axis (decimal years).
        conc_max: Top of the y axis.
        print_title: Add EGRET's title.
        plot_flow_norm, plot_annual, plot_gen_conc: What to draw.
        col, col_pred, col_gen: Colours of means, flow-normalized line and
            Kalman means (R's defaults).
        lw: Line width.
        ax: Axes to draw on; a new figure if None.

    Returns:
        ``(fig, ax)``.
    """
    return _history(elist, "Conc", "GenConc", "FNConc", 1.0, year_start, year_end,
                    conc_max, print_title, plot_flow_norm, plot_annual, plot_gen_conc,
                    "Concentration", "Concentration, in mg/L", col, col_pred, col_gen,
                    lw, ax)


def plot_flux_hist(elist: dict, year_start=None, year_end=None, flux_unit=9,
                   flux_max=None, print_title: bool = True, plot_flow_norm: bool = True,
                   plot_annual: bool = True, plot_gen_flux: bool = False,
                   col: str = "black", col_pred: str = "green", col_gen: str = "red",
                   lw: float = 2, ax=None):
    """Annual mean and flow-normalized flux. Mirrors R ``plotFluxHist()``.

    Args:
        elist: eList dict after model_estimation.
        year_start, year_end: Limit the x axis (decimal years).
        flux_unit: EGRET flux unit number or name. Default 9 (10^6 kg/yr).
        flux_max: Top of the y axis, in ``flux_unit``.
        print_title: Add EGRET's title.
        plot_flow_norm, plot_annual, plot_gen_flux: What to draw.
        col, col_pred, col_gen: Colours.
        lw: Line width.
        ax: Axes to draw on; a new figure if None.

    Returns:
        ``(fig, ax)``.
    """
    unit = get_flux_unit(flux_unit)
    return _history(elist, "Flux", "GenFlux", "FNFlux", unit["factor"], year_start,
                    year_end, flux_max, print_title, plot_flow_norm, plot_annual,
                    plot_gen_flux, "Flux Estimates", unit["label"], col, col_pred,
                    col_gen, lw, ax)


def _time_daily(elist, flux, year_start, year_end, y_max, print_title, plot_gen,
                random_censored, flux_unit, col, lw, ax):
    info_pa = get_pa(elist)
    pa_start, pa_long = info_pa
    sample = elist["Sample"]
    daily = elist["Daily"]
    if plot_gen and "GenConc" not in daily.columns:
        plot_gen = False
    if pa_long != 12:
        sample = select_days(sample, pa_long, pa_start)
        daily = select_days(daily, pa_long, pa_start)
    if year_start is None:
        year_start = int(np.nanmin(sample["DecYear"]))
    if year_end is None:
        year_end = int(np.nanmax(sample["DecYear"]))

    s_year = sample["DecYear"].to_numpy(dtype=float)
    d_year = daily["DecYear"].to_numpy(dtype=float)
    s_keep = (s_year >= year_start) & (s_year <= year_end)
    d_keep = (d_year >= year_start) & (d_year <= year_end)

    factor = 1.0
    q_sample = np.ones(len(sample))
    q_daily = np.ones(len(daily))
    y_label = "Concentration, in mg/L"
    if flux:
        unit = get_flux_unit(flux_unit)
        factor = unit["factor"] * 86.4
        q_sample = sample["Q"].to_numpy(dtype=float)
        q_daily = daily["Q"].to_numpy(dtype=float)
        y_label = unit["label"]

    if plot_gen:
        if flux:
            daily_values = daily["GenFlux"].to_numpy(dtype=float) * factor / 86.4
        else:
            daily_values = daily["GenConc"].to_numpy(dtype=float)
    else:
        daily_values = daily["ConcDay"].to_numpy(dtype=float) * q_daily * factor

    uncen = sample["Uncen"].to_numpy()[s_keep]
    x_sample = s_year[s_keep]
    fig, ax = new_axes(ax, figsize=(9.0, 5.0))

    if random_censored:
        if "rObserved" not in sample.columns:
            sample = make_augmented_sample({"Sample": sample, "INFO": elist["INFO"]})["Sample"]
        y_obs = sample["rObserved"].to_numpy(dtype=float)[s_keep] * q_sample[s_keep] * factor
        ax.plot(x_sample[uncen == 1], y_obs[uncen == 1], "o", color=col, markersize=3)
        ax.plot(x_sample[uncen == 0], y_obs[uncen == 0], "o", color=col, markersize=3,
                markerfacecolor="none")
        y_top_data = np.nanmax(y_obs)
    else:
        y_low = sample["ConcLow"].to_numpy(dtype=float)[s_keep] * q_sample[s_keep] * factor
        y_high = sample["ConcHigh"].to_numpy(dtype=float)[s_keep] * q_sample[s_keep] * factor
        ax.plot(x_sample[uncen == 1], y_high[uncen == 1], "o", color=col, markersize=3)
        y_top_data = np.nanmax(y_high)

    ax.plot(d_year[d_keep], daily_values[d_keep], "-", color=col, linewidth=lw)
    if y_max is None:
        y_max = 1.05 * max(y_top_data, np.nanmax(daily_values[d_keep]))
    ax.set_ylim(0, y_max)
    if not random_censored:
        censored_segments(ax, 0.0, y_low, y_high, x_sample, uncen, color=col, lw=lw)
    date_ticks(ax, year_start, year_end)
    ax.set_ylabel(y_label)
    if print_title:
        what = "Flux" if flux else "Concentration"
        text = f"Observed and Estimated {what} versus Time"
        if pa_long != 12:
            text = text + "\n" + set_season_label(pa_start, pa_long)
        ax.set_title(site_title(elist, text), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_conc_time_daily(elist: dict, year_start=None, year_end=None, conc_max=None,
                         print_title: bool = True, plot_gen_conc: bool = True,
                         random_censored: bool = False, col: str = "black",
                         lw: float = 1, ax=None):
    """Daily estimated concentration with the observations. Mirrors R ``plotConcTimeDaily()``.

    The line is the daily estimate (WRTDS-Kalman ``GenConc`` when it exists
    and ``plot_gen_conc``, else ``ConcDay``); dots are the samples and
    vertical lines the censored ones (or open circles at randomly drawn
    values when ``random_censored``).

    Args:
        elist: eList dict after model_estimation.
        year_start, year_end: x range; default the sample record's years.
        conc_max: Top of the y axis.
        print_title: Add EGRET's title.
        plot_gen_conc: Use GenConc when available. Default True, as in R.
        random_censored: Draw censored samples as random values.
        col, lw: Colour and line width.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    return _time_daily(elist, False, year_start, year_end, conc_max, print_title,
                       plot_gen_conc, random_censored, 3, col, lw, ax)


def plot_flux_time_daily(elist: dict, year_start=None, year_end=None, flux_unit=3,
                         flux_max=None, random_censored: bool = False,
                         print_title: bool = True, plot_gen_flux: bool = True,
                         col: str = "black", lw: float = 1, ax=None):
    """Daily estimated flux with the observed flux. Mirrors R ``plotFluxTimeDaily()``.

    Args:
        elist: eList dict after model_estimation.
        year_start, year_end: x range; default the sample record's years.
        flux_unit: EGRET flux unit. Default 3 (kg/day), as in R.
        flux_max: Top of the y axis.
        random_censored: Draw censored samples as random values.
        print_title: Add EGRET's title.
        plot_gen_flux: Use GenFlux when available.
        col, lw: Colour and line width.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    return _time_daily(elist, True, year_start, year_end, flux_max, print_title,
                       plot_gen_flux, random_censored, flux_unit, col, lw, ax)


def plot_wrtds_kalman(elist: dict, side_by_side: bool = False, flux_unit=9, axes=None):
    """Annual WRTDS and WRTDS-Kalman flux, over time and against each other.

    Mirrors R ``plotWRTDSKalman()``: the left panel shows annual flux by
    year (WRTDS red, WRTDS-K green); the right panel plots one against the
    other with a 1:1 line.

    Args:
        elist: eList dict after wrtds_kalman.
        side_by_side: Arrange the panels left-right (else top-bottom).
        flux_unit: EGRET flux unit. Default 9 (10^6 kg/yr).
        axes: Two axes to draw on.

    Returns:
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    daily = elist["Daily"]
    if "GenFlux" not in daily.columns:
        raise ValueError("This function requires running wrtds_kalman on eList")
    pa_start, pa_long = get_pa(elist)
    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)
    unit = get_flux_unit(flux_unit)
    flux = annual["Flux"].to_numpy(dtype=float) * unit["factor"]
    gen = annual["GenFlux"].to_numpy(dtype=float) * unit["factor"]
    year = annual["DecYear"].to_numpy(dtype=float)
    y_max = 1.1 * np.nanmax(np.concatenate([flux, gen]))

    if axes is None:
        if side_by_side:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    else:
        fig = axes[0].figure

    period = set_season_label(pa_start, pa_long)
    axes[0].plot(year, flux, "o", color="red", markersize=4, label="WRTDS")
    axes[0].plot(year, gen, "o", color="green", markersize=4, label="WRTDS-K")
    axes[0].set_xlim(np.floor(year[0]), np.ceil(year[-1]))
    axes[0].set_ylim(0, y_max)
    axes[0].set_ylabel(f"Annual flux, in {unit['short_name']}")
    axes[0].set_title(site_title(elist, f"{period}\nAnnual Flux Estimates: WRTDS in red, "
                                        "WRTDS-K in green"), fontsize=10)

    axes[1].plot(flux, gen, "o", color="red", markersize=4)
    axes[1].plot([0, y_max], [0, y_max], "-", color="black", linewidth=1)
    axes[1].set_xlim(0, y_max)
    axes[1].set_ylim(0, y_max)
    axes[1].set_xlabel(f"WRTDS estimate of annual flux, in {unit['short_name']}")
    axes[1].set_ylabel(f"WRTDSKalman estimate of annual flux, in {unit['short_name']}")
    axes[1].set_title(f"Comparison of the two flux estimates\n{period}", fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_time_slice(elist: dict, start=None, end=None, conc: bool = True,
                    flux_unit=3, ax=None):
    """WRTDS and WRTDS-Kalman daily estimates over a short stretch, with the data.

    Mirrors R ``plotTimeSlice()``: black is WRTDS, green WRTDS-Kalman;
    detected samples are red and censored ones cyan (at a randomly drawn
    value). The title reports the ratio of the two methods' means.

    Args:
        elist: eList dict after wrtds_kalman.
        start, end: Decimal-year limits.
        conc: Concentration (True) or flux (False).
        flux_unit: EGRET flux unit for the flux version. Default 3 (kg/day).
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    daily = elist["Daily"]
    if "GenConc" not in daily.columns:
        raise ValueError("This function requires running wrtds_kalman on eList")
    sample = make_augmented_sample(elist)["Sample"]
    d_year = daily["DecYear"].to_numpy(dtype=float)
    s_year = sample["DecYear"].to_numpy(dtype=float)
    if start is None:
        start = min(np.nanmin(d_year), np.nanmin(s_year))
    if end is None:
        end = max(np.nanmax(d_year), np.nanmax(s_year))
    d_keep = (d_year >= start) & (d_year <= end)
    s_keep = (s_year >= start) & (s_year <= end)
    uncen = sample["Uncen"].to_numpy()[s_keep]

    if conc:
        y1 = daily["ConcDay"].to_numpy(dtype=float)[d_keep]
        y2 = daily["GenConc"].to_numpy(dtype=float)[d_keep]
        y3 = sample["rObserved"].to_numpy(dtype=float)[s_keep]
        y_label = "Concentration, in mg/L"
        what = "Concentrations"
    else:
        unit = get_flux_unit(flux_unit)
        y1 = daily["FluxDay"].to_numpy(dtype=float)[d_keep] * unit["factor"]
        y2 = daily["GenFlux"].to_numpy(dtype=float)[d_keep] * unit["factor"]
        q = sample["Q"].to_numpy(dtype=float)[s_keep]
        y3 = sample["rObserved"].to_numpy(dtype=float)[s_keep] * q * 86.4 * unit["factor"]
        y_label = unit["label"]
        what = "Flux"
    ratio = np.nanmean(y2) / np.nanmean(y1)

    fig, ax = new_axes(ax, figsize=(9.0, 5.0))
    ax.plot(d_year[d_keep], y1, "-", color="black", linewidth=1)
    ax.plot(d_year[d_keep], y2, "-", color="green", linewidth=1)
    x3 = s_year[s_keep]
    ax.plot(x3[uncen == 1], y3[uncen == 1], "o", color="red", markersize=4)
    ax.plot(x3[uncen == 0], y3[uncen == 0], "o", color="cyan", markersize=4)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1.05 * np.nanmax(np.concatenate([y1, y2, y3])))
    ax.set_ylabel(y_label)
    ax.set_title(site_title(elist, f"{what}, Black is WRTDS, Green is WRTDSKalman\n"
                                   f"Data in red, (rl in blue if <), Ratio of means is "
                                   f"{ratio:.2g}"), fontsize=10)
    fig.tight_layout()
    return fig, ax
