"""
Plots of the fitted WRTDS surface.

Mirrors R ``plotContours()``, ``plotDiffContours()``, ``plotConcQSmooth()``
and ``plotConcTimeSmooth()``. The first two show the surface (or the change
in it between two years) as filled contours over time and discharge; the
last two take slices through the model: concentration against discharge on
up to three dates, and concentration over time at up to three discharges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.model.est_surfaces import run_surv_reg_points
from pyegret.plots.plot_utils import date_ticks, log_pretty3, new_axes, site_title
from pyegret.utils.season import MONTH_ABBREV
from pyegret.utils.units import get_q_unit

__all__ = ["plot_contours", "plot_diff_contours", "plot_conc_q_smooth",
           "plot_conc_time_smooth"]

_SURFACE_NAMES = ["log of Concentration", "Standard Error of log(C)", "Concentration"]


def _colormap(colors):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("pyegret", colors)


def _pretty_levels(low: float, high: float, n: int) -> np.ndarray:
    """Round contour levels spanning low..high, like R's ``pretty(x, n)``."""
    from matplotlib.ticker import MaxNLocator

    locator = MaxNLocator(nbins=n)
    return locator.tick_values(low, high)


def _q_ticks(daily, y_q, q_factor, q_bottom, q_top):
    """R's choice of discharge range and log ticks for a contour plot."""
    q = daily["Q"].to_numpy(dtype=float)
    q_bottom_t = np.nanquantile(q, 0.05) * q_factor
    q_top_t = np.nanquantile(q, 0.95) * q_factor
    if q_bottom is None:
        q_bottom = max(0.9 * y_q[0], q_bottom_t)
    if q_top is None:
        q_top = min(1.1 * y_q[-1], q_top_t)
    return log_pretty3(q_bottom, q_top)


def _flow_duration_surface(daily, log_q_axis, n_x, span):
    """Fraction of days with LogQ below each grid discharge, by season.

    R ``plotContours(flowDuration = TRUE)``: 16 season centres a year, each
    taking the days within ``span`` days of it.
    """
    day = daily["Day"].to_numpy(dtype=int)
    log_q = daily["LogQ"].to_numpy(dtype=float)
    surface = np.zeros((n_x, len(log_q_axis)))
    centers = np.arange(1, 366, 22.9)
    centers = np.asarray(np.floor(centers), dtype=int)
    for ix in range(16):
        good = np.arange(centers[ix] - span, centers[ix] + span + 1)
        good = np.where(good > 0, good, good + 365)
        good = np.where(good < 366, good, good - 365)
        in_window = np.isin(day, good)
        window_q = log_q[in_window]
        freq = np.zeros(len(log_q_axis))
        for j in range(len(log_q_axis)):
            freq[j] = np.sum(window_q < log_q_axis[j]) / len(window_q)
        for row in range(ix, n_x, 16):
            surface[row, :] = freq
    return surface


def plot_contours(elist: dict, year_start: float, year_end: float, q_bottom=None,
                  q_top=None, what_surface: int = 3, q_unit=2, contour_levels=None,
                  span: int = 60, pval: float = 0.05, print_title: bool = True,
                  vert1=None, vert2=None, horiz=None, flow_duration: bool = True,
                  colors=("white", "gray", "blue", "red"), ax=None):
    """Filled contours of the WRTDS surface over time and discharge.

    Mirrors R ``plotContours()``. With ``flow_duration`` the black lines
    mark the ``pval`` and ``1 - pval`` flow percentiles for each season,
    so the part of the surface the river actually visits is visible.

    Args:
        elist: eList dict with a fitted surface.
        year_start, year_end: x range in decimal years.
        q_bottom, q_top: Discharge range (in ``q_unit``). Default: the 5th
            and 95th percentiles of daily discharge, as R chooses.
        what_surface: 1 yHat, 2 SE, 3 concentration (default).
        q_unit: EGRET discharge unit. Default 2 (cms).
        contour_levels: Explicit contour levels.
        span: Half-width in days of the flow-duration season window.
        pval: Flow percentile for the duration lines.
        print_title: Add EGRET's title.
        vert1, vert2: Decimal years at which to draw vertical lines.
        horiz: Discharge at which to draw a horizontal line.
        flow_duration: Draw the flow-percentile lines.
        colors: Colour ramp, R's default white-gray-blue-red.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    if elist.get("surfaces") is None:
        raise ValueError("This function requires running model_estimation on eList")
    index = what_surface - 1
    surface = elist["surfaces"][:, :, index]
    x = np.asarray(elist["vector_year"], dtype=float)
    log_q_axis = np.asarray(elist["vector_logQ"], dtype=float)
    unit = get_q_unit(q_unit)
    y_q = np.exp(log_q_axis) * unit["factor"]
    ticks = _q_ticks(elist["Daily"], y_q, unit["factor"], q_bottom, q_top)

    if contour_levels is None:
        contour_levels = _pretty_levels(np.nanmin(surface), np.nanmax(surface), 5)

    fig, ax = new_axes(ax, figsize=(9.0, 5.5))
    filled = ax.contourf(x, np.log10(y_q), surface.T, levels=contour_levels,
                         cmap=_colormap(colors), extend="both")
    fig.colorbar(filled, ax=ax)

    title_extra = ""
    if flow_duration:
        duration = _flow_duration_surface(elist["Daily"], log_q_axis, len(x), span)
        ax.contour(x, np.log10(y_q), duration.T, levels=[pval, 1 - pval],
                   colors="black", linewidths=2)
        title_extra = (f"\nBlack lines are {pval * 100:.2g} and {(1 - pval) * 100:.2g} "
                       "flow percentiles")
    if vert1 is not None:
        ax.axvline(vert1, color="black", linewidth=2)
    if vert2 is not None:
        ax.axvline(vert2, color="black", linewidth=2)
    if horiz is not None:
        ax.axhline(np.log10(horiz), color="black", linewidth=2)

    ax.set_ylim(np.log10(ticks[0]), np.log10(ticks[-1]))
    ax.set_yticks(np.log10(ticks))
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    date_ticks(ax, year_start, year_end)
    ax.set_ylabel(unit["label"])
    if print_title:
        text = f"Estimated {_SURFACE_NAMES[index]} Surface in Color" + title_extra
        ax.set_title(site_title(elist, text), fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig, ax


def plot_diff_contours(elist: dict, year0: int, year1: int, q_bottom=None, q_top=None,
                       max_diff=None, what_surface: int = 3, q_unit=2, span: int = 60,
                       pval: float = 0.05, print_title: bool = True,
                       plot_percent: bool = False, flow_duration: bool = True,
                       colors=("blue", "white", "red"), ax=None):
    """Change in the surface from one year to another, by season and discharge.

    Mirrors R ``plotDiffContours()``. The x axis runs through one calendar
    year (January 1 to January 1); each point is the surface in ``year1``
    minus the surface at the same time of ``year0`` (or the change in
    percent with ``plot_percent``).

    Args:
        elist: eList dict with a fitted surface.
        year0, year1: Calendar years compared (integers).
        q_bottom, q_top: Discharge range in ``q_unit``.
        max_diff: Symmetric colour range ``[-max_diff, max_diff]``, or a pair.
            Default: the 5th-95th percentile of the differences, as R uses.
        what_surface: 1 yHat, 2 SE, 3 concentration (default).
        q_unit: EGRET discharge unit. Default 2.
        span, pval, flow_duration: As for :func:`plot_contours`.
        print_title: Add EGRET's title.
        plot_percent: Plot percent change instead of difference.
        colors: Colour ramp, R's default blue-white-red.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    if elist.get("surfaces") is None:
        raise ValueError("This function requires running model_estimation on eList")
    index = what_surface - 1
    surfaces = elist["surfaces"][:, :, index]
    vector_year = np.asarray(elist["vector_year"], dtype=float)
    log_q_axis = np.asarray(elist["vector_logQ"], dtype=float)
    start0 = int(np.argmin(np.abs(vector_year - year0)))
    start1 = int(np.argmin(np.abs(vector_year - year1)))
    s0 = surfaces[start0:start0 + 17]
    s1 = surfaces[start1:start1 + 17]
    if plot_percent:
        diff = (s1 - s0) * 100.0 / s0
    else:
        diff = s1 - s0

    if max_diff is None:
        low, high = np.nanquantile(diff, [0.05, 0.95])
    elif np.ndim(max_diff) == 0:
        low, high = -max_diff, max_diff
    else:
        low, high = min(max_diff), max(max_diff)
    levels = _pretty_levels(low, high, 15)

    unit = get_q_unit(q_unit)
    y_q = np.exp(log_q_axis) * unit["factor"]
    ticks = _q_ticks(elist["Daily"], y_q, unit["factor"], q_bottom, q_top)
    x = np.arange(17) / 16.0

    fig, ax = new_axes(ax, figsize=(8.0, 5.5))
    filled = ax.contourf(x, np.log10(y_q), diff.T, levels=levels,
                         cmap=_colormap(colors), extend="both")
    fig.colorbar(filled, ax=ax)
    if flow_duration:
        duration = _flow_duration_surface(elist["Daily"], log_q_axis, 17, span)
        ax.contour(x, np.log10(y_q), duration.T, levels=[pval, 1 - pval],
                   colors="black", linewidths=2)
    ax.set_ylim(np.log10(ticks[0]), np.log10(ticks[-1]))
    ax.set_yticks(np.log10(ticks))
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    month_starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) / 365.0
    ax.set_xticks(month_starts)
    ax.set_xticklabels([m + "1" for m in MONTH_ABBREV] + ["Jan1"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylabel(unit["label"])
    if print_title:
        what = _SURFACE_NAMES[index]
        if plot_percent:
            text = f"Estimated {what} percent change from {year0} to {year1}"
        else:
            text = f"Estimated {what} change from {year0} to {year1}"
        if flow_duration:
            text = text + (f"\nBlack lines are {pval * 100:.2g} and {(1 - pval) * 100:.2g} "
                           "flow percentiles")
        ax.set_title(site_title(elist, text), fontsize=10)
    fig.tight_layout()
    return fig, ax


def _dec_low_high(elist: dict):
    """R's DecLow / DecHigh for the smooth plots: INFO if set, else the samples."""
    info = elist["INFO"]
    if "DecLow" in info.columns and "DecHigh" in info.columns:
        return float(info["DecLow"].iloc[0]), float(info["DecHigh"].iloc[0])
    sample = elist["Sample"]
    dec_year = sample["DecYear"].to_numpy(dtype=float)
    return float(dec_year[0]), float(dec_year[-1])


def plot_conc_q_smooth(elist: dict, date1, date2=None, date3=None, q_low: float = None,
                       q_high: float = None, q_unit=2, conc_max=None, conc_min=None,
                       bw: bool = False, print_title: bool = True, min_num_obs: int = 100,
                       min_num_uncen: int = 50, window_y: float = 7, window_q: float = 2,
                       window_s: float = 0.5, log_scale: bool = False,
                       edge_adjust: bool = True, colors=("black", "red", "green"),
                       lw: float = 2, ax=None):
    """Modelled concentration against discharge on up to three dates.

    Mirrors R ``plotConcQSmooth()``: the WRTDS regression is evaluated at 48
    log-spaced discharges between ``q_low`` and ``q_high`` on each date, so
    the curves show how the concentration-discharge relation has changed.

    Args:
        elist: eList dict with Sample.
        date1, date2, date3: Dates, ``"YYYY-MM-DD"``. ``date2`` and
            ``date3`` are optional.
        q_low, q_high: Discharge range in ``q_unit``.
        q_unit: EGRET discharge unit. Default 2.
        conc_max, conc_min: y range.
        bw: Black lines with different dash styles instead of colours.
        print_title: Add EGRET's title.
        min_num_obs, min_num_uncen, window_y, window_q, window_s, edge_adjust:
            Regression settings, R's defaults.
        log_scale: Log concentration axis as well.
        colors: Line colours.
        lw: Line width.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``; the curves are on ``ax.lines``.
    """
    unit = get_q_unit(q_unit)
    log_q_low = np.log(q_low)
    log_q_high = np.log(q_high)
    q_values = np.exp(np.linspace(log_q_low, log_q_high, 48))
    log_q_model = np.log(q_values / unit["factor"])
    dec_low, dec_high = _dec_low_high(elist)

    fig, ax = new_axes(ax)
    styles = ["-", "-", "-"]
    line_colors = list(colors)
    if bw:
        styles = ["-", "--", ":"]
        line_colors = ["black", "black", "black"]
    curves = []
    for k, date in enumerate((date1, date2, date3)):
        if date is None:
            continue
        stamp = pd.Timestamp(date)
        dec_year = stamp.year + (stamp.dayofyear - 0.5) / 366.0
        result = run_surv_reg_points(
            np.full(48, dec_year), log_q_model, dec_low, dec_high, elist["Sample"],
            window_y=window_y, window_q=window_q, window_s=window_s,
            min_num_obs=min_num_obs, min_num_uncen=min_num_uncen, edge_adjust=edge_adjust,
        )
        conc = result[:, 2]
        curves.append(conc)
        ax.plot(q_values, conc, styles[k], color=line_colors[k], linewidth=lw,
                label=str(stamp.date()))
    ax.set_xscale("log")
    if log_scale:
        ax.set_yscale("log")
    top = conc_max if conc_max is not None else 1.05 * np.nanmax(np.concatenate(curves))
    bottom = conc_min if conc_min is not None else 0.0
    if log_scale and conc_min is None:
        bottom = 0.95 * np.nanmin(np.concatenate(curves))
    ax.set_ylim(bottom, top)
    ax.set_xlim(q_low, q_high)
    ax.set_xlabel(unit["label"])
    ax.set_ylabel("Concentration, in mg/L")
    ax.legend(frameon=False)
    if print_title:
        ax.set_title(site_title(elist, "Estimated Concentration Versus Discharge Relationship\n"
                                       "at specific dates"), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_conc_time_smooth(elist: dict, q1: float, q2=None, q3=None, center_date: str = "07-01",
                          year_start: float = None, year_end: float = None, q_unit=2,
                          conc_max=None, conc_min=None, bw: bool = False,
                          print_title: bool = True, min_num_obs: int = 100,
                          min_num_uncen: int = 50, window_y: float = 10,
                          window_q: float = 2, window_s: float = 0.5, lw: float = 2,
                          log_scale: bool = False, edge_adjust: bool = True,
                          colors=("black", "red", "green"), ax=None):
    """Modelled concentration over time at up to three discharges.

    Mirrors R ``plotConcTimeSmooth()``: the regression is evaluated once a
    year on ``center_date`` (``"MM-DD"``) at each discharge. Note R's default
    ``windowY`` here is 10, not 7.

    Args:
        elist: eList dict with Sample.
        q1, q2, q3: Discharges in ``q_unit``; ``q2`` and ``q3`` optional.
        center_date: Day of the year, ``"MM-DD"``. Default ``"07-01"``.
        year_start, year_end: x range. Default: the sample record.
        Other arguments: as for :func:`plot_conc_q_smooth`.

    Returns:
        ``(fig, ax)``.
    """
    unit = get_q_unit(q_unit)
    dec_low, dec_high = _dec_low_high(elist)
    sample_years = elist["Sample"]["DecYear"].to_numpy(dtype=float)
    if year_start is None:
        year_start = float(np.floor(np.nanmin(sample_years)))
    if year_end is None:
        year_end = float(np.ceil(np.nanmax(sample_years)))

    center = pd.Timestamp("2001-" + center_date)
    y_dec = (center.dayofyear - 0.5) / 366.0
    x = np.arange(np.trunc(year_start) + y_dec, year_end + 1e-9, 1.0)

    fig, ax = new_axes(ax)
    styles = ["-", "-", "-"]
    line_colors = list(colors)
    if bw:
        styles = ["-", "--", ":"]
        line_colors = ["black", "black", "black"]
    curves = []
    for k, q in enumerate((q1, q2, q3)):
        if q is None:
            continue
        log_q = np.full(len(x), np.log(q / unit["factor"]))
        result = run_surv_reg_points(
            x, log_q, dec_low, dec_high, elist["Sample"],
            window_y=window_y, window_q=window_q, window_s=window_s,
            min_num_obs=min_num_obs, min_num_uncen=min_num_uncen, edge_adjust=edge_adjust,
        )
        conc = result[:, 2]
        curves.append(conc)
        ax.plot(x, conc, styles[k], color=line_colors[k], linewidth=lw,
                label=f"{q:g} {unit['short_name']}")
    if log_scale:
        ax.set_yscale("log")
    top = conc_max if conc_max is not None else 1.05 * np.nanmax(np.concatenate(curves))
    bottom = conc_min if conc_min is not None else 0.0
    if log_scale and conc_min is None:
        bottom = 0.95 * np.nanmin(np.concatenate(curves))
    ax.set_ylim(bottom, top)
    date_ticks(ax, year_start, year_end)
    ax.set_ylabel("Concentration, in mg/L")
    ax.legend(frameon=False)
    if print_title:
        month = MONTH_ABBREV[center.month - 1]
        ax.set_title(site_title(elist, f"Estimated Concentration Versus Time\n"
                                       f"{month} {center.day}"), fontsize=10)
    fig.tight_layout()
    return fig, ax
