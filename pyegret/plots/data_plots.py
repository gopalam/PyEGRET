"""
Plots of the raw data, before any model is fitted.

Mirrors R ``plotConcQ()``, ``plotConcTime()``, ``boxConcMonth()``,
``boxQTwice()``, ``plotQTimeDaily()`` and ``multiPlotDataOverview()``.
These are the first look at a record: they show censoring, gaps, the
concentration-discharge relation, and whether the samples cover the range of
flows. Every function returns ``(fig, ax)`` (``(fig, axes)`` for several
panels) and never calls ``plt.show()``.
"""
from __future__ import annotations

import numpy as np

from pyegret.analysis.setup_years import get_pa
from pyegret.data.quality import make_augmented_sample
from pyegret.plots.plot_utils import censored_segments, date_ticks, new_axes, select_days, site_title
from pyegret.utils.season import MONTH_ABBREV, set_season_label
from pyegret.utils.units import get_q_unit

__all__ = [
    "plot_conc_q",
    "plot_conc_time",
    "box_conc_month",
    "box_q_twice",
    "plot_q_time_daily",
    "multi_plot_data_overview",
]


def _season_sample(elist: dict):
    pa_start, pa_long = get_pa(elist)
    sample = elist["Sample"]
    if pa_long != 12:
        sample = select_days(sample, pa_long, pa_start)
    return sample, pa_start, pa_long


def _title_with_season(elist, text, pa_start, pa_long):
    if pa_long != 12:
        text = text + "\n" + set_season_label(pa_start, pa_long)
    return site_title(elist, text)


def _conc_points(ax, elist, sample, x, log_scale, random_censored, conc_max, conc_min, col, lw):
    """Shared by plot_conc_q and plot_conc_time."""
    uncen = sample["Uncen"].to_numpy()
    detected = uncen == 1
    if random_censored:
        if "rObserved" not in sample.columns:
            augmented = dict(elist)
            augmented["Sample"] = sample
            sample = make_augmented_sample(augmented)["Sample"]
        y_high = sample["rObserved"].to_numpy(dtype=float)
        y_low = y_high
    else:
        y_low = sample["ConcLow"].to_numpy(dtype=float)
        y_high = sample["ConcHigh"].to_numpy(dtype=float)

    if log_scale:
        ax.set_yscale("log")
    ax.plot(x[detected], y_high[detected], "o", color=col, markersize=3)
    if random_censored:
        ax.plot(x[~detected], y_high[~detected], "o", color=col, markersize=3,
                markerfacecolor="none")

    top = conc_max
    if top is None:
        top = 1.05 * np.nanmax(y_high)
    bottom = conc_min
    if bottom is None:
        bottom = 0.0
        if log_scale:
            positive = np.concatenate([y_low[y_low > 0], y_high[y_high > 0]])
            bottom = 0.95 * np.nanmin(positive)
    ax.set_ylim(bottom, top)
    if not random_censored:
        censored_segments(ax, bottom, y_low, y_high, x, uncen, color=col, lw=lw)
    ax.set_ylabel("Concentration, in mg/L")


def plot_conc_q(elist: dict, q_unit=2, log_scale: bool = False,
                random_censored: bool = False, conc_max=None, conc_min=None,
                print_title: bool = True, col: str = "black", lw: float = 1, ax=None):
    """Concentration against discharge, discharge on a log axis. Mirrors R ``plotConcQ()``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _season_sample(elist)
    unit = get_q_unit(q_unit)
    x = sample["Q"].to_numpy(dtype=float) * unit["factor"]
    fig, ax = new_axes(ax)
    ax.set_xscale("log")
    _conc_points(ax, elist, sample, x, log_scale, random_censored, conc_max, conc_min, col, lw)
    ax.set_xlabel(unit["label"])
    if print_title:
        ax.set_title(_title_with_season(elist, "Concentration versus Discharge",
                                        pa_start, pa_long), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_conc_time(elist: dict, q_unit=2, year_start=None, year_end=None, q_lower=None,
                   q_upper=None, random_censored: bool = False, conc_max=None,
                   conc_min=None, print_title: bool = True, log_scale: bool = False,
                   col: str = "black", lw: float = 1, ax=None):
    """Concentration against time, optionally for a range of discharge.

    Mirrors R ``plotConcTime()``. ``q_lower`` / ``q_upper`` (in ``q_unit``)
    keep only samples taken at discharges in that range.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _season_sample(elist)
    unit = get_q_unit(q_unit)
    q = sample["Q"].to_numpy(dtype=float) * unit["factor"]
    keep = np.ones(len(sample), dtype=bool)
    text = "Concentration versus Time"
    if q_lower is not None:
        keep = keep & (q > q_lower)
        text = text + f"\nFor Discharge > {q_lower} {unit['unit_name']}"
    if q_upper is not None:
        keep = keep & (q < q_upper)
        text = text + f"\nFor Discharge < {q_upper} {unit['unit_name']}"
    dec_year = sample["DecYear"].to_numpy(dtype=float)
    if year_start is not None:
        keep = keep & (dec_year >= year_start)
    if year_end is not None:
        keep = keep & (dec_year <= year_end)
    subset = sample[keep]
    subset = subset.reset_index(drop=True)
    x = subset["DecYear"].to_numpy(dtype=float)

    fig, ax = new_axes(ax)
    _conc_points(ax, elist, subset, x, log_scale, random_censored, conc_max, conc_min, col, lw)
    start = year_start if year_start is not None else np.floor(np.nanmin(x))
    end = year_end if year_end is not None else np.ceil(np.nanmax(x))
    date_ticks(ax, start, end)
    if print_title:
        ax.set_title(_title_with_season(elist, text, pa_start, pa_long), fontsize=10)
    fig.tight_layout()
    return fig, ax


def box_conc_month(elist: dict, print_title: bool = True, log_scale: bool = False, ax=None):
    """Box plots of sampled concentration (ConcAve) by month. Mirrors R ``boxConcMonth()``.

    Box widths follow the number of samples, as R's ``varwidth = TRUE``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _season_sample(elist)
    conc = sample["ConcAve"].to_numpy(dtype=float)
    month = sample["Month"].to_numpy(dtype=int)
    groups = []
    counts = []
    for m in range(1, 13):
        values = conc[(month == m) & np.isfinite(conc)]
        groups.append(values)
        counts.append(len(values))
    widths = 0.8 * np.sqrt(np.array(counts) / max(max(counts), 1))
    widths = np.where(widths > 0, widths, 0.1)

    fig, ax = new_axes(ax)
    ax.boxplot(groups, widths=widths)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(MONTH_ABBREV)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.02 * np.nanmax(sample["ConcHigh"].to_numpy(dtype=float)))
    ax.set_xlabel("Month")
    ax.set_ylabel("Concentration, in mg/L")
    if print_title:
        ax.set_title(_title_with_season(elist, "Concentration Distribution by Month",
                                        pa_start, pa_long), fontsize=10)
    fig.tight_layout()
    return fig, ax


def box_q_twice(elist: dict, print_title: bool = True, q_unit=2, log_scale: bool = True,
                ax=None):
    """Discharge on sampled days beside discharge on all days. Mirrors R ``boxQTwice()``.

    Shows whether the samples cover the range of flows the daily record
    holds; a sampling program that misses floods shows up here first.

    Returns:
        ``(fig, ax)``.
    """
    pa_start, pa_long = get_pa(elist)
    sample = elist["Sample"]
    daily = elist["Daily"]
    if pa_long != 12:
        sample = select_days(sample, pa_long, pa_start)
        daily = select_days(daily, pa_long, pa_start)
    unit = get_q_unit(q_unit)
    q_sample = sample["Q"].to_numpy(dtype=float) * unit["factor"]
    q_daily = daily["Q"].to_numpy(dtype=float) * unit["factor"]
    q_sample = q_sample[np.isfinite(q_sample)]
    q_daily = q_daily[np.isfinite(q_daily)]
    counts = np.array([len(q_sample), len(q_daily)], dtype=float)
    widths = 0.8 * np.sqrt(counts / np.max(counts))

    fig, ax = new_axes(ax)
    ax.boxplot([q_sample, q_daily], widths=widths)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Sampled", "All"])
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(unit["label"])
    if print_title:
        ax.set_title(_title_with_season(elist, "Comparison of distribution of\n"
                                               "Sampled Discharges and All Daily Discharges",
                                        pa_start, pa_long), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_q_time_daily(elist: dict, year_start=None, year_end=None, q_lower=None,
                      q_unit=1, log_scale: bool = False, print_title: bool = True,
                      lw: float = 3, col: str = "red", ax=None):
    """Daily discharge over time. Mirrors R ``plotQTimeDaily()``.

    With ``q_lower``, only flows above it are drawn (R uses this to show
    floods), and the y axis starts there. R's default unit here is cfs.

    Returns:
        ``(fig, ax)``.
    """
    pa_start, pa_long = get_pa(elist)
    daily = elist["Daily"]
    if pa_long != 12:
        daily = select_days(daily, pa_long, pa_start)
    unit = get_q_unit(q_unit)
    x = daily["DecYear"].to_numpy(dtype=float)
    y = daily["Q"].to_numpy(dtype=float) * unit["factor"]
    if year_start is None:
        year_start = np.nanmin(x)
    if year_end is None:
        year_end = np.nanmax(x)
    keep = (x >= year_start) & (x <= year_end)
    x = x[keep]
    y = y[keep]

    fig, ax = new_axes(ax, figsize=(9.0, 5.0))
    if q_lower is None:
        ax.plot(x, y, "-", color=col, linewidth=lw / 3)
        bottom = 0.0
        if log_scale:
            bottom = np.nanmin(y[y > 0])
    else:
        # R draws a vertical line from q_lower up to each day's flow.
        above = y > q_lower
        ax.vlines(x[above], q_lower, y[above], colors=col, linewidth=lw / 3)
        bottom = q_lower
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylim(bottom, 1.05 * np.nanmax(y))
    date_ticks(ax, year_start, year_end)
    ax.set_ylabel(unit["label"])
    if print_title:
        info = elist["INFO"]
        name = str(info["shortName"].iloc[0]) if "shortName" in info.columns else ""
        text = name
        if pa_long != 12:
            text = text + "\n" + set_season_label(pa_start, pa_long)
        ax.set_title(text, fontsize=10)
    fig.tight_layout()
    return fig, ax


def multi_plot_data_overview(elist: dict, q_unit=2, random_censored: bool = False,
                             log_scale_conc: bool = True, log_scale_q: bool = True):
    """Four views of the raw data on one page. Mirrors R ``multiPlotDataOverview()``.

    Concentration v. discharge, concentration by month, concentration over
    time, and sampled v. all discharge.

    Returns:
        ``(fig, axes)`` with ``axes`` a 2 x 2 array.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_conc_q(elist, q_unit=q_unit, print_title=False, log_scale=log_scale_conc,
                random_censored=random_censored, ax=axes[0, 0])
    box_conc_month(elist, print_title=False, log_scale=log_scale_conc, ax=axes[0, 1])
    plot_conc_time(elist, print_title=False, log_scale=log_scale_conc,
                   random_censored=random_censored, ax=axes[1, 0])
    box_q_twice(elist, print_title=False, q_unit=q_unit, log_scale=log_scale_q, ax=axes[1, 1])
    fig.suptitle(site_title(elist, sep="\n"))
    fig.tight_layout()
    return fig, axes
