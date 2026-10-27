"""
Plots of the fitted model's residuals and its fit to the samples.

Mirrors R ``plotResidPred()``, ``plotResidQ()``, ``plotResidTime()``,
``boxResidMonth()``, ``boxConcThree()``, ``plotConcPred()``,
``plotFluxPred()`` and ``fluxBiasMulti()``. The residuals are the
cross-validated ones ``model_estimation`` stores (``yHat``, ``SE``).

Censored samples are drawn as vertical lines over their interval, or, with
``random_censored=True``, as open circles at values drawn from the fitted
distribution (:func:`~pyegret.data.quality.make_augmented_sample`).
"""
from __future__ import annotations

import numpy as np

from pyegret.analysis.error_stats import flux_bias_stat
from pyegret.analysis.setup_years import get_pa
from pyegret.data.quality import make_augmented_sample
from pyegret.plots.data_plots import box_q_twice
from pyegret.plots.plot_utils import censored_segments, date_ticks, new_axes, select_days, site_title
from pyegret.utils.season import MONTH_ABBREV, set_season_label
from pyegret.utils.units import get_flux_unit, get_q_unit

__all__ = [
    "plot_resid_pred",
    "plot_resid_q",
    "plot_resid_time",
    "box_resid_month",
    "box_conc_three",
    "plot_conc_pred",
    "plot_flux_pred",
    "flux_bias_multi",
]


def _model_sample(elist: dict, random_censored: bool):
    """Sample in the period of analysis, augmented if asked; and the PA."""
    sample = elist["Sample"]
    if "yHat" not in sample.columns or "SE" not in sample.columns:
        raise ValueError("This function requires running model_estimation on eList")
    pa_start, pa_long = get_pa(elist)
    if random_censored and "rResid" not in sample.columns:
        sample = make_augmented_sample(elist)["Sample"]
    if pa_long != 12:
        sample = select_days(sample, pa_long, pa_start)
    return sample, pa_start, pa_long


def _residuals(sample, std_resid: bool, random_censored: bool):
    """(y_low, y_high, uncen) for a residual plot, in log units."""
    y_hat = sample["yHat"].to_numpy(dtype=float)
    se = sample["SE"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy()
    if random_censored:
        y_high = sample["rResid"].to_numpy(dtype=float)
        y_low = y_high
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            y_low = np.log(sample["ConcLow"].to_numpy(dtype=float)) - y_hat
            y_high = np.log(sample["ConcHigh"].to_numpy(dtype=float)) - y_hat
    if std_resid:
        y_low = y_low / se
        y_high = y_high / se
    return y_low, y_high, uncen


def _resid_scatter(ax, x, y_low, y_high, uncen, random_censored, col, lw, h_line=True):
    """Points for detections, segments or open circles for censored values."""
    detected = uncen == 1
    ax.plot(x[detected], y_high[detected], "o", color=col, markersize=3)
    finite = np.concatenate([y_high[np.isfinite(y_high)], y_low[np.isfinite(y_low)]])
    span = np.nanmax(np.abs(finite)) * 1.05
    ax.set_ylim(-span, span)
    if random_censored:
        ax.plot(x[~detected], y_high[~detected], "o", color=col, markersize=3,
                markerfacecolor="none")
    else:
        censored_segments(ax, -span, y_low, y_high, x, uncen, color=col, lw=lw)
    if h_line:
        ax.axhline(0.0, color=col, linewidth=1)


def _finish(fig, ax, elist, print_title, text, pa_start, pa_long):
    if print_title:
        if pa_long != 12:
            text = text + "\n" + set_season_label(pa_start, pa_long)
        ax.set_title(site_title(elist, text), fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_resid_pred(elist: dict, std_resid: bool = False, print_title: bool = True,
                    col: str = "black", lw: float = 1, random_censored: bool = False,
                    ax=None):
    """Residual against estimated log concentration. Mirrors R ``plotResidPred()``.

    Args:
        elist: eList dict after model_estimation.
        std_resid: Divide residuals by SE.
        print_title: Add EGRET's title.
        col, lw: Colour and line width.
        random_censored: Draw censored values as random draws.
        ax: Axes to draw on.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    x = sample["yHat"].to_numpy(dtype=float)
    y_low, y_high, uncen = _residuals(sample, std_resid, random_censored)
    fig, ax = new_axes(ax)
    _resid_scatter(ax, x, y_low, y_high, uncen, random_censored, col, lw)
    ax.set_xlabel("Estimated concentration in natural log units")
    ax.set_ylabel("Standardized residual in natural log units" if std_resid
                  else "Residual in natural log units")
    return _finish(fig, ax, elist, print_title, "Residual versus Estimated Concentration",
                   pa_start, pa_long)


def plot_resid_q(elist: dict, q_unit=2, std_resid: bool = False, print_title: bool = True,
                 col: str = "black", lw: float = 1, random_censored: bool = False, ax=None):
    """Residual against discharge (log axis). Mirrors R ``plotResidQ()``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    unit = get_q_unit(q_unit)
    x = sample["Q"].to_numpy(dtype=float) * unit["factor"]
    y_low, y_high, uncen = _residuals(sample, std_resid, random_censored)
    fig, ax = new_axes(ax)
    ax.set_xscale("log")
    _resid_scatter(ax, x, y_low, y_high, uncen, random_censored, col, lw)
    ax.set_xlabel(unit["label"])
    ax.set_ylabel("Standardized residual in natural log units" if std_resid
                  else "Residual in natural log units")
    return _finish(fig, ax, elist, print_title, "Residual versus Discharge",
                   pa_start, pa_long)


def plot_resid_time(elist: dict, std_resid: bool = False, print_title: bool = True,
                    h_line: bool = True, col: str = "black", lw: float = 1,
                    random_censored: bool = False, ax=None):
    """Residual against time. Mirrors R ``plotResidTime()``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    x = sample["DecYear"].to_numpy(dtype=float)
    y_low, y_high, uncen = _residuals(sample, std_resid, random_censored)
    fig, ax = new_axes(ax)
    _resid_scatter(ax, x, y_low, y_high, uncen, random_censored, col, lw, h_line)
    date_ticks(ax, np.floor(np.nanmin(x)), np.ceil(np.nanmax(x)))
    ax.set_ylabel("Standardized residual in natural log units" if std_resid
                  else "Residual in natural log units")
    return _finish(fig, ax, elist, print_title, "Residual versus Time", pa_start, pa_long)


def box_resid_month(elist: dict, std_resid: bool = False, print_title: bool = True,
                    random_censored: bool = False, ax=None):
    """Box plots of residuals by month. Mirrors R ``boxResidMonth()``.

    Without random draws the residual is ``log(ConcAve) - yHat``; with them
    it is ``rResid``. Box widths are proportional to the square root of the
    number of samples, as R's ``varwidth = TRUE``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    if random_censored:
        resid = sample["rResid"].to_numpy(dtype=float)
    else:
        resid = np.log(sample["ConcAve"].to_numpy(dtype=float)) - sample["yHat"].to_numpy(dtype=float)
    if std_resid:
        resid = resid / sample["SE"].to_numpy(dtype=float)
    month = sample["Month"].to_numpy(dtype=int)

    groups = []
    counts = []
    for m in range(1, 13):
        values = resid[(month == m) & np.isfinite(resid)]
        groups.append(values)
        counts.append(len(values))
    widths = 0.8 * np.sqrt(np.array(counts) / max(max(counts), 1))
    widths = np.where(widths > 0, widths, 0.1)

    fig, ax = new_axes(ax)
    ax.boxplot(groups, widths=widths)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(MONTH_ABBREV)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Standardized Residuals" if std_resid else "Residuals")
    return _finish(fig, ax, elist, print_title, "Residuals by Month", pa_start, pa_long)


def box_conc_three(elist: dict, print_title: bool = True, more_title: str = "WRTDS", ax=None):
    """Box plots of sampled values, estimates on sampled days, and all days.

    Mirrors R ``boxConcThree()``: observed ``ConcAve``, model ``ConcHat`` on
    the sampled days, and ``ConcDay`` on every day.

    Returns:
        ``(fig, ax)``.
    """
    sample = elist["Sample"]
    daily = elist["Daily"]
    groups = [sample["ConcAve"].to_numpy(dtype=float),
              sample["ConcHat"].to_numpy(dtype=float),
              daily["ConcDay"].to_numpy(dtype=float)]
    clean = []
    for values in groups:
        clean.append(values[np.isfinite(values)])
    counts = np.array([len(values) for values in clean], dtype=float)
    widths = 0.8 * np.sqrt(counts / np.max(counts))

    fig, ax = new_axes(ax)
    ax.boxplot(clean, widths=widths)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Sampled\nvalues", "Sampled\nestimates", "All\nestimates"])
    ax.set_ylim(0, 1.1 * max(np.max(values) for values in clean))
    ax.set_ylabel("Concentration, in mg/L")
    if print_title:
        ax.set_title(site_title(elist, f"Comparison of distribution of\nSampled "
                                       f"Values and {more_title} Estimates"), fontsize=10)
    fig.tight_layout()
    return fig, ax


def _pred_scatter(ax, x, y_low, y_high, uncen, log_scale, random_censored, col, lw, y_max):
    detected = uncen == 1
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.plot(x[detected], y_high[detected], "o", color=col, markersize=3)
    if random_censored:
        ax.plot(x[~detected], y_high[~detected], "o", color=col, markersize=3,
                markerfacecolor="none")
    top = y_max
    if top is None:
        top = 1.05 * np.nanmax(np.concatenate([x, y_high[np.isfinite(y_high)]]))
    bottom = 0.0
    if log_scale:
        positive = np.concatenate([x[x > 0], y_high[y_high > 0]])
        bottom = 0.95 * np.nanmin(positive)
    ax.set_xlim(bottom, top)
    ax.set_ylim(bottom, top)
    if not random_censored:
        censored_segments(ax, bottom, y_low, y_high, x, uncen, color=col, lw=lw)
    ax.plot([bottom, top], [bottom, top], "-", color=col, linewidth=1)


def plot_conc_pred(elist: dict, conc_max=None, log_scale: bool = False,
                   print_title: bool = True, col: str = "black", lw: float = 1,
                   random_censored: bool = False, ax=None):
    """Observed against estimated concentration, with a 1:1 line.

    Mirrors R ``plotConcPred()``. The estimate is the cross-validated
    ``ConcHat``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    x = sample["ConcHat"].to_numpy(dtype=float)
    if random_censored:
        y_high = sample["rObserved"].to_numpy(dtype=float)
        y_low = y_high
    else:
        y_low = sample["ConcLow"].to_numpy(dtype=float)
        y_high = sample["ConcHigh"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy()
    fig, ax = new_axes(ax)
    _pred_scatter(ax, x, y_low, y_high, uncen, log_scale, random_censored, col, lw, conc_max)
    ax.set_xlabel("Estimated concentration, in mg/L")
    ax.set_ylabel("Observed concentration, in mg/L")
    return _finish(fig, ax, elist, print_title, "Observed versus Estimated Concentration",
                   pa_start, pa_long)


def plot_flux_pred(elist: dict, flux_unit=3, flux_max=None, print_title: bool = True,
                   col: str = "black", lw: float = 1, log_scale: bool = False,
                   random_censored: bool = False, ax=None):
    """Observed against estimated flux on the sampled days. Mirrors R ``plotFluxPred()``.

    Returns:
        ``(fig, ax)``.
    """
    sample, pa_start, pa_long = _model_sample(elist, random_censored)
    unit = get_flux_unit(flux_unit)
    factor = unit["factor"] * 86.4
    q = sample["Q"].to_numpy(dtype=float)
    x = sample["ConcHat"].to_numpy(dtype=float) * q * factor
    if random_censored:
        y_high = sample["rObserved"].to_numpy(dtype=float) * q * factor
        y_low = y_high
    else:
        y_low = sample["ConcLow"].to_numpy(dtype=float) * q * factor
        y_high = sample["ConcHigh"].to_numpy(dtype=float) * q * factor
    uncen = sample["Uncen"].to_numpy()
    fig, ax = new_axes(ax)
    _pred_scatter(ax, x, y_low, y_high, uncen, log_scale, random_censored, col, lw, flux_max)
    ax.set_xlabel(f"Estimated flux, in {unit['unit_name']}")
    ax.set_ylabel(f"Observed flux, in {unit['unit_name']}")
    return _finish(fig, ax, elist, print_title, "Observed versus Estimated Flux",
                   pa_start, pa_long)


def flux_bias_multi(elist: dict, q_unit=2, flux_unit=3, more_title: str = "WRTDS",
                    random_censored: bool = False, col: str = "black", lw: float = 1):
    """Eight diagnostic panels and the flux bias statistic. Mirrors R ``fluxBiasMulti()``.

    Panels, in R's order: residual v. estimate, residual v. discharge,
    residual v. time, residuals by month, the three concentration
    distributions, observed v. estimated concentration, discharge on sampled
    days v. all days, observed v. estimated flux. The figure title reports
    ``bias3`` from :func:`~pyegret.analysis.error_stats.flux_bias_stat`.

    Returns:
        ``(fig, axes)`` with ``axes`` a 4 x 2 array.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    plot_resid_pred(elist, print_title=False, col=col, lw=lw,
                    random_censored=random_censored, ax=axes[0, 0])
    plot_resid_q(elist, q_unit=q_unit, print_title=False, col=col, lw=lw,
                 random_censored=random_censored, ax=axes[0, 1])
    plot_resid_time(elist, print_title=False, col=col, lw=lw,
                    random_censored=random_censored, ax=axes[1, 0])
    box_resid_month(elist, print_title=False, random_censored=random_censored, ax=axes[1, 1])
    box_conc_three(elist, print_title=False, ax=axes[2, 0])
    plot_conc_pred(elist, print_title=False, col=col, lw=lw,
                   random_censored=random_censored, ax=axes[2, 1])
    box_q_twice(elist, print_title=False, q_unit=q_unit, ax=axes[3, 0])
    plot_flux_pred(elist, flux_unit=flux_unit, print_title=False, col=col, lw=lw,
                   random_censored=random_censored, ax=axes[3, 1])

    bias = flux_bias_stat(elist["Sample"])
    bias3 = float(bias["bias3"].iloc[0])
    title = site_title(elist, f"Model is {more_title} Flux Bias Statistic {bias3:.3g}", sep=", ")
    pa_start, pa_long = get_pa(elist)
    if pa_long != 12:
        title = title + "\n" + set_season_label(pa_start, pa_long)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
