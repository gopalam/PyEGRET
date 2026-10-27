"""
Helpers shared by the EGRET plot functions.

Mirrors R ``censoredSegments()``, ``logPretty1()``, ``logPretty3()`` and
``selectDays()``, plus the small pieces every R plot repeats: the title from
INFO, the period of analysis, and making a new figure when no axes are given.

Every plot function in PyEGRET returns ``(fig, ax)`` and never calls
``plt.show()``; call it yourself, or ``fig.savefig(...)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "new_axes",
    "site_title",
    "censored_segments",
    "log_pretty1",
    "log_pretty3",
    "select_days",
    "date_ticks",
]


def new_axes(ax=None, figsize=(7.0, 5.0)):
    """Return ``(fig, ax)``: the given axes and their figure, or a new pair."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def site_title(elist: dict, text: str = "", sep: str = " ") -> str:
    """``"shortName paramShortName"`` from INFO, then ``text`` on a new line."""
    info = elist["INFO"]
    parts = []
    for column in ("shortName", "paramShortName"):
        if column in info.columns:
            parts.append(str(info[column].iloc[0]))
    title = sep.join(parts)
    if text:
        title = title + "\n" + text
    return title


def censored_segments(ax, y_bottom: float, y_low, y_high, x, uncen,
                      color: str = "black", lw: float = 1.0) -> None:
    """Draw each censored sample as a vertical line over its interval.

    Mirrors R ``censoredSegments()``: a missing lower bound starts at
    ``y_bottom``, a missing upper bound runs to the top of the axes.
    """
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    x = np.asarray(x, dtype=float)
    censored = np.asarray(uncen) == 0
    if not np.any(censored):
        return
    y_top = ax.get_ylim()[1]
    low = np.where(np.isfinite(y_low) & (y_low > -np.inf), y_low, y_bottom)
    if ax.get_yscale() == "log":
        low = np.where(low > 0, low, y_bottom)
    high = np.where(np.isfinite(y_high), y_high, y_top)
    ax.vlines(x[censored], low[censored], high[censored], colors=color, linewidth=lw)


def log_pretty1(x_min: float, x_max: float) -> np.ndarray:
    """Log-axis ticks at every power of ten. Mirrors R ``logPretty1()``."""
    first = int(np.floor(np.log10(x_min)))
    last = int(np.ceil(np.log10(x_max)))
    return 10.0 ** np.arange(first, last + 1)


def log_pretty3(x_min: float, x_max: float) -> np.ndarray:
    """Log-axis ticks at 1, 2, 5, 10, 20, 50, ... Mirrors R ``logPretty3()``.

    Keeps one tick beyond the data at each end, as R does.
    """
    x_min = x_min * 1.00001
    first = int(np.floor(np.log10(x_min)))
    last = int(np.ceil(np.log10(x_max)))
    ticks = []
    for decade in range(first, last):
        for step in (1.0, 2.0, 5.0):
            ticks.append(np.log10(step) + decade)
    ticks.append(float(last))
    ticks = np.array(ticks)
    keep_low = np.log10(x_min) < ticks
    keep_high = np.log10(x_max) > ticks
    shifted_low = np.append(keep_low[1:], keep_low[-1])
    shifted_high = np.insert(keep_high[:-1], 0, keep_high[0])
    keep = shifted_low & shifted_high
    return 10.0 ** ticks[keep]


def select_days(frame: pd.DataFrame, pa_long: int, pa_start: int) -> pd.DataFrame:
    """Keep only rows in the months of the period of analysis.

    Mirrors R ``selectDays()``.
    """
    months = []
    for k in range(pa_long):
        month = pa_start + k
        if month > 12:
            month = month - 12
        months.append(month)
    keep = np.isin(frame["Month"].to_numpy(dtype=int), months)
    subset = frame[keep]
    subset = subset.reset_index(drop=True)
    return subset


def date_ticks(ax, start: float, end: float) -> None:
    """Integer-year ticks for a decimal-year axis, about five of them."""
    span = end - start
    step = 1
    for candidate in (1, 2, 5, 10, 20, 50):
        if span / candidate <= 8:
            step = candidate
            break
    first = int(np.ceil(start / step) * step)
    ticks = np.arange(first, end + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlim(start, end)
