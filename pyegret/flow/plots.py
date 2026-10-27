"""
flowHistory figures: the Quantile-Kendall plot and single-statistic trends.

Styling follows EGRET so the figures stay recognisable to anyone who knows the
R package, with one deliberate change: significance is encoded by marker size
as well as colour. EGRET distinguishes its three p-value classes by colour
alone, which fails for a colourblind reader and in greyscale print.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["plot_quantile_kendall", "plot_flow_trend"]

# Significance classes, in EGRET's order. Colour carries the meaning and size
# repeats it, so neither has to be read alone.
_SIGNIFICANT = {"color": "#B2182B", "size": 26, "label": "p < 0.05"}
_MARGINAL = {"color": "#1A1A1A", "size": 16, "label": "0.05 - 0.1"}
_NOT_SIGNIFICANT = {"color": "#B8B8B8", "size": 9, "label": "p > 0.1"}

_GRID = {"color": "#D8D8D8", "linewidth": 0.6, "zorder": 0}
_INK = "#222222"
_MUTED = "#666666"

# Non-exceedance probabilities worth naming on the x axis.
_TICK_PROBABILITIES = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
_TICK_LABELS = ["0.001", "0.01", "0.1", "0.25", "0.5", "0.75", "0.9", "0.99", "0.999"]


def _classify(p_values: np.ndarray) -> list[dict]:
    """Assign each point to a significance class."""
    classes = []
    for p in p_values:
        if not np.isfinite(p):
            classes.append(_NOT_SIGNIFICANT)
        elif p < 0.05:
            classes.append(_SIGNIFICANT)
        elif p <= 0.1:
            classes.append(_MARGINAL)
        else:
            classes.append(_NOT_SIGNIFICANT)
    return classes


def plot_quantile_kendall(
    results: pd.DataFrame,
    title: str = "Quantile-Kendall",
    subtitle: str | None = None,
    p_column: str = "p_value_adj",
    ax=None,
):
    """Plot the trend in every order statistic against its plotting position.

    The x axis runs from the annual minimum daily discharge on the left to the
    annual maximum on the right, spaced by the normal quantile of
    non-exceedance probability so the tails get the room they need. A flat
    line of points near zero means a stable flow regime; a tilt means the
    distribution is stretching or compressing.

    Args:
        results: Output of
            :func:`~pyegret.flow.quantile_kendall.quantile_kendall`.
        title: Plot title, normally the site name.
        subtitle: Second line, e.g. the period analysed.
        p_column: Which p-value drives the significance classes. Default
            ``"p_value_adj"``, the serial-correlation-adjusted one EGRET
            plots.
        ax: Existing matplotlib axes. A new figure is made if None.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = ax.figure

    z = results["z"].to_numpy(dtype=float)
    slope = results["slope_pct"].to_numpy(dtype=float)
    classes = _classify(results[p_column].to_numpy(dtype=float))

    ax.axhline(0.0, color=_MUTED, linewidth=0.9, zorder=1)
    ax.grid(True, axis="both", **_GRID)
    ax.set_axisbelow(True)

    for style in (_NOT_SIGNIFICANT, _MARGINAL, _SIGNIFICANT):
        mask = np.array([c is style for c in classes])
        if not mask.any():
            continue
        ax.scatter(
            z[mask], slope[mask],
            s=style["size"], c=style["color"],
            label=f"{style['label']}  (n={int(mask.sum())})",
            edgecolors="none", zorder=3,
        )

    ticks = [float(np.sqrt(2) * 0 + v) for v in _TICK_PROBABILITIES]
    from scipy.stats import norm as _norm
    ax.set_xticks([_norm.ppf(p) for p in ticks])
    ax.set_xticklabels(_TICK_LABELS)

    ax.set_xlabel("Daily non-exceedance probability", color=_INK)
    ax.set_ylabel("Trend in discharge (% per year)", color=_INK)
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}",
                 color=_INK, loc="left")

    legend = ax.legend(title="Mann-Kendall", loc="upper left", frameon=False,
                       fontsize=9, title_fontsize=9)
    legend_title = legend.get_title()
    legend_title.set_color(_MUTED)
    for text in legend.get_texts():
        text.set_color(_INK)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AAAAAA")
    ax.tick_params(colors=_MUTED)

    fig.tight_layout()
    return fig, ax


def plot_flow_trend(
    annual: pd.DataFrame,
    istat: int,
    title: str | None = None,
    ax=None,
):
    """Plot one annual flow statistic with its smoothed trend.

    Args:
        annual: Output of
            :func:`~pyegret.flow.annual_series.make_annual_series`.
        istat: Which statistic to draw, 1 through 8.
        title: Plot title. Defaults to the statistic's name.
        ax: Existing matplotlib axes. A new figure is made if None.

    Returns:
        ``(fig, ax)``.

    Raises:
        ValueError: If the requested statistic is not in the frame.
    """
    import matplotlib.pyplot as plt

    series = annual[annual["istat"] == istat]
    if series.empty:
        raise ValueError(f"istat {istat} is not present in the annual series.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
    else:
        fig = ax.figure

    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)

    ax.scatter(series["DecYear"], series["value"], s=22, c="#2C6E9B",
               edgecolors="none", zorder=3, label="annual value")
    ax.plot(series["DecYear"], series["smoothed"], color="#B2182B",
            linewidth=2.0, zorder=4, label="smoothed trend")

    ax.set_yscale("log")
    ax.set_xlabel("Year", color=_INK)
    ax.set_ylabel("Discharge (m$^3$/s)", color=_INK)
    ax.set_title(title or series["statistic"].iloc[0], color=_INK, loc="left")

    legend = ax.legend(loc="best", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_INK)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AAAAAA")
    ax.tick_params(colors=_MUTED)

    fig.tight_layout()
    return fig, ax
