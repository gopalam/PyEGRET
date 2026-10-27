"""
The eList: a plain dict holding one site's record, mirroring R's ``eList``.

In R EGRET an eList is a named list with elements ``INFO``, ``Daily``,
``Sample`` and ``surfaces``. PyEGRET uses an ordinary Python dict with the
same keys, plus the two grid axes that R keeps as attributes of ``surfaces``:

=================  ==========================================================
key                contents
=================  ==========================================================
``"INFO"``         1-row DataFrame of site and parameter metadata
``"Daily"``        DataFrame, one row per day (Date, Q, LogQ, DecYear, ...)
``"Sample"``       DataFrame, one row per water-quality sample, or None
``"surfaces"``     ndarray (n_year, n_logQ, 3): yHat, SE, ConcHat; or None
``"vector_year"``  1-D array, decimal-year axis of ``surfaces``; or None
``"vector_logQ"``  1-D array, log-discharge axis of ``surfaces``; or None
=================  ==========================================================

Functions that fit a model may add further keys (``"window_y_used"``,
``"wall_date"``); each documents the keys it writes.

Example::

    elist = make_elist(INFO, Daily, Sample)
    elist = model_estimation(elist)
    daily = elist["Daily"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_elist", "has_sample", "has_model", "describe_elist", "n_observations",
           "n_censored_vals", "n_discharge", "blank_time"]


def make_elist(
    INFO: pd.DataFrame,
    Daily: pd.DataFrame,
    Sample: pd.DataFrame | None = None,
    surfaces: np.ndarray | None = None,
) -> dict:
    """Build an eList dict from its parts.

    Args:
        INFO: 1-row DataFrame of site and parameter metadata.
        Daily: Daily discharge DataFrame.
        Sample: Water-quality sample DataFrame, or None.
        surfaces: Fitted WRTDS surface array, or None.

    Returns:
        A dict with keys ``INFO``, ``Daily``, ``Sample``, ``surfaces``,
        ``vector_year`` and ``vector_logQ``.
    """
    elist = {
        "INFO": INFO,
        "Daily": Daily,
        "Sample": Sample,
        "surfaces": surfaces,
        "vector_year": None,
        "vector_logQ": None,
    }
    return elist


def has_sample(elist: dict) -> bool:
    """Return True if the eList holds a non-empty Sample DataFrame."""
    sample = elist.get("Sample")
    if sample is None:
        return False
    return len(sample) > 0


def has_model(elist: dict) -> bool:
    """Return True if WRTDS surfaces have been estimated."""
    return elist.get("surfaces") is not None


def describe_elist(elist: dict) -> str:
    """Return a short text summary of an eList (site, dates, samples, model).

    Args:
        elist: eList dict.

    Returns:
        Multi-line summary string. Print it to see it.
    """
    info = elist.get("INFO")
    site = "Unknown"
    if info is not None and "station_nm" in info.columns:
        site = info["station_nm"].iloc[0]

    daily = elist.get("Daily")
    date_min = "?"
    date_max = "?"
    n_days = 0
    if daily is not None and "Date" in daily.columns and len(daily) > 0:
        date_min = daily["Date"].min()
        date_min = date_min.date()
        date_max = daily["Date"].max()
        date_max = date_max.date()
        n_days = len(daily)

    n_sample = 0
    if has_sample(elist):
        n_sample = len(elist["Sample"])
    model_str = "no"
    if has_model(elist):
        model_str = "yes"

    text = (
        "eList\n"
        f"  Site:    {site}\n"
        f"  Daily:   {date_min} to {date_max} ({n_days} days)\n"
        f"  Samples: {n_sample}\n"
        f"  Model:   {model_str}"
    )
    return text


def n_observations(elist: dict):
    """Number of samples. Mirrors R ``nObservations()``; None without a Sample."""
    if elist.get("Sample") is None:
        return None
    return len(elist["Sample"])


def n_censored_vals(elist: dict):
    """Number of censored samples. Mirrors R ``nCensoredVals()``."""
    if elist.get("Sample") is None:
        return None
    return int(np.sum(elist["Sample"]["Uncen"].to_numpy() == 0))


def n_discharge(elist: dict):
    """Number of daily discharge values. Mirrors R ``nDischarge()``."""
    if elist.get("Daily") is None:
        return None
    return len(elist["Daily"])


def blank_time(elist: dict, start_blank, end_blank) -> dict:
    """Blank out the model results over a stretch with no samples.

    Mirrors R ``blankTime()``: sets ConcDay, FluxDay, FNConc and FNFlux (and
    GenConc / GenFlux after WRTDS-Kalman) to NaN from ``start_blank`` to
    ``end_blank`` inclusive, so annual results and plots do not report years
    the samples never saw. Use it for long gaps in sampling.

    Args:
        elist: eList dict with model results in Daily.
        start_blank: First date to blank, ``"YYYY-MM-DD"``.
        end_blank: Last date to blank.

    Returns:
        A new eList; the input is not changed.
    """
    daily = elist["Daily"].copy()
    dates = pd.to_datetime(daily["Date"])
    bad = (dates >= pd.Timestamp(start_blank)) & (dates <= pd.Timestamp(end_blank))
    bad = bad.to_numpy()
    for column in ("ConcDay", "FluxDay", "FNConc", "FNFlux", "GenConc", "GenFlux"):
        if column in daily.columns:
            values = daily[column].to_numpy(dtype=float)
            daily[column] = np.where(bad, np.nan, values)
    elist = dict(elist)
    elist["Daily"] = daily
    return elist
