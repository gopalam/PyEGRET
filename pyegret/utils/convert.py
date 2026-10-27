"""
Small conversion helpers used throughout PyEGRET.

Each one replaces a pandas call chain that would otherwise be repeated in
many places, so the calling code can stay one step per line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["as_float_array", "as_dates", "clean_text"]


def as_float_array(values) -> np.ndarray:
    """Convert values to a float NumPy array; anything non-numeric becomes NaN.

    Args:
        values: Series, list or array of numbers or numeric strings.

    Returns:
        1-D float64 ndarray.
    """
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return np.asarray(numeric, dtype=float)


def as_dates(values, utc: bool = False) -> pd.Series:
    """Convert values to timezone-naive calendar dates (time of day dropped).

    Unparseable entries become NaT. Timezone-aware input is converted to UTC
    before the timezone is dropped, so EGRET's arithmetic against naive
    constants (such as the 1850-01-01 Julian origin) works.

    Args:
        values: Series, list or array of dates or date strings.
        utc: Parse as UTC. Use for strings carrying mixed UTC offsets.

    Returns:
        pandas Series of datetime64[ns], normalised to midnight.
    """
    dates = pd.to_datetime(pd.Series(values), errors="coerce", utc=utc)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("UTC")
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    return dates


def clean_text(values, lower: bool = False) -> pd.Series:
    """Convert values to stripped strings, with missing values as ``""``.

    Args:
        values: Series, list or array of codes or labels.
        lower: Also convert to lower case.

    Returns:
        pandas Series of str, with a fresh 0..n-1 index.
    """
    text = pd.Series(values)
    text = text.reset_index(drop=True)
    text = text.fillna("")
    text = text.astype(str)
    text = text.str.strip()
    if lower:
        text = text.str.lower()
    return text
