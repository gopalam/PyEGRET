"""
Date utility functions for EGRET decimal-year and sequential-date calculations.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

__all__ = [
    "day_of_water_year",
    "water_year",
    "to_decimal_year",
    "month_seq",
    "julian_day",
    "check_date_format",
]

_JULIAN_ORIGIN = pd.Timestamp("1850-01-01")


def _as_series(dates) -> pd.Series:
    """Coerce DatetimeIndex or array-like to a pd.Series of datetime64."""
    if not isinstance(dates, pd.Series):
        return pd.Series(dates)
    return dates


def to_decimal_year(dates: pd.Series) -> pd.Series:
    """Convert a Series of dates to decimal year.

    Formula: year + (day_of_year - 1) / days_in_year
    Leap years use 366, non-leap years use 365.

    Args:
        dates: pandas Series or DatetimeIndex of datetime64 values.

    Returns:
        pandas Series of float64 decimal years.
    """
    dates = _as_series(dates)
    year = dates.dt.year
    day_of_year = dates.dt.day_of_year
    days_in_year = dates.dt.is_leap_year.map({True: 366, False: 365})
    return (year + (day_of_year - 1) / days_in_year).astype("float64")


def month_seq(dates: pd.Series, base_date: pd.Timestamp = None) -> pd.Series:
    """Return EGRET's absolute month sequence, ``(year - 1850) * 12 + month``.

    January 1850 is 1, February 1850 is 2, and so on. The origin matches
    :func:`julian_day`, and the anchoring matters: period-of-analysis windows
    step through the record in strides of 12 starting from a month number, so
    ``MonthSeq`` must map to the calendar month the same way in every record.
    A record-relative sequence would make April mean different things in
    different datasets.

    Args:
        dates: pandas Series or DatetimeIndex of datetime64 values.
        base_date: Optional origin, floored to the month. Given, month 1 is
            the month of ``base_date`` instead of January 1850. Use only when
            you want a record-relative index; EGRET compatibility needs the
            default.

    Returns:
        pandas Series of int64.
    """
    dates = _as_series(dates)
    if base_date is None:
        return ((dates.dt.year - 1850) * 12 + dates.dt.month).astype("int64")
    base_year: int = base_date.year
    base_month: int = base_date.month
    return (
        (dates.dt.year - base_year) * 12 + (dates.dt.month - base_month) + 1
    ).astype("int64")


def day_of_water_year(dates: pd.Series) -> pd.Series:
    """Return EGRET's leap-adjusted day-of-year, in which March 1 is always 61.

    Plain day-of-year shifts by one across the leap boundary: March 1 is day
    60 in a common year and day 61 in a leap year. EGRET removes that shift by
    adding 1 to every day after February 28 in a common year, so day numbers
    line up across years and day 60 is February 29 wherever it exists.

    This matters because flow normalization averages the fitted surface over
    the discharges observed on the same day of the year across all years. On
    an unadjusted index, three quarters of the calendar would be comparing
    common years against leap years one day out of step.

    Args:
        dates: pandas Series or DatetimeIndex of datetime64 values.

    Returns:
        pandas Series of int64 in ``1..366``.
    """
    dates = _as_series(dates)
    doy = dates.dt.day_of_year
    offset = (~dates.dt.is_leap_year).astype("int64")
    adjusted = doy.where(doy <= 59, doy + offset)
    return adjusted.astype("int64")


def water_year(dates: pd.Series) -> pd.Series:
    """Return the USGS water year: October through September, labelled by its end.

    October 1999 through September 2000 is water year 2000.

    Args:
        dates: pandas Series or DatetimeIndex of datetime64 values.

    Returns:
        pandas Series of int64.
    """
    dates = _as_series(dates)
    year = dates.dt.year
    in_oct_to_dec = dates.dt.month >= 10
    labelled = year.where(~in_oct_to_dec, year + 1)
    return labelled.astype("int64")


def julian_day(dates: pd.Series) -> pd.Series:
    """Return integer days elapsed since 1850-01-01 (EGRET convention).

    Args:
        dates: pandas Series or DatetimeIndex of datetime64 values.

    Returns:
        pandas Series of int64 (0 = 1850-01-01).
    """
    dates = _as_series(dates)
    return ((dates - _JULIAN_ORIGIN).dt.days).astype("int64")


def check_date_format(date_str: str) -> bool:
    """Return True if date_str is a valid YYYY-MM-DD string or empty string.

    Args:
        date_str: Candidate date string.

    Returns:
        True if the string is empty or matches YYYY-MM-DD format exactly.
    """
    if date_str == "":
        return True
    try:
        _dt.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
