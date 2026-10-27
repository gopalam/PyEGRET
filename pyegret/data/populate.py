"""
DataFrame population helpers: date columns, sample columns, and Q merging.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pyegret.utils.convert import as_dates, as_float_array

from pyegret.utils.date_utils import (
    day_of_water_year,
    julian_day,
    month_seq,
    to_decimal_year,
    water_year,
)

__all__ = [
    "populate_daily",
    "populate_date_columns",
    "populate_sample_columns",
    "merge_daily_q",
]

_TWO_PI = 2.0 * np.pi


def populate_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard EGRET date columns derived from the ``Date`` column.

    Adds: ``DecYear``, ``Month``, ``Day``, ``MonthSeq``, ``Julian``,
    ``waterYear``, ``SinDY``, ``CosDY``.

    ``Day`` is EGRET's leap-adjusted day of year, not plain day-of-year: every
    day after February 28 in a common year is shifted up by one, so March 1 is
    day 61 in every year and day 60 is February 29. Flow normalization groups
    reference discharges by this column, so the adjustment keeps leap and
    common years in step.

    Args:
        df: DataFrame containing a ``Date`` column of dtype datetime64.

    Returns:
        The same DataFrame with the new columns appended (in-place copy).
    """
    df = df.copy()

    # Normalise Date: coerce to datetime64, strip any timezone so that
    # arithmetic against tz-naive constants (e.g. julian_day origin 1850-01-01)
    # works correctly.  dataretrieval returns UTC-aware timestamps; EGRET works
    # with plain calendar dates.
    dates: pd.Series = pd.to_datetime(df["Date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("UTC")
        dates = dates.dt.tz_localize(None)
    df["Date"] = dates

    df["DecYear"] = to_decimal_year(dates).values
    df["Month"] = dates.dt.month.values
    df["Day"] = day_of_water_year(dates).values
    df["MonthSeq"] = month_seq(dates).values
    df["Julian"] = julian_day(dates).values
    df["waterYear"] = water_year(dates).values
    df["SinDY"] = np.sin(_TWO_PI * df["DecYear"].values)
    df["CosDY"] = np.cos(_TWO_PI * df["DecYear"].values)
    return df


def populate_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived concentration columns to a Sample DataFrame.

    Requires: ``ConcLow``, ``ConcHigh``, ``Uncen`` already present.
    Adds: ``ConcAve``, ``LogConcLow``, ``LogConcHigh``.

    ``LogConcLow`` is set to NaN when ``ConcLow`` <= 0 or is missing.
    Censored samples carry ``ConcLow = 0`` in PyEGRET and ``NA`` in R EGRET;
    both are accepted and mean the same thing, and the regression uses
    ``LogConcHigh`` for those rows either way.

    Args:
        df: Sample DataFrame with at least ``ConcLow``, ``ConcHigh``, ``Uncen``.

    Returns:
        DataFrame with new columns appended.
    """
    df = df.copy()
    cl = df["ConcLow"].to_numpy(dtype=float)
    ch = df["ConcHigh"].to_numpy(dtype=float)
    uncen = df["Uncen"].to_numpy()

    # A right-censored row (a ">" result) has no upper bound. R EGRET writes
    # NA there, and so does PyEGRET, so that it is never mistaken for a value
    # of zero.
    right_censored = (uncen == 0) & ((ch == 0) | np.isnan(ch))
    ch = np.where(right_censored, np.nan, ch)
    df["ConcHigh"] = ch

    # A left-censored row has no lower bound. PyEGRET writes 0 there; R EGRET
    # writes NA. Both mean the same thing, and EGRET's ConcAve treats it as
    # zero (a "< 0.05" result gets ConcAve 0.025), so accept either spelling.
    # A right-censored row gets 1.5 times its lower bound, as in R.
    low_or_zero = np.nan_to_num(cl, nan=0.0)
    df["ConcAve"] = np.where(right_censored, 1.5 * low_or_zero, (low_or_zero + ch) / 2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_low = np.where(cl > 0, np.log(cl), np.nan)
        log_high = np.log(ch)

    df["LogConcLow"] = log_low
    df["LogConcHigh"] = log_high
    return df


def merge_daily_q(
    daily: pd.DataFrame, sample: pd.DataFrame
) -> pd.DataFrame:
    """Merge daily discharge (Q, LogQ) onto the Sample DataFrame by date.

    This is the step R EGRET performs inline inside ``mergeReport()``:
    ``merge(Daily[, c("Date", "Q", "LogQ")], Sample, by = "Date", all.y = TRUE)``.
    Dates must match exactly; a sample with no discharge on its own date keeps
    the row with Q = NaN, and a warning says how many. It is not R's
    ``populateConcentrations`` (that is
    :func:`pyegret.data.quality.populate_concentrations`).

    Uses ``pd.merge_asof`` with a 1-day tolerance and nearest-direction
    matching, so each sample is matched to the closest available daily
    record within ±1 day.

    Args:
        daily: Daily DataFrame with at least ``Date``, ``Q``, ``LogQ``.
        sample: Sample DataFrame with at least a ``Date`` column.

    Returns:
        Copy of ``sample`` with ``Q`` and ``LogQ`` columns added/updated.
    """
    daily_q = daily[["Date", "Q", "LogQ"]]
    daily_q = daily_q.sort_values("Date")
    daily_q = daily_q.drop_duplicates(subset="Date")
    sample_sorted = sample.sort_values("Date")

    # Drop any existing Q / LogQ to avoid suffix collisions
    for col in ("Q", "LogQ"):
        if col in sample_sorted.columns:
            sample_sorted = sample_sorted.drop(columns=[col])

    merged = pd.merge(sample_sorted, daily_q, on="Date", how="left")
    merged = merged.reset_index(drop=True)

    n_missing = int(np.sum(merged["Q"].isna()))
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} Sample dates do not have corresponding flow data. "
            "Not all EGRET functions will work correctly.",
            UserWarning,
            stacklevel=2,
        )
    return merged


def _trailing_mean(values: np.ndarray, n: int) -> np.ndarray:
    """Mean of each value and the n - 1 before it; NaN for the first n - 1.

    Same as R ``stats::filter(x, rep(1/n, n), sides = 1)``: any missing value
    inside the window makes that mean missing.
    """
    result = np.full(len(values), np.nan)
    if len(values) < n:
        return result
    cumulative = np.cumsum(np.concatenate([[0.0], values]))
    result[n - 1:] = (cumulative[n:] - cumulative[:-n]) / n
    return result


def populate_daily(
    dates,
    values,
    q_convert: float = 1.0,
    qualifier=None,
    adjust: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build an EGRET Daily DataFrame from raw dates and discharge values.

    Mirrors R ``populateDaily()`` (without its optional gap filling):

    - ``Q = value / q_convert``, so pass 35.314667 for values in ft3/s.
    - The record is expanded to every calendar day between the first and last
      date; days with no value get Q = NaN rather than being dropped.
    - If ``adjust`` and there are zero-flow days but no negative ones, every
      day gets ``0.001 * mean(Q)`` added, so the log is defined. EGRET
      reports this shift; so does this function when ``verbose``.
    - ``LogQ``, the date columns, and the 7- and 30-day trailing means ``Q7``
      and ``Q30`` (records of 30 days or more) are added.

    Args:
        dates: Dates of the values.
        values: Discharge values, in the units ``q_convert`` divides out.
        q_convert: Divisor converting the input to m3/s. Default 1 (already
            m3/s). EGRET's qUnit 1 (cfs) is 35.314667.
        qualifier: Optional data-qualifier codes, one per value.
        adjust: Shift zero flows as EGRET does. Default True.
        verbose: Print EGRET's messages about zero, negative and missing
            flows. Default True.

    Returns:
        Daily DataFrame, one row per calendar day.
    """
    clean_dates = as_dates(dates)
    raw = pd.DataFrame({
        "Date": clean_dates.to_numpy(),
        "Q": as_float_array(values) / q_convert,
    })
    if qualifier is None:
        raw["Qualifier"] = ""
    else:
        raw["Qualifier"] = np.asarray(qualifier)
    raw = raw.dropna(subset=["Date"])
    raw = raw.drop_duplicates(subset="Date", keep="first")

    all_dates = pd.date_range(raw["Date"].min(), raw["Date"].max(), freq="D")
    daily = pd.DataFrame({"Date": all_dates})
    daily = pd.merge(daily, raw, on="Date", how="left")

    q = daily["Q"].to_numpy(dtype=float)
    n_zero = int(np.sum(q == 0))
    n_negative = int(np.sum(q < 0))
    if n_negative > 0 and verbose:
        print(f"There are {n_negative} negative flow days.")
        print("Many EGRET functions will not work with negative values.")

    q_shift = 0.0
    if adjust and n_zero > 0 and n_negative == 0:
        q_shift = 0.001 * np.nanmean(q)
        if verbose:
            print(f"There are {n_zero} zero flow days and no negative flow days.")
            print(f"All days had {round(q_shift, 4)} cms added to the discharge value.")
    elif adjust and n_negative > 0 and verbose:
        print(f"Adjust is TRUE but there are {n_negative} negative flow days.")
        print("Discharge was not adjusted.")

    q = q + q_shift
    daily["Q"] = q
    with np.errstate(divide="ignore", invalid="ignore"):
        daily["LogQ"] = np.log(q)
    daily = populate_date_columns(daily)
    daily["i"] = np.arange(1, len(daily) + 1)

    if len(daily) >= 30:
        daily["Q7"] = _trailing_mean(q, 7)
        daily["Q30"] = _trailing_mean(q, 30)
    elif verbose:
        print("Rolling means not calculated because there are fewer than 30 days.")

    n_missing = int(np.sum(np.isnan(q)))
    if verbose and n_missing > 0:
        percent = 100.0 * n_missing / len(daily)
        print(f"{percent:.3g}% missing data.")
        print("Many EGRET functions will not work with missing values.")

    return daily
