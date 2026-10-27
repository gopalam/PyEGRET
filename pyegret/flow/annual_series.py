"""
flowHistory: streamflow trend analysis, no water-quality data required.

EGRET is two packages sharing one data structure. WRTDS needs concentration
samples; flowHistory needs only the daily discharge record, and asks a
different question: how has the *flow regime* changed?

The eight statistics are chosen so that different parts of the hydrograph can
move independently, because they usually do. Urbanisation raises flood peaks
while lowering baseflow; a new reservoir does the reverse; a warming snowpack
shifts the whole annual cycle earlier without changing the annual mean at all.
Looking only at mean flow would miss every one of those.

============  ===============================================
``istat``     statistic
============  ===============================================
1             minimum 1-day discharge
2             minimum 7-day mean
3             minimum 30-day mean
4             median of daily means
5             mean of daily means
6             maximum 30-day mean
7             maximum 7-day mean
8             maximum 1-day discharge
============  ===============================================

Each series is smoothed by the same locally weighted regression WRTDS uses -
tricube weights in time, fit to log discharge - with a default half-window of
20 years and the same edge adjustment.

One detail worth knowing: the three low-flow statistics are computed on a year
that starts in **April** rather than October, when the period of analysis is
the default water year. A summer drought spans the July-August boundary
comfortably, but an October-to-September year splits the autumn low-flow
period across two labels and can assign a single drought to two different
years. EGRET calls this the climate year, and it applies only to statistics
1-3.

Mirrors R EGRET's ``makeAnnualSeries``, ``flowDuration`` and ``cumQdate``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.model.tri_cube import tri_cube
from pyegret.utils.convert import as_float_array

__all__ = [
    "ISTAT_NAMES",
    "add_rolling_means",
    "make_annual_series",
    "flow_duration",
    "cum_q_date",
]

ISTAT_NAMES = {
    1: "minimum 1-day discharge",
    2: "minimum 7-day mean",
    3: "minimum 30-day mean",
    4: "median daily discharge",
    5: "mean daily discharge",
    6: "maximum 30-day mean",
    7: "maximum 7-day mean",
    8: "maximum 1-day discharge",
}

# Statistics 1-3 use a year starting in April when the period of analysis is
# the water year, so a summer low-flow period is not split across two labels.
_LOW_FLOW_ISTATS = (1, 2, 3)
_CLIMATE_YEAR_START = 4

# A year needs more than 26 valid days per month to be used at all.
_MIN_DAYS_PER_MONTH = 26

DEFAULT_SMOOTHING_WINDOW = 20.0


def add_rolling_means(daily: pd.DataFrame) -> pd.DataFrame:
    """Add the trailing 7-day and 30-day mean discharge columns.

    Mirrors EGRET's ``populateDaily``, which uses a one-sided (trailing)
    filter: ``Q7`` on a given day is the mean of that day and the six before
    it. The first six days are therefore NaN, and likewise the first 29 for
    ``Q30``.

    Args:
        daily: Daily DataFrame with a ``Q`` column, in date order.

    Returns:
        A copy with ``Q7`` and ``Q30`` added.
    """
    daily = daily.copy()
    q = pd.Series(as_float_array(daily["Q"]), index=daily.index)
    window_7 = q.rolling(7)
    window_30 = q.rolling(30)
    daily["Q7"] = window_7.mean()
    daily["Q30"] = window_30.mean()
    return daily


def _period_windows(daily: pd.DataFrame, pa_start: int, pa_long: int):
    """Yield (start, end) MonthSeq bounds for each period of analysis."""
    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    first, last = month_seq[0], month_seq[-1]

    starts = np.arange(pa_start, last + 1, 12)
    ends = starts + pa_long - 1
    keep = (ends >= first) & (starts <= last)
    return list(zip(starts[keep], ends[keep]))


def _smooth_log_series(
    years: np.ndarray,
    values: np.ndarray,
    window: float,
    edge_adjust: bool,
) -> np.ndarray:
    """Locally weighted linear regression of log(values) on year.

    The same smoother WRTDS uses on its surface, applied here to an annual
    series: at each year, fit a tricube-weighted straight line through
    log discharge and read off the fitted value.
    """
    smoothed = np.full(len(years), np.nan)
    finite = np.isfinite(years) & np.isfinite(values) & (values > 0)
    if finite.sum() < 2:
        return smoothed

    x = years[finite]
    y = np.log(values[finite])
    first, last = x[0], x[-1]

    fitted = np.empty(len(x))
    for i, xi in enumerate(x):
        this_window = window
        if edge_adjust:
            dist_to_edge = min(xi - first, last - xi)
            if dist_to_edge < window:
                this_window = 2.0 * window - dist_to_edge

        weights = tri_cube(x - xi, this_window)
        if weights.sum() <= 0:
            fitted[i] = np.nan
            continue

        # Weighted least squares of y on x, evaluated at xi.
        design = np.column_stack([np.ones_like(x), x])
        sqrt_w = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
        fitted[i] = beta[0] + beta[1] * xi

    smoothed[finite] = np.exp(fitted)
    return smoothed


def make_annual_series(
    elist: dict,
    pa_start: int = 10,
    pa_long: int = 12,
    window: float = DEFAULT_SMOOTHING_WINDOW,
    edge_adjust: bool = True,
) -> pd.DataFrame:
    """Compute and smooth the eight annual flow statistics.

    Mirrors R ``makeAnnualSeries()``, which returns a ``3 x 8 x nYears``
    array. This returns tidy long-form instead, one row per statistic per
    year, which is easier to filter, join and plot.

    Args:
        elist: eList dict with a Daily DataFrame. ``Q7`` and ``Q30`` are added if
            missing.
        pa_start: First month of the period of analysis. Default 10 (water
            year). Statistics 1-3 use April instead when the period is the
            full water year; see the module docstring.
        pa_long: Length of the period in months. Default 12.
        window: Half-window of the smoother, in years. Default 20.
        edge_adjust: Widen the smoothing window near the ends of the record.
            Default True.

    Returns:
        DataFrame with columns ``istat``, ``statistic``, ``Year``,
        ``DecYear``, ``value`` (the statistic, in m3/s) and ``smoothed``
        (the locally weighted fit). Years with insufficient data are omitted.

    Raises:
        ValueError: If Daily is missing or lacks the required columns.
    """
    daily = elist["Daily"]
    if daily is None or daily.empty:
        raise ValueError("eList has no Daily data.")
    for column in ("Q", "DecYear", "MonthSeq"):
        if column not in daily.columns:
            raise ValueError(
                f"Daily is missing {column!r}; run populate_date_columns first."
            )

    daily = daily.sort_values("MonthSeq" if "Date" not in daily.columns else "Date")
    if "Q7" not in daily.columns or "Q30" not in daily.columns:
        daily = add_rolling_means(daily)

    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    dec_year = daily["DecYear"].to_numpy(dtype=float)
    q = as_float_array(daily["Q"])
    q7 = as_float_array(daily["Q7"])
    q30 = as_float_array(daily["Q30"])

    min_days = _MIN_DAYS_PER_MONTH * pa_long
    use_climate_year = (pa_long == 12) and (pa_start == 10)
    low_flow_start = _CLIMATE_YEAR_START if use_climate_year else pa_start

    records: dict[int, dict[float, float]] = {i: {} for i in ISTAT_NAMES}

    def _collect(start_month: int, istats, reducer):
        for start, end in _period_windows(daily, start_month, pa_long):
            in_window = (month_seq >= start) & (month_seq <= end)
            if int(np.sum(np.isfinite(q[in_window]))) <= min_days:
                continue
            centre = float(np.nanmean(dec_year[in_window]))
            for istat in istats:
                records[istat][centre] = reducer(istat, in_window)

    def _low(istat, mask):
        series = {1: q, 2: q7, 3: q30}[istat][mask]
        return float(np.nanmin(series)) if np.any(np.isfinite(series)) else np.nan

    def _high(istat, mask):
        if istat == 4:
            return float(np.nanmedian(q[mask]))
        if istat == 5:
            return float(np.nanmean(q[mask]))
        series = {6: q30, 7: q7, 8: q}[istat][mask]
        return float(np.nanmax(series)) if np.any(np.isfinite(series)) else np.nan

    _collect(low_flow_start, _LOW_FLOW_ISTATS, _low)
    _collect(pa_start, (4, 5, 6, 7, 8), _high)

    frames = []
    for istat, by_year in records.items():
        if not by_year:
            continue
        centres = np.array(sorted(by_year))
        values = np.array([by_year[c] for c in centres], dtype=float)
        frames.append(pd.DataFrame({
            "istat": istat,
            "statistic": ISTAT_NAMES[istat],
            "Year": np.asarray(np.round(centres), dtype=int),
            "DecYear": centres,
            "value": values,
            "smoothed": _smooth_log_series(centres, values, window, edge_adjust),
        }))

    if not frames:
        return pd.DataFrame(
            columns=["istat", "statistic", "Year", "DecYear", "value", "smoothed"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["istat", "DecYear"])
    result = result.reset_index(drop=True)
    return result


def flow_duration(
    elist: dict,
    center_date: str = "09-30",
    span: int = 365,
    verbose: bool = False,
) -> pd.Series:
    """Flow duration curve, optionally restricted to a season.

    Mirrors R ``flowDuration()``. Returns the nine quantiles EGRET reports.

    Args:
        elist: eList dict with a Daily DataFrame carrying the ``Day`` column.
        center_date: Centre of the seasonal window, ``"MM-DD"``.
            Default ``"09-30"``.
        span: Half-width of the window in days. Anything over 182 means the
            whole year, which is the default.
        verbose: Print the result. Default False.

    Returns:
        Series of discharge (m3/s) indexed by
        ``min, 5%, 10%, 25%, 50%, 75%, 90%, 95%, max``.

    Raises:
        ValueError: If Daily is missing or has no ``Day`` column.
    """
    daily = elist["Daily"]
    if daily is None or daily.empty:
        raise ValueError("eList has no Daily data.")
    if "Day" not in daily.columns:
        raise ValueError("Daily is missing 'Day'; run populate_date_columns first.")

    half = min(span, 182)
    centre = pd.Timestamp(f"2001-{center_date}").day_of_year

    offsets = np.arange(centre - half, centre + half + 1)
    good_days = np.where(offsets > 0, offsets, offsets + 365)
    good_days = np.where(good_days < 366, good_days, good_days - 365)

    in_season = daily["Day"].isin(set(good_days.tolist()))
    q = as_float_array(daily.loc[in_season, "Q"])
    q = np.sort(q[np.isfinite(q)])
    if q.size == 0:
        raise ValueError("No discharge values fall inside the requested window.")

    n = q.size
    # EGRET indexes the sorted values directly rather than interpolating.
    positions = [
        0,
        int(np.floor(0.05 * n)) - 1, int(np.floor(0.10 * n)) - 1,
        int(np.floor(0.25 * n)) - 1, int(np.floor(0.50 * n)) - 1,
        int(np.ceil(0.75 * n)) - 1, int(np.ceil(0.90 * n)) - 1,
        int(np.ceil(0.95 * n)) - 1,
        n - 1,
    ]
    positions = [int(np.clip(p, 0, n - 1)) for p in positions]
    labels = ["min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max"]
    duration = pd.Series(q[positions], index=labels, name="Q")

    if verbose:
        print(f"Flow duration, {'full year' if span > 182 else f'centred on {center_date}'}"
              f" ({n} days)")
        print(duration.to_string(float_format=lambda v: f"{v:.4g}"))

    return duration


def cum_q_date(
    elist: dict,
    pa_start: int = 10,
    pa_long: int = 12,
    fract: float = 0.5,
) -> pd.DataFrame:
    """The day each year by which a given fraction of runoff has passed.

    Mirrors R ``cumQdate()``. This is the standard measure of annual
    hydrograph timing: in a snowmelt-dominated basin a warming climate moves
    the half-runoff date earlier, and the trend in that date is a cleaner
    climate signal than the trend in annual volume, which can stay flat while
    the timing shifts by weeks.

    Restricting the period of analysis to the melt season - say
    ``pa_start=3, pa_long=4`` for March through June - asks the same question
    of the melt alone.

    Args:
        elist: eList dict with a Daily DataFrame.
        pa_start: First month of the period of analysis. Default 10.
        pa_long: Length of the period in months. Default 12.
        fract: Fraction of cumulative runoff to locate. Default 0.5.

    Returns:
        DataFrame with one row per year that has sufficient data:

        - ``Year`` - the year the period ends in
        - ``Day`` - EGRET's leap-adjusted calendar day of year
        - ``DayOfPeriod`` - days counted from the start of the period

        ``Day`` is what EGRET reports, and it wraps: for the default water
        year, 25 % of runoff might pass on day 345 (mid-December) and 75 % on
        day 97 (early April), which is chronological but not numerically
        ordered. ``DayOfPeriod`` counts from the first day of the period
        instead, so it can be sorted, differenced and regressed directly.

    Raises:
        ValueError: If Daily is missing, or ``fract`` is outside (0, 1).
    """
    if not 0.0 < fract < 1.0:
        raise ValueError(f"fract must be between 0 and 1, got {fract}.")

    daily = elist["Daily"]
    if daily is None or daily.empty:
        raise ValueError("eList has no Daily data.")

    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    dec_year = daily["DecYear"].to_numpy(dtype=float)
    day = daily["Day"].to_numpy(dtype=int)
    q = as_float_array(daily["Q"])

    min_days = _MIN_DAYS_PER_MONTH * pa_long
    rows = []

    for start, end in _period_windows(daily, pa_start, pa_long):
        mask = (month_seq >= start) & (month_seq <= end)
        q_year = q[mask]
        n_good = int(np.sum(np.isfinite(q_year)))
        if n_good <= min_days:
            continue

        cumulative = np.nancumsum(q_year)
        threshold = fract * cumulative[-1]
        crossing = np.flatnonzero(cumulative > threshold)
        if crossing.size == 0:
            continue

        rows.append({
            "Year": int(np.trunc(dec_year[mask][-1])),
            "Day": int(day[mask][crossing[0]]),
            "DayOfPeriod": int(crossing[0]) + 1,
        })

    return pd.DataFrame(rows, columns=["Year", "Day", "DayOfPeriod"])
